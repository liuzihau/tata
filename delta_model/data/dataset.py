"""Training dataset for the tata delta model.


One example = one (sample, block_idx, i_ref, i_target) tuple, with
0 ≤ i_ref < i_target < n_passes_actual ≤ MAX_ITER. We pre-build the index
at construction time so __getitem__ is a small slice off an in-memory
dict (preload mode, default) or an LRU-cached load (preload=False).

Storage layouts supported:
  • Per-sample files (legacy):   `cache_root/<split>/sample_<hex>.pt`
  • Multi-sample shards (§1.4):  `cache_root/<shard_subdir>/<split>/shard_NNNNN.pt`
                                  with `cache_root/shards_manifest.json` describing
                                  which shard each sample lives in.
The dataset auto-detects shard mode via the presence of `shards_manifest.json`.

Preload mode (default, `preload=True`):
  Load every sample/shard the dataset needs at __init__ and pin it in RAM.
  __getitem__ becomes a pure dict-lookup + tensor slice — no disk I/O on
  the hot path. With ~5000 train samples × ~16 MiB each ≈ ~80 GiB RAM.
  This is the right choice on big-RAM nodes (DGX, 256+ GiB) and was added
  to defeat the random-shuffle locality miss that made the LRU useless.

LRU mode (`preload=False`):
  Falls back to a small bounded cache. Useful for memory-constrained
  environments, sequential samplers, or huge caches that don't fit in RAM.

Yields tensors that the VariantC model + composite_loss expect:
    h_ref              [BLOCK_LENGTH, d_model]    fp16
    h_target           [BLOCK_LENGTH, d_model]    fp16
    prefix_kv          [2, n_kv_heads, PREFIX_WINDOW, d_head]  fp16
    prefix_kv_pad_mask [PREFIX_WINDOW]            bool   True=real, False=front-padded
    substituted_ids    [BLOCK_LENGTH]             int64  (caller does GPU embed lookup)
    mask_tgt           [BLOCK_LENGTH]             bool
    block_start_pos    int                        absolute block start in original seq
    i_ref / i_target / reveal_frac                scalars (for binning)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from . import schema as S


# Bounded-LRU sizes used only when `preload=False`.
_PER_SAMPLE_LRU_MAX = 32      # ~512 MiB cache in per-sample mode
_SHARD_LRU_MAX      = 4       # ~3.2 GiB at shard_size=50; tune to RAM budget


class TataDeltaDataset(Dataset):
    def __init__(
        self,
        cache_root: Path | str,
        *,
        split: str = "train",
        mask_token_id: int = S.LLADA_MASK_TOKEN_ID,
        index_filter: Optional[Callable[[tuple], bool]] = None,
        preload: bool = True,
        shard_lru_max: Optional[int] = None,
        max_prompts: Optional[int] = None,
    ):
        cache_root = Path(cache_root)
        self.cache_root    = cache_root
        self.mask_token_id = mask_token_id
        self.preload       = preload
        # Override for the shard-mode LRU cap. `InterleavedShardSampler`
        # keeps `active_shards` shards resident at once — the LRU must hold
        # all of them or it thrashes. `None` → `_SHARD_LRU_MAX` default.
        self._shard_lru_max_override = shard_lru_max
        # Cap the number of prompts in `split` to the first `max_prompts`
        # by load order (sorted sample paths in per-sample mode; manifest
        # order in shard mode). Applied BEFORE shard loading so a 5k slice
        # of a 20k cache only loads ~5k of shards into preload RAM.
        # `None` (or 0) keeps the entire split.
        self._max_prompts = (
            int(max_prompts) if max_prompts and int(max_prompts) > 0 else None
        )

        shards_manifest = cache_root / "shards_manifest.json"
        if shards_manifest.exists():
            self._init_shard_mode(shards_manifest, split, index_filter)
        else:
            self._init_per_sample_mode(cache_root, split, index_filter)

    # ------------------------------------------------------------------
    # Per-sample mode
    # ------------------------------------------------------------------

    def _init_per_sample_mode(
        self, cache_root: Path, split: str,
        index_filter: Optional[Callable[[tuple], bool]],
    ) -> None:
        self.mode = "per_sample"
        manifest_path = cache_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            sample_paths = sorted(
                cache_root / r["filename"]
                for r in manifest["records"]
                if r["split"] == split and (cache_root / r["filename"]).exists()
            )
        else:
            sample_paths = sorted((cache_root / split).glob("sample_*.pt"))

        if not sample_paths:
            raise FileNotFoundError(
                f"No cached samples found at {cache_root}/{split}. "
                f"Run collect_llada first."
            )
        if self._max_prompts is not None:
            sample_paths = sample_paths[: self._max_prompts]
        self.sample_paths = sample_paths

        # Pre-build the (sample, block, i_ref, i_target) flat index. We have
        # to read each file once anyway to learn n_passes_actual; in preload
        # mode we keep the loaded dict if it contributed any indices.
        self.index: list[tuple[int, int, int, int]] = []
        self._sample_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._lru_max = _PER_SAMPLE_LRU_MAX

        kept = 0
        for s_idx, p in enumerate(sample_paths):
            sample = torch.load(p, map_location="cpu", weights_only=False)
            n_added = self._extend_index_from_sample(s_idx, sample, index_filter)
            if self.preload and n_added > 0:
                self._sample_cache[s_idx] = sample
                kept += 1
            # else: sample drops out of scope; GC reclaims it.

        if self.preload:
            est_mib = kept * 16  # per-sample files are ~16 MiB each
            print(
                f"[dataset] preloaded {kept}/{len(sample_paths)} samples "
                f"into RAM (~{est_mib} MiB) for split={split!r}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Shard mode
    # ------------------------------------------------------------------

    def _init_shard_mode(
        self, shards_manifest_path: Path, split: str,
        index_filter: Optional[Callable[[tuple], bool]],
    ) -> None:
        self.mode = "shard"
        manifest = json.loads(shards_manifest_path.read_text())
        shards_for_split = [r for r in manifest["shards"] if r["split"] == split]
        if not shards_for_split:
            raise FileNotFoundError(
                f"No shards for split={split!r} in {shards_manifest_path}."
            )

        self.shard_paths: list[Path] = []
        self.sample_locator: list[tuple[int, int]] = []
        for r in shards_for_split:
            shard_path = self.cache_root / r["filename"]
            if not shard_path.exists():
                raise FileNotFoundError(f"shard missing: {shard_path}")
            shard_local_idx = len(self.shard_paths)
            self.shard_paths.append(shard_path)
            for within in range(r["n_samples"]):
                self.sample_locator.append((shard_local_idx, within))

        # Cap to the first `max_prompts` prompts (by manifest order). Slicing
        # `sample_locator` keeps `_get_shard()` from touching shards that
        # carry only excluded prompts — preload RAM scales with the kept N,
        # not the full split size. `shard_paths` is left intact because
        # nothing else dereferences out-of-range shard indices once the
        # iteration below skips them.
        if self._max_prompts is not None:
            self.sample_locator = self.sample_locator[: self._max_prompts]

        self._shard_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._lru_max = (
            int(self._shard_lru_max_override)
            if self._shard_lru_max_override is not None
            else _SHARD_LRU_MAX
        )

        # First pass: build flat index across all samples in this split.
        # In preload mode we load each shard exactly once and keep it; the
        # LRU eviction is bypassed by the unbounded `_shard_cache` we pin.
        self.index: list[tuple[int, int, int, int]] = []
        # Per-shard count of indices kept; shards with zero contribution
        # are evicted to save RAM.
        per_shard_kept: dict[int, int] = {}
        for s_idx, (shard_idx, within) in enumerate(self.sample_locator):
            shard = self._get_shard(shard_idx, force_pin=self.preload)
            sample = shard["samples"][within]
            n_added = self._extend_index_from_sample(s_idx, sample, index_filter)
            per_shard_kept[shard_idx] = per_shard_kept.get(shard_idx, 0) + n_added

        if self.preload:
            for shard_idx, n in list(per_shard_kept.items()):
                if n == 0:
                    self._shard_cache.pop(shard_idx, None)
            kept_shards = len(self._shard_cache)
            est_mib = sum(
                self.shard_paths[i].stat().st_size
                for i in self._shard_cache
            ) / (1024 * 1024)
            print(
                f"[dataset] preloaded {kept_shards}/{len(self.shard_paths)} "
                f"shards into RAM (~{est_mib:.0f} MiB) for split={split!r}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _extend_index_from_sample(
        self, s_idx: int, sample: dict,
        index_filter: Optional[Callable[[tuple], bool]],
    ) -> int:
        n_added = 0
        # Read the block count from the sample, not `S.NUM_BLOCKS` — a cache
        # thinned by `data/thin_cache.py` keeps fewer blocks (e.g. 7) while
        # collect / inference still operate on the full 8. `n_passes_actual`
        # already bounds the per-block iteration enumeration, so a thinned
        # cache loads transparently.
        for b in range(len(sample["blocks"])):
            n_actual = int(sample["blocks"][b].get("n_passes_actual", S.MAX_ITER))
            if n_actual < 2:
                continue
            for i_ref in range(n_actual - 1):
                for i_tgt in range(i_ref + 1, n_actual):
                    tup = (s_idx, b, i_ref, i_tgt)
                    if index_filter is None or index_filter(tup):
                        self.index.append(tup)
                        n_added += 1
        return n_added

    def __len__(self) -> int:
        return len(self.index)

    def _get_sample(self, s_idx: int) -> dict:
        if self.mode == "per_sample":
            return self._get_sample_per_sample(s_idx)
        else:
            shard_idx, within = self.sample_locator[s_idx]
            shard = self._get_shard(shard_idx, force_pin=False)
            return shard["samples"][within]

    def _get_sample_per_sample(self, s_idx: int) -> dict:
        cache = self._sample_cache
        if s_idx in cache:
            if not self.preload:
                cache.move_to_end(s_idx)
            return cache[s_idx]
        if self.preload:
            # Should not happen: every sample contributing to self.index was
            # pinned at __init__. If we ever get here it means index/cache
            # are out of sync.
            raise KeyError(
                f"sample s_idx={s_idx} not preloaded — index/cache out of sync"
            )
        sample = torch.load(
            self.sample_paths[s_idx], map_location="cpu", weights_only=False,
        )
        cache[s_idx] = sample
        if len(cache) > self._lru_max:
            cache.popitem(last=False)
        return sample

    def _get_shard(self, shard_idx: int, *, force_pin: bool) -> dict:
        cache = self._shard_cache
        if shard_idx in cache:
            if not (self.preload or force_pin):
                cache.move_to_end(shard_idx)
            return cache[shard_idx]
        shard = torch.load(
            self.shard_paths[shard_idx], map_location="cpu", weights_only=False,
        )
        cache[shard_idx] = shard
        if not (self.preload or force_pin) and len(cache) > self._lru_max:
            cache.popitem(last=False)
        return shard

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        s_idx, b, i_ref, i_tgt = self.index[idx]
        sample = self._get_sample(s_idx)
        block = sample["blocks"][b]

        # Stay in fp16 — caller casts to model dtype (bf16) on GPU.
        h_ref      = block["h_per_pass"][i_ref]
        h_target   = block["h_per_pass"][i_tgt]
        # `prefix_kv` is stored at COLLECT_PREFIX_WINDOW (default 64); slice
        # down to the last `PREFIX_WINDOW` (32) for what training/inference
        # actually uses. This lets us keep collect-time headroom without
        # re-collecting if we later want to ablate larger windows.
        prefix_kv_full = block["prefix_kv"]                    # [2, KV_H, W, d_head], W ≥ PREFIX_WINDOW
        prefix_kv      = prefix_kv_full[:, :, -S.PREFIX_WINDOW:, :]   # [2, KV_H, 32, d_head]
        # Pad mask: True = real prefix slot, False = front-padded zero.
        # Older v2 caches (built before pad-and-mask) won't have this field;
        # default to all-True since those caches always had full prefixes.
        mask_full = block.get("prefix_kv_pad_mask")
        if mask_full is None:
            prefix_kv_pad_mask = torch.ones(S.PREFIX_WINDOW, dtype=torch.bool)
        else:
            prefix_kv_pad_mask = mask_full[-S.PREFIX_WINDOW:]
        reveal_tgt = block["reveal_per_pass"][i_tgt]
        mask_tgt   = ~reveal_tgt

        gen_ids   = sample["generated_token_ids"]
        block_ids = gen_ids[b * S.BLOCK_LENGTH : (b + 1) * S.BLOCK_LENGTH]
        substituted_ids = torch.where(
            reveal_tgt, block_ids,
            torch.full_like(block_ids, self.mask_token_id),
        )

        block_start_pos = int(sample["prompt_len"]) + b * S.BLOCK_LENGTH

        return {
            "h_ref":              h_ref,
            "h_target":           h_target,
            "prefix_kv":          prefix_kv,
            "prefix_kv_pad_mask": prefix_kv_pad_mask,
            "substituted_ids":    substituted_ids,
            "mask_tgt":           mask_tgt,
            "block_start_pos":    block_start_pos,
            "i_ref":              i_ref,
            "i_target":           i_tgt,
            "reveal_frac":        float(reveal_tgt.float().mean().item()),
        }


    # ------------------------------------------------------------------
    # Shard groupings for locality-aware sampling
    # ------------------------------------------------------------------

    def get_shard_groups(self) -> "dict[int, list[int]]":
        """For shard-mode datasets: return `{shard_id: [k1, k2, ...]}`
        where each `ki` is an index into `self.index` (the flat list of
        `(s_idx, b, i_ref, i_target)` tuples).

        Used by `BlockShardSampler` to keep consecutive accesses inside the
        same shard, so the shard LRU cache stays warm. Raises if called on
        a per-sample dataset (no shard structure to group by).
        """
        if self.mode != "shard":
            raise RuntimeError(
                "get_shard_groups() requires shard-mode dataset; "
                f"this dataset is in {self.mode!r} mode. "
                "Run `delta_model.data.repack` first."
            )
        groups: "dict[int, list[int]]" = {}
        for k, (s_idx, _b, _ir, _it) in enumerate(self.index):
            shard_id = self.sample_locator[s_idx][0]
            groups.setdefault(shard_id, []).append(k)
        return groups

    # ------------------------------------------------------------------
    # Sampling weights (T2; engineering.md §3.4)
    # ------------------------------------------------------------------

    def compute_index_weights(
        self, *, ref0_weight_multiplier: float,
    ) -> torch.Tensor:
        """Per-index sampling weight for use with WeightedRandomSampler.

        Pairs with `i_ref == 0` get weight `ref0_weight_multiplier`; all
        others get `1.0`. The standard `WeightedRandomSampler` normalizes
        these to a probability distribution, so absolute scale doesn't
        matter — only the ratio.

        Why: at inference, `h_ref` is captured at pass 0 and refreshed
        only via rollback, so ~80–95% of delta forwards have `i_ref = 0`.
        At training, only 5/15 = 33% of enumerated pairs have `i_ref = 0`.
        This sampler closes that train-inference distribution mismatch.

        Returns `torch.Tensor[float32]` of shape `(len(self.index),)`.
        """
        if ref0_weight_multiplier <= 0:
            raise ValueError(
                "ref0_weight_multiplier must be > 0; "
                f"got {ref0_weight_multiplier!r}"
            )
        weights = torch.ones(len(self.index), dtype=torch.float32)
        if ref0_weight_multiplier == 1.0:
            return weights
        for k, (_s, _b, i_ref, _i_tgt) in enumerate(self.index):
            if i_ref == 0:
                weights[k] = float(ref0_weight_multiplier)
        return weights


class BlockShardSampler(torch.utils.data.Sampler):
    """Locality-aware weighted sampler for shard-mode datasets.

    Why it exists: at 20000-sample scale (T5), the cache is ~400 GB on disk
    and we can't `preload=True` it into RAM. The dataset's shard LRU
    (default 4 shards ≈ 4 GB) keeps memory bounded, but a *random* shuffle
    forces ~B distinct shard loads per batch (cache miss rate ≈ 100%).
    This sampler keeps consecutive accesses inside one shard for
    `samples_per_shard_visit` indices before moving on, dropping the I/O
    cost from "1 shard load per batch" to "1 shard load per
    (samples_per_shard_visit // batch_size) batches".

    Weighting (T2): per-index weights from
    `TataDeltaDataset.compute_index_weights(...)` are pre-sliced per shard
    and used inside each visit via `torch.multinomial(replacement=True)`.
    Expected sample frequency across all shards stays proportional to
    weights (each shard is visited the same number of times per round).

    Each "round":
      1. Permute the shard order (deterministic if `generator` is set).
      2. For each shard, draw `samples_per_shard_visit` indices with
         replacement, weighted by that shard's slice of `weights`.
      3. Yield each index.
    Rounds continue until `num_samples` indices have been emitted.
    """

    def __init__(
        self,
        shard_groups: "dict[int, list[int]]",
        weights: torch.Tensor,
        *,
        samples_per_shard_visit: int,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__(None)
        if not shard_groups:
            raise ValueError("BlockShardSampler: shard_groups is empty")
        if samples_per_shard_visit <= 0:
            raise ValueError(
                f"samples_per_shard_visit must be > 0; got {samples_per_shard_visit}"
            )

        # Materialize shard slices in a stable order so reruns with the
        # same generator seed produce the same access pattern.
        self.shard_ids: list[int] = sorted(shard_groups.keys())
        self._shard_indices: list[torch.Tensor] = []
        self._shard_weights: list[torch.Tensor] = []
        for sid in self.shard_ids:
            idxs = torch.as_tensor(shard_groups[sid], dtype=torch.long)
            self._shard_indices.append(idxs)
            self._shard_weights.append(weights[idxs].clone().float())

        self.samples_per_shard_visit = int(samples_per_shard_visit)
        if num_samples is None:
            # Default: one round = each shard visited once.
            num_samples = len(self.shard_ids) * self.samples_per_shard_visit
        self.num_samples = int(num_samples)
        self.generator = generator

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        n_yielded = 0
        n_shards  = len(self.shard_ids)
        while n_yielded < self.num_samples:
            perm = torch.randperm(n_shards, generator=self.generator)
            for shard_pos in perm.tolist():
                idxs = self._shard_indices[shard_pos]
                w    = self._shard_weights[shard_pos]
                draw = torch.multinomial(
                    w, self.samples_per_shard_visit,
                    replacement=True, generator=self.generator,
                )
                # Use indexing + tolist to avoid a Python loop over each draw.
                picked = idxs[draw].tolist()
                for k in picked:
                    yield k
                    n_yielded += 1
                    if n_yielded >= self.num_samples:
                        return


class InterleavedShardSampler(torch.utils.data.Sampler):
    """Diversity-preserving weighted sampler for shard-mode datasets.

    Why it replaces `BlockShardSampler`: BlockShardSampler kept an entire
    batch inside one shard (~shard_size prompts), producing correlated,
    high-variance gradients. The three-way M1.5 comparison made the cost
    explicit — same recipe, same ~11.8k-prompt cache, only the loader
    differing:

        preload + WeightedRandomSampler   val/kl 0.280   train/kl 0.15
        BlockShardSampler (spsv 512)      val/kl 0.332   train/kl 0.21

    This sampler keeps a *window* of `active_shards` shards resident at
    once and interleaves draws across all of them, so every batch the
    DataLoader cuts spans ~`active_shards · shard_size` prompts instead of
    one shard's worth — recovering random-sampler gradient quality while
    the in-RAM working set stays bounded at `active_shards` shards (set the
    dataset's `shard_lru_max >= active_shards`).

    I/O is the same as BlockShardSampler — each shard is still loaded once
    per round; only the working set grows from 1 to `active_shards` shards.

    Weighting (T2): per-index weights from
    `TataDeltaDataset.compute_index_weights(...)` are pooled across the
    window and used inside `torch.multinomial(replacement=True)`, so the
    i_ref bias is preserved.

    Each round:
      1. Permute the shard order.
      2. Slide a non-overlapping window of `active_shards` shards across
         the permutation.
      3. For each window, pool the indices + weights of its shards, draw
         `chunk_size` indices (weighted, with replacement), shuffle them,
         and yield. The shuffle is what mixes the shards within every
         batch the DataLoader subsequently cuts.
    Rounds repeat until `num_samples` indices have been emitted.
    """

    def __init__(
        self,
        shard_groups: "dict[int, list[int]]",
        weights: torch.Tensor,
        *,
        active_shards: int,
        chunk_size: int | None = None,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__(None)
        if not shard_groups:
            raise ValueError("InterleavedShardSampler: shard_groups is empty")
        if active_shards < 1:
            raise ValueError(f"active_shards must be >= 1; got {active_shards}")

        self.shard_ids: list[int] = sorted(shard_groups.keys())
        self._shard_indices: list[torch.Tensor] = []
        self._shard_weights: list[torch.Tensor] = []
        for sid in self.shard_ids:
            idxs = torch.as_tensor(shard_groups[sid], dtype=torch.long)
            self._shard_indices.append(idxs)
            self._shard_weights.append(weights[idxs].clone().float())

        n_shards = len(self.shard_ids)
        self.active_shards = min(int(active_shards), n_shards)

        total_pairs = sum(int(t.numel()) for t in self._shard_indices)
        # Auto chunk_size: one window emits ≈ the number of pairs it holds,
        # so one full round ≈ one epoch over the dataset.
        if chunk_size is None:
            mean_pairs_per_shard = max(1, round(total_pairs / n_shards))
            chunk_size = self.active_shards * mean_pairs_per_shard
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
        self.chunk_size = int(chunk_size)

        self.num_samples = (
            int(num_samples) if num_samples is not None else total_pairs
        )
        self.generator = generator

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        n_yielded = 0
        n_shards  = len(self.shard_ids)
        while n_yielded < self.num_samples:
            perm = torch.randperm(n_shards, generator=self.generator).tolist()
            for w_start in range(0, n_shards, self.active_shards):
                window = perm[w_start : w_start + self.active_shards]
                pool_idx = torch.cat([self._shard_indices[s] for s in window])
                pool_w   = torch.cat([self._shard_weights[s] for s in window])
                draw = torch.multinomial(
                    pool_w, self.chunk_size,
                    replacement=True, generator=self.generator,
                )
                # Shuffle the chunk so the DataLoader's batch boundaries mix
                # all `active_shards` shards instead of running shard-by-shard.
                picked = pool_idx[draw]
                picked = picked[torch.randperm(
                    picked.numel(), generator=self.generator,
                )]
                for k in picked.tolist():
                    yield k
                    n_yielded += 1
                    if n_yielded >= self.num_samples:
                        return


def make_train_val_filter(val_frac: float, seed: int = 42) -> tuple[Callable, Callable]:
    """Return (train_filter, val_filter) that partition by sample index hash.

    Stable across runs (seed-only), and within-sample so all of one prompt's
    pairs go to the same side — keeps the val held-out at the prompt level.
    """
    import random
    rng = random.Random(seed)
    cache: dict[int, bool] = {}

    def is_val(s_idx: int) -> bool:
        if s_idx not in cache:
            cache[s_idx] = rng.random() < val_frac
        return cache[s_idx]

    def train_filter(t: tuple) -> bool:
        return not is_val(t[0])

    def val_filter(t: tuple) -> bool:
        return is_val(t[0])

    return train_filter, val_filter

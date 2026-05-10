"""Training dataset for the tata delta model.

One example = one (sample, block_idx, i_ref, i_target) tuple, with
0 ≤ i_ref < i_target < n_passes_actual ≤ MAX_ITER. We pre-build the index
at construction time so __getitem__ is a small slice off a per-worker
LRU-cached sample (or shard, in shard mode).

Storage layouts supported:
  • Per-sample files (legacy):   `cache_root/<split>/sample_<hex>.pt`
  • Multi-sample shards (§1.4):  `cache_root/<shard_subdir>/<split>/shard_NNNNN.pt`
                                  with `cache_root/shards_manifest.json` describing
                                  which shard each sample lives in.
The dataset auto-detects shard mode via the presence of `shards_manifest.json`
and switches the LRU to cache shards (larger entries, fewer of them).

Yields tensors that the VariantC model + composite_loss expect:
    h_ref           [BLOCK_LENGTH, d_model]    fp16
    h_target        [BLOCK_LENGTH, d_model]    fp16
    prefix_kv       [2, n_kv_heads, PREFIX_WINDOW, d_head]  fp16
    substituted_ids [BLOCK_LENGTH]             int64  (caller does GPU embed lookup)
    mask_tgt        [BLOCK_LENGTH]             bool
    block_start_pos int                        absolute block start in original seq
    i_ref / i_target / reveal_frac             scalars (for binning)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from . import schema as S


# Per-worker LRU sizes. In per-sample mode each cache entry is ~16 MiB; in
# shard mode each entry is ~`shard_size * 16` MiB (so a much smaller LRU).
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
    ):
        cache_root = Path(cache_root)
        self.cache_root    = cache_root
        self.mask_token_id = mask_token_id

        shards_manifest = cache_root / "shards_manifest.json"
        if shards_manifest.exists():
            self._init_shard_mode(shards_manifest, split, index_filter)
        else:
            self._init_per_sample_mode(cache_root, split, index_filter)

    # ------------------------------------------------------------------
    # Per-sample mode (legacy, one .pt per prompt)
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
        self.sample_paths = sample_paths

        # Pre-build the (sample, block, i_ref, i_target) flat index. We
        # need n_passes_actual per block, so we have to read each file once.
        self.index: list[tuple[int, int, int, int]] = []
        for s_idx, p in enumerate(sample_paths):
            sample = torch.load(p, map_location="cpu", weights_only=False)
            self._extend_index_from_sample(
                s_idx, sample, index_filter,
            )

        # Sample LRU.
        self._sample_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._lru_max = _PER_SAMPLE_LRU_MAX

    # ------------------------------------------------------------------
    # Shard mode (multi-sample shards from repack.py)
    # ------------------------------------------------------------------

    def _init_shard_mode(
        self, shards_manifest_path: Path, split: str,
        index_filter: Optional[Callable[[tuple], bool]],
    ) -> None:
        self.mode = "shard"
        manifest = json.loads(shards_manifest_path.read_text())
        # `manifest["shards"]` is a flat list across all splits; filter to ours.
        shards_for_split = [r for r in manifest["shards"] if r["split"] == split]
        if not shards_for_split:
            raise FileNotFoundError(
                f"No shards for split={split!r} in {shards_manifest_path}."
            )

        # Map shard manifest entries to absolute paths.
        self.shard_paths: list[Path] = []
        # Per-(shard_idx, within_idx) → flat sample_idx for the index.
        # `self.sample_locator[s_idx] = (shard_idx_in_self.shard_paths, within_idx)`.
        self.sample_locator: list[tuple[int, int]] = []
        for r in shards_for_split:
            shard_path = self.cache_root / r["filename"]
            if not shard_path.exists():
                raise FileNotFoundError(f"shard missing: {shard_path}")
            shard_local_idx = len(self.shard_paths)
            self.shard_paths.append(shard_path)
            for within in range(r["n_samples"]):
                self.sample_locator.append((shard_local_idx, within))

        # Build the flat index by walking each sample once. We have to load
        # each shard once for n_passes_actual; reuse the LRU during this pass
        # so we don't load the same shard twice.
        self._shard_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._lru_max = _SHARD_LRU_MAX

        self.index: list[tuple[int, int, int, int]] = []
        for s_idx, (shard_idx, within) in enumerate(self.sample_locator):
            shard = self._get_shard(shard_idx)
            sample = shard["samples"][within]
            self._extend_index_from_sample(s_idx, sample, index_filter)

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _extend_index_from_sample(
        self, s_idx: int, sample: dict,
        index_filter: Optional[Callable[[tuple], bool]],
    ) -> None:
        for b in range(S.NUM_BLOCKS):
            n_actual = int(sample["blocks"][b].get("n_passes_actual", S.MAX_ITER))
            if n_actual < 2:
                continue
            for i_ref in range(n_actual - 1):
                for i_tgt in range(i_ref + 1, n_actual):
                    tup = (s_idx, b, i_ref, i_tgt)
                    if index_filter is None or index_filter(tup):
                        self.index.append(tup)

    def __len__(self) -> int:
        return len(self.index)

    def _get_sample(self, s_idx: int) -> dict:
        if self.mode == "per_sample":
            return self._get_sample_per_sample(s_idx)
        else:
            shard_idx, within = self.sample_locator[s_idx]
            shard = self._get_shard(shard_idx)
            return shard["samples"][within]

    def _get_sample_per_sample(self, s_idx: int) -> dict:
        cache = self._sample_cache
        if s_idx in cache:
            cache.move_to_end(s_idx)
            return cache[s_idx]
        sample = torch.load(
            self.sample_paths[s_idx], map_location="cpu", weights_only=False,
        )
        cache[s_idx] = sample
        if len(cache) > self._lru_max:
            cache.popitem(last=False)
        return sample

    def _get_shard(self, shard_idx: int) -> dict:
        cache = self._shard_cache
        if shard_idx in cache:
            cache.move_to_end(shard_idx)
            return cache[shard_idx]
        shard = torch.load(
            self.shard_paths[shard_idx], map_location="cpu", weights_only=False,
        )
        cache[shard_idx] = shard
        if len(cache) > self._lru_max:
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
        prefix_kv  = block["prefix_kv_last32"]
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
            "h_ref":           h_ref,
            "h_target":        h_target,
            "prefix_kv":       prefix_kv,
            "substituted_ids": substituted_ids,
            "mask_tgt":        mask_tgt,
            "block_start_pos": block_start_pos,
            "i_ref":           i_ref,
            "i_target":        i_tgt,
            "reveal_frac":     float(reveal_tgt.float().mean().item()),
        }


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

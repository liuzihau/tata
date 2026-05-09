"""Training dataset for the tata delta model.

One example = one (sample_path, block_idx, i_ref, i_target) tuple, with
0 ≤ i_ref < i_target < n_passes_actual ≤ MAX_ITER. We pre-build the index
at construction time so __getitem__ is mmap-load + small slice.

Yields tensors that the VariantC model + composite_loss expect:
    h_ref       [BLOCK_LENGTH, d_model]   fp32
    h_target    [BLOCK_LENGTH, d_model]   fp32
    prev_emb    [BLOCK_LENGTH, d_model]   fp32
    prefix_kv   [2, n_kv_heads, PREFIX_WINDOW, d_head]  fp16  (dtype-preserved
                                                              for cross-attn)
    mask_tgt    [BLOCK_LENGTH]            bool
    i_ref / i_target / reveal_frac       scalars (for binning)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from peft_project.tata.delta_model.data import schema as S


class TataDeltaDataset(Dataset):
    def __init__(
        self,
        cache_root: Path | str,
        token_embed: nn.Embedding,
        *,
        split: str = "train",
        mask_token_id: int = S.LLADA_MASK_TOKEN_ID,
        index_filter: Optional[Callable[[tuple], bool]] = None,
    ):
        cache_root = Path(cache_root)
        manifest_path = cache_root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self.sample_paths = sorted(
                cache_root / r["filename"]
                for r in manifest["records"]
                if r["split"] == split and (cache_root / r["filename"]).exists()
            )
        else:
            # Fallback: glob the split directory directly.
            self.sample_paths = sorted((cache_root / split).glob("sample_*.pt"))

        if not self.sample_paths:
            raise FileNotFoundError(
                f"No cached samples found at {cache_root}/{split}. Run collect_llada first."
            )

        self.token_embed   = token_embed
        self.mask_token_id = mask_token_id

        # Pre-build the (sample, block, i_ref, i_target) flat index.
        # Each sample's manifest tells us n_passes_actual per block; we read
        # the file once at index time to avoid sampling pad slots.
        self.index: list[tuple[int, int, int, int]] = []
        for s_idx, p in enumerate(self.sample_paths):
            sample = torch.load(p, map_location="cpu", weights_only=False)
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

    def __getitem__(self, idx: int) -> dict:
        s_idx, b, i_ref, i_tgt = self.index[idx]
        # mmap=True keeps memory low when many workers read in parallel.
        sample = torch.load(
            self.sample_paths[s_idx], map_location="cpu",
            mmap=True, weights_only=False,
        )
        block = sample["blocks"][b]

        h_ref    = block["h_per_pass"][i_ref].to(torch.float32)        # [32, d]
        h_target = block["h_per_pass"][i_tgt].to(torch.float32)         # [32, d]
        prefix_kv = block["prefix_kv_last32"]                            # [2, KV_H, 32, d_head] fp16
        reveal_tgt = block["reveal_per_pass"][i_tgt]                     # [32] bool
        mask_tgt   = ~reveal_tgt                                         # [32] bool

        # prev_emb = token-embed of revealed positions, mask-token-embed elsewhere.
        gen_ids = sample["generated_token_ids"]                          # [256]
        block_ids = gen_ids[b * S.BLOCK_LENGTH : (b + 1) * S.BLOCK_LENGTH]
        substituted_ids = torch.where(
            reveal_tgt, block_ids, torch.full_like(block_ids, self.mask_token_id),
        )
        with torch.no_grad():
            prev_emb = self.token_embed(substituted_ids).to(torch.float32)  # [32, d]

        return {
            "h_ref":       h_ref,
            "h_target":    h_target,
            "prev_emb":    prev_emb,
            "prefix_kv":   prefix_kv.clone(),       # decouple from mmap
            "mask_tgt":    mask_tgt,
            "i_ref":       i_ref,
            "i_target":    i_tgt,
            "reveal_frac": float(reveal_tgt.float().mean().item()),
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

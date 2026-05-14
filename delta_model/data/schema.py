"""Cache file format constants.

The cache directory contains one .pt file per sample. Every reader and
writer in this package agrees on the shapes / dtypes defined here — bump
SCHEMA_VERSION when changing the on-disk layout so old shards don't get
silently mis-read.

Two prefix-KV window sizes:
  • `COLLECT_PREFIX_WINDOW` — the window collect_llada records on disk.
    Default 64 so we have headroom for ablations without re-collecting.
  • `PREFIX_WINDOW` — what training/inference actually consumes (last
    `PREFIX_WINDOW` of the stored slice). Set to 32 for the M1 design.

Dataset slices the stored window down to `PREFIX_WINDOW` before returning,
so models see the same shape regardless of how much was recorded.
"""
from __future__ import annotations

import torch

# Bumped 1 → 2 when the prefix-KV cache field was renamed
# `prefix_kv_last32` → `prefix_kv` and the on-disk window grew from 32 to
# COLLECT_PREFIX_WINDOW. v1 caches will fail the schema-version check.
SCHEMA_VERSION   = 2

GEN_LENGTH            = 256
BLOCK_LENGTH          = 32
NUM_BLOCKS            = GEN_LENGTH // BLOCK_LENGTH        # 8
MAX_ITER              = 6
PREFIX_WINDOW         = 32                                # what training reads
COLLECT_PREFIX_WINDOW = 64                                # what collect stores

# LLaDA-8B (no GQA: n_kv_heads == n_heads).
N_KV_HEADS_LLADA = 32
D_HEAD_LLADA     = 128
D_MODEL_LLADA    = 4096
N_LAYERS_LLADA   = 32

LLADA_MASK_TOKEN_ID = 126336

DTYPE_KV     = torch.float16
DTYPE_HIDDEN = torch.float16


def expected_block_shapes(
    *, n_kv_heads: int, d_head: int, d_model: int,
    prefix_window: int = COLLECT_PREFIX_WINDOW,
    max_iter: int = MAX_ITER,
) -> dict:
    """Authoritative per-block tensor shapes. Used by collect + tests.

    Defaults to `COLLECT_PREFIX_WINDOW` for the prefix_kv field — that's
    what's actually written to disk. Pass `prefix_window=PREFIX_WINDOW` to
    validate post-slicing dataset output instead.

    `max_iter` defaults to the schema `MAX_ITER` (6). A cache thinned by
    `data/thin_cache.py` stores fewer iterations per block and records the
    real count in `meta["max_iter"]` — pass that here so T3 validates
    against the actual on-disk shape.
    """
    # `prefix_kv_pad_mask` (shape `(prefix_window,)`, bool) is also written
    # by collect_llada but is OPTIONAL for legacy caches and validated
    # separately in `test_collect_roundtrip` so older v2 caches (built
    # before pad-and-mask) don't fail T3.
    return {
        "prefix_kv":          (2, n_kv_heads, prefix_window, d_head),
        "h_per_pass":         (max_iter, BLOCK_LENGTH, d_model),
        "reveal_per_pass":    (max_iter, BLOCK_LENGTH),
    }

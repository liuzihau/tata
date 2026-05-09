"""Cache file format constants.

The cache directory contains one .pt file per sample. Every reader and
writer in this package agrees on the shapes / dtypes defined here — bump
SCHEMA_VERSION when changing the on-disk layout so old shards don't get
silently mis-read.
"""
from __future__ import annotations

import torch

SCHEMA_VERSION   = 1

GEN_LENGTH       = 256
BLOCK_LENGTH     = 32
NUM_BLOCKS       = GEN_LENGTH // BLOCK_LENGTH        # 8
MAX_ITER         = 6
PREFIX_WINDOW    = 32                                # last-32 prefix-KV slice

# LLaDA-8B (no GQA: n_kv_heads == n_heads).
N_KV_HEADS_LLADA = 32
D_HEAD_LLADA     = 128
D_MODEL_LLADA    = 4096
N_LAYERS_LLADA   = 32

LLADA_MASK_TOKEN_ID = 126336

DTYPE_KV     = torch.float16
DTYPE_HIDDEN = torch.float16


def expected_block_shapes(*, n_kv_heads: int, d_head: int, d_model: int) -> dict:
    """Authoritative per-block tensor shapes. Used by collect + tests."""
    return {
        "prefix_kv_last32": (2, n_kv_heads, PREFIX_WINDOW, d_head),
        "h_per_pass":       (MAX_ITER, BLOCK_LENGTH, d_model),
        "reveal_per_pass":  (MAX_ITER, BLOCK_LENGTH),
    }

"""Cache-format roundtrip tests.

Run after `collect_llada.py` produces at least one shard. Asserts:
  - on-disk schema version matches the constants in `data/schema.py`
  - per-block tensors have the documented shapes / dtypes
  - reveal pattern is monotone (a position never un-reveals)

Invocation (from inside the tata repo root):
    python -m delta_model.sanity.test_collect_roundtrip \\
        cache_v1/llada/test/sample_*.pt
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import torch

from ..data import schema as S


def check_one(path: Path) -> list[str]:
    """Return a list of error messages for one sample (empty if OK)."""
    errs: list[str] = []
    s = torch.load(path, map_location="cpu")

    meta = s.get("meta", {})
    if meta.get("schema_version") != S.SCHEMA_VERSION:
        errs.append(
            f"schema_version mismatch: got {meta.get('schema_version')!r}, "
            f"expected {S.SCHEMA_VERSION}"
        )

    if "blocks" not in s or len(s["blocks"]) != S.NUM_BLOCKS:
        errs.append(f"expected {S.NUM_BLOCKS} blocks, got {len(s.get('blocks', []))}")
        return errs

    n_kv_heads    = meta.get("n_kv_heads",    S.N_KV_HEADS_LLADA)
    d_head        = meta.get("d_head",        S.D_HEAD_LLADA)
    d_model       = meta.get("d_model",       S.D_MODEL_LLADA)
    prefix_window = meta.get("prefix_window", S.COLLECT_PREFIX_WINDOW)
    expected = S.expected_block_shapes(
        n_kv_heads=n_kv_heads, d_head=d_head, d_model=d_model,
        prefix_window=prefix_window,
    )

    for b_idx, b in enumerate(s["blocks"]):
        for key, exp_shape in expected.items():
            t = b.get(key)
            if t is None:
                errs.append(f"block {b_idx}: missing field '{key}'")
                continue
            if tuple(t.shape) != exp_shape:
                errs.append(
                    f"block {b_idx} '{key}': shape {tuple(t.shape)}, expected {exp_shape}"
                )

        # dtype checks
        if b["prefix_kv"].dtype != S.DTYPE_KV:
            errs.append(
                f"block {b_idx} prefix_kv dtype "
                f"{b['prefix_kv'].dtype}, expected {S.DTYPE_KV}"
            )
        # `prefix_kv_pad_mask` is optional for older v2 caches. If present,
        # must be bool of length `prefix_window`.
        m = b.get("prefix_kv_pad_mask")
        if m is not None:
            if m.dtype != torch.bool:
                errs.append(
                    f"block {b_idx} prefix_kv_pad_mask dtype "
                    f"{m.dtype}, expected torch.bool"
                )
            if tuple(m.shape) != (prefix_window,):
                errs.append(
                    f"block {b_idx} prefix_kv_pad_mask shape "
                    f"{tuple(m.shape)}, expected ({prefix_window},)"
                )
        if b["h_per_pass"].dtype != S.DTYPE_HIDDEN:
            errs.append(
                f"block {b_idx} h_per_pass dtype "
                f"{b['h_per_pass'].dtype}, expected {S.DTYPE_HIDDEN}"
            )
        if b["reveal_per_pass"].dtype != torch.bool:
            errs.append(
                f"block {b_idx} reveal_per_pass dtype "
                f"{b['reveal_per_pass'].dtype}, expected torch.bool"
            )

        # Monotone reveal: revealed positions stay revealed across passes.
        n_actual = b.get("n_passes_actual", S.MAX_ITER)
        rev = b["reveal_per_pass"][:n_actual]
        for i in range(rev.shape[0] - 1):
            if torch.any(rev[i] & ~rev[i + 1]):
                errs.append(
                    f"block {b_idx}: position un-revealed between pass {i} and {i+1}"
                )
                break

    if "prompt_token_ids" not in s or s["prompt_token_ids"].dtype != torch.long:
        errs.append("prompt_token_ids missing or wrong dtype")
    if "generated_token_ids" not in s or s["generated_token_ids"].shape != (S.GEN_LENGTH,):
        errs.append(
            f"generated_token_ids shape {s.get('generated_token_ids', torch.zeros(0)).shape}"
            f", expected ({S.GEN_LENGTH},)"
        )
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="cache .pt paths or globs")
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.paths:
        matches = sorted(glob(raw))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(raw))

    n_ok = n_bad = 0
    for path in paths:
        errs = check_one(path)
        if errs:
            n_bad += 1
            print(f"[BAD] {path}")
            for e in errs:
                print(f"      - {e}")
        else:
            n_ok += 1
            print(f"[OK ] {path}")

    print(f"\n{n_ok} OK, {n_bad} bad.")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()

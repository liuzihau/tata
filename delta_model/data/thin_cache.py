"""Thin an existing per-sample cache *in place* to reclaim disk.

The collected cache stores more than M1.5 training actually consumes:
  • all 8 generation blocks — but block 7 (the last) can be dropped from
    *training* without changing what inference generates;
  • 6 recorded iterations per block — but iteration 5 only contributes the
    highest-gap (i_target=5) pairs, the least useful ones (shared mass
    plateaus by i >= 3, per design.md);
  • a 64-wide prefix-KV window — but training and inference only ever read
    the last 32 (`S.PREFIX_WINDOW`); the extra 32 was ablation headroom.

This script rewrites each `sample_*.pt` in place (atomic per file,
resumable — already-thinned files are skipped) keeping only the first
`--keep_blocks` blocks, the first `--keep_iters` recorded iterations, and
the last `--prefix_window` prefix-KV slots. Each sample's `meta` (and the
cache `manifest.json`) gets a `thinned` provenance record.

It does NOT touch the backbone, collect, inference, or GSM8K eval — those
still operate on the full 8-block / 256-token generation. Only the
*training cache reader* is affected, and `dataset.py` reads the block
count from each sample (not `S.NUM_BLOCKS`), so a thinned cache loads
transparently. `test_collect_roundtrip` (T3) validates against the
`meta`-recorded counts.

By default ONLY the `train/` split is thinned — the `test/` split is left
at full resolution so validation (which reads `test/` when
`data.val_split=test`) and GSM8K eval score every run on an identical,
un-thinned held-out set. The comparison stays fair: thinning is the
independent variable, the measurement is held fixed. (Pass
`--splits train,test` to thin both anyway.) Note this means val sees
block-7 / gap-5 pairs a thinned-train model never trained on — that's
intentional, it surfaces the cost of thinning in `val/mse_by_gap_*`.

WHAT YOU LOSE (deliberate trade — review before running):
  • the delta model never trains on block-7 positions (highest RoPE
    positions) — inference still decodes block 7, just un-specialised;
  • the highest-gap training pairs (i_target=5) are gone — `bins_gap`
    above `keep_iters - 1` will be empty in val metrics.

Typical disk workflow (per-sample cache, no shards yet):

    python -m delta_model.data.thin_cache --cache_root cache_v1_20k/llada
    python -m delta_model.data.repack --cache_root cache_v1_20k/llada \\
        --shard_size 50 --output_subdir shards --rm_source

The first step shrinks per-sample files in place; the second packs them
into shards and (with --rm_source) deletes each per-sample file once its
shard is durably written — so peak disk stays at ~(thinned cache + one
shard) rather than ~2x.

CLI:
    python -m delta_model.data.thin_cache \\
        --cache_root cache_v1_20k/llada \\
        --keep_blocks 7 --keep_iters 5 --prefix_window 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from . import schema as S


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _compact(t: torch.Tensor) -> torch.Tensor:
    """Return a contiguous copy backed by its *own* right-sized storage.

    A plain slice (`t[:5]`, `t[..., -32:, :]`) is only a view onto the
    original oversized buffer — `torch.save` would still serialise the
    full buffer. `.contiguous().clone()` forces a fresh numel-sized
    allocation, which is the entire point of this script.
    """
    return t.contiguous().clone()


def thin_sample(
    sample: dict, *, keep_blocks: int, keep_iters: int, prefix_window: int,
) -> dict | None:
    """Thin one loaded sample dict. Returns the modified dict, or `None`
    if it is already at (or below) the target shape — so the caller can
    skip the rewrite and the run stays resumable."""
    blocks = sample["blocks"]
    if not blocks:
        return None

    cur_blocks = len(blocks)
    cur_iters  = int(blocks[0]["h_per_pass"].shape[0])
    cur_window = int(blocks[0]["prefix_kv"].shape[2])
    if (cur_blocks <= keep_blocks
            and cur_iters <= keep_iters
            and cur_window <= prefix_window):
        return None  # already thinned (or never needed it)

    new_blocks = []
    for b in blocks[:keep_blocks]:
        nb = dict(b)  # shallow copy; we overwrite the tensor fields below
        nb["h_per_pass"]      = _compact(b["h_per_pass"][:keep_iters])
        nb["reveal_per_pass"] = _compact(b["reveal_per_pass"][:keep_iters])
        nb["prefix_kv"]       = _compact(b["prefix_kv"][:, :, -prefix_window:, :])
        m = b.get("prefix_kv_pad_mask")
        if m is not None:
            nb["prefix_kv_pad_mask"] = _compact(m[-prefix_window:])
        if "n_passes_actual" in b:
            nb["n_passes_actual"] = int(min(b["n_passes_actual"], keep_iters))
        new_blocks.append(nb)
    sample["blocks"] = new_blocks

    meta = dict(sample.get("meta", {}))
    meta["num_blocks"]    = keep_blocks
    meta["max_iter"]      = keep_iters
    meta["prefix_window"] = prefix_window
    meta["thinned"] = {
        "from": {"num_blocks": cur_blocks, "max_iter": cur_iters,
                 "prefix_window": cur_window},
        "keep_blocks": keep_blocks,
        "keep_iters": keep_iters,
        "prefix_window": prefix_window,
    }
    sample["meta"] = meta
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache_root", type=Path, required=True,
                    help="cache directory containing train/ and test/")
    ap.add_argument("--keep_blocks", type=int, default=7,
                    help="keep generation blocks [0, keep_blocks). Default 7 "
                         "— drops the last block from the training cache.")
    ap.add_argument("--keep_iters", type=int, default=5,
                    help="keep recorded iterations [0, keep_iters) per block "
                         "and cap n_passes_actual to it. Default 5 "
                         "(must be >= 2 to form any (i_ref, i_target) pair).")
    ap.add_argument("--prefix_window", type=int, default=S.PREFIX_WINDOW,
                    help=f"keep the last N prefix-KV slots. Default "
                         f"{S.PREFIX_WINDOW} (= what training/inference read). "
                         f"Must be >= S.PREFIX_WINDOW ({S.PREFIX_WINDOW}).")
    ap.add_argument("--splits", type=str, default="train",
                    help="comma-separated splits to thin. Default 'train' "
                         "only — the test split is left full-resolution so "
                         "validation + GSM8K score every run on an identical "
                         "held-out set. Pass 'train,test' to thin both.")
    args = ap.parse_args()

    if args.prefix_window < S.PREFIX_WINDOW:
        ap.error(f"--prefix_window must be >= S.PREFIX_WINDOW ({S.PREFIX_WINDOW}) "
                 "— training slices to the last S.PREFIX_WINDOW and would read "
                 "past the stored window otherwise")
    if args.keep_blocks < 1:
        ap.error("--keep_blocks must be >= 1")
    if args.keep_iters < 2:
        ap.error("--keep_iters must be >= 2 (need >= 2 iterations per block "
                 "to form a single (i_ref, i_target) training pair)")

    cache_root = args.cache_root.resolve()
    if not cache_root.exists():
        raise FileNotFoundError(f"--cache_root {cache_root} does not exist")

    print(f"[thin] cache_root={cache_root}")
    print(f"[thin] target: keep_blocks={args.keep_blocks} "
          f"keep_iters={args.keep_iters} prefix_window={args.prefix_window}",
          flush=True)

    n_thinned = n_skipped = 0
    bytes_before = bytes_after = 0
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        split_dir = cache_root / split
        if not split_dir.is_dir():
            print(f"[thin] {split}: no directory at {split_dir}, skipping")
            continue
        paths = sorted(split_dir.glob("sample_*.pt"))
        print(f"[thin] {split}: {len(paths)} samples", flush=True)
        for i, p in enumerate(paths):
            size_before = p.stat().st_size
            sample = torch.load(p, map_location="cpu", weights_only=False)
            out = thin_sample(
                sample,
                keep_blocks=args.keep_blocks,
                keep_iters=args.keep_iters,
                prefix_window=args.prefix_window,
            )
            if out is None:
                n_skipped += 1
            else:
                _atomic_save(out, p)
                n_thinned += 1
                bytes_before += size_before
                bytes_after  += p.stat().st_size
            if (i + 1) % 200 == 0 or (i + 1) == len(paths):
                print(f"[thin] {split}: {i+1}/{len(paths)} "
                      f"(thinned {n_thinned}, skipped {n_skipped})", flush=True)

    # Update the cache manifest's prefix_window + leave a provenance note,
    # so downstream tooling and a future re-collect see consistent metadata.
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["prefix_window"] = args.prefix_window
        manifest["thinned"] = {
            "keep_blocks": args.keep_blocks,
            "keep_iters": args.keep_iters,
            "prefix_window": args.prefix_window,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[thin] updated {manifest_path}")
    else:
        print(f"[thin] no manifest.json at {manifest_path} (skipping manifest update)")

    saved = bytes_before - bytes_after
    if n_thinned:
        print(
            f"[thin] done. thinned={n_thinned} skipped={n_skipped}. "
            f"rewritten files: {bytes_before/1e9:.1f} GB → "
            f"{bytes_after/1e9:.1f} GB (saved {saved/1e9:.1f} GB, "
            f"{100*saved/max(1, bytes_before):.0f}%).",
            flush=True,
        )
    else:
        print(f"[thin] done. nothing to do — all {n_skipped} samples already "
              f"at or below the target shape.", flush=True)


if __name__ == "__main__":
    main()

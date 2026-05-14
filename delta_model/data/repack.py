"""Repack per-sample .pt cache files into multi-sample shards.

Background: collect_llada writes one .pt per prompt (~16 MiB). With a
shuffled training index of ~120 entries per sample and a small per-worker
LRU, the dataset hits disk on most __getitem__ calls. Packing N samples
into one shard (~N·16 MiB) lets the LRU cache shards instead of singletons
— with N=50 a 4-shard LRU keeps ~200 samples (24K indices) hot in RAM.

Shard format (one .pt file):
    {
        "samples": [sample_dict_0, sample_dict_1, ..., sample_dict_{N-1}],
        "shard_meta": {
            "split":         "train" | "test",
            "shard_index":   int,
            "n_samples":     int,
            "schema_version": int,
        },
    }
Each `sample_dict_i` is a verbatim copy of the original per-sample dict
(keys: prompt_token_ids, generated_token_ids, prompt_len, blocks, meta,
record), so existing dataset code can read it without translation.

Output also writes a `shards_manifest.json` next to the shards that the
dataset uses to discover the new layout. The original per-sample files
are NOT deleted — repack is non-destructive. To revert, just delete the
shards directory + manifest.

CLI:
    python -m delta_model.data.repack \\
        --cache_root cache_v1/llada \\
        --shard_size 50 \\
        --output_subdir shards
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


def repack_split(
    cache_root: Path, split: str, shard_size: int, output_subdir: str,
    *, rm_source: bool = False,
) -> list[dict]:
    """Pack `cache_root/<split>/sample_*.pt` into shards under
    `cache_root/<output_subdir>/<split>/shard_*.pt`. Returns a list of
    shard manifest records ({split, shard_index, filename, n_samples,
    sample_ids:[...]}).

    `rm_source=True` unlinks each per-sample file *after* the shard
    containing it has been atomically written — keeps peak disk usage at
    ~(cache size + one shard) instead of ~2x. Safe to interrupt: a
    half-written shard never deletes its sources (the unlink happens
    post-`_atomic_save`).
    """
    src_dir = cache_root / split
    out_dir = cache_root / output_subdir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(src_dir.glob("sample_*.pt"))
    if not sample_paths:
        print(f"[repack] {split}: no per-sample files found at {src_dir}")
        return []

    print(f"[repack] {split}: {len(sample_paths)} samples → "
          f"shards of size {shard_size} in {out_dir}")

    records: list[dict] = []
    buf: list[dict] = []
    sample_ids_in_buf: list[str] = []
    paths_in_buf: list[Path] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not buf:
            return
        shard_path = out_dir / f"shard_{shard_idx:05d}.pt"
        _atomic_save(
            {
                "samples": buf,
                "shard_meta": {
                    "split":          split,
                    "shard_index":    shard_idx,
                    "n_samples":      len(buf),
                    "schema_version": S.SCHEMA_VERSION,
                },
            },
            shard_path,
        )
        records.append({
            "split":        split,
            "shard_index":  shard_idx,
            "filename":     str(shard_path.relative_to(cache_root)),
            "n_samples":    len(buf),
            "sample_ids":   list(sample_ids_in_buf),
        })
        print(f"[repack] {split}: wrote {shard_path.name} "
              f"({len(buf)} samples, {shard_path.stat().st_size / 1e6:.1f} MB)")
        # The shard is durably on disk now — only then is it safe to drop
        # the per-sample sources it absorbed.
        if rm_source:
            for sp in paths_in_buf:
                try:
                    sp.unlink()
                except OSError:
                    pass
        shard_idx += 1
        buf.clear()
        sample_ids_in_buf.clear()
        paths_in_buf.clear()

    for p in sample_paths:
        sample = torch.load(p, map_location="cpu", weights_only=False)
        buf.append(sample)
        sample_ids_in_buf.append(p.stem)  # e.g. "sample_<hex>"
        paths_in_buf.append(p)
        if len(buf) >= shard_size:
            flush()
    flush()

    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache_root", type=Path, required=True,
                    help="cache directory containing train/ and test/")
    ap.add_argument("--shard_size", type=int, default=50,
                    help="samples per shard (default 50 ≈ 800 MB / shard)")
    ap.add_argument("--output_subdir", type=str, default="shards",
                    help="subdirectory under cache_root to write shards into")
    ap.add_argument("--splits", type=str, default="train,test",
                    help="comma-separated list of splits to repack")
    ap.add_argument("--rm_source", action="store_true",
                    help="unlink each per-sample file after the shard "
                         "containing it is written. Keeps peak disk at "
                         "~(cache + one shard) instead of ~2x. Safe to "
                         "interrupt — sources are only dropped post-save.")
    args = ap.parse_args()

    cache_root = args.cache_root.resolve()
    if not cache_root.exists():
        raise FileNotFoundError(f"--cache_root {cache_root} does not exist")

    all_records: list[dict] = []
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        all_records.extend(repack_split(
            cache_root, split, args.shard_size, args.output_subdir,
            rm_source=args.rm_source,
        ))

    manifest_path = cache_root / "shards_manifest.json"
    manifest = {
        "shard_size":     args.shard_size,
        "output_subdir":  args.output_subdir,
        "schema_version": S.SCHEMA_VERSION,
        "shards":         all_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[repack] manifest: {manifest_path}")
    print(f"[repack] done. {len(all_records)} shards across "
          f"{len(set(r['split'] for r in all_records))} splits.")


if __name__ == "__main__":
    main()

"""Rebuild `shards_manifest.json` from the on-disk shard files.

Use when the manifest got truncated or overwritten (e.g. `repack.py
--splits test` overwrote a manifest that previously held train records,
before the `--merge_manifest` flag landed) and you don't want to
re-pack from per-sample files (which may be gone after `--rm_source`).

Scans `<cache_root>/<output_subdir>/<split>/shard_*.pt` for every split
under `<cache_root>/<output_subdir>/`, reads `n_samples` from each
shard's payload, and writes a fresh manifest.

CLI:
    python -m delta_model.data.rebuild_manifest \\
        --cache_root cache_v1_20k/llada \\
        --output_subdir shards
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from . import schema as S


def _shard_n_samples(shard_path: Path) -> int:
    """Read just the sample count from a shard file. Loads the whole
    payload because torch.save serialises the dict in one blob — there's
    no streaming partial-load path that's faster here."""
    d = torch.load(shard_path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        raise ValueError(f"{shard_path}: expected dict, got {type(d).__name__}")
    if "samples" not in d:
        raise ValueError(f"{shard_path}: missing 'samples' key")
    return len(d["samples"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache_root", type=Path, required=True,
                    help="cache directory containing the shards/ subtree")
    ap.add_argument("--output_subdir", type=str, default="shards",
                    help="subdirectory under cache_root that holds the "
                         "per-split shard directories (default: shards)")
    ap.add_argument("--shard_size", type=int, default=50,
                    help="recorded in the manifest header for reference "
                         "(default: 50). Does not have to match each "
                         "shard's actual n_samples — that's read per-shard.")
    args = ap.parse_args()

    cache_root = args.cache_root.resolve()
    shards_root = cache_root / args.output_subdir
    if not shards_root.is_dir():
        sys.exit(f"[rebuild] {shards_root} does not exist or is not a directory")

    split_dirs = sorted(p for p in shards_root.iterdir() if p.is_dir())
    if not split_dirs:
        sys.exit(f"[rebuild] no split subdirectories under {shards_root}")

    records: list[dict] = []
    for split_dir in split_dirs:
        split = split_dir.name
        shard_paths = sorted(split_dir.glob("shard_*.pt"))
        if not shard_paths:
            print(f"[rebuild] {split}: no shard_*.pt files, skipping")
            continue
        print(f"[rebuild] {split}: {len(shard_paths)} shard(s)")
        for i, p in enumerate(shard_paths):
            n = _shard_n_samples(p)
            rel = str(p.relative_to(cache_root)).replace("\\", "/")
            records.append({
                "split":       split,
                "shard_index": i,
                "filename":    rel,
                "n_samples":   n,
            })
            print(f"[rebuild]   {split} #{i:5d}  {p.name}  n={n}")

    if not records:
        sys.exit("[rebuild] no shards found under any split — refusing to "
                 "write an empty manifest")

    manifest = {
        "shard_size":     args.shard_size,
        "output_subdir":  args.output_subdir,
        "schema_version": S.SCHEMA_VERSION,
        "shards":         records,
        "note":           "rebuilt by rebuild_manifest.py from on-disk shards",
    }
    manifest_path = cache_root / "shards_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[rebuild] wrote {manifest_path} with {len(records)} record(s) "
          f"across splits {sorted({r['split'] for r in records})}")


if __name__ == "__main__":
    main()

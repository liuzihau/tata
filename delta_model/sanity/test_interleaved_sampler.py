"""T5 sanity — InterleavedShardSampler behaviour (no cache / GPU needed).

Checks, on a synthetic shard layout:
  1. Emits exactly `num_samples` indices.
  2. Within a DataLoader-sized batch (256), indices come from many shards
     (interleaving works) — not one, which was the BlockShardSampler
     failure mode that cratered m1_5_data20k (val/kl 0.332 vs 0.280).
  3. Over enough rounds every index is reachable (full coverage).
  4. T2 weights bias the draw (ref0-weighted indices ~3× over-represented).

Run:
    python -m delta_model.sanity.test_interleaved_sampler

Pass criteria: prints "✓ InterleavedShardSampler sanity passed".
"""
from __future__ import annotations

import torch

from ..data.dataset import InterleavedShardSampler


def _fake_shard_groups(n_shards: int, pairs_per_shard: int):
    """Build `{shard_id: [flat_index, ...]}` with contiguous index ranges."""
    groups: dict[int, list[int]] = {}
    k = 0
    for sid in range(n_shards):
        groups[sid] = list(range(k, k + pairs_per_shard))
        k += pairs_per_shard
    return groups, k  # (groups, total_pairs)


def main() -> None:
    n_shards, pairs_per_shard = 40, 100
    groups, total = _fake_shard_groups(n_shards, pairs_per_shard)
    shard_of = {i: sid for sid, idxs in groups.items() for i in idxs}

    # Every 5th index is a "ref0" pair carrying 3× weight (mirrors T2).
    weights = torch.ones(total)
    ref0 = set(range(0, total, 5))
    for i in ref0:
        weights[i] = 3.0

    active_shards = 8
    gen = torch.Generator(); gen.manual_seed(0)
    sampler = InterleavedShardSampler(
        groups, weights, active_shards=active_shards,
        num_samples=total, generator=gen,
    )

    # 1. Exact count.
    emitted = list(sampler)
    assert len(emitted) == total, f"expected {total} emitted, got {len(emitted)}"
    print(f"  emitted {len(emitted)} indices (== num_samples) ✓")

    # 2. Interleaving — a 256-wide batch should touch >= active_shards shards.
    batch = emitted[:256]
    shards_in_batch = {shard_of[i] for i in batch}
    assert len(shards_in_batch) >= active_shards, (
        f"first batch touched only {len(shards_in_batch)} shards; "
        f"expected >= active_shards={active_shards} (interleaving broken)"
    )
    print(f"  first 256-batch spans {len(shards_in_batch)} shards "
          f"(>= active_shards={active_shards}) ✓")

    # 3 + 4. Coverage and weighting over many rounds.
    gen2 = torch.Generator(); gen2.manual_seed(1)
    big = InterleavedShardSampler(
        groups, weights, active_shards=active_shards,
        num_samples=total * 20, generator=gen2,
    )
    counts = torch.zeros(total)
    for k in big:
        counts[k] += 1
    n_unseen = int((counts == 0).sum())
    assert n_unseen == 0, f"{n_unseen} indices never sampled over 20 rounds"

    ref0_mask = torch.zeros(total, dtype=torch.bool)
    ref0_mask[list(ref0)] = True
    ratio = float(counts[ref0_mask].mean() / counts[~ref0_mask].mean())
    assert 2.5 < ratio < 3.5, f"ref0 over-sampling ratio {ratio:.2f} not ≈ 3.0"
    print(f"  full coverage over 20 rounds; ref0 weight ratio {ratio:.2f} ≈ 3 ✓")

    # Auto chunk_size: one window ≈ active_shards · mean_pairs_per_shard.
    assert sampler.chunk_size == active_shards * pairs_per_shard, (
        f"auto chunk_size {sampler.chunk_size} != "
        f"{active_shards * pairs_per_shard}"
    )
    print(f"  auto chunk_size={sampler.chunk_size} "
          f"(active_shards · mean_pairs_per_shard) ✓")

    print("✓ InterleavedShardSampler sanity passed")


if __name__ == "__main__":
    main()

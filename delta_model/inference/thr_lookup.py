"""Per-(gap, reveal_bin) threshold lookup for `generate_with_delta`.

Train-time observation (v2 reveal-bin diagnostic): mse_by_reveal_0 ≈ 0.69
vs mse_by_reveal_4 ≈ 1.95 on the same checkpoint. The 2.8× ratio is
partly *true* reveal-fraction difficulty and partly *gap leakage* — the
high-reveal bin pools (i_target − i_ref) ∈ {1, 2, 3} together, the
low-reveal bin is gap=1 only. The 2-D `mse_by_gap_{g}_reveal_{rb}`
field in `<ckpt>/metrics.jsonl` (added 2026-05-19) disentangles them.

This module turns those binned diagnostics into an *inference-time*
policy: pick the per-position confidence threshold per cell, so easy
cells (gap=1, low reveal) trust the delta more (lower threshold → more
delta commits → fewer backbone rollbacks), and hard cells (gap=3, high
reveal) trust it less (higher threshold → rollback before a bad commit
propagates).

Two public entry points:

  • `build_lookup_from_metrics(metrics, *, default, bins_reveal, ...)`
       → ThresholdLookup
    Reads the last `val/mse_by_gap_{g}_reveal_{rb}` row from a JSONL
    (or accepts a dict). Maps MSE → threshold via a calibration knob:
    cells with low MSE get a lower threshold (we trust the delta);
    cells with high MSE get a higher one.

  • `ThresholdLookup.__call__(gap, reveal_frac) -> float`
    The signature `generate_with_delta` accepts when its
    `per_pos_threshold` argument is callable.

Calibration is intentionally simple (linear remap of MSE within
[base_threshold, max_threshold]). The right next step is to log
`shared_mass_by_(gap, reveal_bin)` and replace MSE with a target
false-commit-rate per cell — left as a follow-up.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


# Default reveal-bin edges; must match `cfg.log.bins_reveal` used at train.
_DEFAULT_BINS_REVEAL: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _reveal_bin(reveal_frac: float, bins: Sequence[float]) -> int:
    """Map a reveal fraction in [0, 1] to its bin index in [0, len(bins)-2]."""
    r = float(reveal_frac)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= r <= hi:
            return i
    return max(0, len(bins) - 2)   # clamp out-of-range to the last bin


class ThresholdLookup:
    """Callable mapping `(gap, reveal_frac) -> per_pos_threshold`.

    Constructed with an explicit `table: dict[(int, int), float]`. Cells
    not in the table fall back to `default`. Gap > max_known_gap (the
    train-inference mismatch when no rollbacks fire) clamps to the
    highest-gap row available — almost always the hardest, so this
    biases toward more rollbacks, which is the safer default.
    """

    def __init__(
        self,
        table: dict[tuple[int, int], float],
        *,
        default: float,
        bins_reveal: Sequence[float] = _DEFAULT_BINS_REVEAL,
    ):
        self.table = dict(table)
        self.default = float(default)
        self.bins_reveal = list(bins_reveal)
        self._max_gap = max((g for (g, _) in table.keys()), default=0)

    def __call__(self, gap: int, reveal_frac: float) -> float:
        g = max(1, int(gap))
        # Clamp gap above max-known to the worst trained-gap row.
        if g > self._max_gap and self._max_gap > 0:
            g = self._max_gap
        rb = _reveal_bin(reveal_frac, self.bins_reveal)
        return self.table.get((g, rb), self.default)

    def __repr__(self) -> str:
        return (
            f"ThresholdLookup(cells={len(self.table)}, default={self.default:.3f}, "
            f"max_gap={self._max_gap})"
        )


def _load_last_val_row(metrics_jsonl: Path) -> dict:
    """Find the most recent JSONL row containing val/* keys."""
    last_val_row: dict | None = None
    for line in metrics_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if any(k.startswith("val/") for k in row):
            last_val_row = row
    if last_val_row is None:
        raise ValueError(
            f"{metrics_jsonl}: no rows contain val/* keys yet — did the "
            f"first val tick run? (default `val_every=500` in v2 configs)"
        )
    return last_val_row


def build_lookup_from_metrics(
    metrics: dict | str | Path,
    *,
    base_threshold: float = 0.65,
    max_threshold:  float = 0.95,
    min_support: int = 32,
    bins_reveal: Sequence[float] = _DEFAULT_BINS_REVEAL,
) -> ThresholdLookup:
    """Build a `ThresholdLookup` from train.py's val metrics.

    `metrics` can be:
      • a path to `<ckpt_dir>/metrics.jsonl` — the last row with any
        `val/...` key is used;
      • a dict with `val/mse_by_gap_{g}_reveal_{rb}` keys.

    Calibration: per cell, threshold = `base + (max - base) * normalized_mse`,
    where normalized_mse ∈ [0, 1] across all *populated* cells (those
    with `n_by_gap_g_reveal_rb >= min_support`). So the easiest cell
    gets `base_threshold`, the hardest gets `max_threshold`. Sparse
    cells fall back to `default = (base + max) / 2`.

    The base/max bracket sets the absolute range; tune both with the
    same threshold-sweep harness already in `gsm8k_e2e.py`:
      • `base=0.65 max=0.95` (default) is a reasonable starting band
        for v2-trained checkpoints
      • narrow the band (e.g. base=0.75 max=0.85) when the BCE head is
        well-calibrated (T9 result) — there's less to gain from
        per-cell variation
    """
    if isinstance(metrics, (str, Path)):
        metrics = _load_last_val_row(Path(metrics))

    # Pull the 2D cells: keys look like "val/mse_by_gap_{g}_reveal_{rb}".
    cells: dict[tuple[int, int], tuple[float, int]] = {}
    prefix_mse = "val/mse_by_gap_"
    for k, v in metrics.items():
        if not k.startswith(prefix_mse):
            continue
        try:
            # "val/mse_by_gap_2_reveal_3" → (2, 3)
            tail = k[len(prefix_mse):]
            g_str, _reveal, rb_str = tail.split("_")
            g, rb = int(g_str), int(rb_str)
        except (ValueError, IndexError):
            continue
        n = int(metrics.get(f"val/n_by_gap_{g}_reveal_{rb}", 0))
        if n < min_support:
            continue
        cells[(g, rb)] = (float(v), n)

    default = 0.5 * (base_threshold + max_threshold)
    if not cells:
        # No 2D cells in metrics — return a constant-default lookup so the
        # caller can fall through to the legacy single-threshold path.
        return ThresholdLookup({}, default=default, bins_reveal=bins_reveal)

    mses = [mse for (mse, _) in cells.values()]
    mse_lo, mse_hi = min(mses), max(mses)
    span = max(mse_hi - mse_lo, 1e-6)
    table: dict[tuple[int, int], float] = {
        (g, rb): base_threshold
                  + (max_threshold - base_threshold) * (mse - mse_lo) / span
        for (g, rb), (mse, _) in cells.items()
    }
    return ThresholdLookup(table, default=default, bins_reveal=bins_reveal)


# Backwards-compat helper for one-off CLI use.
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics_jsonl", type=Path, required=True)
    ap.add_argument("--base", type=float, default=0.65)
    ap.add_argument("--max",  type=float, default=0.95)
    args = ap.parse_args()
    lk = build_lookup_from_metrics(
        args.metrics_jsonl, base_threshold=args.base, max_threshold=args.max,
    )
    print(lk)
    print("table:")
    for (g, rb), thr in sorted(lk.table.items()):
        print(f"  gap={g}  reveal_bin={rb}  thr={thr:.3f}")

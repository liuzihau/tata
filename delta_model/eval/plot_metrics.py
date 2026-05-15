"""Plot training metrics from local JSONL logs.

`train.py` writes a `metrics.jsonl` into each checkpoint directory, mirroring
every `wandb.log` payload one record per line. This script reads one or more
of those files and draws matplotlib plots — useful for quick on-machine
inspection without going through wandb's web UI, and especially for
multi-run comparison (trial-2 vs M1.5 vs M1.6 vs T5 on one figure).

CLI examples:

    # All common metrics in one run, popup window
    python -m delta_model.eval.plot_metrics \\
        ckpts/m1_5_tier1_llada_variant_c/metrics.jsonl

    # Compare three runs on a specific metric, save to PNG
    python -m delta_model.eval.plot_metrics \\
        ckpts/m1_llada_variant_c/metrics.jsonl \\
        ckpts/m1_5_tier1_llada_variant_c/metrics.jsonl \\
        ckpts/m1_6_lambdamse0_llada_variant_c/metrics.jsonl \\
        --metric train/kl val/kl \\
        --out plots/kl_comparison.png

    # Smooth noisy curves with a running mean
    python -m delta_model.eval.plot_metrics \\
        ckpts/m1_5_tier1_llada_variant_c/metrics.jsonl \\
        --metric train/loss train/kl --smooth 25 --out plots/m1_5_train.png

    # Filter by prefix (regex)
    python -m delta_model.eval.plot_metrics \\
        ckpts/m1_5_tier1_llada_variant_c/metrics.jsonl \\
        --metric "val/mse_by_gap_.*" --out plots/m1_5_val_gap.png
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str | Path) -> dict[str, tuple[list[int], list[float]]]:
    """Read a metrics.jsonl, return {metric_name: (sorted_steps, values)}.

    Returns step-sorted, step-deduplicated series. `train.py` opens the
    JSONL in append-mode, so a restarted run produces overlapping step
    ranges in the same file (e.g. 0..1350 from a killed run, then 0..N
    from the relaunch). Plotting in file order then draws a horizontal
    line from the old end back to the new start ("begin → last-point"
    artifact). We dedup by step, *last occurrence in file order wins* —
    matches the user-intuitive "show me the most recent training
    trajectory."

    Skips malformed lines (resume artifacts, partial writes during a
    crash) and non-numeric values.
    """
    # {metric -> {step -> value}}, where later writes to the same step
    # overwrite earlier ones (later in file = more recent run).
    per_metric: dict[str, dict[int, float]] = {}
    n_records = 0
    n_step_decreases = 0
    last_step = -(1 << 60)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = rec.get("step")
            if step is None:
                continue
            n_records += 1
            step = int(step)
            if step < last_step:
                n_step_decreases += 1
            last_step = step
            for k, v in rec.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    per_metric.setdefault(k, {})[step] = float(v)

    if n_step_decreases:
        print(
            f"[plot] {path}: {n_step_decreases} step-decrease event(s) "
            f"detected in {n_records} records — looks like the file contains "
            f"history from multiple training sessions. Deduping by step "
            f"(later writes win).",
            flush=True,
        )

    result: dict[str, tuple[list[int], list[float]]] = {}
    for name, d in per_metric.items():
        steps = sorted(d.keys())
        values = [d[s] for s in steps]
        result[name] = (steps, values)
    return result


def _filter_metric_names(
    available: list[str], patterns: list[str] | None,
) -> list[str]:
    """Resolve `--metric` args: exact match first, then regex fallback."""
    if patterns is None:
        return sorted(available)
    matched: list[str] = []
    for pat in patterns:
        if pat in available:
            if pat not in matched:
                matched.append(pat)
            continue
        try:
            rx = re.compile(f"^{pat}$")
        except re.error:
            continue
        for name in available:
            if rx.match(name) and name not in matched:
                matched.append(name)
    return matched


def _running_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    cum = [0.0]
    for v in values:
        cum.append(cum[-1] + v)
    return [(cum[i + window] - cum[i]) / window for i in range(len(values) - window + 1)]


def _run_label(path: Path) -> str:
    """Take the checkpoint folder name as the run label (`ckpts/<name>/metrics.jsonl`
    → `<name>`). Falls back to the filename stem if the parent isn't a ckpt dir."""
    if path.parent.name and path.name == "metrics.jsonl":
        return path.parent.name
    return path.stem


def main() -> None:
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", help="one or more metrics.jsonl files")
    p.add_argument(
        "--metric", nargs="+", default=None,
        help="metric names to plot (exact or regex). Defaults to all metrics "
             "common across the provided runs.",
    )
    p.add_argument(
        "--out", default=None,
        help="if set, save to this PNG (parent dirs created). Otherwise shows "
             "interactively via plt.show().",
    )
    p.add_argument(
        "--smooth", type=int, default=0,
        help="running-mean window over the value series; 0 = off. Useful for "
             "noisy single-batch train metrics (try 25 at log_every=50).",
    )
    p.add_argument(
        "--cols", type=int, default=3,
        help="subplot columns (default 3).",
    )
    p.add_argument(
        "--ylog", action="store_true",
        help="use a log scale on the y axis (good for losses spanning orders of magnitude).",
    )
    args = p.parse_args()

    runs: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"[plot] WARN: skipping missing {path}")
            continue
        runs[_run_label(path)] = load_jsonl(path)

    if not runs:
        print("[plot] no runs loaded; nothing to do.")
        return

    # Pick metrics: explicit (with regex fallback) or intersection of all runs.
    if args.metric is None:
        common: set | None = None
        for series in runs.values():
            keys = set(series.keys())
            common = keys if common is None else (common & keys)
        metrics = sorted(common) if common else []
    else:
        all_names: set[str] = set()
        for s in runs.values():
            all_names.update(s.keys())
        metrics = _filter_metric_names(sorted(all_names), args.metric)

    if not metrics:
        print(
            "[plot] no metrics to plot. "
            f"Available across runs: {sorted({k for s in runs.values() for k in s})}"
        )
        return

    n = len(metrics)
    cols = max(1, min(args.cols, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i // cols][i % cols]
        any_data = False
        for run_name, series in runs.items():
            if metric not in series:
                continue
            steps, values = series[metric]
            # Only slice `steps` when smoothing actually ran. Sparse metrics
            # like gsm8k/* (logged every ~2000 steps) often have fewer points
            # than the smoothing window — `_running_mean` returns values
            # unchanged in that case, so we must keep `steps` unchanged too
            # or matplotlib hits a "shapes (0,) vs (N,)" mismatch.
            if args.smooth > 1 and len(values) >= args.smooth:
                values = _running_mean(values, args.smooth)
                steps = steps[args.smooth - 1:]
            ax.plot(steps, values, label=run_name, linewidth=1.2)
            any_data = True
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        if args.ylog:
            ax.set_yscale("log")
        if any_data:
            ax.legend(fontsize=7, loc="best")

    # Hide unused axes.
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")

    fig.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[plot] saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

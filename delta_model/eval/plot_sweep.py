"""Plot a per_pos_threshold sweep produced by `eval/gsm8k_e2e.py`.

`gsm8k_e2e.py --out_json ...` writes a JSON list, one dict per
`per_pos_threshold`, with accuracy / speedup / rollback fields. This
script turns one (or more) of those files into a 3-panel figure:

  1. Accuracy–speed tradeoff — `accuracy_hybrid` vs `speedup_ratio`, one
     threshold-ordered polyline. The shaded top-right "win zone" (faster
     than vanilla AND at least as accurate) is the headline: if no point
     lands in it, there is no good operating point.
  2. Accuracy & speedup vs threshold — dual axis: `accuracy_hybrid` as
     black-edged yellow bars (left), `speedup_ratio` as a thick blue line
     (right, linear). The left axis carries a dashed vanilla-accuracy
     reference; the speedup axis needs none (1.0x is trivial). Every
     point is text-labelled (`+2.5%`, `1.07x`) so the twin axes can't
     mislead — the reader always has the exact numbers.
  3. Rollback cost vs threshold — `mean_rollbacks` as bars; the mechanism
     behind the speedup collapse.

Pass several sweep JSONs to overlay them (grouped bars + one polyline
per file, one colour per file).

CLI:
    python -m delta_model.eval.plot_sweep \\
        eval_results/m1_5_daya20k_preload_best.json \\
        --out plots/m1_5_data20k_sweep.png

    # overlay two sweeps
    python -m delta_model.eval.plot_sweep \\
        eval_results/sweep_a.json eval_results/sweep_b.json \\
        --out plots/sweep_ab.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sweep(path: str | Path) -> list[dict]:
    """Read a sweep JSON (list of per-threshold dicts), sorted by threshold."""
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty JSON list of sweep rows")
    for r in rows:
        # `accuracy_delta` is written by gsm8k_e2e; recompute defensively.
        r.setdefault(
            "accuracy_delta", r["accuracy_hybrid"] - r["accuracy_vanilla"],
        )
    return sorted(rows, key=lambda r: r["per_pos_threshold"])


def _col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows]


def _vanilla_acc(rows: list[dict]) -> float:
    # Constant within one sweep (fixed test set); average defensively.
    v = _col(rows, "accuracy_vanilla")
    return sum(v) / len(v)


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", help="one or more sweep JSON files")
    p.add_argument(
        "--out", default=None,
        help="save to this PNG (parent dirs created); else plt.show()",
    )
    p.add_argument("--title", default=None, help="override the figure title")
    args = p.parse_args()

    sweeps: dict[str, list[dict]] = {}
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"[plot] WARN: skipping missing {path}")
            continue
        sweeps[path.stem] = load_sweep(path)
    if not sweeps:
        print("[plot] no sweeps loaded; nothing to do.")
        return

    labels = list(sweeps)
    n_files = len(labels)
    multi = n_files > 1
    palette = plt.get_cmap("tab10")
    file_color = {lab: palette(i % 10) for i, lab in enumerate(labels)}

    # Vanilla baseline — should be identical across files (same test set).
    vanillas = {lab: _vanilla_acc(rows) for lab, rows in sweeps.items()}
    van = sum(vanillas.values()) / n_files
    if multi and (max(vanillas.values()) - min(vanillas.values())) > 0.01:
        print(f"[plot] WARN: vanilla accuracy differs across files "
              f"({vanillas}); drawing the mean ({van:.3f}).")

    # Quick stdout summary — where each sweep's good operating points are.
    for lab, rows in sweeps.items():
        best = max(rows, key=lambda r: r["accuracy_hybrid"])
        faster = [r for r in rows if r["speedup_ratio"] >= 1.0]
        print(
            f"[plot] {lab}: best acc {best['accuracy_hybrid']:.3f} "
            f"@ thr={best['per_pos_threshold']:g} "
            f"(speedup {best['speedup_ratio']:.2f}x) | "
            + (f"{len(faster)} threshold(s) faster than vanilla"
               if faster else "NO threshold beats vanilla speed")
        )

    # Darker gold-yellow bars (black-edged), strong Okabe-Ito blue for the
    # speedup line. Both y-axes stay black — only the line itself is blue.
    BAR_COLOR  = "#C1AC58"   # accuracy bars     (darker gold-yellow)
    SPD_COLOR  = "#0072B2"   # speedup LINE only (Okabe-Ito blue)
    BASE_COLOR = "#555555"   # accuracy baseline reference line
    RB_COLOR   = "#7d6db3"   # rollback bars

    fig, (ax_par, ax_mid, ax_rb) = plt.subplots(1, 3, figsize=(17, 4.8))

    # ===================================================================
    # Panel 1 — accuracy–speed tradeoff
    # ===================================================================
    all_spd, all_acc = [], []
    for lab, rows in sweeps.items():
        spd = _col(rows, "speedup_ratio")
        acc = _col(rows, "accuracy_hybrid")
        thr = _col(rows, "per_pos_threshold")
        all_spd += spd
        all_acc += acc
        c = file_color[lab] if multi else "#1f77b4"
        ax_par.plot(spd, acc, "-o", color=c, ms=6, lw=1.5, zorder=3,
                    label=lab if multi else None)
        for s, a, t in zip(spd, acc, thr):
            ax_par.annotate(f"thr={t:g}", (s, a), textcoords="offset points",
                            xytext=(6, 5), fontsize=8, color=c)

    xlo, xhi = max(0.0, min(all_spd) - 0.1), max(all_spd) * 1.25
    ylo, yhi = -0.03, max(0.85, max(all_acc) * 1.12)
    ax_par.set_xlim(xlo, xhi)
    ax_par.set_ylim(ylo, yhi)
    ax_par.axhline(van, ls="--", color=BASE_COLOR, lw=1)
    ax_par.axvline(1.0, ls="--", color=BASE_COLOR, lw=1)
    if xhi > 1.0 and van < yhi:
        ax_par.add_patch(Rectangle(
            (1.0, van), xhi - 1.0, yhi - van,
            facecolor="tab:green", alpha=0.10, edgecolor="none", zorder=0,
        ))
        ax_par.text(1.0 + (xhi - 1.0) * 0.5, (van + yhi) / 2,
                    "win zone\nfaster & ≥ vanilla", rotation=90,
                    ha="center", va="center", fontsize=7.5, color="#3a7a3a")
    ax_par.text(xlo, van, f" vanilla {van:.3f}", va="bottom", ha="left",
                fontsize=8, color=BASE_COLOR)
    ax_par.set_xlabel("speedup ratio  (vanilla walltime / hybrid)")
    ax_par.set_ylabel("accuracy (hybrid)")
    ax_par.set_title("Accuracy–speed tradeoff", fontsize=10)
    ax_par.grid(alpha=0.25)
    if multi:
        ax_par.legend(fontsize=7, loc="lower left")

    # ===================================================================
    # Panels 2 & 3 share the threshold x-axis
    # ===================================================================
    thr_all = sorted({t for rows in sweeps.values()
                      for t in _col(rows, "per_pos_threshold")})
    x = np.arange(len(thr_all))
    thr_pos = {t: i for i, t in enumerate(thr_all)}
    bar_w = 0.8 / n_files

    def _grouped(key: str):
        """Yield (xpos, values, label, color) per file, with bars offset so
        they don't overlap when multiple files are overlaid."""
        for i, (lab, rows) in enumerate(sweeps.items()):
            xs = np.array([thr_pos[t] for t in _col(rows, "per_pos_threshold")],
                          dtype=float)
            xs = xs + (i - (n_files - 1) / 2) * bar_w
            yield xs, np.array(_col(rows, key)), lab, file_color[lab]

    # ===================================================================
    # Panel 2 — accuracy (bars) + speedup (line), dual axis
    # ===================================================================
    ax_acc = ax_mid
    ax_spd = ax_mid.twinx()

    for xs, acc, lab, fc in _grouped("accuracy_hybrid"):
        dlt = _col(sweeps[lab], "accuracy_delta")
        # All accuracy bars share the yellow; multi-file falls back to the
        # per-file colour only so overlaid sweeps stay distinguishable.
        colors = fc if multi else BAR_COLOR
        ax_acc.bar(xs, acc, width=bar_w * 0.92, color=colors,
                   edgecolor="black", linewidth=2, zorder=2,
                   label=lab if multi else None)
        for xi, a, d in zip(xs, acc, dlt):
            ax_acc.annotate(f"{d * 100:+.1f}%", (xi, a),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7.5)
    for xs, spd, lab, fc in _grouped("speedup_ratio"):
        lc = fc if multi else SPD_COLOR
        ax_spd.plot(xs, spd, "-o", color=lc, ms=6, lw=2.6, zorder=4)
        for xi, s in zip(xs, spd):
            ax_spd.annotate(f"{s:.2f}x", (xi, s), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5, color=lc)

    # The accuracy axis keeps a dashed vanilla-baseline reference, labelled
    # at the top-right of the plot region. The speedup axis gets none —
    # 1.0x is trivial and every point is text-labelled with its exact ratio.
    ax_acc.axhline(van, ls="--", color=BASE_COLOR, lw=1)
    ax_acc.text(0.98, 0.97, f"vanilla accuracy {van:.2f}",
                transform=ax_acc.transAxes, va="top", ha="right",
                fontsize=8, color=BASE_COLOR)

    # Accuracy axis: 0.7x the smallest bar .. 1.3x the tallest, so the
    # speedup line has headroom above the bars. Accuracy can't exceed 1.0,
    # so blank any tick label above it. (NOTE: a non-zero bottom truncates
    # the bars — harmless here since min accuracy ≈ 0, but if a future
    # sweep has a high worst-case accuracy, clamp the bottom to 0.)
    min_acc, max_acc = min(all_acc), max(all_acc)
    ax_acc.set_ylim(0.7 * min_acc, 1.5 * max_acc)
    ax_acc.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: "" if v > 1.0 + 1e-9 else f"{v:.2f}"
    ))
    # Speedup axis: 0 .. at least 1.2x the fastest point.
    ax_spd.set_ylim(0, max(all_spd) * 1.2)

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([f"{t:g}" for t in thr_all])
    ax_acc.set_xlabel("per_pos_threshold")
    ax_acc.set_ylabel("accuracy (hybrid)  —  bars")
    # Both y-axes stay black; only the plotted line is blue.
    ax_spd.set_ylabel("speedup ratio  —  line")
    ax_acc.set_title("Accuracy & speedup vs threshold", fontsize=10)
    ax_acc.grid(alpha=0.25, axis="y")
    if multi:
        ax_acc.legend(fontsize=7, loc="upper right")

    # ===================================================================
    # Panel 3 — rollback cost vs threshold
    # ===================================================================
    max_rb = 0.0
    for xs, rb, lab, fc in _grouped("mean_rollbacks"):
        ax_rb.bar(xs, rb, width=bar_w * 0.92,
                  color=(fc if multi else RB_COLOR), alpha=0.85,
                  label=lab if multi else None)
        for xi, r in zip(xs, rb):
            ax_rb.annotate(f"{r:.0f}", (xi, r), textcoords="offset points",
                           xytext=(0, 3), ha="center", fontsize=7.5)
        max_rb = max(max_rb, float(rb.max()))
    ax_rb.set_xticks(x)
    ax_rb.set_xticklabels([f"{t:g}" for t in thr_all])
    ax_rb.set_xlabel("per_pos_threshold")
    ax_rb.set_ylabel("mean rollbacks / problem")
    ax_rb.set_ylim(0, max_rb * 1.15)
    ax_rb.set_title("Rollback cost vs threshold", fontsize=10)
    ax_rb.grid(alpha=0.25, axis="y")
    if multi:
        ax_rb.legend(fontsize=7)

    # ---- Figure title ----
    if args.title:
        title = args.title
    elif not multi:
        nprob = sweeps[labels[0]][0].get("n_problems", "?")
        title = f"per_pos_threshold sweep — {labels[0]}  (n={nprob})"
    else:
        title = "per_pos_threshold sweep — " + " vs ".join(labels)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[plot] saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

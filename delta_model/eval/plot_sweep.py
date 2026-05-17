"""Plot a per_pos_threshold sweep produced by `eval/gsm8k_e2e.py`.

`gsm8k_e2e.py --out_json ...` writes a JSON list, one dict per
`per_pos_threshold`, with accuracy / speedup / rollback fields. This
script turns one (or more) of those files into a 2×2 figure:

  ┌─────────────────────────────┬──────────────────────────────┐
  │  (0,0) Accuracy–speed       │  (0,1) Accuracy vs threshold │
  │        tradeoff (Pareto)    │        — bars, one per file  │
  ├─────────────────────────────┼──────────────────────────────┤
  │  (1,0) Speedup vs threshold │  (1,1) Rollback cost         │
  │        — line, one per file │        vs threshold — bars   │
  └─────────────────────────────┴──────────────────────────────┘

Pareto (top-left): one threshold-ordered polyline per file. The Y axis
zooms to the actual accuracy range (`min*0.9` floor) so overlapping
sweeps stay distinguishable. Vanilla-accuracy and 1.0x-speedup
references are dashed lines.

Accuracy panel (top-right): grouped bars per threshold. Each bar
text-labelled with `accuracy_delta` (`+2.5%` etc.) so the reader gets
the exact hybrid-vs-vanilla gap independent of axis scale.

Speedup panel (bottom-left): line+marker per file, text-labelled with
the absolute ratio (`1.07x`). A dashed 1.0x reference marks parity.

Rollback panel (bottom-right): mean rollbacks per problem per
threshold — the mechanism behind the speedup collapse.

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

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_par = axes[0, 0]   # accuracy–speed Pareto
    ax_acc = axes[0, 1]   # accuracy bars vs threshold
    ax_spd = axes[1, 0]   # speedup line vs threshold
    ax_rb  = axes[1, 1]   # rollback bars vs threshold

    # ===================================================================
    # Panel (0,0) — accuracy–speed Pareto
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
    # Y-axis zooms to the actual accuracy range. min*0.9 floor with a
    # safety net so vanilla baseline + a small headroom above the max
    # point are always visible. Was: ylo=-0.03 which squashed all points
    # near the top of the panel when overlaying multiple sweeps.
    min_acc, max_acc = min(all_acc), max(all_acc)
    ylo = max(0.0, min(min_acc, van) * 0.9)
    yhi = max(max_acc, van) + max(0.02, (max_acc - ylo) * 0.10)
    ax_par.set_xlim(xlo, xhi)
    ax_par.set_ylim(ylo, yhi)
    ax_par.axhline(van, ls="--", color=BASE_COLOR, lw=1)
    ax_par.axvline(1.0, ls="--", color=BASE_COLOR, lw=1)
    if xhi > 1.0 and van < yhi:
        # Shaded "win zone" (faster than vanilla AND ≥ vanilla accuracy).
        # The in-plot text label was removed — it overlapped data points
        # when sweeps were overlaid. The shading + the dashed reference
        # lines already encode the meaning unambiguously.
        ax_par.add_patch(Rectangle(
            (1.0, van), xhi - 1.0, yhi - van,
            facecolor="tab:green", alpha=0.10, edgecolor="none", zorder=0,
        ))
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
    # Panel (0,1) — accuracy bars vs threshold
    # ===================================================================
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
    ax_acc.axhline(van, ls="--", color=BASE_COLOR, lw=1)
    ax_acc.text(0.98, 0.97, f"vanilla accuracy {van:.2f}",
                transform=ax_acc.transAxes, va="top", ha="right",
                fontsize=8, color=BASE_COLOR)

    # Accuracy axis: zoom to the actual range with 10% headroom. min*0.9
    # floor matches the Pareto panel so vertical comparisons across the
    # two are meaningful. Clamp ticks above 1.0 since accuracy ≤ 1.
    acc_lo = max(0.0, min(min_acc, van) * 0.9)
    acc_hi = min(1.0, max(max_acc, van) + max(0.02, (max_acc - acc_lo) * 0.10))
    ax_acc.set_ylim(acc_lo, acc_hi)
    ax_acc.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: "" if v > 1.0 + 1e-9 else f"{v:.2f}"
    ))
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([f"{t:g}" for t in thr_all])
    ax_acc.set_xlabel("per_pos_threshold")
    ax_acc.set_ylabel("accuracy (hybrid)")
    ax_acc.set_title("Accuracy vs threshold", fontsize=10)
    ax_acc.grid(alpha=0.25, axis="y")
    if multi:
        ax_acc.legend(fontsize=7, loc="upper right")

    # ===================================================================
    # Panel (1,0) — speedup line vs threshold (split from accuracy panel)
    # ===================================================================
    for xs, spd, lab, fc in _grouped("speedup_ratio"):
        lc = fc if multi else SPD_COLOR
        ax_spd.plot(xs, spd, "-o", color=lc, ms=6, lw=2.6, zorder=4,
                    label=lab if multi else None)
        for xi, s in zip(xs, spd):
            ax_spd.annotate(f"{s:.2f}x", (xi, s), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5, color=lc)
    # 1.0x reference: anything below this line is slower than vanilla.
    ax_spd.axhline(1.0, ls="--", color=BASE_COLOR, lw=1)
    ax_spd.text(0.98, 0.04, "1.0x = vanilla",
                transform=ax_spd.transAxes, va="bottom", ha="right",
                fontsize=8, color=BASE_COLOR)
    ax_spd.set_xticks(x)
    ax_spd.set_xticklabels([f"{t:g}" for t in thr_all])
    ax_spd.set_xlabel("per_pos_threshold")
    ax_spd.set_ylabel("speedup ratio  (vanilla walltime / hybrid)")
    ax_spd.set_ylim(0, max(all_spd) * 1.2)
    ax_spd.set_title("Speedup vs threshold", fontsize=10)
    ax_spd.grid(alpha=0.25, axis="y")
    if multi:
        ax_spd.legend(fontsize=7, loc="upper right")

    # ===================================================================
    # Panel (1,1) — rollback cost vs threshold
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

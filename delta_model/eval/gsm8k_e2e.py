"""End-to-end GSM8K accuracy + speedup eval for the hybrid runner.

Compares the hybrid `generate_with_delta` against the vanilla
`generate_with_prefix_cache` baseline on the same `seed`-shuffled
GSM8K test slice. Logs accuracy delta, wall-clock speedup, and mean
rollback count per problem.

CLI (run from inside the tata repo root):
    python -m delta_model.eval.gsm8k_e2e \\
        --delta_ckpt ckpts/m1_llada_variant_c/step_0020000.pt \\
        --n_problems 200 \\
        --per_pos_thresholds 0.70,0.80,0.85,0.90,0.95
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data import schema as S
from ..llada_runtime import load_llada
from ..inference.hybrid_runner import generate_with_delta
from ..models.variant_c import VariantC


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> Optional[str]:
    """Best-effort GSM8K answer extraction. Try '#### N' marker first,
    then last number in the text."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def _format_prompt_llada(tokenizer, question: str) -> torch.Tensor:
    msg = [{"role": "user", "content": question}]
    p = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    return torch.tensor(tokenizer(p)["input_ids"], dtype=torch.long).unsqueeze(0).cuda()


def _vanilla_prefix_cache_generate(model, prompt: torch.Tensor, *,
                                     gen_length: int, block_length: int,
                                     threshold: float | None,
                                     factor: float | None,
                                     mask_id: int):
    """Run Fast-dLLM v1 generate_with_prefix_cache on this prompt.

    Imported lazily so this module doesn't pin the path at import time.
    Exactly one of `threshold` / `factor` must be set; the other is None.
    """
    from generate import generate_with_prefix_cache  # noqa: WPS433
    out = generate_with_prefix_cache(
        model, prompt,
        steps=128, gen_length=gen_length, block_length=block_length,
        temperature=0.0, remasking="low_confidence",
        mask_id=mask_id, threshold=threshold, factor=factor,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def run_gsm8k_eval(
    *,
    backbone_model=None,
    delta_model=None,
    delta_ckpt: str | None = None,
    fast_dllm_path: str | None = None,
    n_problems: int = 200,
    per_pos_threshold: float = 0.85,
    seed: int = 42,
    threshold: float | None = None,
    factor: float | None = 1.0,
    inner_loop_max_iter: int | None = None,
    show_progress: bool = True,
) -> dict:
    """Run hybrid + vanilla on the same GSM8K slice. Returns metrics dict.

    Decoding mode: pass exactly one of `threshold` (Fast-dLLM fixed) or
    `factor` (Fast-dLLM dynamic, default 1.0 = paper-recommended). Must
    match whatever was used at collect time, or the delta model's training
    distribution won't match what it sees here.
    """
    if (threshold is None) == (factor is None):
        raise ValueError(
            "run_gsm8k_eval: pass exactly one of `threshold` or `factor` "
            f"(got threshold={threshold!r}, factor={factor!r})"
        )
    from datasets import load_dataset

    if backbone_model is None or delta_model is None:
        # Standalone invocation path. Train.py passes both in, eval CLI loads here.
        if backbone_model is None:
            backbone_model, tokenizer = load_llada(fast_dllm_path=fast_dllm_path)
        else:
            tokenizer = backbone_model._tokenizer  # set by load_llada wrapper if avail
        if delta_model is None:
            assert delta_ckpt is not None, "Need delta_ckpt or in-memory delta_model"
            ckpt = torch.load(delta_ckpt, map_location="cpu", weights_only=False)
            cfg_m = ckpt["cfg"]["model"]
            # Read backbone hyperparams off the loaded LLaDA so RoPE / RMSNorm
            # match between the variantc and what produced h_target.
            bb_cfg = {
                "rope_theta":  float(getattr(backbone_model.config, "rope_theta", 1e6)),
                "rms_eps":     float(getattr(backbone_model.config, "layer_norm_eps", 1e-5)),
                "max_seq_len": int(getattr(backbone_model.config, "max_sequence_length", 8192)),
            }
            delta_model = VariantC(
                d_model=cfg_m["d_model"], n_heads=cfg_m["n_heads"],
                n_layers=cfg_m["n_layers"],
                d_ff_inner=cfg_m.get("d_ff_inner"),
                dropout=cfg_m.get("dropout", 0.0),
                rope_theta=bb_cfg["rope_theta"],
                rms_eps=bb_cfg["rms_eps"],
                max_seq_len=bb_cfg["max_seq_len"],
            ).cuda().eval()
            delta_model.load_state_dict(ckpt["model"])
    else:
        tokenizer = backbone_model._tokenizer
    delta_model.eval()

    # Pull frozen norm + lm_head + token_embed off the backbone.
    transformer = backbone_model.model.transformer
    final_norm  = transformer.ln_f.cuda()
    lm_head     = transformer.ff_out.cuda()
    token_embed = transformer.wte.cuda()

    ds = load_dataset("gsm8k", "main", split="test")
    if n_problems < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_problems))

    hits_hybrid = hits_vanilla = 0
    rollbacks: list[int] = []
    backbone_calls: list[int] = []
    vanilla_backbone_calls: list[int] = []
    delta_forwards: list[int] = []
    disagreements:  list[int] = []
    rollback_rates: list[float] = []          # disagreements / delta_forwards, per problem
    revealed_at_finish: list[float] = []     # block_length = fully decoded
    n_blocks_finished:  list[int]  = []      # of NUM_BLOCKS blocks per problem
    t_hybrid = t_vanilla = 0.0

    # Counter for vanilla backbone forwards. Registered on the top-level
    # `backbone_model` only for the duration of the vanilla call below — so
    # hybrid forwards (which already self-report via stats["backbone_forwards"])
    # are NOT double-counted here.
    _van_counter = {"n": 0}
    def _van_count_hook(module, inputs):
        _van_counter["n"] += 1

    pbar = tqdm(
        ds, desc=f"gsm8k thr={per_pos_threshold:.2f}",
        dynamic_ncols=True, disable=not show_progress, leave=show_progress,
    )
    for i, prob in enumerate(pbar):
        prompt_ids = _format_prompt_llada(tokenizer, prob["question"])
        gold_num = extract_final_number(prob["answer"])

        # Hybrid run.
        t0 = time.time()
        out_ids, stats = generate_with_delta(
            backbone_model, delta_model, final_norm, lm_head, token_embed,
            prompt_ids,
            per_pos_threshold=per_pos_threshold,
            threshold=threshold, factor=factor,
            inner_loop_max_iter=inner_loop_max_iter,
        )
        t_hybrid += time.time() - t0
        gen_text_h = tokenizer.decode(
            out_ids[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True,
        )
        pred_h = extract_final_number(gen_text_h)
        hits_hybrid += int(pred_h is not None and pred_h == gold_num)
        rollbacks.append(stats["rollbacks"])
        backbone_calls.append(stats["backbone_forwards"])
        # Delta-pass disagreement rate. `disagreements` is the count of delta
        # passes where Fast-dLLM wanted more tokens than the per-pos confidence
        # gate let through (and hence forced a rollback on the next pass).
        # rollback_rate = disagreements / delta_forwards is the per-pass
        # probability that the small model's commit set got vetoed at ≥ 1
        # position — independent of how many backbone forwards that triggered.
        n_dlt = int(stats.get("delta_forwards", 0))
        n_dis = int(stats.get("disagreements", 0))
        delta_forwards.append(n_dlt)
        disagreements.append(n_dis)
        rollback_rates.append(n_dis / n_dlt if n_dlt > 0 else 0.0)
        # Cascading-failure diagnostic — mean revealed positions per block
        # (= S.BLOCK_LENGTH iff every block finished cleanly).
        revealed = stats.get("per_block_revealed_at_finish", [])
        if revealed:
            revealed_at_finish.append(float(np.mean(revealed)))
            n_blocks_finished.append(
                int(sum(1 for r in revealed if r == S.BLOCK_LENGTH))
            )

        # Vanilla run. Hook is registered only here so hybrid forwards
        # (counted by hybrid_runner via stats["backbone_forwards"]) are not
        # double-counted into the vanilla column.
        _van_counter["n"] = 0
        _van_handle = backbone_model.register_forward_pre_hook(_van_count_hook)
        t0 = time.time()
        try:
            out_v = _vanilla_prefix_cache_generate(
                backbone_model, prompt_ids,
                gen_length=S.GEN_LENGTH, block_length=S.BLOCK_LENGTH,
                threshold=threshold, factor=factor,
                mask_id=S.LLADA_MASK_TOKEN_ID,
            )
        finally:
            _van_handle.remove()
        t_vanilla += time.time() - t0
        vanilla_backbone_calls.append(_van_counter["n"])
        gen_text_v = tokenizer.decode(
            out_v[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=True,
        )
        pred_v = extract_final_number(gen_text_v)
        hits_vanilla += int(pred_v is not None and pred_v == gold_num)

        # Live readout: per-problem decode time hybrid vs vanilla (running
        # means), the speedup ratio, and running accuracy. The bar itself
        # shows the current problem number / total + ETA.
        done = i + 1
        pbar.set_postfix({
            "hyb":   f"{t_hybrid / done:.2f}s",
            "van":   f"{t_vanilla / done:.2f}s",
            "x":     f"{t_vanilla / max(1e-6, t_hybrid):.2f}",
            "acc_h": f"{hits_hybrid / done:.2f}",
            "acc_v": f"{hits_vanilla / done:.2f}",
            "bb_h":  f"{np.mean(backbone_calls):.1f}",
            "bb_v":  f"{np.mean(vanilla_backbone_calls):.1f}",
            "rb%":   f"{100 * np.mean(rollback_rates):.0f}",
        }, refresh=False)

    n = len(ds)
    out = {
        "n_problems":            n,
        "accuracy_hybrid":       hits_hybrid / n,
        "accuracy_vanilla":      hits_vanilla / n,
        "accuracy_delta":        (hits_hybrid - hits_vanilla) / n,
        "speedup_ratio":         (t_vanilla / max(1e-6, t_hybrid)),
        "mean_rollbacks":             float(np.mean(rollbacks)),
        "mean_backbone_calls":        float(np.mean(backbone_calls)),
        "mean_vanilla_backbone_calls": float(np.mean(vanilla_backbone_calls)),
        "mean_delta_forwards":        float(np.mean(delta_forwards)),
        "mean_disagreements":         float(np.mean(disagreements)),
        # Per-problem rollback rate averaged across problems (NOT
        # sum(disagreements) / sum(delta_forwards), which would over-weight
        # problems with more delta passes). Bounded in [0, 1]; high values
        # mean the per-pos confidence gate is vetoing most delta-pass
        # commits — the small model isn't being trusted on this slice.
        "rollback_rate":              float(np.mean(rollback_rates)),
        "per_pos_threshold":          per_pos_threshold,
        "inner_loop_max_iter":        inner_loop_max_iter,
    }
    if revealed_at_finish:
        # Diagnostic for the cascading-failure mode (mass at block_length =
        # all blocks finished cleanly; mass below block_length = blocks
        # hit budget with mask tokens still present → downstream blocks
        # see corrupted context).
        out["mean_revealed_per_block"]  = float(np.mean(revealed_at_finish))
        out["mean_blocks_finished"]     = float(np.mean(n_blocks_finished))
        out["block_length"]             = S.BLOCK_LENGTH
        out["num_blocks_per_problem"]   = S.NUM_BLOCKS
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--delta_ckpt", required=True)
    p.add_argument("--fast_dllm_path", default=None)
    p.add_argument("--n_problems", type=int, default=200)
    p.add_argument(
        "--per_pos_thresholds", default="0.70,0.80,0.85,0.90,0.95",
        help="Comma-separated per-position confidence thresholds to sweep "
             "(§3.2 agreement decoding). A position commits only when both "
             "Fast-dLLM wants it AND c_pos[i] >= threshold; any disagreement "
             "forces a backbone rollback at the next iter.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Fast-dLLM fixed-threshold decoding. Mutually exclusive with "
             "--factor. If neither is set, defaults to --factor 1.0.",
    )
    p.add_argument(
        "--factor", type=float, default=None,
        help="Fast-dLLM dynamic per-rank threshold. Mutually exclusive with "
             "--threshold. Must match the mode used at collect time.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    p.add_argument(
        "--inner_loop_max_iter", type=int, default=None,
        help="Hard cap on the agreement-decoding inner while loop. "
             "Default = None → resolves to 2 * BLOCK_LENGTH (= 64), "
             "high enough that every block finishes naturally but "
             "bounded enough to catch degenerate cases. The 6 in collect "
             "is a recording budget, NOT a loop budget — see "
             "hybrid_runner.generate_with_delta docstring. To reproduce "
             "the pre-2026-05-13 behavior of the T4 sweep, pass "
             "--inner_loop_max_iter 6.",
    )
    args = p.parse_args()

    if args.threshold is not None and args.factor is not None:
        p.error("--threshold and --factor are mutually exclusive")
    if args.threshold is None and args.factor is None:
        decoding_threshold: float | None = None
        decoding_factor: float | None = 1.0
    else:
        decoding_threshold = args.threshold
        decoding_factor = args.factor
    print(
        f"[gsm8k] decoding mode: "
        + (f"factor={decoding_factor}" if decoding_factor is not None
           else f"threshold={decoding_threshold}"),
        flush=True,
    )

    rows = []
    for thr in [float(x) for x in args.per_pos_thresholds.split(",") if x.strip()]:
        print(f"[gsm8k] per_pos_threshold={thr}", flush=True)
        m = run_gsm8k_eval(
            delta_ckpt=args.delta_ckpt,
            fast_dllm_path=args.fast_dllm_path,
            n_problems=args.n_problems,
            per_pos_threshold=thr,
            seed=args.seed,
            threshold=decoding_threshold,
            factor=decoding_factor,
            inner_loop_max_iter=args.inner_loop_max_iter,
        )
        print(json.dumps(m, indent=2), flush=True)
        rows.append(m)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"[gsm8k] saved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()

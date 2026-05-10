"""§3.1 verification — partial-forward vs full-forward equivalence.

CONTEXT:
    `collect_llada.collect_one_sample` runs a full forward at pass 0 and
    *partial* forwards (over `x[:, s:]` with cached prefix KV) at passes ≥ 1.
    If LLaDA's modeling code does not auto-derive `position_ids` as
    `[past_seq_len, past_seq_len+1, …]` when called with `past_key_values`,
    then RoPE rotates the block tokens from absolute position 0 instead of
    `s`. Every `h_per_pass[i ≥ 1]` in the cache would then be silently
    corrupted, the delta model trains on the wrong distribution, and e2e
    GSM8K collapses while in-distribution val metrics still look fine.

THIS TEST:
    Build a synthetic `x` (real prompt + all-mask gen region), run BOTH:
        (A) `model(x, use_cache=True)`               — full forward
        (B) `model(x[:, s:], past_key_values=cache_to_s, use_cache=True)`
            where cache_to_s is the prefix KV from (A) sliced to `[:s]`.
    Compare last-block hidden states at the block region `[s:s+BL]`. They
    should be numerically equivalent (within fp16 noise) iff the partial
    forward applies RoPE at absolute positions `s..s+BL-1` like the full
    forward does.

PASS:  max abs diff ≤ 5e-3   → §3.1 ruled out as a bug.
FAIL:  max abs diff > 5e-3   → bug confirmed; the cache is poisoned.

If FAIL: the fix is to pass
    position_ids=torch.arange(s, x.shape[1], device=device).unsqueeze(0)
explicitly to the partial forward in `collect_llada.py:332`, then
re-collect from scratch.

Run (from inside the tata repo root, GPU + Fast-dLLM required):
    python -m delta_model.sanity.test_partial_full_forward_equivalence \\
        --fast_dllm_path external/Fast-dLLM/v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..data import schema as S
from ..llada_runtime import load_llada


# fp16 noise tolerance. Hidden states are saved in fp16 in collect, so
# anything above ~1e-2 in absolute terms is structural divergence, not noise.
_FP16_NOISE_TOL = 5e-3


def _find_last_block(model: torch.nn.Module) -> torch.nn.Module:
    last = None
    for m in model.modules():
        if type(m).__name__ in ("LLaDASequentialBlock", "LLaDALlamaBlock"):
            last = m
    if last is None:
        raise RuntimeError(
            "No LLaDA transformer blocks found — model class names changed?"
        )
    return last


def _format_prompt(tokenizer, question: str) -> torch.Tensor:
    msg = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=False,
    )
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast_dllm_path", type=str, default=None,
                    help="Fast-dLLM v1 root (overrides env / default).")
    ap.add_argument("--block_idx", type=int, default=0,
                    help="which block to test (0..NUM_BLOCKS-1). The block's "
                         "absolute start is `prompt_len + block_idx * BLOCK_LENGTH`.")
    ap.add_argument("--prompt", type=str, default=None,
                    help="Prompt text to use. Default: first non-comment line "
                         "of delta_model/data/sample_prompts.txt.")
    args = ap.parse_args()

    if args.block_idx < 0 or args.block_idx >= S.NUM_BLOCKS:
        raise ValueError(f"--block_idx must be in [0, {S.NUM_BLOCKS-1}]")

    print("[sanity] loading LLaDA …", flush=True)
    model, tokenizer = load_llada(fast_dllm_path=args.fast_dllm_path)
    model.eval()
    device = model.device

    # Pick a prompt.
    if args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompts_file = (
            Path(__file__).resolve().parents[1] / "data" / "sample_prompts.txt"
        )
        raw = [
            ln.strip() for ln in prompts_file.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not raw:
            raise RuntimeError(f"No usable prompt found in {prompts_file}")
        prompt_text = raw[0]
        print(f"[sanity] using sample prompt: {prompt_text!r}", flush=True)

    prompt_ids = _format_prompt(tokenizer, prompt_text).to(device)   # [1, Lp]
    Lp = int(prompt_ids.shape[1])
    if Lp < S.PREFIX_WINDOW:
        raise ValueError(
            f"prompt too short ({Lp} tokens); pick one with ≥ {S.PREFIX_WINDOW}."
        )

    BL = S.BLOCK_LENGTH
    GL = S.GEN_LENGTH
    s = Lp + args.block_idx * BL                                     # block start abs pos
    e = s + BL

    # Build x: prompt + all-mask gen region (matches the state at pass 0 of collect).
    x = torch.full((1, Lp + GL), S.LLADA_MASK_TOKEN_ID,
                   dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids

    print(f"[sanity] prompt_len={Lp}  block_idx={args.block_idx}  "
          f"block_start_pos={s}  seq_len={x.shape[1]}", flush=True)

    # Hook to capture last-block (= last layer pre-final-norm) hidden states.
    capture: dict[str, torch.Tensor | None] = {"latest": None}

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        capture["latest"] = h.detach().clone()

    last_block = _find_last_block(model)
    handle = last_block.register_forward_hook(_hook)

    try:
        # ---- Path A: full forward over all of x.
        out_full = model(x, use_cache=True)
        h_full_seq = capture["latest"]                                # [1, Lp+GL, d_model]
        h_full_block = h_full_seq[:, s:e, :].clone()                  # [1, BL, d_model]
        past_key_values = out_full.past_key_values

        # Slice prefix KV to keep only [:s] — same op collect_llada does.
        past_to_s = [
            tuple(t[:, :, :s, :] for t in pkv) for pkv in past_key_values
        ]

        # ---- Path B1: partial forward WITHOUT position_ids (the broken path
        # that collect_llada used pre-§3.1 fix). Reproduces the original bug.
        capture["latest"] = None
        _ = model(x[:, s:], past_key_values=past_to_s, use_cache=True)
        h_b1_seq = capture["latest"]
        h_b1_block = h_b1_seq[:, :BL, :].clone()

        # ---- Path B2: partial forward WITH explicit position_ids. This
        # mirrors the post-§3.1 fix in collect_llada.py:332.
        position_ids = torch.arange(
            s, x.shape[1], device=device,
        ).unsqueeze(0)
        capture["latest"] = None
        _ = model(
            x[:, s:], past_key_values=past_to_s, use_cache=True,
            position_ids=position_ids,
        )
        h_b2_seq = capture["latest"]
        h_b2_block = h_b2_seq[:, :BL, :].clone()
    finally:
        handle.remove()

    base_max = float(h_full_block.abs().max().item())

    def _report(tag: str, h_other: torch.Tensor) -> tuple[float, float]:
        diff = (h_full_block.float() - h_other.float()).abs()
        max_d  = float(diff.max().item())
        mean_d = float(diff.mean().item())
        rel    = max_d / max(base_max, 1e-9)
        ok     = max_d <= _FP16_NOISE_TOL
        marker = "✓" if ok else "✗"
        print(
            f"[sanity] {marker} {tag}: max={max_d:.4e}  mean={mean_d:.4e}  "
            f"relative={rel:.4e}"
        )
        return max_d, mean_d

    print(f"[sanity] |h_full_block|.max = {base_max:.4e}")
    print(f"[sanity] tolerance = {_FP16_NOISE_TOL:.0e}")
    print()
    print("[sanity] Path B1 — partial forward, NO position_ids (collect-pre-fix path):")
    b1_max, _ = _report("B1 vs A (full)", h_b1_block)
    print()
    print("[sanity] Path B2 — partial forward, WITH position_ids (collect-post-fix path):")
    b2_max, _ = _report("B2 vs A (full)", h_b2_block)
    print()

    b1_ok = b1_max <= _FP16_NOISE_TOL
    b2_ok = b2_max <= _FP16_NOISE_TOL

    if b1_ok and b2_ok:
        print(
            "[sanity] ✓ PASS — both partial-forward paths agree with the full "
            "forward. §3.1 was never a bug on this LLaDA build."
        )
        sys.exit(0)
    elif (not b1_ok) and b2_ok:
        print(
            "[sanity] ✓ FIX VERIFIED — without position_ids the partial forward "
            "is broken (B1 diverges), but passing position_ids=arange(s, len) "
            "recovers full-forward equivalence (B2 within tolerance)."
        )
        print(
            "[sanity]   Action: confirm collect_llada.py passes position_ids "
            "to the partial forward, then re-collect the cache from scratch."
        )
        sys.exit(0)
    elif (not b1_ok) and (not b2_ok):
        print(
            "[sanity] ✗ FAIL — both partial paths diverge from full. position_ids "
            "alone does not fix it; LLaDA's modeling code likely needs deeper "
            "audit (cache_position kwarg? attention_mask? layer_past slicing?)."
        )
        sys.exit(1)
    else:
        print(
            "[sanity] ✗ FAIL — B1 (no position_ids) passes but B2 (with position_ids) "
            "diverges. Unexpected; LLaDA may interpret position_ids differently. "
            "Print q/k positions in the RoPE module to investigate."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

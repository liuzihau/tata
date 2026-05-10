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

        # ---- Path B: partial forward over x[:, s:] with cached prefix.
        # NOTE: this mirrors the call in collect_llada.py:332 — no
        # `position_ids` argument is passed, so this test reproduces the
        # exact code path used to build the training cache.
        capture["latest"] = None
        _ = model(x[:, s:], past_key_values=past_to_s, use_cache=True)
        h_partial_seq = capture["latest"]                             # [1, GL+Lp-s, d_model]
        h_partial_block = h_partial_seq[:, :BL, :].clone()            # [1, BL, d_model]
    finally:
        handle.remove()

    # Compare.
    diff = (h_full_block.float() - h_partial_block.float()).abs()
    max_diff  = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    base_max  = float(h_full_block.abs().max().item())
    rel_max   = max_diff / max(base_max, 1e-9)

    print(f"[sanity] |h_full_block|.max  = {base_max:.4e}")
    print(f"[sanity] |h_part_block|.max  = {float(h_partial_block.abs().max().item()):.4e}")
    print(f"[sanity] |full - partial|: max = {max_diff:.4e}  mean = {mean_diff:.4e}  "
          f"relative-to-max = {rel_max:.4e}")

    if max_diff <= _FP16_NOISE_TOL:
        print(
            f"[sanity] ✓ PASS — partial and full forward agree to within "
            f"{_FP16_NOISE_TOL:.0e}. §3.1 position-id alignment is fine."
        )
        sys.exit(0)
    else:
        print(
            f"[sanity] ✗ FAIL — divergence {max_diff:.3e} > {_FP16_NOISE_TOL:.0e}."
        )
        print(
            f"[sanity]   Consistent with the §3.1 position-id bug: the partial "
            f"forward at iter ≥ 1 is rotating Q/K from absolute position 0 "
            f"instead of {s}, so all `h_per_pass[i ≥ 1]` in the cache are "
            f"computed at wrong positions."
        )
        print(
            "[sanity]   Likely fix: in `collect_llada.py:332`, pass\n"
            "[sanity]     position_ids=torch.arange(s, x.shape[1], device=x.device).unsqueeze(0)\n"
            "[sanity]   to the partial forward, then re-collect the cache from scratch."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

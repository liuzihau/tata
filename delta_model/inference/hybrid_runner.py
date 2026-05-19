"""Hybrid backbone-then-delta runner.

Inference engine that mirrors Fast-dLLM v1 `generate_with_prefix_cache`
(`v1/llada/generate.py:132`) but swaps the in-block backbone forward for
the delta model at iter ≥ 1. §3.2 agreement decoding gates per-position
commits via the delta model's per-position confidence head:

  At each delta pass, run Fast-dLLM's standard token-transfer selection
  on logits = lm_head(final_norm(h_ref + Δh_pred)) → returns positions
  Fast-dLLM wants to commit (top-K by per-position confidence with the
  dynamic-rank or fixed-threshold rule). We then intersect this set with
  the per-position confidence head's signal: a position commits only if
  BOTH (a) Fast-dLLM wants it AND (b) c_pos[i] ≥ per_pos_threshold. Any
  disagreement (Fast-dLLM wants more positions than agreement-set has)
  forces a backbone rollback at the start of the *next* iter.

Rollback uses a *partial* backbone forward with the cached prefix from
pass 0, matching collect's iter ≥ 1 path (see improvements.md §3.1) so
the refreshed h_ref stays in the same distribution the delta model was
trained on. A rollback then commits tokens via *vanilla* Fast-dLLM on the
backbone's own logits — no confidence gate, because that gate only judges
the *delta model's* predictions, not a real backbone forward. So a
rollback always advances the block by ≥ 1 token; it can't livelock
paying a backbone forward that then commits nothing.

Returned stats include rollback / disagreement counts so the eval
harness can plot accuracy / speed / per-position-threshold curves.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import torch
import torch.nn as nn

from ..data import schema as S
from ..llada_runtime import (
    _get_num_transfer_tokens,
    _get_transfer_index,
    _get_transfer_index_dynamic,
)


def _find_last_block(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if type(m).__name__ in ("LLaDASequentialBlock", "LLaDALlamaBlock"):
            last = m
    try:
        return last
    except UnboundLocalError as e:
        raise RuntimeError("No LLaDA blocks found in backbone.") from e


@torch.no_grad()
def generate_with_delta(
    model,                                  # full LLaDA backbone (eval mode)
    delta_model: nn.Module,
    final_norm: nn.Module,
    lm_head: nn.Linear,
    token_embed: nn.Embedding,
    prompt: torch.Tensor,                   # [1, S], on model.device
    *,
    gen_length: int = S.GEN_LENGTH,
    block_length: int = S.BLOCK_LENGTH,
    max_iter_per_block: int = S.MAX_ITER,
    inner_loop_max_iter: int | None = None,
    per_pos_threshold: "float | Callable[[int, float], float]" = 0.85,
    threshold: float | None = None,         # Fast-dLLM fixed-threshold mode
    factor: float | None = 1.0,             # Fast-dLLM dynamic-threshold mode
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = S.LLADA_MASK_TOKEN_ID,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Returns (final_token_ids, stats).

    Decoding mode is controlled by exactly one of `threshold` or `factor`
    (the other must be None). The default factor=1.0 matches the collect
    pipeline's default and the Fast-dLLM paper's recommended setting.

    `per_pos_threshold` ∈ (0, 1] gates per-position commits via the delta
    model's per-position confidence head. Set to 0 to commit everything
    Fast-dLLM wants (fastest, most lenient). Set to 1 to gate out every
    delta commit — each delta pass then disagrees and the following
    rollback advances the block via vanilla Fast-dLLM, i.e. it degrades
    gracefully to the backbone-only baseline (one wasted delta forward
    per rollback, but no livelock). Useful sweep: {0.7, 0.8, 0.85, 0.9, 0.95}.

    `inner_loop_max_iter` is the hard upper bound on the agreement-decoding
    inner while loop. `None` (default) resolves to `2 * block_length`
    (= 64 at block_length=32) — high enough that every block finishes
    naturally, low enough to bound runtime in degenerate cases (e.g.
    per_pos_threshold so high that no position ever commits).

    This is intentionally **decoupled** from `max_iter_per_block`. The
    "6" in collect is a *recording-budget* (cap on how many `h_per_pass`
    snapshots get stored per block, for storage / training-distribution
    reasons), not a *loop-budget*: collect's own decoding loop runs
    until the block is fully decoded (cf. `collect_llada.py`). Inference
    should match — the delta model has no iter-index input and can't
    tell whether it's at training-iter-3 or inference-iter-15. To
    reproduce the old behavior (pre-2026-05-13), pass
    `inner_loop_max_iter=max_iter_per_block` (= 6).

    `max_iter_per_block` is still used as Fast-dLLM's per-iter transfer
    quota in `threshold` mode (unchanged).

    stats keys: rollbacks (int — backbone passes triggered by a delta
                disagreement; each commits ≥ 1 token via vanilla Fast-dLLM),
                backbone_forwards (int), delta_forwards (int),
                disagreements (int — delta passes where Fast-dLLM wanted more
                than the per-pos head agreed to), walltime (float, sec),
                per_block_passes (list[int]),
                per_block_revealed_at_finish (list[int] — non-mask positions
                in each block when its inner loop exited; reaches
                `block_length` for cleanly-finished blocks).
    """
    if (threshold is None) == (factor is None):
        raise ValueError(
            "generate_with_delta: pass exactly one of `threshold` or `factor` "
            f"(got threshold={threshold!r}, factor={factor!r})"
        )
    device = model.device
    B = prompt.shape[0]
    assert B == 1, "hybrid_runner currently only supports B=1"
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt

    # Inner-loop hard cap. None resolves to `2 * block_length` (=64 at
    # block_length=32) — generous enough that every block finishes
    # naturally, bounded enough to catch degenerate cases. Decoupled from
    # `max_iter_per_block` (which is a collect-side recording budget;
    # see docstring).
    loop_cap = (
        int(inner_loop_max_iter)
        if inner_loop_max_iter is not None
        else 2 * int(block_length)
    )

    stats = {
        "rollbacks": 0, "backbone_forwards": 0, "delta_forwards": 0,
        "disagreements": 0, "per_block_passes": [],
        "per_block_revealed_at_finish": [],
    }
    t0 = time.time()

    # Hook to capture last-layer block hidden state during backbone forwards.
    hook_state = {"latest": None, "s": 0, "e": 0, "is_full_forward": False}
    last_block = _find_last_block(model)

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if hook_state["is_full_forward"]:
            slice_h = h[:, hook_state["s"]:hook_state["e"], :]
        else:
            slice_h = h[:, :block_length, :]
        hook_state["latest"] = slice_h.detach().clone()           # [1, BL, D]

    handle = last_block.register_forward_hook(_hook)

    try:
        for nb in range(num_blocks):
            s = Lp + nb * block_length
            e = s + block_length

            block_mask_index = (x[:, s:e] == mask_id)
            num_transfer_tokens = _get_num_transfer_tokens(
                block_mask_index, max_iter_per_block,
            )

            # ---- Pass 0: backbone full forward ----
            hook_state["is_full_forward"] = True
            hook_state["s"] = s; hook_state["e"] = e
            out = model(x, use_cache=True)
            stats["backbone_forwards"] += 1
            past_key_values = out.past_key_values
            logits = out.logits

            # Slice prefix KV to keep only [:s].
            past_key_values = [
                tuple(t[:, :, :s, :] for t in pkv) for pkv in past_key_values
            ]

            # Last-PREFIX_WINDOW prefix KV for the delta model. Mirror
            # collect_llada's front-pad behavior: when block start `s` is
            # under PREFIX_WINDOW (short prompt at block 0), zero-pad the
            # leading slots and record which slots are real via the pad
            # mask below. Without this, the slice yields fewer than
            # PREFIX_WINDOW tokens and SDPA in the delta model broadcast-
            # fails against the fixed-size mask.
            W = S.PREFIX_WINDOW
            valid_len = min(int(s), W)
            pad_len   = W - valid_len
            real_K = past_key_values[-1][0][0, :, s - valid_len:s, :]
            real_V = past_key_values[-1][1][0, :, s - valid_len:s, :]
            if pad_len > 0:
                pad_shape = (real_K.shape[0], pad_len, real_K.shape[2])
                zeros_K = torch.zeros(pad_shape, dtype=real_K.dtype, device=real_K.device)
                zeros_V = torch.zeros(pad_shape, dtype=real_V.dtype, device=real_V.device)
                last_K = torch.cat([zeros_K, real_K], dim=1)
                last_V = torch.cat([zeros_V, real_V], dim=1)
            else:
                last_K, last_V = real_K, real_V
            prefix_kv_slice = torch.stack([last_K, last_V], dim=0).unsqueeze(0)

            # h_ref: last-block output at the masked block region.
            h_ref = hook_state["latest"]                                         # [1, BL, D]

            # Pass 0 token transfer (mask-only, restricted to current block).
            global_mask_index = (x == mask_id)
            global_mask_index[:, e:] = False
            if factor is not None:
                x0, transfer_index = _get_transfer_index_dynamic(
                    logits, temperature, remasking, global_mask_index, x,
                    factor=factor,
                )
            else:
                quota = None if threshold is not None else num_transfer_tokens[:, 0]
                x0, transfer_index = _get_transfer_index(
                    logits, temperature, remasking, global_mask_index, x,
                    quota, threshold,
                )
            x = torch.where(transfer_index, x0, x)
            n_passes = 1
            # `i_ref_step` tracks which pass's hidden state the current
            # h_ref reflects (= n_passes at the moment it was captured).
            # gap = n_passes - i_ref_step at the start of each delta pass.
            # Updated whenever a rollback refreshes h_ref.
            i_ref_step = 0

            # Pad mask: leading `pad_len` slots were zero-padded above
            # (short-prompt case at block 0); the trailing `valid_len`
            # slots are real KV. For prompts ≥ PREFIX_WINDOW tokens this
            # collapses to all-True (the pre-existing behavior).
            prefix_kv_pad_mask = torch.zeros(
                (1, W), dtype=torch.bool, device=device,
            )
            if valid_len > 0:
                prefix_kv_pad_mask[:, pad_len:] = True

            # ---- Passes 1..loop_cap-1: delta + agreement decode ----
            force_rollback_next = False
            while n_passes < loop_cap:
                if (x[:, s:e] == mask_id).sum().item() == 0:
                    break

                # If the previous (delta) pass disagreed, this pass is a
                # ROLLBACK: a pure backbone Fast-dLLM step.
                #
                # The per-position confidence gate is the *delta model's*
                # self-assessment — "can the small model match LLaDA here" —
                # so it does NOT apply to a real backbone forward. A rollback
                # follows vanilla Fast-dLLM on the backbone's own logits
                # (factor / threshold), which commits ≥ 1 token. That
                # guarantees forward progress: a rollback always pays one
                # backbone forward AND advances the block, instead of paying
                # the forward and then re-gating it away (the old livelock:
                # at high per_pos_threshold every rollback committed nothing,
                # so the block spun to loop_cap — ~62 backbone forwards/block,
                # 10x slower than vanilla).
                #
                # The partial forward + token transfer mirror collect's
                # iter ≥ 1 path exactly, so the refreshed h_ref AND the
                # committed tokens both stay in the delta model's training
                # distribution.
                if force_rollback_next:
                    hook_state["is_full_forward"] = False
                    hook_state["s"] = s; hook_state["e"] = e
                    out_rb = model(
                        x[:, s:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    stats["backbone_forwards"] += 1
                    stats["rollbacks"] += 1
                    # Refresh the delta model's anchor for the *next* delta
                    # pass. After the commit below, h_ref lags the reveal
                    # state — the in-distribution i_ref < i_target setup.
                    h_ref = hook_state["latest"]
                    # h_ref now reflects the just-completed rollback pass,
                    # so the next delta forward sees gap=1 again.
                    i_ref_step = n_passes
                    force_rollback_next = False

                    # Vanilla Fast-dLLM transfer on the backbone's own
                    # logits, restricted to the current block. No gate.
                    logits_rb = out_rb.logits                       # covers x[:, s:]
                    mask_rb = (x[:, s:] == mask_id)
                    mask_rb[:, block_length:] = False
                    if factor is not None:
                        x0_rb, transfer_rb = _get_transfer_index_dynamic(
                            logits_rb, temperature, remasking, mask_rb,
                            x[:, s:], factor=factor,
                        )
                    else:
                        quota = (
                            None if threshold is not None
                            else num_transfer_tokens[
                                :, min(n_passes, num_transfer_tokens.size(1) - 1)
                            ]
                        )
                        x0_rb, transfer_rb = _get_transfer_index(
                            logits_rb, temperature, remasking, mask_rb,
                            x[:, s:], quota, threshold,
                        )
                    new_suffix = torch.where(transfer_rb, x0_rb, x[:, s:])
                    x = torch.cat([x[:, :s], new_suffix], dim=1)
                    n_passes += 1
                    continue

                # Build prev_emb at the current state of the block.
                block_ids = x[0, s:e]
                prev_emb = token_embed(block_ids).to(
                    device=device, dtype=h_ref.dtype,
                ).unsqueeze(0)                                                  # [1, BL, D]

                # Delta forward — produces Δh and per-position confidence.
                block_start_pos_t = torch.tensor(
                    [s], dtype=torch.long, device=device,
                )
                delta_h, c_pos = delta_model(
                    h_ref.to(delta_model.delta_head.proj.weight.dtype),
                    prev_emb.to(delta_model.delta_head.proj.weight.dtype),
                    prefix_kv_slice.to(torch.float16),
                    block_start_pos_t,
                    prefix_kv_pad_mask=prefix_kv_pad_mask,
                )
                stats["delta_forwards"] += 1

                # Compute h_pred and logits for token transfer.
                h_pred_block = h_ref + delta_h.to(h_ref.dtype)                  # [1, BL, D]
                logits_block = lm_head(final_norm(h_pred_block))                # [1, BL, V]

                # Step 1: Fast-dLLM's standard transfer selection on
                # delta-derived logits.
                mask_blk = (x[:, s:e] == mask_id)                               # [1, BL]
                if factor is not None:
                    x0_blk, transfer_blk = _get_transfer_index_dynamic(
                        logits_block, temperature, remasking, mask_blk,
                        x[:, s:e], factor=factor,
                    )
                else:
                    quota = (
                        None if threshold is not None
                        else num_transfer_tokens[:, min(n_passes, num_transfer_tokens.size(1) - 1)]
                    )
                    x0_blk, transfer_blk = _get_transfer_index(
                        logits_block, temperature, remasking, mask_blk,
                        x[:, s:e], quota, threshold,
                    )
                # transfer_blk: [1, BL] bool — positions Fast-dLLM wants.

                # Step 2: per-position confidence gate. `per_pos_threshold`
                # may be a float (global gate) OR a callable
                # `(gap, reveal_frac) -> float` that returns a per-state
                # threshold (see delta_model/inference/thr_lookup.py).
                if callable(per_pos_threshold):
                    gap_now = n_passes - i_ref_step
                    reveal_frac_now = float(
                        (x[:, s:e] != mask_id).float().mean().item()
                    )
                    thr_now = float(per_pos_threshold(gap_now, reveal_frac_now))
                else:
                    thr_now = float(per_pos_threshold)
                per_pos_pass = (c_pos.float() >= thr_now)                        # [1, BL]
                agreement_blk = transfer_blk & per_pos_pass                      # [1, BL]

                # Step 3: commit ONLY agreement-set positions this pass.
                new_blk = torch.where(agreement_blk, x0_blk, x[:, s:e])
                x = torch.cat([x[:, :s], new_blk, x[:, e:]], dim=1)

                # Step 4: any disagreement → force rollback next pass.
                n_want   = int(transfer_blk.sum().item())
                n_agree  = int(agreement_blk.sum().item())
                if n_want > n_agree:
                    force_rollback_next = True
                    stats["disagreements"] += 1

                n_passes += 1

            stats["per_block_passes"].append(n_passes)
            stats["per_block_revealed_at_finish"].append(
                int((x[:, s:e] != mask_id).sum().item())
            )
    finally:
        handle.remove()

    stats["walltime"] = time.time() - t0
    return x, stats

"""Hybrid backbone-then-delta runner.

Inference engine that mirrors Fast-dLLM v1 `generate_with_prefix_cache`
(`v1/llada/generate.py:132`) but swaps the in-block backbone forward
for the delta model at passes ≥ 1. A confidence-head threshold triggers
a rollback — a fresh full backbone forward — which refreshes `h_ref`
and the prefix-KV slice the delta model attends to.

Returned stats include rollback counts (per block + total) so the eval
harness can plot accuracy / speed trade-offs by `conf_threshold`.
"""
from __future__ import annotations

import time
from typing import Any

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
    conf_threshold: float = 0.85,
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

    stats keys: rollbacks (int), backbone_forwards (int), delta_forwards (int),
                walltime (float, sec), per_block_passes (list[int]).
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

    stats = {
        "rollbacks": 0, "backbone_forwards": 0, "delta_forwards": 0,
        "per_block_passes": [],
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

            # Last-32 prefix KV for the delta model.
            last_K = past_key_values[-1][0][0, :, s - S.PREFIX_WINDOW:s, :]
            last_V = past_key_values[-1][1][0, :, s - S.PREFIX_WINDOW:s, :]
            prefix_kv_slice = torch.stack([last_K, last_V], dim=0).unsqueeze(0)   # [1, 2, KV_H, 32, d_head]

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

            # ---- Passes 1..max_iter_per_block-1: delta model w/ rollback ----
            while n_passes < max_iter_per_block:
                if (x[:, s:e] == mask_id).sum().item() == 0:
                    break

                # Build prev_emb at the current state of the block.
                block_ids = x[0, s:e]
                # mask_id positions get the mask embedding via token_embed lookup.
                prev_emb = token_embed(block_ids).to(
                    device=device, dtype=h_ref.dtype,
                ).unsqueeze(0)                                                  # [1, BL, D]

                # Delta forward.
                block_start_pos_t = torch.tensor(
                    [s], dtype=torch.long, device=device,
                )
                delta_h, c_pred = delta_model(
                    h_ref.to(delta_model.delta_head.proj.weight.dtype),
                    prev_emb.to(delta_model.delta_head.proj.weight.dtype),
                    prefix_kv_slice.to(torch.float16),
                    block_start_pos_t,
                )
                stats["delta_forwards"] += 1

                if c_pred.item() < conf_threshold:
                    # ---- Rollback: full backbone forward to refresh h_ref ----
                    hook_state["is_full_forward"] = True
                    hook_state["s"] = s; hook_state["e"] = e
                    out = model(x, use_cache=True)
                    stats["backbone_forwards"] += 1
                    stats["rollbacks"] += 1
                    past_key_values = out.past_key_values
                    logits = out.logits
                    past_key_values = [
                        tuple(t[:, :, :s, :] for t in pkv) for pkv in past_key_values
                    ]
                    last_K = past_key_values[-1][0][0, :, s - S.PREFIX_WINDOW:s, :]
                    last_V = past_key_values[-1][1][0, :, s - S.PREFIX_WINDOW:s, :]
                    prefix_kv_slice = torch.stack(
                        [last_K, last_V], dim=0,
                    ).unsqueeze(0)
                    h_ref = hook_state["latest"]
                else:
                    # Use the delta model's prediction as h_target.
                    h_pred_block = h_ref + delta_h.to(h_ref.dtype)              # [1, BL, D]
                    logits_block = lm_head(final_norm(h_pred_block))            # [1, BL, V]
                    # Pad logits up to the full sequence shape get_transfer_index expects.
                    # Easier path: use a block-only mask over the block and call _get_transfer_index
                    # with x[:, s:e] only.
                    mask_blk = (x[:, s:e] == mask_id)
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
                    new_blk = torch.where(transfer_blk, x0_blk, x[:, s:e])
                    x = torch.cat([x[:, :s], new_blk, x[:, e:]], dim=1)

                n_passes += 1

            stats["per_block_passes"].append(n_passes)
    finally:
        handle.remove()

    stats["walltime"] = time.time() - t0
    return x, stats

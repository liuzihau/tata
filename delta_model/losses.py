"""Composite training loss for the delta model.

L = λ_mse · MSE(...)                                  ← space switched by `mse_space`
  + λ_kl  · KL(p_actual ‖ p_predicted) at mask positions
  + λ_conf· BCE(c_pos, c_label_per_pos) at mask positions only

`mse_space` (T1; engineering.md §3.2) selects which space the MSE term
lives in:
    "raw"        — MSE(Δh_pred,                  h_target − h_ref).
                   Legacy behaviour; what trial 1/2 trained against.
                   Default kwarg here for API back-compat (test_zero_init
                   tests this branch and asserts the closed-form value).
    "final_norm" — MSE(final_norm(h_ref + Δh_pred), final_norm(h_target)).
                   M1.5 T1 — aligns MSE gradient with the directions
                   `lm_head` actually reads. The frozen RMSNorm strips
                   many h-coords before lm_head, so raw-h MSE wastes
                   gradient on those coords. Trial-2 diagnostic: train-val
                   KL gap ~4× while train-val raw-h MSE gap is only ~1.2×
                   — MSE has saturated at the fp16 noise floor in raw-h
                   space. The new default in M1 config is "final_norm"
                   with λ_mse rescaled 0.1 → 1.0 (post-norm MSE sits at
                   O(1), raw MSE was O(10²)).

The rest of the recipe is unchanged. p_actual / p_predicted are the
softmaxed lm_head outputs; c_label_per_pos = sum_v min(p_actual_v,
p_pred_v) is the per-position shared-mass that the conf head learns
to predict via BCE.

`final_norm` and `lm_head` are the *frozen* backbone components. Pass them
in via the closure / module — losses.py never instantiates them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tiered_conf_label(
    p_actual: torch.Tensor,        # [B, T, V]  detached
    p_pred:   torch.Tensor,        # [B, T, V]  detached
    *,
    topk: int = 10,
) -> torch.Tensor:
    """v3 Change 1 — tiered confidence target in [0, 1], shape [B, T].

        c_label = (1/3)·s_top1 + (1/3)·s_topk + (1/3)·s_all

    - s_top1  : binary — 1.0 iff argmax(p_actual) == argmax(p_pred).
    - s_topk  : normalized overlap on p_actual's top-k support,
                Σ_topk min(p_a,p_p) / Σ_topk p_a  — 1.0 on a perfect
                in-support match.
    - s_all   : normalized overlap on the full support; Σ p_a = 1 so this
                is just the sum-of-min (= legacy shared_mass).

    Equal 1/3 weights ⇒ if the top-1 disagrees, c_label ≤ 2/3 regardless
    of the rest of the match — the label tracks the greedy commit
    outcome instead of a tail artefact.
    """
    # s_top1 — binary argmax agreement.
    s_top1 = (p_actual.argmax(-1) == p_pred.argmax(-1)).to(p_actual.dtype)

    # s_all — full-support normalized overlap (denominator = Σ p_a = 1).
    s_all = torch.minimum(p_actual, p_pred).sum(-1)

    # s_topk — normalized overlap restricted to p_actual's top-k tokens.
    topk_vals, topk_idx = p_actual.topk(topk, dim=-1)               # [B,T,k]
    p_pred_topk  = p_pred.gather(-1, topk_idx)                       # [B,T,k]
    overlap_topk = torch.minimum(topk_vals, p_pred_topk).sum(-1)     # [B,T]
    denom_topk   = topk_vals.sum(-1).clamp_min(1e-9)                 # [B,T]
    s_topk       = overlap_topk / denom_topk                         # [B,T]

    c_label = (s_top1 + s_topk + s_all) / 3.0
    return c_label.clamp_(0.0, 1.0)


def composite_loss(
    delta_h_pred: torch.Tensor,         # [B, T, d_model]
    c_pos:        torch.Tensor,         # [B, T]                     per-position conf in (0, 1)
    h_ref:        torch.Tensor,         # [B, T, d_model]            from dataset
    h_target:     torch.Tensor,         # [B, T, d_model]
    mask_tgt:     torch.Tensor,         # [B, T]   bool              True = mask
    final_norm:   nn.Module,            # frozen backbone final RMSNorm/LayerNorm
    lm_head:      nn.Linear,            # frozen backbone LM head
    *,
    lambda_mse:  float = 1.0,
    lambda_kl:   float = 0.1,
    lambda_conf: float = 0.1,
    mse_space:   str   = "raw",         # "raw" (legacy) | "final_norm" (T1, M1 default)
    # v3 Change 1 — confidence-head target.
    conf_target: str   = "shared_mass", # "shared_mass" (legacy) | "tiered" (v3)
    conf_topk:   int   = 10,            # k for the tiered target's s_topk
    # v3 Change 4 — symmetric KL on the delta objective.
    kl_mode:           str   = "forward",   # "forward" (legacy) | "symmetric" (v3)
    lambda_kl_backward: float = 0.5,        # weight on KL(p_pred ‖ p_actual)
    kl_backward_clamp:  float = 10.0,       # max of the per-token backward-KL log-ratio
) -> dict[str, torch.Tensor]:
    """Returns a dict containing 'loss' (scalar w/ grad) and detached
    component scalars suitable for logging."""
    h_pred = h_ref + delta_h_pred                                       # [B, T, D]

    # Targets: forward h_target through frozen norm + lm_head once. We
    # also keep `fn_target` around in case the MSE space is "final_norm",
    # so we don't redo the RMSNorm.
    with torch.no_grad():
        fn_target     = final_norm(h_target)                           # [B, T, D]
        logits_actual = lm_head(fn_target)                              # [B, T, V]
        log_p_actual  = F.log_softmax(logits_actual.float(), dim=-1)
        p_actual      = log_p_actual.exp()

    fn_pred      = final_norm(h_pred)                                   # [B, T, D] — grad flows through Δh_pred
    logits_pred  = lm_head(fn_pred)                                     # [B, T, V]
    log_p_pred   = F.log_softmax(logits_pred.float(), dim=-1)

    # MSE term — chosen space.
    if mse_space == "raw":
        delta_h_target = h_target - h_ref                              # [B, T, D]
        mse = F.mse_loss(delta_h_pred, delta_h_target)
    elif mse_space == "final_norm":
        # fn_target is detached (computed under no_grad); grad flows
        # back through fn_pred → final_norm → h_pred → delta_h_pred.
        mse = F.mse_loss(fn_pred, fn_target)
    else:
        raise ValueError(
            f"composite_loss: unknown mse_space={mse_space!r}; "
            "expected 'raw' or 'final_norm'."
        )

    # Forward KL(p_actual ‖ p_predicted) at mask positions only. This is
    # the metric of record — `kl` in the return dict keeps meaning the
    # forward KL so `train/kl` / `val/kl` plots stay comparable across
    # every run, regardless of `kl_mode`.
    kl_fwd_per_pos = (p_actual * (log_p_actual - log_p_pred)).sum(-1)   # [B, T]
    mask_f     = mask_tgt.to(kl_fwd_per_pos.dtype)                      # [B, T]
    mask_denom = mask_f.sum().clamp_min(1.0)                            # scalar
    kl         = (kl_fwd_per_pos * mask_f).sum() / mask_denom

    # v3 Change 4 — backward KL(p_predicted ‖ p_actual). Mode-seeking /
    # zero-forcing: punishes p_pred mass on tokens p_actual considers
    # unlikely (the spurious-argmax case forward KL is blind to). The
    # per-token log-ratio is clamped on the max side — it diverges when
    # p_actual(v) → 0 with p_pred(v) > 0; that divergence IS the signal
    # but the clamp bounds the gradient where it would otherwise spike.
    if kl_mode == "symmetric":
        log_ratio = (log_p_pred - log_p_actual).clamp(max=kl_backward_clamp)
        kl_bwd_per_pos = (log_p_pred.exp() * log_ratio).sum(-1)         # [B, T]
        kl_backward = (kl_bwd_per_pos * mask_f).sum() / mask_denom
    elif kl_mode == "forward":
        kl_backward = torch.zeros((), device=kl.device, dtype=kl.dtype)
    else:
        raise ValueError(
            f"composite_loss: unknown kl_mode={kl_mode!r}; "
            "expected 'forward' or 'symmetric'."
        )

    # Per-position confidence-head label (detached — only c_pos trains).
    # `clamp_(0,1)` guards fp32-softmax roundoff: p_actual / p_pred each
    # drift from exactly 1.0 by ~1e-7, so an overlap can land at 1+ε,
    # and F.binary_cross_entropy's CUDA kernel hard-asserts target∈[0,1].
    with torch.no_grad():
        p_pred_detached = log_p_pred.exp()
        if conf_target == "shared_mass":
            c_label_per_pos = (
                torch.minimum(p_actual, p_pred_detached).sum(-1).clamp_(0.0, 1.0)
            )   # [B, T]
        elif conf_target == "tiered":
            c_label_per_pos = _tiered_conf_label(
                p_actual, p_pred_detached, topk=conf_topk,
            )   # [B, T]  (already clamped to [0, 1])
        else:
            raise ValueError(
                f"composite_loss: unknown conf_target={conf_target!r}; "
                "expected 'shared_mass' or 'tiered'."
            )

    # Per-position BCE, masked to mask positions only. c_pos is in the
    # model dtype (bf16); c_label_per_pos is fp32 from the softmax upcast
    # above — compute BCE in fp32 for stability.
    bce_per_pos = F.binary_cross_entropy(
        c_pos.float(), c_label_per_pos.detach().float(),
        reduction="none",
    )                                                                   # [B, T]
    bce = (bce_per_pos * mask_f).sum() / mask_denom

    # Aggregate metrics for logging — averaged over mask positions only,
    # so they're comparable to the old block-level scalars.
    with torch.no_grad():
        c_pred_mean  = (c_pos.float()         * mask_f).sum() / mask_denom
        c_label_mean = (c_label_per_pos       * mask_f).sum() / mask_denom

    # The grad-bearing total. `kl` is always the forward KL; the backward
    # term enters only under kl_mode="symmetric" (kl_backward is exactly
    # 0 otherwise, so this line is correct in both modes).
    total = (
        lambda_mse * mse
        + lambda_kl * kl
        + lambda_kl_backward * kl_backward
        + lambda_conf * bce
    )
    return {
        "loss":         total,
        "mse":          mse.detach(),
        "kl":           kl.detach(),
        "kl_backward":  kl_backward.detach(),
        "bce":          bce.detach(),
        "c_label_mean": c_label_mean.detach(),
        "c_pred_mean":  c_pred_mean.detach(),
    }

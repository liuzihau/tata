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

    # KL(p_actual ‖ p_predicted) at mask positions only.
    kl_per_pos = (p_actual * (log_p_actual - log_p_pred)).sum(-1)       # [B, T]
    mask_f     = mask_tgt.to(kl_per_pos.dtype)                          # [B, T]
    mask_denom = mask_f.sum().clamp_min(1.0)                            # scalar
    kl         = (kl_per_pos * mask_f).sum() / mask_denom

    # Per-position shared-mass label (detached — only c_pos trains).
    # Clamp guards against fp32-softmax roundoff: p_actual.sum() and
    # p_pred.sum() each drift from exactly 1.0 by ~1e-7, so the sum-of-min
    # can land at 1+ε. F.binary_cross_entropy's CUDA kernel hard-asserts
    # `target ∈ [0, 1]`, aborting the whole step.
    with torch.no_grad():
        p_pred_detached = log_p_pred.exp()
        c_label_per_pos = (
            torch.minimum(p_actual, p_pred_detached).sum(-1).clamp_(0.0, 1.0)
        )   # [B, T]

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

    total = lambda_mse * mse + lambda_kl * kl + lambda_conf * bce
    return {
        "loss":         total,
        "mse":          mse.detach(),
        "kl":           kl.detach(),
        "bce":          bce.detach(),
        "c_label_mean": c_label_mean.detach(),
        "c_pred_mean":  c_pred_mean.detach(),
    }

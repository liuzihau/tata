"""Composite training loss for the delta model.

L = λ_mse · MSE(Δh_pred, Δh_target)
  + λ_kl  · KL(p_actual ‖ p_predicted) at mask positions
  + λ_conf· BCE(c_pos, c_label_per_pos) at mask positions only

where:
    Δh_target        = h_target_actual − h_ref_actual                 (i_ref < i_target ≤ MAX_ITER-1)
    p_actual         = softmax(lm_head(final_norm(h_target_actual)))
    p_predicted      = softmax(lm_head(final_norm(h_ref_actual + Δh_pred)))
    c_label_per_pos  = shared_mass(p_actual, p_predicted) per token
                       (= sum_v min(p_actual_v, p_pred_v); detached)
    c_pos            = sigmoid output of the per-position confidence head, [B, T]

Compared to the legacy block-aggregate variant (one scalar `c_label` per
sample, one `c_pred`), §3.2 trains the confidence head per position so
inference can decide *per-position* whether to commit a token. BCE is
masked to mask positions only — revealed positions don't contribute to
the conf gradient (their tokens are already committed; nothing to predict).

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
) -> dict[str, torch.Tensor]:
    """Returns a dict containing 'loss' (scalar w/ grad) and detached
    component scalars suitable for logging."""
    delta_h_target = h_target - h_ref                                  # [B, T, D]
    mse = F.mse_loss(delta_h_pred, delta_h_target)

    h_pred = h_ref + delta_h_pred                                       # [B, T, D]

    # Targets: forward h_target through frozen norm + lm_head once.
    with torch.no_grad():
        logits_actual = lm_head(final_norm(h_target))                  # [B, T, V]
        log_p_actual  = F.log_softmax(logits_actual.float(), dim=-1)
        p_actual      = log_p_actual.exp()

    logits_pred  = lm_head(final_norm(h_pred))                          # [B, T, V]
    log_p_pred   = F.log_softmax(logits_pred.float(), dim=-1)

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

"""Composite training loss for the delta model.

L = λ_mse · MSE(Δh_pred, Δh_target)
  + λ_kl  · KL(p_actual ‖ p_predicted) at mask positions
  + λ_conf· BCE(c_pred, c_label)

where:
    Δh_target  = h_target_actual − h_ref_actual                 (i_ref < i_target ≤ MAX_ITER-1)
    p_actual   = softmax(lm_head(final_norm(h_target_actual)))
    p_predicted= softmax(lm_head(final_norm(h_ref_actual + Δh_pred)))
    c_label    = mean shared_mass(p_actual, p_predicted) over mask positions
                  (detached — only c_pred trains)
    c_pred     = sigmoid(...)  (already squashed by ConfHead)

`final_norm` and `lm_head` are the *frozen* backbone components. Pass them
in via the closure / module — losses.py never instantiates them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def composite_loss(
    delta_h_pred: torch.Tensor,         # [B, T, d_model]
    c_pred:       torch.Tensor,         # [B]                        in (0, 1)
    h_ref:        torch.Tensor,         # [B, T, d_model]            from dataset
    h_target:     torch.Tensor,         # [B, T, d_model]
    mask_tgt:     torch.Tensor,         # [B, T]   bool              True = mask, KL evaluated here
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
    denom      = mask_f.sum().clamp_min(1.0)
    kl         = (kl_per_pos * mask_f).sum() / denom

    # BCE on confidence head; soft target = mean shared mass at mask positions.
    with torch.no_grad():
        p_pred_detached = log_p_pred.exp()
        shared = torch.minimum(p_actual, p_pred_detached).sum(-1)       # [B, T]
        per_seq_denom = mask_f.sum(-1).clamp_min(1.0)                   # [B]
        c_label = (shared * mask_f).sum(-1) / per_seq_denom             # [B]
    bce = F.binary_cross_entropy(c_pred, c_label.detach())

    total = lambda_mse * mse + lambda_kl * kl + lambda_conf * bce
    return {
        "loss":         total,
        "mse":          mse.detach(),
        "kl":           kl.detach(),
        "bce":          bce.detach(),
        "c_label_mean": c_label.detach().mean(),
        "c_pred_mean":  c_pred.detach().mean(),
    }

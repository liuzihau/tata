"""Step-0 sanity: with zero-init delta head, the model emits Δh ≡ 0.

This means:
  - h_pred == h_ref  (delta head is zeros, so the only outputs from it
    are zeros regardless of feature values).
  - MSE(Δh_pred, Δh_target) == ‖Δh_target‖²_mean, i.e. the "h_ref reuse"
    baseline.
  - KL(p_actual ‖ p_predicted) reduces to KL(p(h_target) ‖ p(h_ref)).

We don't need real backbone weights — this test stubs final_norm + lm_head
and just checks the model's invariant.

Run (from inside the tata repo root):
    python -m delta_model.sanity.test_zero_init
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ..data import schema as S
from ..losses import composite_loss
from ..models.variant_c import VariantC


def main() -> None:
    torch.manual_seed(0)
    B = 2
    d_model = 256                  # tiny dims for the test (model is dim-agnostic)
    n_kv_heads = 4
    d_head = 64

    model = VariantC(
        d_model=d_model, n_heads=4, n_layers=2, d_ff_inner=512,
        max_seq_len=512,
    )

    # Stub final_norm + lm_head with random frozen weights.
    final_norm = nn.LayerNorm(d_model)
    lm_head    = nn.Linear(d_model, 1024)
    for p in (*final_norm.parameters(), *lm_head.parameters()):
        p.requires_grad_(False)

    h_ref     = torch.randn(B, S.BLOCK_LENGTH, d_model)
    h_target  = h_ref + 0.1 * torch.randn_like(h_ref)
    prev_emb  = torch.randn(B, S.BLOCK_LENGTH, d_model)
    prefix_kv = torch.randn(B, 2, n_kv_heads, S.PREFIX_WINDOW, d_head,
                              dtype=torch.float16)

    # n_kv_heads * d_head must equal d_model so prefix_kv K/V flow into
    # cross-attn directly without reshaping.
    assert n_kv_heads * d_head == d_model, "test config: kv_heads*d_head must equal d_model"

    mask_tgt = torch.ones(B, S.BLOCK_LENGTH, dtype=torch.bool)
    block_start_pos = torch.tensor([64, 96], dtype=torch.long)  # arbitrary in-range positions
    # All-True pad mask for the synthetic prefix_kv (no padding in this test).
    prefix_kv_pad_mask = torch.ones(B, S.PREFIX_WINDOW, dtype=torch.bool)

    delta_h, c_pred = model(
        h_ref, prev_emb, prefix_kv, block_start_pos,
        prefix_kv_pad_mask=prefix_kv_pad_mask,
    )
    print(f"delta_h.abs().max() = {delta_h.abs().max().item():.6e}")
    assert delta_h.abs().max().item() == 0.0, "Δh head not zero-initialized"

    loss_dict = composite_loss(
        delta_h, c_pred, h_ref, h_target, mask_tgt,
        final_norm, lm_head,
    )
    expected_mse = ((h_target - h_ref) ** 2).mean()
    print(
        f"composite MSE term = {loss_dict['mse'].item():.6e} | "
        f"baseline ‖Δh‖²_mean = {expected_mse.item():.6e}"
    )
    assert torch.allclose(loss_dict["mse"], expected_mse, atol=1e-7), (
        "MSE at step 0 should equal the h_ref-reuse baseline"
    )

    # Loss must be finite and non-negative.
    assert torch.isfinite(loss_dict["loss"]), "loss is non-finite"
    assert loss_dict["loss"] >= 0, "loss is negative — sign error?"

    # KL must be ≥ 0 (it's a divergence).
    assert loss_dict["kl"] >= 0, f"KL went negative: {loss_dict['kl']}"

    print("✓ zero-init sanity passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise

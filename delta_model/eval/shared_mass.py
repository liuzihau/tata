"""Shared-mass overlap metric.

shared_mass(p, q) = Σ_x min(p(x), q(x))      ∈ [0, 1]

This equals 1 − 0.5·‖p − q‖₁, i.e. the speculative-decoding closed-form
acceptance rate when q is the draft and p the target. We use it both
as the soft label for the conf head and as a per-bin diagnostic
(higher = better delta-model prediction quality).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def shared_mass(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute shared mass along the last dim. Both tensors must be valid
    probability distributions (non-negative, sum-to-1 along last dim).
    Returns a tensor of shape p.shape[:-1]."""
    return torch.minimum(p, q).sum(dim=-1)


@torch.no_grad()
def shared_mass_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor
                              ) -> torch.Tensor:
    """Convenience: softmax both then compare."""
    pa = torch.softmax(logits_a.float(), dim=-1)
    pb = torch.softmax(logits_b.float(), dim=-1)
    return shared_mass(pa, pb)

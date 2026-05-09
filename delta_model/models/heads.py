"""Output heads shared across delta-model variants.

`DeltaHead`  — zero-init linear that emits Δh. At step 0 the model is
              equivalent to "reuse h_ref verbatim" (the no-training
              baseline).
`ConfHead`   — pooled feature → scalar in [0, 1]. Trained with BCE
              against shared-mass on mask positions; consumed at
              inference to decide rollback.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeltaHead(nn.Module):
    """Zero-init Δh projection. Bias and weight both zero so step 0 emits
    Δh = 0, putting the model exactly at the "h_ref reuse" baseline."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConfHead(nn.Module):
    """Pool block features and project to a scalar confidence in [0, 1]."""

    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        h = hidden if hidden is not None else d_model // 4
        self.proj = nn.Sequential(
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(pooled).squeeze(-1))

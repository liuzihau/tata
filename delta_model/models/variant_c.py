"""Variant C — 2-stream sequence-concat delta model (M1 default).

Inputs (per batch element):
    h_ref     [B, 32, d_model]                        last-layer hidden at i_ref
    prev_emb  [B, 32, d_model]                        input embedding at i_target
    prefix_kv [B, 2, n_kv_heads, 32, d_head]          last-32 prefix KV (fp16)

Outputs:
    delta_h   [B, 32, d_model]                        predicted Δh
    c_pred    [B]                                     confidence ∈ (0, 1)

The model concatenates h_ref + prev_emb along the sequence axis (length
2*BLOCK_LENGTH = 64), runs a small stack of (self-attn, cross-attn-into-
prefix, FFN) layers, then takes the second half (the prev_emb side) as
features for the Δh head and a pooled feature vector for the conf head.

The Δh head is zero-init (`heads.DeltaHead`), so step 0 = "h_ref reuse"
baseline. Variants A/B will reuse the same head + signature; only the
internal mixing changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import schema as S
from .heads import ConfHead, DeltaHead


class _CrossAttnBlock(nn.Module):
    """One layer: self-attn, cross-attn-into-prefix-KV, FFN. Pre-norm
    residual style throughout."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
            kdim=d_model, vdim=d_model,
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # x:  [B, 2*BL, d_model]
        # kv: [B, 32,    d_model]   — packed K=V=this for cross-attn (see model.forward)
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, kv, kv, need_weights=False)[0]
        x = x + self.ffn(self.norm3(x))
        return x


class VariantC(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = S.D_MODEL_LLADA,
        n_heads: int = 32,
        n_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model      = d_model
        self.block_length = S.BLOCK_LENGTH
        d_ff = d_ff if d_ff is not None else 4 * d_model

        # Segment embed distinguishes h_ref tokens from prev_emb tokens.
        self.seg_emb = nn.Embedding(2, d_model)
        # Position embed across the full 2*BLOCK_LENGTH concat sequence.
        self.pos_emb = nn.Embedding(2 * S.BLOCK_LENGTH, d_model)

        # We project the prefix KV from [n_kv_heads, d_head] = d_model
        # into the cross-attn d_model. With LLaDA's no-GQA setup this is
        # already d_model, but we keep an explicit projection for safety
        # (a no-op `Linear(d_model, d_model)` adds negligible params).
        self.prefix_proj = nn.Linear(d_model, d_model)

        self.layers = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.delta_head = DeltaHead(d_model)
        self.conf_head  = ConfHead(d_model)

    @staticmethod
    def _pack_prefix_kv(prefix_kv: torch.Tensor) -> torch.Tensor:
        """[B, 2, n_kv_heads, 32, d_head] → [B, 32, d_model] (mean of K and V).

        For the cross-attention bank we use the average of K and V at each
        position — keeping both would double the seq dim with no clear
        information gain at this resolution. (Open to revisiting; a simple
        ablation would be K-only / V-only / concat.)
        """
        B, two, kv_h, T, d_head = prefix_kv.shape
        assert two == 2, f"prefix_kv first dim must be 2 (K, V), got {two}"
        # Mean over (K, V), then flatten heads.
        kv_mean = prefix_kv.mean(dim=1)             # [B, kv_h, T, d_head]
        kv = kv_mean.permute(0, 2, 1, 3).reshape(B, T, kv_h * d_head)
        return kv

    def forward(
        self,
        h_ref:     torch.Tensor,   # [B, 32, d_model]
        prev_emb:  torch.Tensor,   # [B, 32, d_model]
        prefix_kv: torch.Tensor,   # [B, 2, n_kv_heads, 32, d_head], fp16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, BL, D = h_ref.shape
        device = h_ref.device

        # Concat along seq axis, add segment + position embeddings.
        x = torch.cat([h_ref, prev_emb], dim=1)                       # [B, 2*BL, D]
        seg_ids = torch.cat([
            torch.zeros(BL, dtype=torch.long, device=device),
            torch.ones (BL, dtype=torch.long, device=device),
        ])                                                            # [2*BL]
        pos_ids = torch.arange(2 * BL, device=device)                 # [2*BL]
        x = x + self.seg_emb(seg_ids)[None] + self.pos_emb(pos_ids)[None]

        # Cross-attn bank from the prefix KV. Cast to model dtype.
        kv = self._pack_prefix_kv(prefix_kv.to(x.dtype))             # [B, 32, D]
        kv = self.prefix_proj(kv)

        for layer in self.layers:
            x = layer(x, kv)
        x = self.final_norm(x)

        feats   = x[:, BL:, :]                                        # [B, 32, D]
        delta_h = self.delta_head(feats)                              # [B, 32, D]
        pooled  = feats.mean(dim=1)                                   # [B, D]
        c_pred  = self.conf_head(pooled)                              # [B]
        return delta_h, c_pred

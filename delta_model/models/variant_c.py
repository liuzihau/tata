"""Variant C — 2-stream sequence-concat delta model (M1 default).

Architecture, post §1.5 + §1.6 alignment with the LLaDA backbone:

  Inputs (per batch element):
      h_ref            [B, 32, d_model]                       last-layer hidden at i_ref
      prev_emb         [B, 32, d_model]                       input embedding at i_target
      prefix_kv        [B, 2, n_kv_heads, 32, d_head]         last-32 prefix KV (fp16),
                                                              already RoPE-rotated by the backbone
      block_start_pos  [B] long                               absolute start position of this block
                                                              in the original sequence
                                                              ( = prompt_len + nb * BLOCK_LENGTH )

  Outputs:
      delta_h   [B, 32, d_model]
      c_pred    [B]   ∈ (0, 1)

  Block structure (per layer): pre-RMSNorm → RoPE self-attn → +x → pre-RMSNorm
  → RoPE cross-attn (queries from x, K/V passed in untouched from prefix_kv)
  → +x → pre-RMSNorm → SwiGLU FFN → +x. Final RMSNorm before the heads.

  Positional encoding: RoPE applied to projected Q and K inside each attention,
  with absolute positions matching what the backbone used:
    - self-attn:  row i of both halves rotated at `block_start_pos + i`
    - cross-attn: queries rotated at `block_start_pos + i`; prefix_kv keys
                   pass through as-is (already rotated by backbone at
                   `block_start_pos - 32 .. block_start_pos - 1`).

  The Δh head is zero-init (`heads.DeltaHead`), so step 0 = "h_ref reuse"
  baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import schema as S
from .heads import ConfHead, DeltaHead


# ---------------------------------------------------------------------------
# Primitives: RMSNorm, SwiGLU, RoPE
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Root-mean-square layer norm matching LLaDA's convention."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f = x.float()
        rms_inv = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms_inv).to(in_dtype) * self.weight


class _SwiGLU(nn.Module):
    """SwiGLU FFN matching LLaDA's MLP shape (gate × up, then down)."""

    def __init__(self, d_model: int, d_ff_inner: int):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff_inner, bias=False)
        self.down    = nn.Linear(d_ff_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class _RoPE(nn.Module):
    """Rotary positional embedding. No buffers — `inv_freq` (head_dim/2 floats)
    is recomputed per call on the right device, and sin/cos are computed in
    fp32 then cast to `x.dtype` only at the final multiply. This sidesteps
    the bf16-downcast problem cleanly and matches LLaDA's `rope_full_precision`
    behavior.

    `apply` rotates a `[B, n_heads, T, head_dim]` tensor by per-row absolute
    positions `[B, T]`. Same `theta` and "concat-half" rotate convention as
    LLaDA's RotaryEmbedding."""

    def __init__(self, head_dim: int, theta: float = 1e6, max_seq_len: int = 16384):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim    = head_dim
        self.theta       = theta
        self.max_seq_len = max_seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate `x: [B, n_heads, T, head_dim]` by per-row `positions: [B, T]`."""
        device = x.device
        head_dim = x.shape[-1]
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim)
        )                                                     # [head_dim/2]
        pos = positions.to(torch.float)                       # [B, T]
        freqs = torch.einsum("bt,d->btd", pos, inv_freq)      # [B, T, head_dim/2]
        angles = torch.cat([freqs, freqs], dim=-1)            # [B, T, head_dim]
        cos = angles.cos().unsqueeze(1).to(x.dtype)           # [B, 1, T, head_dim]
        sin = angles.sin().unsqueeze(1).to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class _SelfAttnRoPE(nn.Module):
    """Self-attention with RoPE on both Q and K."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(self, x: torch.Tensor, rope: _RoPE,
                positions: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]; positions: [B, T]
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))
        q = rope.apply(q, positions)
        k = rope.apply(k, positions)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(self._merge(out))


class _CrossAttnRoPE(nn.Module):
    """Cross-attention into externally-supplied K/V (already RoPE-rotated by
    backbone). Only Q gets RoPE applied here."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split_q(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(self, x: torch.Tensor,
                k_external: torch.Tensor, v_external: torch.Tensor,
                rope: _RoPE, q_positions: torch.Tensor) -> torch.Tensor:
        # x: [B, T_q, D]; k_external/v_external: [B, n_heads, T_k, head_dim]
        q = self._split_q(self.q_proj(x))
        q = rope.apply(q, q_positions)
        out = F.scaled_dot_product_attention(
            q, k_external, v_external,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(self._merge(out))


# ---------------------------------------------------------------------------
# Block + top-level model
# ---------------------------------------------------------------------------

class _DeltaBlock(nn.Module):
    """Pre-norm: self-attn (RoPE) → cross-attn (RoPE on Q only) → SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff_inner: int,
                 dropout: float, rms_eps: float):
        super().__init__()
        self.norm_self  = _RMSNorm(d_model, eps=rms_eps)
        self.self_attn  = _SelfAttnRoPE(d_model, n_heads, dropout=dropout)
        self.norm_cross = _RMSNorm(d_model, eps=rms_eps)
        self.cross_attn = _CrossAttnRoPE(d_model, n_heads, dropout=dropout)
        self.norm_ffn   = _RMSNorm(d_model, eps=rms_eps)
        self.ffn        = _SwiGLU(d_model, d_ff_inner)

    def forward(self, x: torch.Tensor,
                k_external: torch.Tensor, v_external: torch.Tensor,
                rope: _RoPE,
                self_positions: torch.Tensor, q_positions: torch.Tensor,
                ) -> torch.Tensor:
        x = x + self.self_attn(self.norm_self(x),  rope, self_positions)
        x = x + self.cross_attn(self.norm_cross(x), k_external, v_external,
                                 rope, q_positions)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class VariantC(nn.Module):
    def __init__(
        self,
        *,
        d_model:    int = S.D_MODEL_LLADA,
        n_heads:    int = 32,
        n_layers:   int = 2,
        d_ff_inner: int | None = None,
        dropout:    float = 0.0,
        rope_theta: float = 1e6,
        rms_eps:    float = 1e-5,
        max_seq_len: int = 16384,
    ):
        super().__init__()
        self.d_model      = d_model
        self.block_length = S.BLOCK_LENGTH
        self.n_heads      = n_heads
        head_dim          = d_model // n_heads
        # SwiGLU param budget ≈ matches a GELU MLP at 4·d_model when
        # d_ff_inner ≈ 8/3·d_model. Caller can override.
        if d_ff_inner is None:
            d_ff_inner = ((8 * d_model // 3) + 31) // 32 * 32

        # Segment embed distinguishes h_ref tokens from prev_emb tokens.
        self.seg_emb = nn.Embedding(2, d_model)

        # RoPE module (no learned params; just sin/cos buffers).
        self.rope = _RoPE(
            head_dim=head_dim, theta=rope_theta, max_seq_len=max_seq_len,
        )

        self.layers = nn.ModuleList([
            _DeltaBlock(d_model, n_heads, d_ff_inner, dropout, rms_eps)
            for _ in range(n_layers)
        ])
        self.final_norm = _RMSNorm(d_model, eps=rms_eps)

        self.delta_head = DeltaHead(d_model)
        self.conf_head  = ConfHead(d_model)

    @staticmethod
    def _split_prefix_kv(prefix_kv: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 2, n_kv_heads, T, d_head] → (K, V) each [B, n_kv_heads, T, d_head].

        K is already RoPE-rotated by the backbone; V is unrotated. Both pass
        straight into cross-attn — no learned projection / re-rotation."""
        K = prefix_kv[:, 0]
        V = prefix_kv[:, 1]
        return K, V

    def forward(
        self,
        h_ref:           torch.Tensor,   # [B, 32, d_model]
        prev_emb:        torch.Tensor,   # [B, 32, d_model]
        prefix_kv:       torch.Tensor,   # [B, 2, n_kv_heads, 32, d_head]
        block_start_pos: torch.Tensor,   # [B] long, abs start pos of this block
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, BL, D = h_ref.shape
        device = h_ref.device

        # Segment embed for the two streams.
        seg_ids = torch.cat([
            torch.zeros(BL, dtype=torch.long, device=device),
            torch.ones (BL, dtype=torch.long, device=device),
        ])                                                            # [2*BL]
        x = torch.cat([h_ref, prev_emb], dim=1)                       # [B, 2*BL, D]
        x = x + self.seg_emb(seg_ids)[None]

        # Per-row absolute positions: both halves share the same intra-block
        # offsets 0..BL-1 (h_ref[i] and prev_emb[i] are at the SAME absolute
        # token position s+i; seg_emb is what distinguishes them).
        offsets = torch.arange(BL, device=device)                     # [BL]
        bsp = block_start_pos.to(device).unsqueeze(1)                 # [B, 1]
        intra_block = bsp + offsets[None]                             # [B, BL]
        positions_full = torch.cat([intra_block, intra_block], dim=1) # [B, 2*BL]

        # Cast prefix_kv to the same dtype as x and split into K, V.
        K_ext, V_ext = self._split_prefix_kv(prefix_kv.to(x.dtype))

        for layer in self.layers:
            x = layer(x, K_ext, V_ext, self.rope,
                      positions_full, positions_full)
        x = self.final_norm(x)

        feats   = x[:, BL:, :]                                        # prev_emb half
        delta_h = self.delta_head(feats)                              # [B, 32, D]
        pooled  = feats.mean(dim=1)                                   # [B, D]
        c_pred  = self.conf_head(pooled)                              # [B]
        return delta_h, c_pred

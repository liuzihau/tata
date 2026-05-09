"""LLaDA + Fast-dLLM v1 runtime helpers — self-contained for tata.

Originally these lived in `probe_runner/`; we copy them here so tata
has no cross-package dependency. If you change behavior here please
mirror to probe_runner (or vice versa) so they don't drift.

Contents:
    resolve_fast_dllm_path  — find Fast-dLLM v1 root via CLI / env / default
    _add_fast_dllm_to_path  — inject Fast-dLLM v1 LLaDA dir into sys.path
    load_llada              — load LLaDA-8B-Instruct via Fast-dLLM's wrapper
    _add_gumbel_noise       — Fast-dLLM Gumbel-noise utility
    _get_num_transfer_tokens — Fast-dLLM per-step transfer count schedule
    _get_transfer_index     — Fast-dLLM low-confidence remask token picker
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fast-dLLM v1 path resolution
# ---------------------------------------------------------------------------

DEFAULT_FAST_DLLM_RELATIVE = Path("external") / "Fast-dLLM" / "v1"


def resolve_fast_dllm_path(explicit: str | os.PathLike | None = None) -> Path:
    """Find Fast-dLLM v1 root (the dir containing `llada/` and `dream/`).

    Priority:
      1. `explicit` arg (CLI --fast_dllm_path)
      2. env var FAST_DLLM_V1_PATH
      3. default ./external/Fast-dLLM/v1 (relative to cwd)
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
    elif os.environ.get("FAST_DLLM_V1_PATH"):
        candidate = Path(os.environ["FAST_DLLM_V1_PATH"]).expanduser().resolve()
    else:
        candidate = (Path.cwd() / DEFAULT_FAST_DLLM_RELATIVE).resolve()

    if not (candidate / "llada" / "model" / "modeling_llada.py").exists():
        raise FileNotFoundError(
            f"Fast-dLLM v1 not found at {candidate}.\n"
            f"Expected file: {candidate / 'llada' / 'model' / 'modeling_llada.py'}\n\n"
            f"Fix one of:\n"
            f"  1. Pass --fast_dllm_path /your/path/to/Fast-dLLM/v1 .\n"
            f"  2. Export FAST_DLLM_V1_PATH=/your/path/to/Fast-dLLM/v1 .\n"
            f"  3. Place Fast-dLLM at ./external/Fast-dLLM/v1 .\n"
        )
    return candidate


def _add_fast_dllm_to_path(fast_dllm_path: str | Path | None = None) -> None:
    """Add Fast-dLLM v1's LLaDA module dir to sys.path so its
    `model.modeling_llada` resolves."""
    root = resolve_fast_dllm_path(fast_dllm_path)
    fast_dllm_llada = root / "llada"
    if str(fast_dllm_llada) not in sys.path:
        sys.path.insert(0, str(fast_dllm_llada))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_llada(
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    fast_dllm_path: str | Path | None = None,
):
    """Load LLaDA via Fast-dLLM v1's LLaDAModelLM. Returns (model, tokenizer)."""
    _add_fast_dllm_to_path(fast_dllm_path)
    from model.modeling_llada import LLaDAModelLM  # noqa: WPS433  (Fast-dLLM module path)
    from transformers import AutoTokenizer

    model = LLaDAModelLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype,
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Fast-dLLM v1 token-transfer utilities (verbatim copy)
# ---------------------------------------------------------------------------

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    device = block_mask_index.device
    dtype = torch.long
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num = base.unsqueeze(1).expand(-1, steps).to(dtype)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    return num + add_mask.to(dtype)


def _get_transfer_index(logits, temperature, remasking, mask_index, x,
                         num_transfer_tokens, threshold=None):
    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device,
    ).clamp(min=0)
    _, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index

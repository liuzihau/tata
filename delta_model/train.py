"""tata delta-model training loop.

CLI (run from inside the tata repo root):
    python -m delta_model.train \\
        --config delta_model/configs/m1_llada_variant_c.yaml \\
        [--override key=value ...]
    python -m delta_model.train --resume_from <ckpt_path>

Loads frozen backbone components (token_embed, final_norm, lm_head) once
at startup, builds train + val Datasets, runs the AdamW + cosine
warmup loop with wandb logging, periodic validation, periodic GSM8K
subset eval, and rolling checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Avoid /dev/shm bus errors on shared HPC nodes: route DataLoader worker
# tensors through the file-system strategy instead of SysV shared memory.
torch.multiprocessing.set_sharing_strategy("file_system")

from .data import schema as S
from .data.dataset import (
    BlockShardSampler,
    InterleavedShardSampler,
    TataDeltaDataset,
    make_train_val_filter,
)
from .losses import composite_loss
from .models.variant_c import VariantC


# ---------------------------------------------------------------------------
# Wallclock timers (debug / bottleneck hunting)
# ---------------------------------------------------------------------------

class StepTimers:
    """Cumulative section timers with CUDA-sync correctness.

    Use the `time(tag)` contextmanager to wrap a code section. Call
    `summary_and_reset()` periodically to get a one-line breakdown and
    zero out the accumulators.
    """

    def __init__(self, sync_cuda: bool = True):
        self.acc: dict[str, float] = defaultdict(float)
        self.cnt: dict[str, int] = defaultdict(int)
        self.sync_cuda = sync_cuda and torch.cuda.is_available()

    @contextmanager
    def time(self, tag: str):
        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                torch.cuda.synchronize()
            self.acc[tag] += time.perf_counter() - t0
            self.cnt[tag] += 1

    def summary_and_reset(self) -> str:
        total = sum(self.acc.values()) or 1e-9
        order = sorted(self.acc.items(), key=lambda kv: -kv[1])
        parts = []
        for k, v in order:
            mean_ms = 1000.0 * v / max(1, self.cnt[k])
            pct     = 100.0 * v / total
            parts.append(f"{k}:{mean_ms:5.1f}ms({pct:2.0f}%)")
        out = " ".join(parts)
        self.acc.clear(); self.cnt.clear()
        return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class _Cfg(dict):
    """Dict with attribute access (so we can write `cfg.data.batch_size`)."""
    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return _Cfg(v) if isinstance(v, dict) else v


def _load_config(path: str, overrides: list[str]) -> _Cfg:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides or []:
        k, _, v = o.partition("=")
        # Coerce to int / float / bool / str heuristically.
        try:
            v_cast = yaml.safe_load(v)
        except yaml.YAMLError:
            v_cast = v
        keys = k.split(".")
        d = cfg
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = v_cast
    return _Cfg(cfg)


# ---------------------------------------------------------------------------
# Backbone components
# ---------------------------------------------------------------------------

def _load_backbone_components(
    model_type: str, fast_dllm_path: str | None,
    *, device: torch.device | str, dtype: torch.dtype,
) -> tuple[nn.Embedding, nn.Module, nn.Linear]:
    """Returns (token_embed, final_norm, lm_head) — all frozen, on `device`.

    Token embed is now GPU-resident (the dataset no longer needs it; lookups
    happen once per batch on GPU in the train/val loop). Held in `dtype`
    (typically bf16) to save VRAM.
    """
    if model_type == "llada":
        from .llada_runtime import load_llada
        model, _tok = load_llada(fast_dllm_path=fast_dllm_path)
    else:
        raise NotImplementedError(f"Backbone '{model_type}' not wired in M1.")

    # LLaDA topology under Fast-dLLM v1 wrappers:
    #   model.model.transformer.wte      → token embedding
    #   model.model.transformer.ln_f     → final norm
    #   model.model.transformer.ff_out   → lm_head (weight-tied to wte in LLaDA)
    transformer = model.model.transformer
    src_token_embed = transformer.wte
    final_norm     = transformer.ln_f
    lm_head        = transformer.ff_out  # nn.Linear-like

    for p in model.parameters():
        p.requires_grad_(False)

    # Build a GPU embedding for fast post-batch lookups. Copy weights before
    # the source `model` falls out of scope and gets GC'd.
    vocab_size, d_model = src_token_embed.weight.shape
    token_embed = nn.Embedding(vocab_size, d_model).to(device, dtype=dtype)
    with torch.no_grad():
        token_embed.weight.data.copy_(
            src_token_embed.weight.data.detach().to(device=device, dtype=dtype)
        )
    token_embed.weight.requires_grad_(False)

    final_norm = final_norm.to(device)
    lm_head    = lm_head.to(device)

    # Surface backbone hyperparams that variantc needs to stay in the same
    # numerical regime (RoPE theta, RMSNorm eps, max position).
    bb_cfg = {
        "rope_theta":     float(getattr(model.config, "rope_theta", 1e6)),
        "rms_eps":        float(getattr(model.config, "layer_norm_eps", 1e-5)),
        "max_seq_len":    int(getattr(model.config, "max_sequence_length", 8192)),
    }
    return token_embed, final_norm, lm_head, bb_cfg


# ---------------------------------------------------------------------------
# Lr schedule (warmup + cosine)
# ---------------------------------------------------------------------------

def _lr_schedule(step: int, *, warmup: int, total: int, base_lr: float,
                 kind: str = "cosine") -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    if kind == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    if kind == "linear":
        return base_lr * max(0.0, 1.0 - progress)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_val(model, val_dl, token_embed, final_norm, lm_head, *,
            loss_kwargs: dict,
            bins_gap, bins_reveal,
            max_batches: int = 50,
            device="cuda", dtype=torch.bfloat16) -> dict:
    """Compute val loss components + per-bin breakdowns.

    `loss_kwargs` is the exact dict passed to `composite_loss` in the
    train loop, so val and train never diverge on the loss recipe.

    Per-bin MSE is computed in the *same* space as the loss
    (`loss_kwargs["mse_space"]`) so the binned diagnostic stays
    comparable to `val/mse`.

    Also logs `top1_agree` — mean over masked positions of
    `argmax(p_pred) == argmax(p_actual)`. This is a *diagnostic only*
    (it is teacher-forced, hence exposure-bias-blind); never use it for
    checkpoint selection (v3 plan, Change 2).
    """
    mse_space = loss_kwargs.get("mse_space", "raw")
    model.eval()
    sums = defaultdict(float); counts = defaultdict(int)
    bin_sums = defaultdict(lambda: defaultdict(float))
    bin_counts = defaultdict(lambda: defaultdict(int))

    for bi, batch in enumerate(val_dl):
        if bi >= max_batches:
            break
        batch_dev = {
            k: v.to(device, dtype=dtype) if isinstance(v, torch.Tensor)
                                            and v.dtype.is_floating_point
            else (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        prev_emb = token_embed(batch_dev["substituted_ids"])
        delta_h, c_pos = model(
            batch_dev["h_ref"], prev_emb, batch_dev["prefix_kv"],
            batch_dev["block_start_pos"],
            prefix_kv_pad_mask=batch_dev.get("prefix_kv_pad_mask"),
        )
        loss_dict = composite_loss(
            delta_h, c_pos,
            batch_dev["h_ref"], batch_dev["h_target"], batch_dev["mask_tgt"],
            final_norm, lm_head,
            **loss_kwargs,
        )
        for k, v in loss_dict.items():
            sums[k]   += float(v.item())
            counts[k] += 1

        # v3 — top1_agree diagnostic: argmax(p_pred) vs argmax(p_actual)
        # at mask positions. Teacher-forced, so exposure-bias-blind —
        # logged for the diagnostic gap vs free-running metrics, never
        # used as a selector.
        with torch.no_grad():
            h_pred_t1   = batch_dev["h_ref"] + delta_h
            logits_pred = lm_head(final_norm(h_pred_t1))
            logits_act  = lm_head(final_norm(batch_dev["h_target"]))
            agree = (logits_pred.argmax(-1) == logits_act.argmax(-1))    # [B, T]
            m_t1  = batch_dev["mask_tgt"].bool()
            denom_t1 = int(m_t1.sum().item())
            if denom_t1 > 0:
                sums["top1_agree"]   += float(agree[m_t1].float().mean().item())
                counts["top1_agree"] += 1

        # Per-(gap, reveal_bin) sums of MSE per row, in the loss's space.
        gap = (batch["i_target"] - batch["i_ref"]).numpy()              # [B]
        reveal = batch["reveal_frac"].numpy()                            # [B]
        with torch.no_grad():
            if mse_space == "final_norm":
                h_pred_v   = batch_dev["h_ref"] + delta_h
                fn_pred_v  = final_norm(h_pred_v)
                fn_tgt_v   = final_norm(batch_dev["h_target"])
                row_mse    = ((fn_pred_v - fn_tgt_v) ** 2).mean(dim=(1, 2))   # [B]
            else:
                row_mse = ((delta_h - (batch_dev["h_target"] - batch_dev["h_ref"]))
                            ** 2).mean(dim=(1, 2))                           # [B]
            row_mse_np = row_mse.float().cpu().numpy()
        for r_idx in range(len(gap)):
            g = int(gap[r_idx])
            if g in bins_gap:
                bin_sums["gap"][g]   += float(row_mse_np[r_idx])
                bin_counts["gap"][g] += 1
            # reveal bin (find the bucket index)
            r = float(reveal[r_idx])
            rb_found = None
            for bi_, (lo, hi) in enumerate(zip(bins_reveal[:-1], bins_reveal[1:])):
                if lo <= r <= hi:
                    bin_sums["reveal"][bi_]   += float(row_mse_np[r_idx])
                    bin_counts["reveal"][bi_] += 1
                    rb_found = bi_
                    break
            # 2D (gap, reveal_bin) binning — disentangles "is late reveal
            # hard because of more committed tokens?" from "is it hard
            # because larger-gap pairs leak into the high-reveal bin?".
            # Many of the 3×5 cells are empty by construction (e.g. gap=3
            # rarely lands in reveal_0); we only emit populated cells.
            if rb_found is not None and g in bins_gap:
                key = (g, rb_found)
                bin_sums["gap_reveal"][key]   += float(row_mse_np[r_idx])
                bin_counts["gap_reveal"][key] += 1

    out = {}
    for k, total in sums.items():
        out[k] = total / max(1, counts[k])
    for g, total in bin_sums["gap"].items():
        out[f"mse_by_gap_{g}"] = total / max(1, bin_counts["gap"][g])
    for bi_, total in bin_sums["reveal"].items():
        out[f"mse_by_reveal_{bi_}"] = total / max(1, bin_counts["reveal"][bi_])
    # 2D cells: emit count too so a future thr-lookup builder can weight
    # by support and skip noisy sparse cells.
    for (g, rb_), total in bin_sums["gap_reveal"].items():
        n = bin_counts["gap_reveal"][(g, rb_)]
        out[f"mse_by_gap_{g}_reveal_{rb_}"]   = total / max(1, n)
        out[f"n_by_gap_{g}_reveal_{rb_}"]     = int(n)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_wandb(cfg: _Cfg):
    if cfg.log.framework != "wandb":
        return None
    import wandb  # noqa: WPS433
    wandb.init(
        project=cfg.log.project,
        group=cfg.log.group,
        name=cfg.run_name,
        config=dict(cfg),
        tags=[cfg.model.variant, cfg.backbone.model, "M1"],
    )
    return wandb


def _ckpt_save(model, opt, step, cfg, *, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:07d}.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "cfg":   dict(cfg),
    }, path)
    # rotate — NOTE: only touches `step_*.pt`; `best_*.pt` files are preserved.
    ckpts = sorted(out_dir.glob("step_*.pt"))
    keep = cfg.checkpoint.keep_last
    for old in ckpts[:-keep] if keep > 0 else []:
        old.unlink()
    return path


class _BestTracker:
    """Track best-by-metric and persist the corresponding checkpoint.

    For each registered metric we keep exactly one `.pt` file on disk:
    `<ckpt_dir>/best_<label>_step<NNNN>.pt`. When a new best is observed,
    the prior file (matching `best_<label>_step*.pt`) is unlinked and a
    new one is written. The running best is also persisted to a sidecar
    `best_metrics.json` so resumed runs don't lose history.

    Modes:
        "min" — lower is better (e.g. val/kl)
        "max" — higher is better (e.g. gsm8k/accuracy_hybrid)
    """

    def __init__(self, ckpt_dir: Path, *, log=print):
        self.ckpt_dir = ckpt_dir
        self.log = log
        self.sidecar = ckpt_dir / "best_metrics.json"
        if self.sidecar.exists():
            try:
                self.state = json.loads(self.sidecar.read_text())
            except json.JSONDecodeError:
                self.state = {}
        else:
            self.state = {}

    def maybe_save(self, *, label: str, value: float | None, mode: str,
                   step: int, model, opt, cfg) -> bool:
        """Save a new best-checkpoint if `value` improves on `label`'s prior
        best. Returns True if a new file was written.

        `label` is the filename tag (e.g. 'gsm8k', 'val_kl') — used in
        `best_<label>_step<NNNN>.pt`. Whatever string you pass becomes part
        of the file path, so keep it short and shell-safe.
        """
        if value is None or not math.isfinite(value):
            return False
        prior = self.state.get(label, {})
        prior_value = prior.get("value")
        is_better = (
            prior_value is None
            or (mode == "min" and value < prior_value)
            or (mode == "max" and value > prior_value)
        )
        if not is_better:
            return False
        # Delete any prior best file for this label (we only keep the latest).
        for old in self.ckpt_dir.glob(f"best_{label}_step*.pt"):
            try:
                old.unlink()
            except OSError:
                pass
        new_path = self.ckpt_dir / f"best_{label}_step{step}.pt"
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "opt":   opt.state_dict(),
            "cfg":   dict(cfg),
            "best_label": label,
            "best_value": value,
            "best_mode":  mode,
        }, new_path)
        self.state[label] = {
            "value": float(value), "step": int(step), "mode": mode,
        }
        self.sidecar.write_text(json.dumps(self.state, indent=2))
        prior_repr = f"{prior_value:.4f}" if isinstance(prior_value, (int, float)) else "—"
        self.log(
            f"[best] {label}={value:.4f} @ step {step} (prior {prior_repr})  "
            f"→  {new_path.name}"
        )
        return True

    def maybe_save_topk(self, *, label: str, value: float | None, mode: str,
                        step: int, model, opt, cfg, k: int = 3) -> bool:
        """Keep the top-`k` checkpoints by `label` (v3 phase-1 selection).

        Unlike `maybe_save` (one file per label), this retains up to `k`
        `best_<label>_step<N>.pt` files. When a new value beats the worst
        of the current k, the new checkpoint is written and the worst is
        evicted. State for `label` is stored as a list of
        `{value, step, mode}`.
        """
        if value is None or not math.isfinite(value):
            return False
        entries = self.state.get(label, [])
        if not isinstance(entries, list):   # migrate any stale scalar state
            entries = []
        better = (lambda v, w: v < w) if mode == "min" else (lambda v, w: v > w)

        if len(entries) >= k:
            worst = (max if mode == "min" else min)(
                entries, key=lambda e: e["value"],
            )
            if not better(value, worst["value"]):
                return False

        new_path = self.ckpt_dir / f"best_{label}_step{step}.pt"
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "opt":   opt.state_dict(),
            "cfg":   dict(cfg),
            "best_label": label,
            "best_value": value,
            "best_mode":  mode,
        }, new_path)
        entries.append({"value": float(value), "step": int(step), "mode": mode})
        # Sort best-first, evict everything past rank k.
        entries.sort(key=lambda e: e["value"], reverse=(mode == "max"))
        for e in entries[k:]:
            old = self.ckpt_dir / f"best_{label}_step{e['step']}.pt"
            try:
                old.unlink()
            except OSError:
                pass
        entries = entries[:k]
        self.state[label] = entries
        self.sidecar.write_text(json.dumps(self.state, indent=2))
        self.log(
            f"[best] {label}={value:.4f} @ step {step} → top-{k}: "
            + ", ".join(f"{e['value']:.4f}@{e['step']}" for e in entries)
        )
        return True


def run_conf_phase(cfg, resume_from: str) -> None:
    """v3 phase 2 (Change 3) — train ONLY the confidence head.

    The delta + trunk are loaded frozen from a phase-1 checkpoint; only
    `conf_head_per_pos` trains. Supervision is the free-running DAgger
    label from `inference/conf_rollout.py` — the conf head learns to
    predict commit safety on the delta's *own* free-running states, not
    on cached teacher-forced ones.

    Selection: every `log.val_every` steps a small hybrid GSM8K eval
    measures `rollback_rate`; the checkpoint is scored
    `accuracy_hybrid` when `rollback_rate ∈ [0.5, 0.7]` (the healthy
    band) and `accuracy_hybrid − 1.0` otherwise, so an in-band
    checkpoint always wins. Best is saved as `best_conf_step<N>.pt`.
    """
    import torch.nn.functional as F
    from .llada_runtime import load_llada
    from .inference.conf_rollout import collect_dagger_records, records_to_batches
    from .eval.gsm8k_e2e import run_gsm8k_eval

    device = "cuda"
    dtype  = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[cfg.optim.precision]

    # --- backbone (full model — the rollout needs real forwards) ----------
    print("[conf] loading backbone …", flush=True)
    backbone, tokenizer = load_llada(fast_dllm_path=cfg.backbone.fast_dllm_path)
    backbone = backbone.to(device)
    backbone.eval()
    # run_gsm8k_eval reads `backbone_model._tokenizer` when both the
    # backbone and delta model are passed in — load_llada doesn't attach
    # it, so do it here.
    backbone._tokenizer = tokenizer
    for p_ in backbone.parameters():
        p_.requires_grad_(False)
    transformer = backbone.model.transformer
    token_embed, final_norm, lm_head = (
        transformer.wte, transformer.ln_f, transformer.ff_out,
    )
    bb_cfg = {
        "rope_theta":  float(getattr(backbone.config, "rope_theta", 1e6)),
        "rms_eps":     float(getattr(backbone.config, "layer_norm_eps", 1e-5)),
        "max_seq_len": int(getattr(backbone.config, "max_sequence_length", 8192)),
    }

    # --- delta model — phase-1 weights, frozen except the conf head -------
    ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
    model = VariantC(
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff_inner=getattr(cfg.model, "d_ff_inner", None),
        dropout=cfg.model.dropout,
        rope_theta=bb_cfg["rope_theta"], rms_eps=bb_cfg["rms_eps"],
        max_seq_len=bb_cfg["max_seq_len"],
        detach_conf_features=bool(getattr(cfg.model, "detach_conf_features", False)),
    ).to(device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for p_ in model.conf_head_per_pos.parameters():
        p_.requires_grad_(True)
    conf_params = [p_ for p_ in model.conf_head_per_pos.parameters()]
    n_conf = sum(p_.numel() for p_ in conf_params)
    print(f"[conf] phase-1 ckpt={resume_from}  trainable conf-head params={n_conf}",
          flush=True)

    opt = torch.optim.AdamW(
        conf_params, lr=cfg.optim.lr, betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )

    # --- rollout prompts (GSM8K train split) ------------------------------
    from datasets import load_dataset
    n_prompts   = int(getattr(cfg, "conf_rollout_prompts", 64))
    ds = load_dataset("gsm8k", "main", split="train").shuffle(seed=cfg.seed)
    ds = ds.select(range(min(n_prompts, len(ds))))
    prompts = [
        torch.tensor(
            tokenizer(tokenizer.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                add_generation_prompt=True, tokenize=False))["input_ids"],
            dtype=torch.long,
        ).unsqueeze(0).to(device)
        for r in ds
    ]
    print(f"[conf] {len(prompts)} rollout prompts", flush=True)

    # --- logging / checkpoints --------------------------------------------
    wandb = _setup_wandb(cfg)
    ckpt_dir = Path(cfg.checkpoint.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_fp = (ckpt_dir / "metrics.jsonl").open("a", buffering=1)

    def log_metrics(step: int, payload: dict) -> None:
        if wandb is not None:
            wandb.log(payload, step=step)
        metrics_fp.write(json.dumps({"step": int(step), **payload}) + "\n")

    best_tracker = _BestTracker(ckpt_dir, log=lambda s: print(s, flush=True))

    # --- DAgger loop ------------------------------------------------------
    max_steps  = int(cfg.optim.max_steps)
    conf_batch = int(getattr(cfg, "conf_batch_size", 64))
    factor     = float(getattr(cfg.loss, "decode_factor", 1.0))
    conf_topk  = int(getattr(cfg.loss, "conf_topk", 10))
    gen = torch.Generator(); gen.manual_seed(int(cfg.seed))

    # Collect the DAgger buffer ONCE. The rollout is delta-only (ungated)
    # and the delta is frozen in phase 2 → the free-running trajectory,
    # and hence the buffer, is fixed. No refresh needed; phase 2 then
    # epochs over this buffer.
    model.eval()
    records = collect_dagger_records(
        backbone, model, final_norm, lm_head, token_embed, prompts,
        factor=factor, conf_topk=conf_topk,
    )
    print(f"[conf] collected {len(records)} DAgger records", flush=True)
    if not records:
        raise RuntimeError("conf phase: rollout produced no records")

    step = 0
    while step < max_steps:
        model.train()   # only conf_head_per_pos has requires_grad
        for batch in records_to_batches(records, conf_batch,
                                        device=device, generator=gen):
            _delta_h, c_pos = model(
                batch["h_ref"].to(dtype),
                batch["prev_emb"].to(dtype),
                batch["prefix_kv"].to(torch.float16),
                batch["block_start_pos"],
                prefix_kv_pad_mask=batch["prefix_kv_pad_mask"],
            )
            mask_f = batch["mask_tgt"].to(c_pos.dtype)
            denom  = mask_f.sum().clamp_min(1.0)
            bce_pp = F.binary_cross_entropy(
                c_pos.float(), batch["c_label"].float(), reduction="none",
            )
            loss = (bce_pp * mask_f).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conf_params, cfg.optim.grad_clip)
            opt.step(); opt.zero_grad()
            step += 1

            if step % cfg.log.log_every == 0:
                log_metrics(step, {"conf/bce": float(loss.item())})

            if step % cfg.log.val_every == 0:
                model.eval()
                try:
                    gsm = run_gsm8k_eval(
                        backbone_model=backbone, delta_model=model,
                        n_problems=cfg.log.gsm8k_subset,
                        per_pos_threshold=getattr(
                            cfg.log, "gsm8k_per_pos_threshold", 0.75),
                        seed=cfg.seed, show_progress=False,
                    )
                    rb  = float(gsm.get("rollback_rate", 1.0))
                    acc = float(gsm.get("accuracy_hybrid", 0.0))
                    # In-band rollback rate always outranks out-of-band.
                    in_band = 0.5 <= rb <= 0.7
                    score   = acc if in_band else acc - 1.0
                    log_metrics(step, {f"gsm8k/{k}": v for k, v in gsm.items()})
                    best_tracker.maybe_save(
                        label="conf", value=score, mode="max",
                        step=step, model=model, opt=opt, cfg=cfg,
                    )
                    print(f"[conf] step {step}: acc={acc:.3f} rb_rate={rb:.3f} "
                          f"in_band={in_band}", flush=True)
                except Exception as e:
                    print(f"[conf] gsm8k eval failed (continuing): {e}", flush=True)
                model.train()

            if step >= max_steps:
                break

    _ckpt_save(model, opt, step, cfg, out_dir=ckpt_dir)
    metrics_fp.close()
    print(f"[conf] done — {step} steps. checkpoints in {ckpt_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--resume_from", default=None)
    p.add_argument(
        "--phase", choices=["joint", "delta", "conf"], default="joint",
        help="v3 training phase. joint = legacy v2 behavior (delta + conf "
             "trained together). delta = phase 1 (delta only: lambda_conf "
             "forced to 0, hybrid-GSM8K off, delta-only GSM8K on, top-3 "
             "checkpoints tracked). conf = phase 2 (freeze all but the "
             "conf head, BCE-only on free-running DAgger labels; requires "
             "--resume_from a phase-1 checkpoint).",
    )
    args = p.parse_args()

    phase = args.phase
    if phase == "conf" and not args.resume_from:
        p.error("--phase conf requires --resume_from <phase-1 checkpoint>")

    cfg = _load_config(args.config, args.override)

    # v3 phase 2 is a separate training mode — dispatch before the
    # teacher-forced dataset / dataloader machinery (which it never uses).
    if phase == "conf":
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        run_conf_phase(cfg, args.resume_from)
        return

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = "cuda"
    dtype  = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[cfg.optim.precision]

    print("[train] loading backbone components …", flush=True)
    token_embed, final_norm, lm_head, bb_cfg = _load_backbone_components(
        cfg.backbone.model, cfg.backbone.fast_dllm_path,
        device=device, dtype=dtype,
    )
    print(f"[train] backbone hyperparams: {bb_cfg}", flush=True)

    print("[train] building Datasets …", flush=True)
    preload = bool(getattr(cfg.data, "preload", True))

    # Sampler config — read early because the interleaving sampler needs the
    # shard LRU sized to `active_shards` *before* the dataset is built.
    shard_sampler_kind = str(getattr(cfg.data, "shard_sampler", "interleaved"))
    active_shards      = int(getattr(cfg.data, "active_shards", 8))

    # Validation source:
    #   "test"    → validate on the hash-stable cache `test/` split (default),
    #               and train on the *full* train split. The test split is
    #               identical across cache sizes, so val/kl is finally
    #               comparable between e.g. the 5k and 20k caches.
    #   "holdout" → legacy: carve `val_frac` out of the train split via
    #               `make_train_val_filter`. The held-out prompts differ
    #               between caches, so cross-cache val/kl is NOT comparable.
    val_split = str(getattr(cfg.data, "val_split", "test"))
    if val_split == "test":
        train_filter = val_filter = None
        val_ds_split = "test"
    elif val_split == "holdout":
        train_filter, val_filter = make_train_val_filter(
            cfg.data.val_frac, seed=cfg.data.shuffle_seed,
        )
        val_ds_split = "train"
    else:
        raise ValueError(
            f"data.val_split must be 'test' or 'holdout'; got {val_split!r}"
        )

    # When the interleaving sampler is active the train dataset's shard LRU
    # must hold every shard in the active window simultaneously, else it
    # thrashes. +1 for headroom across window boundaries.
    train_shard_lru = (
        active_shards + 1
        if (shard_sampler_kind == "interleaved" and not preload)
        else None
    )
    # `data.train_max_prompts` (optional, default = full split) caps the
    # train side to the first N prompts by load order, applied *before*
    # shard loading so preload RAM scales with N rather than the full cache.
    # Used by the v2 5k/10k/20k bracket runs to derive 5k / 10k / 20k train
    # subsets from a single cache_v1_20k cache. Val stays full.
    train_max_prompts = int(getattr(cfg.data, "train_max_prompts", 0))
    train_ds = TataDeltaDataset(
        cfg.data.cache_root, split="train",
        mask_token_id=cfg.data.mask_token_id, index_filter=train_filter,
        preload=preload, shard_lru_max=train_shard_lru,
        max_prompts=train_max_prompts if train_max_prompts > 0 else None,
    )
    val_ds = TataDeltaDataset(
        cfg.data.cache_root, split=val_ds_split,
        mask_token_id=cfg.data.mask_token_id, index_filter=val_filter,
        preload=preload,
    )
    print(
        f"[train] val_split={val_split!r} → val from {val_ds_split!r} split. "
        f"train pairs={len(train_ds)}  val pairs={len(val_ds)}",
        flush=True,
    )

    # Sampler selection:
    #   1. shard mode + preload=False → a shard-aware sampler:
    #        - "interleaved" (default): InterleavedShardSampler keeps
    #          `active_shards` shards resident and mixes draws across all of
    #          them, so every batch spans ~active_shards · shard_size prompts.
    #          Recovers random-sampler gradient quality at bounded RAM.
    #        - "block" (legacy): BlockShardSampler — one shard per batch.
    #          Kept only to reproduce the m1_5_data20k / m2_t5 runs; it gave
    #          correlated gradients (val/kl 0.332 vs 0.280 preload).
    #   2. preload=True or per-sample, ref0_mult ≠ 1.0 → WeightedRandomSampler.
    #   3. otherwise → standard shuffle.
    ref0_mult = float(getattr(cfg.data, "ref0_weight_multiplier", 1.0))
    weights = train_ds.compute_index_weights(ref0_weight_multiplier=ref0_mult)
    use_shard_sampler = (train_ds.mode == "shard" and not preload)
    n_ref0 = int((weights == ref0_mult).sum().item()) if ref0_mult != 1.0 else 0

    dl_common = dict(
        batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True, persistent_workers=cfg.data.num_workers > 0,
    )

    if use_shard_sampler and shard_sampler_kind == "interleaved":
        shard_groups = train_ds.get_shard_groups()
        gen = torch.Generator(); gen.manual_seed(int(cfg.seed))
        chunk_size = getattr(cfg.data, "shard_chunk_size", None)
        sampler = InterleavedShardSampler(
            shard_groups, weights,
            active_shards=active_shards,
            chunk_size=int(chunk_size) if chunk_size is not None else None,
            num_samples=len(train_ds),       # match nominal epoch length
            generator=gen,
        )
        train_dl = DataLoader(train_ds, sampler=sampler, **dl_common)
        print(
            f"[train] InterleavedShardSampler: {len(shard_groups)} shards, "
            f"active_shards={sampler.active_shards}, chunk_size={sampler.chunk_size} "
            f"(≈{sampler.chunk_size // cfg.data.batch_size} batches/window). "
            f"ref0_mult={ref0_mult} ({n_ref0} ref0 pairs / {len(weights)} total).",
            flush=True,
        )
    elif use_shard_sampler and shard_sampler_kind == "block":
        spsv = int(getattr(cfg.data, "samples_per_shard_visit", 4096))
        shard_groups = train_ds.get_shard_groups()
        gen = torch.Generator(); gen.manual_seed(int(cfg.seed))
        sampler = BlockShardSampler(
            shard_groups, weights,
            samples_per_shard_visit=spsv,
            num_samples=len(train_ds),       # match nominal epoch length
            generator=gen,
        )
        train_dl = DataLoader(train_ds, sampler=sampler, **dl_common)
        print(
            f"[train] BlockShardSampler (legacy): {len(shard_groups)} shards, "
            f"{spsv} samples/visit ({spsv // cfg.data.batch_size} batches/visit). "
            f"ref0_mult={ref0_mult} ({n_ref0} ref0 pairs / {len(weights)} total).",
            flush=True,
        )
    elif use_shard_sampler:
        raise ValueError(
            f"data.shard_sampler must be 'interleaved' or 'block'; "
            f"got {shard_sampler_kind!r}"
        )
    elif ref0_mult != 1.0:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(train_ds), replacement=True,
        )
        train_dl = DataLoader(train_ds, sampler=sampler, **dl_common)
        n_other = len(weights) - n_ref0
        p_ref0 = (n_ref0 * ref0_mult) / (n_ref0 * ref0_mult + n_other)
        print(
            f"[train] i_ref-biased sampler: ref0_weight_multiplier={ref0_mult} "
            f"→ P(i_ref=0) ≈ {p_ref0:.2f} ({n_ref0} ref0 pairs, {n_other} others)",
            flush=True,
        )
    else:
        dl_kwargs = dict(dl_common); dl_kwargs["shuffle"] = True
        train_dl = DataLoader(train_ds, **dl_kwargs)
    # Mirror the training loader's worker count exactly: if cfg.data.num_workers
    # is 0 (e.g. constrained /dev/shm), val must also be 0, otherwise the first
    # val pass spawns a worker and SIGBUSes on shm.
    val_workers = (cfg.data.num_workers // 2) if cfg.data.num_workers > 1 else 0
    val_dl = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=val_workers,
        pin_memory=cfg.data.pin_memory,
    )

    print("[train] building model …", flush=True)
    model = VariantC(
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff_inner=getattr(cfg.model, "d_ff_inner", None),
        dropout=cfg.model.dropout,
        rope_theta=bb_cfg["rope_theta"],
        rms_eps=bb_cfg["rms_eps"],
        max_seq_len=bb_cfg["max_seq_len"],
        detach_conf_features=bool(getattr(cfg.model, "detach_conf_features", False)),
    ).to(device, dtype=dtype)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )

    step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = int(ckpt["step"])
        print(f"[train] resumed from {args.resume_from} at step {step}", flush=True)

    wandb = _setup_wandb(cfg)
    ckpt_dir = Path(cfg.checkpoint.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Local JSONL mirror of every wandb.log payload. Append-mode so resumes
    # don't truncate the existing history. Plot with
    # `python -m delta_model.eval.plot_metrics <ckpt_dir>/metrics.jsonl`.
    metrics_path = ckpt_dir / "metrics.jsonl"
    metrics_fp = metrics_path.open("a", buffering=1)   # line-buffered
    print(f"[train] metrics → {metrics_path}", flush=True)

    def log_metrics(step: int, payload: dict) -> None:
        """Mirror to both wandb (if active) and the local JSONL log."""
        if wandb is not None:
            wandb.log(payload, step=step)
        metrics_fp.write(json.dumps({"step": int(step), **payload}) + "\n")

    # Best-checkpoint tracker. Saves
    #   best_val_kl_step<N>.pt   (lower is better)
    #   best_gsm8k_step<N>.pt    (higher is better)
    # next to the rolling `step_*.pt` files. The peak GSM8K checkpoint
    # otherwise gets rotated away by `keep_last` (cf. M1.5: peaked at
    # step 5k, only 16k/18k/20k checkpoints survived to final eval).
    best_tracker = _BestTracker(ckpt_dir, log=lambda s: print(s, flush=True))
    if best_tracker.state:
        print(f"[train] resumed best_metrics: {best_tracker.state}", flush=True)

    log_keys = ("loss", "mse", "kl", "kl_backward", "bce",
                "c_label_mean", "c_pred_mean")

    # v3 loss knobs (Change 1 + Change 4). Pulled once; passed verbatim to
    # every composite_loss call (train loop + run_val) so the two never
    # diverge. Legacy defaults reproduce pre-v3 behaviour exactly.
    loss_kwargs = dict(
        lambda_mse=cfg.loss.lambda_mse,
        lambda_kl=cfg.loss.lambda_kl,
        lambda_conf=cfg.loss.lambda_conf,
        mse_space=getattr(cfg.loss, "mse_space", "raw"),
        conf_target=getattr(cfg.loss, "conf_target", "shared_mass"),
        conf_topk=int(getattr(cfg.loss, "conf_topk", 10)),
        kl_mode=getattr(cfg.loss, "kl_mode", "forward"),
        lambda_kl_backward=float(getattr(cfg.loss, "lambda_kl_backward", 0.5)),
        kl_backward_clamp=float(getattr(cfg.loss, "kl_backward_clamp", 10.0)),
    )
    # v3 phase 1 (delta): the conf head is not trained. Force its loss
    # weight to 0 so BCE contributes no gradient, regardless of the
    # config value.
    if phase == "delta":
        loss_kwargs["lambda_conf"] = 0.0
        print("[train] phase=delta → lambda_conf forced to 0 (conf head untrained)",
              flush=True)
    print("[train] starting", flush=True)
    t0 = time.time()
    train_iter = iter(train_dl)
    timers = StepTimers()
    pbar = tqdm(
        total=cfg.optim.max_steps, initial=step,
        desc="train", dynamic_ncols=True,
    )
    while step < cfg.optim.max_steps:
        with timers.time("data"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)

        with timers.time("h2d"):
            batch_dev = {
                k: v.to(device, dtype=dtype) if isinstance(v, torch.Tensor)
                                                and v.dtype.is_floating_point
                else (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            with torch.no_grad():
                prev_emb = token_embed(batch_dev["substituted_ids"])

        with timers.time("fwd"):
            delta_h, c_pos = model(
                batch_dev["h_ref"], prev_emb, batch_dev["prefix_kv"],
                batch_dev["block_start_pos"],
                prefix_kv_pad_mask=batch_dev.get("prefix_kv_pad_mask"),
            )

        with timers.time("loss"):
            loss_dict = composite_loss(
                delta_h, c_pos,
                batch_dev["h_ref"], batch_dev["h_target"], batch_dev["mask_tgt"],
                final_norm, lm_head,
                **loss_kwargs,
            )

        with timers.time("bwd"):
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)

        with timers.time("opt"):
            lr = _lr_schedule(
                step, warmup=cfg.optim.warmup_steps,
                total=cfg.optim.max_steps, base_lr=cfg.optim.lr,
                kind=cfg.optim.scheduler,
            )
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step(); opt.zero_grad()

        pbar.set_postfix(
            loss=f"{float(loss_dict['loss'].item()):.3e}",
            mse=f"{float(loss_dict['mse'].item()):.2e}",
            kl=f"{float(loss_dict['kl'].item()):.2e}",
            bce=f"{float(loss_dict['bce'].item()):.2e}",
            lr=f"{lr:.1e}",
            refresh=False,
        )

        if step % cfg.log.log_every == 0:
            payload = {f"train/{k}": float(loss_dict[k].item()) for k in log_keys}
            payload["lr"] = lr
            payload["throughput/samples_per_sec"] = (
                cfg.data.batch_size / max(1e-6, time.time() - t0)
            )
            t0 = time.time()
            log_metrics(step, payload)
            if wandb is None:
                pbar.write(
                    f"[train] step={step:6d} lr={lr:.2e} "
                    + " ".join(f"{k}={float(loss_dict[k].item()):.4e}" for k in log_keys),
                )
            pbar.write(f"[time] step={step:6d} {timers.summary_and_reset()}")

        if step > 0 and step % cfg.log.val_every == 0:
            with timers.time("val"):
                val_metrics = run_val(
                    model, val_dl, token_embed, final_norm, lm_head,
                    loss_kwargs=loss_kwargs,
                    bins_gap=cfg.log.bins_gap,
                    bins_reveal=cfg.log.bins_reveal,
                    device=device, dtype=dtype,
                )
            log_metrics(step, {f"val/{k}": v for k, v in val_metrics.items()})
            best_tracker.maybe_save(
                label="val_kl", value=val_metrics.get("kl"), mode="min",
                step=step, model=model, opt=opt, cfg=cfg,
            )
            if wandb is None:
                pbar.write(f"[val ] step={step:6d} {val_metrics}")

        if (step > 0 and cfg.log.gsm8k_every > 0
                and step % cfg.log.gsm8k_every == 0):
            try:
                from .eval.gsm8k_e2e import run_gsm8k_eval
                # v3 phase=delta: the conf head is untrained, so the hybrid
                # gate is meaningless — run delta-only GSM8K instead (the
                # free-running phase-1 selection metric) and track top-3.
                # Other phases keep the hybrid eval + single-best tracker.
                delta_only_eval = (phase == "delta")
                gsm = run_gsm8k_eval(
                    backbone_model=None,         # let the eval load its own
                    delta_model=model,
                    n_problems=cfg.log.gsm8k_subset,
                    per_pos_threshold=getattr(
                        cfg.log, "gsm8k_per_pos_threshold", 0.85,
                    ),
                    delta_only=delta_only_eval,
                    inner_loop_max_iter=getattr(
                        cfg.log, "gsm8k_inner_loop_max_iter", None,
                    ),
                    seed=cfg.seed,
                    show_progress=False,   # keep the train loop's pbar clean
                )
                log_metrics(step, {f"gsm8k/{k}": v for k, v in gsm.items()})
                if delta_only_eval:
                    best_tracker.maybe_save_topk(
                        label="delta_gsm8k",
                        value=gsm.get("accuracy_hybrid"), mode="max",
                        step=step, model=model, opt=opt, cfg=cfg, k=3,
                    )
                else:
                    best_tracker.maybe_save(
                        label="gsm8k", value=gsm.get("accuracy_hybrid"),
                        mode="max", step=step, model=model, opt=opt, cfg=cfg,
                    )
            except Exception as e:
                pbar.write(f"[gsm8k] mid-train eval failed (continuing): {e}")

        if step > 0 and step % cfg.checkpoint.every == 0:
            path = _ckpt_save(model, opt, step, cfg, out_dir=ckpt_dir)
            pbar.write(f"[ckpt] {path}")

        step += 1
        pbar.update(1)

    pbar.close()
    final = _ckpt_save(model, opt, step, cfg, out_dir=ckpt_dir)
    metrics_fp.close()
    print(f"[train] done. final ckpt: {final}", flush=True)


if __name__ == "__main__":
    main()

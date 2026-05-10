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
            lambda_mse, lambda_kl, lambda_conf, bins_gap, bins_reveal,
            max_batches: int = 50,
            device="cuda", dtype=torch.bfloat16) -> dict:
    """Compute val loss components + per-bin breakdowns."""
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
        delta_h, c_pred = model(
            batch_dev["h_ref"], prev_emb, batch_dev["prefix_kv"],
            batch_dev["block_start_pos"],
        )
        loss_dict = composite_loss(
            delta_h, c_pred,
            batch_dev["h_ref"], batch_dev["h_target"], batch_dev["mask_tgt"],
            final_norm, lm_head,
            lambda_mse=lambda_mse, lambda_kl=lambda_kl, lambda_conf=lambda_conf,
        )
        for k, v in loss_dict.items():
            sums[k]   += float(v.item())
            counts[k] += 1

        # Per-(gap, reveal_bin) sums of MSE per row.
        gap = (batch["i_target"] - batch["i_ref"]).numpy()              # [B]
        reveal = batch["reveal_frac"].numpy()                            # [B]
        with torch.no_grad():
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
            for bi_, (lo, hi) in enumerate(zip(bins_reveal[:-1], bins_reveal[1:])):
                if lo <= r <= hi:
                    bin_sums["reveal"][bi_]   += float(row_mse_np[r_idx])
                    bin_counts["reveal"][bi_] += 1
                    break

    out = {}
    for k, total in sums.items():
        out[k] = total / max(1, counts[k])
    for g, total in bin_sums["gap"].items():
        out[f"mse_by_gap_{g}"] = total / max(1, bin_counts["gap"][g])
    for bi_, total in bin_sums["reveal"].items():
        out[f"mse_by_reveal_{bi_}"] = total / max(1, bin_counts["reveal"][bi_])
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
    # rotate
    ckpts = sorted(out_dir.glob("step_*.pt"))
    keep = cfg.checkpoint.keep_last
    for old in ckpts[:-keep] if keep > 0 else []:
        old.unlink()
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--resume_from", default=None)
    args = p.parse_args()

    cfg = _load_config(args.config, args.override)

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
    train_filter, val_filter = make_train_val_filter(
        cfg.data.val_frac, seed=cfg.data.shuffle_seed,
    )
    preload = bool(getattr(cfg.data, "preload", True))
    train_ds = TataDeltaDataset(
        cfg.data.cache_root, split="train",
        mask_token_id=cfg.data.mask_token_id, index_filter=train_filter,
        preload=preload,
    )
    val_ds = TataDeltaDataset(
        cfg.data.cache_root, split="train",
        mask_token_id=cfg.data.mask_token_id, index_filter=val_filter,
        preload=preload,
    )
    print(f"[train] train pairs={len(train_ds)}  val pairs={len(val_ds)}",
          flush=True)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        drop_last=True, persistent_workers=cfg.data.num_workers > 0,
    )
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

    log_keys = ("loss", "mse", "kl", "bce", "c_label_mean", "c_pred_mean")
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
            delta_h, c_pred = model(
                batch_dev["h_ref"], prev_emb, batch_dev["prefix_kv"],
                batch_dev["block_start_pos"],
            )

        with timers.time("loss"):
            loss_dict = composite_loss(
                delta_h, c_pred,
                batch_dev["h_ref"], batch_dev["h_target"], batch_dev["mask_tgt"],
                final_norm, lm_head,
                lambda_mse=cfg.loss.lambda_mse,
                lambda_kl=cfg.loss.lambda_kl,
                lambda_conf=cfg.loss.lambda_conf,
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

        if step % cfg.log.log_every == 0 and wandb is not None:
            payload = {f"train/{k}": float(loss_dict[k].item()) for k in log_keys}
            payload["lr"] = lr
            payload["throughput/samples_per_sec"] = (
                cfg.data.batch_size / max(1e-6, time.time() - t0)
            )
            wandb.log(payload, step=step)
            t0 = time.time()
            pbar.write(f"[time] step={step:6d} {timers.summary_and_reset()}")
        elif step % cfg.log.log_every == 0:
            pbar.write(
                f"[train] step={step:6d} lr={lr:.2e} "
                + " ".join(f"{k}={float(loss_dict[k].item()):.4e}" for k in log_keys),
            )
            pbar.write(f"[time] step={step:6d} {timers.summary_and_reset()}")

        if step > 0 and step % cfg.log.val_every == 0:
            with timers.time("val"):
                val_metrics = run_val(
                    model, val_dl, token_embed, final_norm, lm_head,
                    lambda_mse=cfg.loss.lambda_mse,
                    lambda_kl=cfg.loss.lambda_kl,
                    lambda_conf=cfg.loss.lambda_conf,
                    bins_gap=cfg.log.bins_gap,
                    bins_reveal=cfg.log.bins_reveal,
                    device=device, dtype=dtype,
                )
            if wandb is not None:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()},
                          step=step)
            else:
                pbar.write(f"[val ] step={step:6d} {val_metrics}")

        if (step > 0 and cfg.log.gsm8k_every > 0
                and step % cfg.log.gsm8k_every == 0):
            try:
                from .eval.gsm8k_e2e import run_gsm8k_eval
                gsm = run_gsm8k_eval(
                    backbone_model=None,         # let the eval load its own
                    delta_model=model,
                    n_problems=cfg.log.gsm8k_subset,
                    seed=cfg.seed,
                )
                if wandb is not None:
                    wandb.log({f"gsm8k/{k}": v for k, v in gsm.items()},
                              step=step)
            except Exception as e:
                pbar.write(f"[gsm8k] mid-train eval failed (continuing): {e}")

        if step > 0 and step % cfg.checkpoint.every == 0:
            path = _ckpt_save(model, opt, step, cfg, out_dir=ckpt_dir)
            pbar.write(f"[ckpt] {path}")

        step += 1
        pbar.update(1)

    pbar.close()
    final = _ckpt_save(model, opt, step, cfg, out_dir=ckpt_dir)
    print(f"[train] done. final ckpt: {final}", flush=True)


if __name__ == "__main__":
    main()

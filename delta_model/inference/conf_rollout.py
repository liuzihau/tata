"""v3 Change 3 — free-running DAgger labelling for the confidence head.

The v2 conf head is trained to predict a *teacher-forced* quantity
(`shared_mass` from cached `h_ref` / reveal pattern). At inference it is
used as a gate on *free-running* states (`prev_emb` built from the
delta's own prior commits). T9 / T10 showed those diverge — the conf
head becomes a good shared-mass predictor and a bad commit gate.

This module fixes the supervision distribution. It rolls the delta
decoder forward on real prompts and, at every delta pass, records:

  - the conf head's inputs at that *free-running* state
    (`h_ref`, `prev_emb`, `prefix_kv`, `block_start_pos`, pad mask);
  - a tiered free-running label — the Change-1 top1/top10/all comparison
    between the delta's distribution and the *backbone's* distribution
    **at the same free-running state** (one extra partial backbone
    forward per delta pass).

Phase-2 conf training (train.py `--phase conf`) then trains the conf
head's BCE against those records. This is DAgger: the supervision
distribution is the delta policy's own state distribution.

The rollout commits the delta's tokens ungated (`delta_only` style — no
conf gate, no rollback) so it visits the delta's free-running states.
The delta is frozen throughout (phase 2 only trains the conf head), so
the recorded `(inputs, label)` pairs are stationary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import schema as S
from ..llada_runtime import _get_transfer_index, _get_transfer_index_dynamic
from ..losses import _tiered_conf_label
from .hybrid_runner import _find_last_block


@torch.no_grad()
def collect_dagger_records(
    backbone: nn.Module,
    delta_model: nn.Module,
    final_norm: nn.Module,
    lm_head: nn.Linear,
    token_embed: nn.Embedding,
    prompts: list[torch.Tensor],         # each [1, Lp] on backbone.device
    *,
    gen_length: int = S.GEN_LENGTH,
    block_length: int = S.BLOCK_LENGTH,
    factor: float | None = 1.0,
    threshold: float | None = None,
    conf_topk: int = 10,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = S.LLADA_MASK_TOKEN_ID,
    inner_loop_cap: int | None = None,
    max_records: int | None = None,
) -> list[dict]:
    """Roll the delta decoder forward on `prompts`; return one record per
    delta pass. Each record is a dict of CPU tensors:

        h_ref               [BL, d_model]   fp16
        prev_emb            [BL, d_model]   fp16
        prefix_kv           [2, n_kv, BL, d_head]  fp16
        prefix_kv_pad_mask  [BL]  bool
        block_start_pos     int
        mask_tgt            [BL]  bool   — positions masked at this pass
        c_label             [BL]  float  — tiered free-running label

    Records with no masked positions are skipped (nothing to supervise).
    """
    device = backbone.device
    W = S.PREFIX_WINDOW
    loop_cap = (int(inner_loop_cap) if inner_loop_cap is not None
                else 2 * int(block_length))

    hook_state = {"latest": None}
    last_block = _find_last_block(backbone)

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        hook_state["latest"] = h.detach()

    handle = last_block.register_forward_hook(_hook)
    records: list[dict] = []

    def _transfer(logits, mask_index, x_seq):
        """Fast-dLLM token transfer (factor or threshold mode)."""
        if factor is not None:
            return _get_transfer_index_dynamic(
                logits, temperature, remasking, mask_index, x_seq, factor=factor,
            )
        return _get_transfer_index(
            logits, temperature, remasking, mask_index, x_seq, None, threshold,
        )

    try:
        for prompt in prompts:
            Lp = int(prompt.shape[1])
            num_blocks = gen_length // block_length
            x = torch.full((1, Lp + gen_length), mask_id,
                           dtype=torch.long, device=device)
            x[:, :Lp] = prompt

            for nb in range(num_blocks):
                s = Lp + nb * block_length
                e = s + block_length

                # ---- pass 0: full backbone forward ----
                out = backbone(x, use_cache=True)
                past_to_s = [tuple(t[:, :, :s, :] for t in pkv)
                             for pkv in out.past_key_values]
                h_block = hook_state["latest"][:, s:e, :]              # [1,BL,D]
                h_ref = h_block.clone()

                # prefix KV for the delta model (front-padded if s < W).
                valid = min(s, W); pad = W - valid
                rk = past_to_s[-1][0][0, :, s - valid:s, :]
                rv = past_to_s[-1][1][0, :, s - valid:s, :]
                if pad > 0:
                    z = torch.zeros((rk.shape[0], pad, rk.shape[2]),
                                    dtype=rk.dtype, device=device)
                    rk = torch.cat([z, rk], dim=1)
                    rv = torch.cat([z, rv], dim=1)
                prefix_kv = torch.stack([rk, rv], dim=0).unsqueeze(0)   # [1,2,nkv,W,dh]
                pad_mask = torch.zeros((1, W), dtype=torch.bool, device=device)
                if valid > 0:
                    pad_mask[:, pad:] = True

                # pass-0 commit on the backbone's own logits.
                gmask = (x == mask_id); gmask[:, e:] = False
                x0, ti = _transfer(out.logits, gmask, x)
                x = torch.where(ti, x0, x)

                block_start_pos = torch.tensor([s], dtype=torch.long, device=device)
                n_passes = 1

                # ---- delta passes (delta_only rollout) ----
                while n_passes < loop_cap:
                    mask_blk = (x[:, s:e] == mask_id)                   # [1,BL]
                    if int(mask_blk.sum().item()) == 0:
                        break

                    prev_emb = token_embed(x[0, s:e]).to(
                        device=device, dtype=h_ref.dtype,
                    ).unsqueeze(0)                                      # [1,BL,D]

                    delta_h, _c = delta_model(
                        h_ref.to(delta_model.delta_head.proj.weight.dtype),
                        prev_emb.to(delta_model.delta_head.proj.weight.dtype),
                        prefix_kv.to(torch.float16),
                        block_start_pos,
                        prefix_kv_pad_mask=pad_mask,
                    )
                    h_pred = h_ref + delta_h.to(h_ref.dtype)
                    logits_delta = lm_head(final_norm(h_pred))          # [1,BL,V]

                    # ---- LABEL: backbone distribution at this free-running
                    # state (one partial forward), then the tiered compare.
                    out_bb = backbone(x[:, s:], past_key_values=past_to_s,
                                      use_cache=True)
                    logits_bb = out_bb.logits[:, :block_length, :]      # [1,BL,V]
                    p_actual = F.softmax(logits_bb.float(), dim=-1)
                    p_pred   = F.softmax(logits_delta.float(), dim=-1)
                    c_label  = _tiered_conf_label(
                        p_actual, p_pred, topk=conf_topk,
                    )[0]                                                # [BL]

                    records.append({
                        "h_ref":              h_ref[0].half().cpu(),
                        "prev_emb":           prev_emb[0].half().cpu(),
                        "prefix_kv":          prefix_kv[0].half().cpu(),
                        "prefix_kv_pad_mask": pad_mask[0].cpu(),
                        "block_start_pos":    int(s),
                        "mask_tgt":           mask_blk[0].cpu(),
                        "c_label":            c_label.float().cpu(),
                    })
                    if max_records is not None and len(records) >= max_records:
                        return records

                    # commit ungated (delta_only) — Fast-dLLM transfer on
                    # the delta's logits.
                    x0b, tib = _transfer(logits_delta, mask_blk, x[:, s:e])
                    x = torch.cat([x[:, :s],
                                   torch.where(tib, x0b, x[:, s:e]),
                                   x[:, e:]], dim=1)
                    n_passes += 1
    finally:
        handle.remove()

    return records


def records_to_batches(records: list[dict], batch_size: int,
                        *, device="cuda", shuffle: bool = True,
                        generator: torch.Generator | None = None):
    """Yield device batches from `collect_dagger_records` output. Each
    batch dict has the keys VariantC.forward + the BCE loss consume:
    h_ref, prev_emb, prefix_kv, prefix_kv_pad_mask, block_start_pos,
    mask_tgt, c_label."""
    n = len(records)
    order = (torch.randperm(n, generator=generator).tolist()
             if shuffle else list(range(n)))
    for i in range(0, n - batch_size + 1, batch_size):
        idx = order[i:i + batch_size]
        chunk = [records[j] for j in idx]
        yield {
            "h_ref":              torch.stack([r["h_ref"] for r in chunk]).to(device),
            "prev_emb":           torch.stack([r["prev_emb"] for r in chunk]).to(device),
            "prefix_kv":          torch.stack([r["prefix_kv"] for r in chunk]).to(device),
            "prefix_kv_pad_mask": torch.stack([r["prefix_kv_pad_mask"] for r in chunk]).to(device),
            "block_start_pos":    torch.tensor([r["block_start_pos"] for r in chunk],
                                               dtype=torch.long, device=device),
            "mask_tgt":           torch.stack([r["mask_tgt"] for r in chunk]).to(device),
            "c_label":            torch.stack([r["c_label"] for r in chunk]).to(device),
        }

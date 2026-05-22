"""Cache builder for the LLaDA delta-model training set.

Runs LLaDA + Fast-dLLM v1 in *prefix-cache* mode (matching the inference
engine our delta model will swap into) and captures, per sample:

    - last-32-token prefix KV at the last layer at each block's pass-1
    - per-pass last-layer hidden state at the block's positions, for
      passes 0..MAX_ITER-1
    - the reveal pattern at the start of each pass

Data is partitioned into a hash-stable test set (default 800 problems) and
a scalable train set (default 5000 problems). The test partition is
deterministic across collect runs so increasing n_train never overlaps
with the held-out test set.

CLI (run from inside the tata repo root):
    python -m delta_model.data.collect_llada \\
        --n_train 5000 --n_test 800 \\
        --output_root cache_v1/llada \\
        --fast_dllm_path external/Fast-dLLM/v1 \\
        --subset_ratios '{"chat":0.35,"math":0.35,"code":0.20,"stem":0.10}'

Prereq: `huggingface-cli login` once on the machine that runs collect
(Nemotron Post-Training v2 is gated).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import schema as S
from ..llada_runtime import (
    _get_num_transfer_tokens,
    _get_transfer_index,
    _get_transfer_index_dynamic,
    load_llada,
)
from ..models.variant_c import VariantC


def _load_delta_for_collect(delta_ckpt: str, backbone) -> torch.nn.Module:
    """Load a VariantC (delta + conf head) for in-the-loop gated-hybrid
    rollout (v4 DAgger). Pass a v3 **phase-2** checkpoint — it carries
    the phase-1 delta AND the phase-2-trained conf head, so the rollout
    can gate exactly as the deployed hybrid does. Backbone hyperparams
    are read off `backbone.config` so RoPE / RMSNorm match the cache.
    """
    ckpt = torch.load(delta_ckpt, map_location="cpu", weights_only=False)
    cm = ckpt["cfg"]["model"]
    delta = VariantC(
        d_model=cm["d_model"], n_heads=cm["n_heads"], n_layers=cm["n_layers"],
        d_ff_inner=cm.get("d_ff_inner"), dropout=0.0,
        detach_conf_features=bool(cm.get("detach_conf_features", False)),
        rope_theta=float(getattr(backbone.config, "rope_theta", 1e6)),
        rms_eps=float(getattr(backbone.config, "layer_norm_eps", 1e-5)),
        max_seq_len=int(getattr(backbone.config, "max_sequence_length", 8192)),
    ).to(backbone.device, dtype=torch.bfloat16).eval()
    delta.load_state_dict(ckpt["model"])
    return delta


# ---------------------------------------------------------------------------
# Nemotron loader with hash-stable splits
# ---------------------------------------------------------------------------

def _stable_hash(text: str) -> int:
    """Deterministic 32-bit hash of a string. Used to partition prompts
    into a fixed test set independent of dataset row order or n_train."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _default_prompt_extractor(record: dict) -> str:
    """Best-effort extractor for the prompt text in a Nemotron record.

    Nemotron Post-Training v2 records vary across subsets — the conversation
    field is usually `messages` (list of role/content dicts) but some rows
    expose `input` or `prompt` strings directly. We grab the first user
    turn or fall back to whatever string field looks promising.
    """
    if "messages" in record and record["messages"]:
        for m in record["messages"]:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        # No explicit user turn — concatenate everything.
        return "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m)
            for m in record["messages"]
        )
    for key in ("input", "prompt", "question", "query", "instruction"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError(
        f"Could not find a prompt field in record. Available keys: {list(record.keys())}"
    )


def _load_and_split_nemotron(
    subset_ratios: dict[str, float],
    n_train: int,
    n_test: int,
    seed: int,
    prompt_extractor: Callable[[dict], str] = _default_prompt_extractor,
) -> tuple[list[dict], list[dict]]:
    """Load Nemotron Post-Training v2 and split into stable test + train.

    Each returned record is a dict {"prompt": str, "subset": str,
    "split": "train"|"test", "id": int}. The split is hash-stable: the
    same `n_test` test problems are returned regardless of `n_train`.

    Deduplication: Nemotron has many records sharing the same user prompt
    text (multiple candidate assistant responses per user query, plus
    cross-subset overlap on chat-formatted math/code problems). Since our
    on-disk filename is `sample_<id>.pt` and `id` is derived from the
    user-prompt-text hash, duplicate prompts would clobber to the same
    file and waste the n_train budget. We dedup globally by hash before
    allocating the per-subset budget so `n_train + n_test` slots each
    map to a unique prompt.
    """
    from datasets import load_dataset

    total_frac = sum(f for f in subset_ratios.values() if f > 0)
    if total_frac <= 0:
        raise ValueError("subset_ratios must contain at least one positive entry")

    train_records: list[dict] = []
    test_records: list[dict] = []
    # Global dedup set — shared across subsets so cross-subset duplicates
    # (e.g. the same chat-math prompt in both subsets) are caught.
    seen_hashes: set[int] = set()

    for subset, frac in subset_ratios.items():
        if frac <= 0:
            continue
        print(f"[nemotron] loading subset='{subset}' …", flush=True)
        ds = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            split=subset,
        )

        # Extract prompts and dedup by hash. Track intra-subset duplicates
        # (multiple records with the same user prompt — typically multiple
        # assistant candidates per user query) separately from cross-subset
        # duplicates (prompts already seen in an earlier subset).
        unique_prompts: list[str] = []
        unique_hashes:  list[int] = []
        n_dup_intra = 0
        n_dup_cross = 0
        local_seen: set[int] = set()
        for r in ds:
            p = prompt_extractor(r)
            h = _stable_hash(p)
            if h in local_seen:
                n_dup_intra += 1
                continue
            local_seen.add(h)
            if h in seen_hashes:
                n_dup_cross += 1
                continue
            seen_hashes.add(h)
            unique_prompts.append(p)
            unique_hashes.append(h)
        print(
            f"[nemotron] {subset}: {len(ds)} raw → {len(unique_prompts)} unique "
            f"(dropped {n_dup_intra} intra-subset, {n_dup_cross} cross-subset)",
            flush=True,
        )

        # Deterministic order via hash of the prompt text.
        order = sorted(range(len(unique_prompts)),
                       key=lambda i: unique_hashes[i])

        q_test = int(round(n_test * frac / total_frac))
        q_train = int(round(n_train * frac / total_frac))

        if q_test + q_train > len(unique_prompts):
            raise ValueError(
                f"subset '{subset}' has {len(unique_prompts)} unique prompts "
                f"after dedup but we asked for {q_test}+{q_train}. Lower "
                f"n_train, raise subset_ratios for other subsets, or pick "
                f"a subset with more unique data."
            )

        for idx in order[:q_test]:
            test_records.append({
                "prompt": unique_prompts[idx], "subset": subset, "split": "test",
                "id": unique_hashes[idx],
            })
        for idx in order[q_test:q_test + q_train]:
            train_records.append({
                "prompt": unique_prompts[idx], "subset": subset, "split": "train",
                "id": unique_hashes[idx],
            })

    rng = random.Random(seed)
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    print(
        f"[nemotron] split: {len(train_records)} train + {len(test_records)} test "
        f"({len(seen_hashes)} unique prompts seen across all subsets)",
        flush=True,
    )
    return train_records, test_records


# ---------------------------------------------------------------------------
# Per-sample collection — fork of generate_with_prefix_cache
# ---------------------------------------------------------------------------

def _find_last_block(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last LLaDABlock module (we hook its output to capture
    last-layer hidden states)."""
    blocks: list[torch.nn.Module] = []
    for m in model.modules():
        if type(m).__name__ in ("LLaDASequentialBlock", "LLaDALlamaBlock"):
            blocks.append(m)
    if not blocks:
        raise RuntimeError(
            "No LLaDA transformer blocks found — has the model architecture changed?"
        )
    return blocks[-1]


def _format_prompt_llada(tokenizer, question: str) -> torch.Tensor:
    """Apply LLaDA's chat template to a user message. Returns [1, S] long
    tensor on the model's device."""
    msg = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        msg, add_generation_prompt=True, tokenize=False,
    )
    ids = tokenizer(prompt)["input_ids"]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()


@torch.no_grad()
def collect_one_sample(
    model,
    tokenizer,
    prompt_text: str,
    *,
    gen_length: int = S.GEN_LENGTH,
    block_length: int = S.BLOCK_LENGTH,
    max_iter: int = S.MAX_ITER,
    prefix_window: int = S.COLLECT_PREFIX_WINDOW,
    threshold: float | None = None,
    factor: float | None = 1.0,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = S.LLADA_MASK_TOKEN_ID,
    delta_model=None,
    final_norm=None,
    lm_head=None,
    token_embed=None,
    gate_threshold: float = 0.75,
) -> dict | None:
    """Run prefix-cache generation on one prompt and return the cache dict
    described in `data/schema.py`. Returns None only for empty prompts.

    Gated-hybrid rollout (v4 — DAgger for the delta). When `delta_model`
    is given (a v3 phase-2 VariantC: delta + trained conf head), each
    in-block iteration ≥ 1 commits like the *deployed hybrid policy*:
    the delta proposes tokens, and per position the conf head's
    `c_pos ≥ gate_threshold` decides whether the **delta's** token or the
    **backbone's** token is committed. The reveal pattern / committed IDs
    then follow the gated free-running trajectory, while `h_per_pass[i]`
    stays the *backbone's* hidden at that state — the
    exposure-bias-correcting training target. `delta_model = None`
    (default) → pure teacher-forced collect, identical to the legacy path.

    `prefix_window` is the number of last-prefix slots cached per block.
    Default = `S.COLLECT_PREFIX_WINDOW` (64) so we have headroom over the
    training-time `S.PREFIX_WINDOW` (32). When the prompt is shorter than
    `prefix_window`, the prefix_kv tensor is zero-padded at the front and
    `prefix_kv_pad_mask` records which slots are real — so short prompts
    are no longer dropped. The dataset slices to the last `S.PREFIX_WINDOW`
    of both tensors and the model's cross-attention applies the mask.

    Decoding mode is controlled by exactly one of:
      - `threshold` (float in (0,1]): commit positions with confidence ≥ thr.
      - `factor`    (float > 0):       per-rank dynamic threshold (Fast-dLLM
                                       paper-recommended). Default 1.0.
    Pass the other as None.
    """
    if (threshold is None) == (factor is None):
        raise ValueError(
            "collect_one_sample: pass exactly one of `threshold` or `factor` "
            f"(got threshold={threshold!r}, factor={factor!r})"
        )
    if prefix_window < S.PREFIX_WINDOW:
        raise ValueError(
            f"prefix_window ({prefix_window}) must be ≥ S.PREFIX_WINDOW "
            f"({S.PREFIX_WINDOW}); training slices to the last {S.PREFIX_WINDOW} "
            "tokens of the stored prefix_kv."
        )
    prompt_ids = _format_prompt_llada(tokenizer, prompt_text)
    prompt_len = int(prompt_ids.shape[1])
    if prompt_len < 1:
        return None

    num_blocks = gen_length // block_length
    assert num_blocks == S.NUM_BLOCKS, (
        f"schema expects {S.NUM_BLOCKS} blocks, got {num_blocks}"
    )

    # Initial sequence: prompt + all-mask suffix.
    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device,
    )
    x[:, :prompt_len] = prompt_ids

    # --- Hook to capture last-block (= last-layer pre-norm) hidden states ---
    # Pass 0 is a full forward over the whole sequence; we slice the block
    # region [s:e] from its output. Iter ≥ 1 uses a partial forward over
    # x[:, s:] (Fast-dLLM prefix-cache style) — the block region is the
    # first `block_length` rows of that output.
    hook_state = {"latest": None, "pass_idx": 0, "s": 0, "e": 0}

    def _last_block_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # [1, T, d_model]
        if hook_state["pass_idx"] == 0:
            slice_h = h[:, hook_state["s"]:hook_state["e"], :]
        else:
            slice_h = h[:, :block_length, :]
        hook_state["latest"] = (
            slice_h.detach().to("cpu", dtype=S.DTYPE_HIDDEN, copy=True).squeeze(0)
        )

    handle = _find_last_block(model).register_forward_hook(_last_block_hook)

    # Per-block accumulators.
    blocks_out: list[dict] = []

    try:
        for nb in range(num_blocks):
            s = prompt_len + nb * block_length
            e = s + block_length

            block_mask_index = (x[:, s:e] == mask_id)
            num_transfer_tokens = _get_num_transfer_tokens(
                block_mask_index, max(1, max_iter),
            )

            h_per_pass: list[torch.Tensor] = []           # each [block_length, d_model]
            reveal_per_pass: list[torch.Tensor] = []      # each [block_length] bool
            past_key_values = None
            recorded_passes = 0
            i = 0

            while True:
                # Stop conditions (in order):
                #   1. block fully decoded → done with this block.
                #   2. recorded_passes == max_iter → keep decoding (so that the
                #      next block sees a fully-decoded prefix), but no longer
                #      record into h_per_pass / reveal_per_pass.
                fully_decoded = (x[:, s:e] == mask_id).sum().item() == 0
                if fully_decoded:
                    break

                # Reveal state at the START of this pass.
                if recorded_passes < max_iter:
                    pre_reveal = (x[0, s:e] != mask_id).to(
                        dtype=torch.bool, device="cpu",
                    )
                    reveal_per_pass.append(pre_reveal)

                hook_state["pass_idx"] = i
                hook_state["s"] = s
                hook_state["e"] = e

                # Fast-dLLM prefix-cache decoding:
                #   pass 0 of each block: full forward over x; cache prefix [:s].
                #   pass ≥ 1: partial forward over x[:, s:] with cached prefix.
                # This matches what the vanilla `generate_with_prefix_cache`
                # baseline does at inference, so h_per_pass[i] captured here
                # is what the baseline would compute at the same iter. Training
                # the delta model on these targets keeps the hidden-state
                # distribution aligned between collect, baseline, and the
                # hybrid runner's pass-0 full forward.
                if i == 0:
                    out = model(x, use_cache=True)
                    past_key_values = out.past_key_values

                    # Capture up to `prefix_window` last-prefix-token KV at
                    # positions [s-valid_len, s). When `s < prefix_window`
                    # (short prompt at block 0), we have fewer valid slots;
                    # front-pad with zeros and record validity in
                    # `prefix_kv_pad_mask` so cross-attn at training/inference
                    # masks the padded positions out.
                    valid_len = min(s, prefix_window)
                    pad_len   = prefix_window - valid_len
                    last_K_v = past_key_values[-1][0][0, :, s - valid_len:s, :]
                    last_V_v = past_key_values[-1][1][0, :, s - valid_len:s, :]
                    if pad_len > 0:
                        last_K = F.pad(last_K_v, (0, 0, pad_len, 0))
                        last_V = F.pad(last_V_v, (0, 0, pad_len, 0))
                    else:
                        last_K = last_K_v
                        last_V = last_V_v
                    prefix_kv = torch.stack(
                        [last_K, last_V], dim=0,
                    ).to("cpu", dtype=S.DTYPE_KV, copy=True)
                    prefix_kv_pad_mask = torch.zeros(prefix_window, dtype=torch.bool)
                    prefix_kv_pad_mask[pad_len:] = True

                    # Slice the cache to [:s] for use at iter ≥ 1. This is
                    # the "prefix cache" the Fast-dLLM baseline reuses across
                    # iterations within the block.
                    past_key_values = [
                        tuple(t[:, :, :s, :] for t in pkv)
                        for pkv in past_key_values
                    ]
                    logits = out.logits

                    # Pass-0 token transfer: logits are full-sequence, restrict
                    # the transfer mask to the current block.
                    global_mask_index = (x == mask_id)
                    global_mask_index[:, e:] = False
                    if factor is not None:
                        x0, transfer_index = _get_transfer_index_dynamic(
                            logits, temperature, remasking, global_mask_index, x,
                            factor=factor,
                        )
                    else:
                        quota = (
                            None if threshold is not None
                            else num_transfer_tokens[:, min(i, num_transfer_tokens.size(1) - 1)]
                        )
                        x0, transfer_index = _get_transfer_index(
                            logits, temperature, remasking, global_mask_index, x,
                            quota, threshold,
                        )
                    x = torch.where(transfer_index, x0, x)

                else:
                    # Partial forward over the suffix x[:, s:] with cached
                    # prefix. This is the prefix-cache decoding pattern the
                    # baseline uses; h_per_pass[i] for i ≥ 1 is captured
                    # under this same path (via the hook) so the training
                    # target distribution aligns with the baseline. This
                    # backbone forward happens regardless of scheduled
                    # sampling — h_per_pass[i] is always the backbone's.
                    out = model(
                        x[:, s:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = out.logits   # [1, GL - nb*BL, V] — covers x[:, s:]

                    if delta_model is None:
                        # Teacher-forced (legacy) — backbone transfer.
                        # Pass-≥1 logits are suffix-only; restrict the mask
                        # to the first `block_length` rows.
                        mask_index_blk = (x[:, s:] == mask_id)
                        mask_index_blk[:, block_length:] = False
                        if factor is not None:
                            x0_full, transfer_full = _get_transfer_index_dynamic(
                                logits, temperature, remasking, mask_index_blk,
                                x[:, s:], factor=factor,
                            )
                        else:
                            quota = (
                                None if threshold is not None
                                else num_transfer_tokens[:, min(i, num_transfer_tokens.size(1) - 1)]
                            )
                            x0_full, transfer_full = _get_transfer_index(
                                logits, temperature, remasking, mask_index_blk,
                                x[:, s:], quota, threshold,
                            )
                        new_suffix = torch.where(transfer_full, x0_full, x[:, s:])
                        x = torch.cat([x[:, :s], new_suffix], dim=1)
                    else:
                        # v4 gated-hybrid rollout — commit like the deployed
                        # policy: delta token where the conf gate passes,
                        # backbone token where it vetoes. h_per_pass[i] (the
                        # hook above) stays the backbone's hidden.
                        ddt = delta_model.delta_head.proj.weight.dtype
                        h_ref_g  = h_per_pass[0].to(
                            model.device, dtype=torch.bfloat16,
                        ).unsqueeze(0)                                # [1, BL, D]
                        prev_emb = token_embed(x[0, s:e]).unsqueeze(0)
                        pkv_g = prefix_kv[..., -S.PREFIX_WINDOW:, :].to(
                            model.device,
                        ).unsqueeze(0)
                        pad_g = prefix_kv_pad_mask[-S.PREFIX_WINDOW:].to(
                            model.device,
                        ).unsqueeze(0)
                        bsp = torch.tensor(
                            [s], dtype=torch.long, device=model.device,
                        )
                        delta_h, c_pos = delta_model(
                            h_ref_g.to(ddt), prev_emb.to(ddt),
                            pkv_g.to(torch.float16), bsp,
                            prefix_kv_pad_mask=pad_g,
                        )
                        h_pred = h_ref_g + delta_h.to(h_ref_g.dtype)
                        delta_logits = lm_head(final_norm(h_pred))    # [1, BL, V]
                        bb_logits_blk = logits[:, :block_length, :]   # [1, BL, V]
                        mask_blk = (x[:, s:e] == mask_id)

                        quota = (
                            None if threshold is not None
                            else num_transfer_tokens[:, min(i, num_transfer_tokens.size(1) - 1)]
                        )
                        if factor is not None:
                            x0_d, transfer_d = _get_transfer_index_dynamic(
                                delta_logits, temperature, remasking, mask_blk,
                                x[:, s:e], factor=factor,
                            )
                            x0_b, _transfer_b = _get_transfer_index_dynamic(
                                bb_logits_blk, temperature, remasking, mask_blk,
                                x[:, s:e], factor=factor,
                            )
                        else:
                            x0_d, transfer_d = _get_transfer_index(
                                delta_logits, temperature, remasking, mask_blk,
                                x[:, s:e], quota, threshold,
                            )
                            x0_b, _transfer_b = _get_transfer_index(
                                bb_logits_blk, temperature, remasking, mask_blk,
                                x[:, s:e], quota, threshold,
                            )

                        # Gate: positions the delta wants AND the conf head
                        # trusts (c_pos ≥ gate_threshold) commit the delta's
                        # token; delta-wanted-but-vetoed positions commit the
                        # backbone's token (the rollback's effect, applied
                        # per-position). Every delta-wanted position is thus
                        # committed — never an ungated delta token.
                        per_pos_pass = (c_pos.float() >= gate_threshold)
                        commit_delta = transfer_d & per_pos_pass
                        commit_bb    = transfer_d & (~per_pos_pass)
                        new_blk = x[:, s:e].clone()
                        new_blk = torch.where(commit_delta, x0_d, new_blk)
                        new_blk = torch.where(commit_bb,    x0_b, new_blk)
                        x = torch.cat([x[:, :s], new_blk, x[:, e:]], dim=1)

                if recorded_passes < max_iter:
                    h_per_pass.append(hook_state["latest"])
                    recorded_passes += 1

                i += 1

            # Pad to MAX_ITER if the block decoded in fewer passes. The padded
            # entries duplicate the last recorded pass — these slots are
            # never used for training because the dataset filters
            # `i < n_passes_actual`.
            n_passes_actual = len(h_per_pass)
            assert n_passes_actual == len(reveal_per_pass) and n_passes_actual >= 1
            while len(h_per_pass) < S.MAX_ITER:
                h_per_pass.append(h_per_pass[-1].clone())
                reveal_per_pass.append(reveal_per_pass[-1].clone())

            blocks_out.append({
                "prefix_kv":          prefix_kv,                          # [2, n_kv_heads, prefix_window, d_head]
                "prefix_kv_pad_mask": prefix_kv_pad_mask,                 # [prefix_window] bool — True = real, False = padded
                "h_per_pass":         torch.stack(h_per_pass, dim=0),     # [MAX_ITER, 32, d_model]
                "reveal_per_pass":    torch.stack(reveal_per_pass, dim=0),# [MAX_ITER, 32]
                "n_passes_actual":    int(n_passes_actual),
            })

    finally:
        handle.remove()

    generated = x[0, prompt_len:].to("cpu", dtype=torch.long)
    assert generated.shape[0] == gen_length

    return {
        "prompt_token_ids":    prompt_ids[0].cpu().to(torch.long),
        "generated_token_ids": generated,
        "prompt_len":          prompt_len,
        "blocks":              blocks_out,
        "meta": {
            "model": "llada",
            "n_kv_heads":     S.N_KV_HEADS_LLADA,
            "d_head":         S.D_HEAD_LLADA,
            "d_model":        S.D_MODEL_LLADA,
            "gen_length":     gen_length,
            "block_length":   block_length,
            "max_iter":       max_iter,
            "prefix_window":  prefix_window,
            "fast_dllm_mode": "prefix_cache",
            "decoding": (
                {"mode": "factor",    "factor":    factor}    if factor is not None
                else {"mode": "threshold", "threshold": threshold}
            ),
            # v4 provenance: None = pure teacher-forced collect; a float
            # = the conf-gate threshold used for the gated-hybrid rollout
            # (this sample's trajectory followed the deployed policy).
            "gate_threshold": (
                float(gate_threshold) if delta_model is not None else None
            ),
            "schema_version": S.SCHEMA_VERSION,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_prompts_file(path: Path, seed: int) -> tuple[list[dict], list[dict]]:
    """Load prompts from a plain-text file (one prompt per line).

    Used for the smoke test path that bypasses Nemotron — no HF auth
    required. Every prompt becomes a 'test' split record.
    """
    raw = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    records = [{
        "prompt": p, "subset": "file", "split": "test", "id": _stable_hash(p),
    } for p in raw]
    return [], records  # no train, all test (smoke-test convention)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_test", type=int, default=800)
    p.add_argument(
        "--subset_ratios",
        type=str,
        default='{"chat":0.35,"math":0.35,"code":0.20,"stem":0.10}',
        help="JSON dict over {chat, math, code, stem}. Will be normalized.",
    )
    p.add_argument("--output_root", type=Path, default=Path("cache_v1/llada"))
    p.add_argument("--fast_dllm_path", type=str, default=None)
    p.add_argument("--shuffle_seed", type=int, default=42)
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold for parallel decoding (Fast-dLLM "
             "fixed-threshold mode). Mutually exclusive with --factor. "
             "If neither is set, defaults to --factor 1.0.",
    )
    p.add_argument(
        "--factor", type=float, default=None,
        help="Dynamic per-rank threshold factor (Fast-dLLM paper-recommended "
             "mode). Mutually exclusive with --threshold. Default behavior "
             "(neither flag) is --factor 1.0.",
    )
    p.add_argument(
        "--prefix_window", type=int, default=S.COLLECT_PREFIX_WINDOW,
        help=(
            "Number of last-prefix tokens whose K/V we record per block. "
            f"Default {S.COLLECT_PREFIX_WINDOW} (= S.COLLECT_PREFIX_WINDOW). "
            f"Must be ≥ S.PREFIX_WINDOW ({S.PREFIX_WINDOW}). Drop to 32 for "
            "smoke testing on prompts shorter than 64 tokens — the dataset "
            "always slices to the last S.PREFIX_WINDOW regardless of stored "
            "size, so as long as `prefix_window >= S.PREFIX_WINDOW`, the "
            "training pipeline reads the cache correctly."
        ),
    )
    p.add_argument(
        "--start_index", type=int, default=0,
        help="Resume from this index in the (combined train+test) record list.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap for testing (overrides n_train+n_test).",
    )
    p.add_argument(
        "--prompts_file", type=Path, default=None,
        help="If set, load prompts from this file (one per line) instead of "
             "Nemotron. No HuggingFace auth needed. Used for smoke tests.",
    )
    p.add_argument(
        "--conf_ckpt", type=str, default=None,
        help="v4 gated-hybrid rollout (DAgger for the delta): a v3 "
             "phase-2 checkpoint — a full VariantC carrying the phase-1 "
             "delta AND the phase-2-trained conf head. When set, the "
             "TRAIN split is collected by rolling out the gated hybrid "
             "policy (delta token where the conf gate passes, backbone "
             "token where it vetoes), so reveal trajectories match the "
             "deployed policy. h_per_pass targets stay the backbone's "
             "hidden states. Unset (default) = pure teacher-forced "
             "collect (legacy).",
    )
    p.add_argument(
        "--collect_gate_threshold", type=float, default=0.75,
        help="conf-gate threshold for the gated-hybrid rollout — a "
             "position commits the delta's token iff c_pos ≥ this, else "
             "the backbone's token. Only used when --conf_ckpt is set; "
             "pick the per_pos_threshold you expect to deploy at.",
    )
    args = p.parse_args()

    if not (0.0 <= args.collect_gate_threshold <= 1.0):
        p.error("--collect_gate_threshold must be in [0, 1]")

    # Mutual exclusion + default. Both default to None at the CLI; if neither
    # was passed we use factor=1.0 (paper-recommended mode).
    if args.threshold is not None and args.factor is not None:
        p.error("--threshold and --factor are mutually exclusive")
    if args.threshold is None and args.factor is None:
        decoding_threshold: float | None = None
        decoding_factor: float | None = 1.0
    else:
        decoding_threshold = args.threshold
        decoding_factor = args.factor
    print(
        f"[collect] decoding mode: "
        + (f"factor={decoding_factor}" if decoding_factor is not None
           else f"threshold={decoding_threshold}"),
        flush=True,
    )
    print(
        f"[collect] prefix_window={args.prefix_window} "
        f"(training reads last {S.PREFIX_WINDOW}); "
        f"prompts shorter than {args.prefix_window} tokens will be skipped.",
        flush=True,
    )

    out_root = args.output_root
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    if args.prompts_file is not None:
        train_records, test_records = _load_prompts_file(
            args.prompts_file, seed=args.shuffle_seed,
        )
        subset_ratios = {"file": 1.0}
        print(
            f"[collect] using --prompts_file {args.prompts_file} "
            f"({len(test_records)} prompts, all to 'test' split)",
            flush=True,
        )
    else:
        subset_ratios = json.loads(args.subset_ratios)
        train_records, test_records = _load_and_split_nemotron(
            subset_ratios=subset_ratios,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.shuffle_seed,
        )
    all_records = test_records + train_records  # collect test first so eval is unblocked early
    if args.limit is not None:
        all_records = all_records[: args.limit]

    print(f"[collect] target: {len(all_records)} samples → {out_root}", flush=True)

    # Save the manifest (record metadata) so dataset.py can discover splits
    # without re-loading Nemotron.
    manifest = {
        "subset_ratios": subset_ratios,
        "n_train":  len(train_records),
        "n_test":   len(test_records),
        "shuffle_seed": args.shuffle_seed,
        "schema_version": S.SCHEMA_VERSION,
        "prefix_window": args.prefix_window,
        "decoding": (
            {"mode": "factor",    "factor":    decoding_factor}
            if decoding_factor is not None
            else {"mode": "threshold", "threshold": decoding_threshold}
        ),
        "dagger_rollout": {
            "conf_ckpt":      args.conf_ckpt,
            "gate_threshold": float(args.collect_gate_threshold),
            "applies_to":     "train split only" if args.conf_ckpt else None,
        },
        "records":  [
            {
                "split": r["split"], "subset": r["subset"], "id": r["id"],
                "filename": f"{r['split']}/sample_{r['id']:08x}.pt",
            }
            for r in all_records
        ],
    }
    with open(out_root / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("[collect] loading LLaDA …", flush=True)
    model, tokenizer = load_llada(fast_dllm_path=args.fast_dllm_path)

    # v4 gated-hybrid rollout — load the delta+conf model + the frozen
    # backbone heads it needs (final_norm / lm_head / token_embed) once.
    delta_model = sched_final_norm = sched_lm_head = sched_token_embed = None
    if args.conf_ckpt is not None:
        print(f"[collect] gated-hybrid rollout ON (train split only): "
              f"gate_threshold={args.collect_gate_threshold} "
              f"conf_ckpt={args.conf_ckpt}", flush=True)
        delta_model = _load_delta_for_collect(args.conf_ckpt, model)
        _tr = model.model.transformer
        sched_final_norm = _tr.ln_f
        sched_lm_head    = _tr.ff_out
        sched_token_embed = _tr.wte

    runtimes: list[float] = []
    skipped = 0
    for idx, rec in enumerate(all_records[args.start_index:], start=args.start_index):
        out_path = out_root / rec["split"] / f"sample_{rec['id']:08x}.pt"
        if out_path.exists():
            print(f"[collect] {idx:5d} {rec['split']:5s} {rec['subset']:5s} "
                  f"id={rec['id']:08x} (already exists, skip)", flush=True)
            continue
        try:
            t0 = time.time()
            # Gated-hybrid rollout applies to the TRAIN split only — the
            # test split stays pure teacher-forced so it remains a clean,
            # fixed held-out set comparable across runs (and to v3).
            rec_delta = delta_model if rec["split"] == "train" else None
            data = collect_one_sample(
                model, tokenizer, rec["prompt"],
                prefix_window=args.prefix_window,
                threshold=decoding_threshold, factor=decoding_factor,
                delta_model=rec_delta,
                final_norm=sched_final_norm,
                lm_head=sched_lm_head,
                token_embed=sched_token_embed,
                gate_threshold=args.collect_gate_threshold,
            )
            if data is None:
                skipped += 1
                print(f"[collect] {idx:5d} {rec['split']:5s} skipped (prompt too short)",
                      flush=True)
                continue
            data["record"] = {k: rec[k] for k in ("split", "subset", "id")}
            _atomic_save(data, out_path)
            dt = time.time() - t0
            runtimes.append(dt)
            print(
                f"[collect] {idx:5d} {rec['split']:5s} {rec['subset']:5s} "
                f"id={rec['id']:08x} done in {dt:5.1f}s "
                f"(mean {np.mean(runtimes):4.1f}s)",
                flush=True,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[collect] {idx:5d} FAILED: {e}", flush=True)

    print(f"[collect] done. processed={len(runtimes)} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()

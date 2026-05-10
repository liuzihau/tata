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
    """
    from datasets import load_dataset

    total_frac = sum(f for f in subset_ratios.values() if f > 0)
    if total_frac <= 0:
        raise ValueError("subset_ratios must contain at least one positive entry")

    train_records: list[dict] = []
    test_records: list[dict] = []

    for subset, frac in subset_ratios.items():
        if frac <= 0:
            continue
        print(f"[nemotron] loading subset='{subset}' …", flush=True)
        ds = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v2",
            split=subset,
        )

        # Deterministic order via hash of the prompt text.
        prompts = [prompt_extractor(r) for r in ds]
        hashes = [_stable_hash(p) for p in prompts]
        order = sorted(range(len(ds)), key=lambda i: hashes[i])

        q_test = int(round(n_test * frac / total_frac))
        q_train = int(round(n_train * frac / total_frac))

        if q_test + q_train > len(ds):
            raise ValueError(
                f"subset '{subset}' has {len(ds)} rows but we asked for "
                f"{q_test}+{q_train}. Lower n_train or subset_ratios[{subset}]."
            )

        for i, idx in enumerate(order[:q_test]):
            test_records.append({
                "prompt": prompts[idx], "subset": subset, "split": "test",
                "id": hashes[idx],
            })
        for i, idx in enumerate(order[q_test:q_test + q_train]):
            train_records.append({
                "prompt": prompts[idx], "subset": subset, "split": "train",
                "id": hashes[idx],
            })

    rng = random.Random(seed)
    rng.shuffle(train_records)
    rng.shuffle(test_records)
    print(
        f"[nemotron] split: {len(train_records)} train + {len(test_records)} test",
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
    threshold: float | None = None,
    factor: float | None = 1.0,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = S.LLADA_MASK_TOKEN_ID,
) -> dict | None:
    """Run prefix-cache generation on one prompt and return the cache dict
    described in `data/schema.py`. Returns None if the prompt is too short
    (< COLLECT_PREFIX_WINDOW tokens) — those samples are dropped.

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
    prompt_ids = _format_prompt_llada(tokenizer, prompt_text)
    prompt_len = int(prompt_ids.shape[1])
    if prompt_len < S.COLLECT_PREFIX_WINDOW:
        # Can't take the last-`COLLECT_PREFIX_WINDOW` prefix slice at block 0
        # if the prompt itself is shorter than that.
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
    # We always do a full forward (see iter ≥ 1 branch below), so the hook
    # always slices the block region [s:e] from the full-sequence output.
    hook_state = {"latest": None, "s": 0, "e": 0}

    def _last_block_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output  # [1, seq_len, d_model]
        slice_h = h[:, hook_state["s"]:hook_state["e"], :]
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

                hook_state["s"] = s
                hook_state["e"] = e

                # §3.1 — full-forward at every iter.
                #
                # Why not the partial-forward optimization?
                # The pre-§3.1 path (forward `x[:, s:]` with `past_key_values`
                # sliced to `[:s]`) silently rotated new K at positions
                # `[0, new_len)` instead of `[s, s+new_len)`. We tried
                # Fast-dLLM v1's intended `replace_position` API as a fix —
                # both paths were verified to diverge from a full forward by
                # max abs ~6–8 (1.8–2.4% relative) on T4
                # (`delta_model/sanity/test_partial_full_forward_equivalence.py`).
                # The most likely reason is that LLaDA's attention bias /
                # position handling assumes new tokens are at the tail of the
                # sequence, which is incompatible with mid-sequence block
                # decoding. Rather than monkey-patch the modeling code, we
                # use a full forward at every iter — guaranteed correct, ~30%
                # slower at collect time.
                #
                # `use_cache=True` is needed at iter 0 only, to extract the
                # last-32 prefix KV for the delta model's cross-attn input.
                if i == 0:
                    out = model(x, use_cache=True)
                    past_key_values = out.past_key_values

                    # Capture last-32 prefix KV from the last layer at
                    # positions [s-32, s). Cache stores K/V *unrotated*
                    # (`present` is captured before RoPE in attention), so
                    # this is the right content for variantc's cross-attn
                    # K/V passthrough (which expects backbone-RoPE'd K — see
                    # §1.5 in improvements.md).
                    # Capture the last COLLECT_PREFIX_WINDOW (64) tokens of
                    # prefix KV. Training reads only the last `PREFIX_WINDOW`
                    # (32); the extra is headroom for window-size ablations
                    # without re-collecting.
                    last_K = past_key_values[-1][0][0, :, s - S.COLLECT_PREFIX_WINDOW:s, :]
                    last_V = past_key_values[-1][1][0, :, s - S.COLLECT_PREFIX_WINDOW:s, :]
                    prefix_kv = torch.stack(
                        [last_K, last_V], dim=0,
                    ).to("cpu", dtype=S.DTYPE_KV, copy=True)
                else:
                    out = model(x)

                logits = out.logits

                # Token transfer for current block. Logits cover the full
                # sequence; we restrict transfer to the current block by
                # zeroing out `global_mask_index` outside [..., e).
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
                "prefix_kv":        prefix_kv,                          # [2, n_kv_heads, COLLECT_PREFIX_WINDOW, d_head]
                "h_per_pass":       torch.stack(h_per_pass, dim=0),     # [MAX_ITER, 32, d_model]
                "reveal_per_pass":  torch.stack(reveal_per_pass, dim=0),# [MAX_ITER, 32]
                "n_passes_actual":  int(n_passes_actual),
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
            "fast_dllm_mode": "prefix_cache",
            "decoding": (
                {"mode": "factor",    "factor":    factor}    if factor is not None
                else {"mode": "threshold", "threshold": threshold}
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
    args = p.parse_args()

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
        "decoding": (
            {"mode": "factor",    "factor":    decoding_factor}
            if decoding_factor is not None
            else {"mode": "threshold", "threshold": decoding_threshold}
        ),
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
            data = collect_one_sample(
                model, tokenizer, rec["prompt"],
                threshold=decoding_threshold, factor=decoding_factor,
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

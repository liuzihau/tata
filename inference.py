"""Interactive single-prompt comparison: hybrid (tata) vs vanilla Fast-dLLM.

Prompts the user for a question on stdin, runs the prompt twice — once
through `generate_with_delta` (tata's hybrid decoder, uses the delta
model + Fast-dLLM prefix-cache) and once through Fast-dLLM v1's
`generate_with_prefix_cache` (vanilla baseline; same factor / threshold
/ mask_id, no delta model). Prints both outputs side-by-side plus
wall-time, backbone-forward count, rollback count, and the speedup
ratio.

Run from the tata repo root:

    python inference.py

Override the checkpoint / decoding knobs via env vars:

    DELTA_CKPT=ckpts/tata_release/best_val_kl_step7500.pt \\
    PER_POS_THRESHOLD=0.75 \\
    FACTOR=1.0 \\
    python inference.py
"""
import os
import sys
import time

import torch

from delta_model.llada_runtime import load_llada
from delta_model.inference.hybrid_runner import generate_with_delta
from delta_model.models.variant_c import VariantC
import delta_model.data.schema as S


# --- knobs (env-overridable) ----------------------------------------------

DELTA_CKPT        = os.environ.get(
    "DELTA_CKPT", "ckpts/tata_release/best_val_kl_step7500.pt",
)
FAST_DLLM_PATH    = os.environ.get("FAST_DLLM_PATH", "external/Fast-dLLM/v1")
PER_POS_THRESHOLD = float(os.environ.get("PER_POS_THRESHOLD", "0.75"))
FACTOR            = float(os.environ.get("FACTOR", "1.0"))
GEN_LENGTH        = int(os.environ.get("GEN_LENGTH",  str(S.GEN_LENGTH)))
BLOCK_LENGTH      = int(os.environ.get("BLOCK_LENGTH", str(S.BLOCK_LENGTH)))


# --- load backbone + delta -------------------------------------------------

print(f"[init] loading backbone + delta ckpt={DELTA_CKPT}", flush=True)
backbone, tokenizer = load_llada(fast_dllm_path=FAST_DLLM_PATH)
ckpt = torch.load(DELTA_CKPT, map_location="cpu", weights_only=False)

# Some LLaDA builds rename these config fields — fall back the same way
# train.py / gsm8k_e2e.py do.
bb_cfg = {
    "rope_theta":  float(getattr(backbone.config, "rope_theta", 1e6)),
    "rms_eps":     float(getattr(backbone.config, "layer_norm_eps", 1e-5)),
    "max_seq_len": int(getattr(backbone.config, "max_sequence_length", 8192)),
}
delta = VariantC(
    d_model=ckpt["cfg"]["model"]["d_model"],
    n_heads=ckpt["cfg"]["model"]["n_heads"],
    n_layers=ckpt["cfg"]["model"]["n_layers"],
    d_ff_inner=ckpt["cfg"]["model"].get("d_ff_inner"),
    dropout=0.0,
    detach_conf_features=bool(
        ckpt["cfg"]["model"].get("detach_conf_features", False)
    ),
    **bb_cfg,
).cuda().to(torch.bfloat16)
delta.load_state_dict(ckpt["model"], strict=True)
delta.eval()

t = backbone.model.transformer


# --- prompt ----------------------------------------------------------------

raw = input("Enter your question: ").strip()
if not raw:
    sys.exit("[inference] empty prompt; nothing to generate")

prompt = "Q: " + raw
msg = [{"role": "user", "content": prompt}]
ids = torch.tensor(
    tokenizer(
        tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False),
    )["input_ids"],
    dtype=torch.long,
).unsqueeze(0).cuda()
print(f"[init] prompt tokens={ids.shape[1]}  gen_length={GEN_LENGTH}  "
      f"block_length={BLOCK_LENGTH}  factor={FACTOR}  "
      f"per_pos_threshold={PER_POS_THRESHOLD}",
      flush=True)


# --- run 1: tata hybrid ----------------------------------------------------

print("\n[run] hybrid (tata) decoding ...", flush=True)
torch.cuda.synchronize()
t0_h = time.time()
hyb_ids, stats = generate_with_delta(
    backbone, delta, t.ln_f, t.ff_out, t.wte, ids,
    per_pos_threshold=PER_POS_THRESHOLD,
    factor=FACTOR,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
)
torch.cuda.synchronize()
t_hybrid = time.time() - t0_h
hyb_text = tokenizer.decode(
    hyb_ids[0, ids.shape[1]:].tolist(), skip_special_tokens=True,
)


# --- run 2: vanilla Fast-dLLM (same factor + KV cache style) --------------

# generate_with_prefix_cache lives in Fast-dLLM v1; load it lazily so the
# import path matches gsm8k_e2e.py's behavior.
sys.path.insert(0, FAST_DLLM_PATH)
from generate import generate_with_prefix_cache  # noqa: E402

# Count vanilla backbone forwards via a temporary forward_pre_hook so the
# stats line below is directly comparable to the hybrid `backbone_forwards`.
_van_count = {"n": 0}
def _van_hook(module, inputs):
    _van_count["n"] += 1
_van_handle = backbone.register_forward_pre_hook(_van_hook)

print("[run] vanilla Fast-dLLM decoding ...", flush=True)
torch.cuda.synchronize()
t0_v = time.time()
try:
    van_out = generate_with_prefix_cache(
        backbone, ids,
        steps=128,
        gen_length=GEN_LENGTH,
        block_length=BLOCK_LENGTH,
        temperature=0.0,
        remasking="low_confidence",
        mask_id=S.LLADA_MASK_TOKEN_ID,
        threshold=None,
        factor=FACTOR,
    )
finally:
    _van_handle.remove()
torch.cuda.synchronize()
t_vanilla = time.time() - t0_v
van_ids = van_out[0] if isinstance(van_out, tuple) else van_out
van_text = tokenizer.decode(
    van_ids[0, ids.shape[1]:].tolist(), skip_special_tokens=True,
)


# --- side-by-side report ---------------------------------------------------

bar = "─" * 78
print(f"\n{bar}\n[hybrid output — tata]\n{bar}")
print(hyb_text or "<empty>")
print(f"\n{bar}\n[vanilla output — Fast-dLLM]\n{bar}")
print(van_text or "<empty>")

print(f"\n{bar}\n[stats]")
print(f"  hybrid:   {t_hybrid:6.2f}s  "
      f"backbone={stats['backbone_forwards']:>3d}  "
      f"delta={stats['delta_forwards']:>3d}  "
      f"rollbacks={stats['rollbacks']:>3d}")
print(f"  vanilla:  {t_vanilla:6.2f}s  "
      f"backbone={_van_count['n']:>3d}")
print(f"  speedup:  {t_vanilla / max(t_hybrid, 1e-6):.2f}x  "
      f"(vanilla / hybrid wall-time)")
print(f"  agreement: {'identical' if hyb_text == van_text else 'different'}")

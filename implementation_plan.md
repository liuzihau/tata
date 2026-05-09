# tata delta-model — M1 implementation plan

Companion to `scoping.md`. Scope: **M1 only** — LLaDA-8B,
**variant C** (2-stream sequence concat), **5000 samples** of cached
data, last-layer hidden as `h_ref`, last-32-token prefix-KV slice,
`max_iter = 6`. Variants A/B, Dream-7B, multi-layer fusion, full
prefix KV are deferred to M2 and only sketched here.

This doc is meant to be detailed enough that an AI agent can produce
working training + eval code from it without further design clarification.
Tensor shapes, dtypes, and signatures are explicit. Numerical defaults
and code organization are concrete. Where M2 bolts on, I leave a
named TODO.

---

## 0. Status & assumptions

- Base model: LLaDA-8B via Fast-dLLM v1 (`external/Fast-dLLM/v1`).
- Backbone dims: `n_layers=32`, `n_heads=32`, `n_kv_heads=32`
  (no GQA), `d_model=4096`, `d_head=128`.
- Generation config: `gen_length=256`, `block_length=32` ⇒ `num_blocks=8`.
  `max_iter_per_block = 6`. **No prompt-length cap** — since we only
  cache the last 32 tokens of prefix KV, long prompts cost the same
  as short ones. Cap is whatever the LLaDA backbone accepts (4096
  positions in practice).
- Hardware target: 1× A100 80GB (or H100 80GB) for both data
  collection and training. Dataset (~80 GB at 5000 samples) fits on
  local disk.
- Python 3.10, PyTorch 2.4+, transformers 4.44+, h5py, datasets,
  wandb. Reuse `probe_runner/requirements.txt` as the seed.
- One-time prereq: `huggingface-cli login` on the collect machine
  (Nemotron Post-Training v2 is gated).

---

## 1. Repo layout

All new code lives under `delta_model/` at the root of the standalone
**tata** repo. Existing references in `scoping.md` §"File / module
layout" stand; the concrete M1-only file list is:

```
tata/                                  ← repo root, cwd for all commands
  scoping.md
  implementation_plan.md                (this doc)
  delta_model/
    __init__.py
    data/
      collect_llada.py                  ★ collects cache shards
      dataset.py                        ★ training Dataset
      schema.py                         tensor shapes / dtype constants
    models/
      heads.py                          Δh head (zero-init), conf head
      variant_c.py                      ★ M1 model
      # variant_b.py, variant_a.py     M2
    losses.py                           ★ MSE + KL + BCE composite
    train.py                            ★ training loop
    inference/
      hybrid_runner.py                  ★ generate_with_prefix_cache + Δ
    eval/
      shared_mass.py                    overlap metric (reused)
      gsm8k_e2e.py                      ★ end-to-end accuracy
    configs/
      m1_llada_variant_c.yaml           ★ default M1 config
    sanity/
      test_zero_init.py                 first-step sanity check
      test_collect_roundtrip.py         shape / dtype assertions
```

★ = must exist for M1 to run.

---

## 2. Cache file format

One `.pt` file per sample at `cache_root/llada/sample_NNNN.pt`,
saved with `torch.save`. Shard directory = the model name. Each file
contains a single dict with these keys; dtypes and shapes are exact.

```python
# Top-level keys per sample file
{
  "prompt_token_ids":   LongTensor[prompt_len],          # int64, variable length
  "generated_token_ids": LongTensor[256],                # int64, gen_length=256
  "prompt_len":         int,                             # python scalar
  "blocks": [                                            # list of length 8
    {
      "prefix_kv_last32": HalfTensor[2, 32, 32, 128],    # [K|V, n_kv_heads, 32_tok, d_head]
      "h_per_pass":       HalfTensor[6, 32, 4096],       # [pass, block_len, d_model]
      "reveal_per_pass":  BoolTensor[6, 32],             # mask=False at start of pass i
    },
    ...                                                  # 8 entries total
  ],
  "meta": {
    "model": "llada",
    "n_kv_heads": 32, "d_head": 128, "d_model": 4096,
    "gen_length": 256, "block_length": 32, "max_iter": 6,
    "fast_dllm_mode": "prefix_cache",                    # see §3
    "schema_version": 1,
  },
}
```

Constants live in `data/schema.py`:

```python
SCHEMA_VERSION   = 1
GEN_LENGTH       = 256
BLOCK_LENGTH     = 32
NUM_BLOCKS       = GEN_LENGTH // BLOCK_LENGTH        # 8
MAX_ITER         = 6
PREFIX_WINDOW    = 32                                # last-32 slice
N_KV_HEADS_LLADA = 32
D_HEAD_LLADA     = 128
D_MODEL_LLADA    = 4096
DTYPE_KV         = torch.float16
DTYPE_HIDDEN     = torch.float16
```

**Why `.pt` per sample, not webdataset / sharded h5:** at 5000 samples
× 16 MB = 80 GB, single-sample random access is fast enough on local
NVMe (microseconds via `mmap`). Switching to webdataset shards is a
one-line change at the `Dataset` level if I/O later bottlenecks.

---

## 3. collect_llada.py

**Goal:** run LLaDA + Fast-dLLM v1 on Nemotron-3 prompts, capture the
per-sample cache described in §2.

**CLI:**

```
python -m delta_model.data.collect_llada \
    --n_samples 5000 \
    --output_root cache_v1/llada \
    --fast_dllm_path external/Fast-dLLM/v1 \
    --nemotron_subsets chat,math,code \
    --shuffle_seed 42
```

**Implementation outline:**

1. Load LLaDA via `probe_runner.llada_runner.load_llada(fast_dllm_path)`.
   Reuse the loader as-is.
2. Load Nemotron Post-Training v2 prompts. The dataset is gated, so
   the user must `huggingface-cli login` once on the collect machine.
   The dataset has **four splits**: `stem`, `chat`, `math`, `code`
   — no built-in train/test split, so we make our own with a fixed,
   stable test set:

   ```python
   from datasets import load_dataset, concatenate_datasets

   def load_nemotron_pool(subset_ratios: dict[str, float]):
       """Returns a single shuffled HF Dataset over the requested mix.

       subset_ratios is a normalized dict over {"stem","chat","math","code"};
       e.g. {"chat":0.4,"math":0.4,"code":0.2,"stem":0.0}.
       """
       parts = []
       for split, frac in subset_ratios.items():
           if frac <= 0: continue
           ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2",
                             split=split)
           # Hash-based stable test partition: same 800 problems
           # always go to test regardless of n_train.
           ds = ds.add_column("__hash__",
                              [stable_hash(p["id"]) for p in ds])
           parts.append((split, ds, frac))
       return parts

   def split_train_test(parts, n_train: int, n_test: int = 800,
                         seed: int = 42):
       """Per subset: top-`q_test` by hash → test, next `q_train` → train.
       This way enlarging n_train never touches test."""
       train, test = [], []
       total_frac = sum(f for _,_,f in parts)
       for split, ds, frac in parts:
           ds = ds.sort("__hash__")
           q_test  = int(round(n_test  * frac / total_frac))
           q_train = int(round(n_train * frac / total_frac))
           test_part  = ds.select(range(q_test))
           train_part = ds.select(range(q_test, q_test + q_train))
           train.append(train_part); test.append(test_part)
       train = concatenate_datasets(train).shuffle(seed=seed)
       test  = concatenate_datasets(test).shuffle(seed=seed)
       return train, test
   ```

   Stable hash recipe: `int(hashlib.sha1(prompt_id.encode()).hexdigest()[:8], 16)`.
   The key invariant: `stable_hash` is deterministic across runs and
   independent of `n_train`, so the test set never overlaps with
   training as the dataset grows. `n_train=5000`, `n_test=800` are
   the M1 defaults; `n_train` can be raised at M2 / M3 without
   re-running M1's eval baseline.

   **Default ratio for M1**: `{"chat":0.35, "math":0.35, "code":0.20,
   "stem":0.10}` (math-heavy because GSM8K is the headline eval).
   User-adjustable via `cfg.data.subset_ratios`.

   **Fallback if Nemotron is unreachable**: GSM8K-train mixed with
   ShareGPT chat (both public-no-login), keep the same hash-based
   test split logic.
3. For each prompt:
   a. Apply chat template → `prompt_ids` (truncate to `PROMPT_MAX`).
   b. Run a customized `generate_with_prefix_cache` that records
      probe data. The reference is `Fast-dLLM/v1/llada/generate.py:132`
      — fork it to a local `_collect_prefix_cache` function that
      additionally captures, per block:
      - the **last-32 prefix KV** at last layer (slice
        `past_key_values[-1][..., -32:, :]` after the pass-1
        full forward),
      - per pass `i ∈ {0..5}`: the last-layer hidden state at the
        block's masked positions (hook the final transformer block
        like `probe_runner/hooks.py` does), the reveal pattern.
      Stop the in-block loop after `MAX_ITER = 6` passes regardless
      of whether all positions are revealed (truncation is
      acceptable for training data).
   c. Pad / cap any pass with `< 6` actual passes by:
      - copying the last pass's hidden into trailing pass slots, AND
      - marking the pad slots in a `n_passes_actual: int` per-block
        field (add to schema as v2 if we want strict bookkeeping).
      For M1 we only train on `(i_ref, i_target)` pairs where both
      indices < `n_passes_actual` — so padding is unobserved.
4. Save with `torch.save(sample_dict, out_path)`. Atomic write
   pattern: write to `*.pt.tmp`, then `os.rename`.

**Hooks reuse:** `probe_runner/hooks.py:ProbeHooks` already captures
per-pass last-layer hidden via the `intra_block` flag. Cleanest
collect script imports it and configures `record_blocks_set = all`,
`intra_block = True`, drops the attention / v_norm capture.

**Throughput estimate:** LLaDA full-decode of 256 tokens ≈ 3-5 s on
A100 (Fast-dLLM dual-cache). 5000 samples ≈ 4-7 hours. Single GPU
is fine; no need to distribute collect.

**Sanity assertions** (`sanity/test_collect_roundtrip.py`):

```python
def test_one_sample():
    s = torch.load("cache_v1/llada/sample_0000.pt")
    assert s["meta"]["schema_version"] == SCHEMA_VERSION
    assert len(s["blocks"]) == NUM_BLOCKS
    for b in s["blocks"]:
        assert b["prefix_kv_last32"].shape == (2, 32, 32, 128)
        assert b["prefix_kv_last32"].dtype == torch.float16
        assert b["h_per_pass"].shape   == (6, 32, 4096)
        assert b["h_per_pass"].dtype   == torch.float16
        assert b["reveal_per_pass"].shape == (6, 32)
        assert b["reveal_per_pass"].dtype == torch.bool
        # Reveal pattern is monotone (revealed positions stay revealed).
        for i in range(5):
            assert torch.all(b["reveal_per_pass"][i+1] | ~b["reveal_per_pass"][i])
```

---

## 4. dataset.py

```python
class TataDeltaDataset(Dataset):
    """One example = (sample, block, i_ref, i_target).

    With 5000 samples × 8 blocks × C(6,2)=15 pairs = 600 000 examples.
    """
    def __init__(
        self,
        cache_root: Path,                  # e.g. cache_v1/llada
        tokenizer,                         # for token-id → embedding table
        token_embed: nn.Embedding,         # base model's token embed (frozen, on CPU is fine)
        mask_token_id: int = 126336,       # LLaDA mask
        index_filter: Optional[Callable] = None,  # for train/val split
    ):
        self.sample_paths = sorted(Path(cache_root).glob("sample_*.pt"))
        self.token_embed = token_embed
        self.mask_token_id = mask_token_id
        # Pre-build the (sample, block, i_ref, i_target) index.
        self.index = []
        for s_idx, p in enumerate(self.sample_paths):
            for b in range(NUM_BLOCKS):
                for i_ref in range(MAX_ITER - 1):
                    for i_tgt in range(i_ref + 1, MAX_ITER):
                        self.index.append((s_idx, b, i_ref, i_tgt))
        if index_filter is not None:
            self.index = [t for t in self.index if index_filter(t)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        s_idx, b, i_ref, i_tgt = self.index[idx]
        sample = torch.load(self.sample_paths[s_idx], map_location="cpu",
                            mmap=True)        # mmap'd, no full read
        block = sample["blocks"][b]

        h_ref       = block["h_per_pass"][i_ref].float()        # [32, 4096]
        h_target    = block["h_per_pass"][i_tgt].float()        # [32, 4096]
        prefix_kv   = block["prefix_kv_last32"]                 # [2, 32, 32, 128]
        reveal_tgt  = block["reveal_per_pass"][i_tgt]           # [32] bool
        mask_tgt    = ~reveal_tgt                               # [32] bool

        # prev_emb[j] = token_embed[token_id_at_j] if revealed, else mask_embed
        block_token_ids = sample["generated_token_ids"][
            b * BLOCK_LENGTH : (b + 1) * BLOCK_LENGTH
        ]                                                       # [32]
        substituted_ids = torch.where(reveal_tgt, block_token_ids,
                                       torch.full_like(block_token_ids,
                                                        self.mask_token_id))
        prev_emb = self.token_embed(substituted_ids).float()    # [32, 4096]

        return {
            "h_ref":       h_ref,        # [32, 4096] fp32
            "h_target":    h_target,     # [32, 4096] fp32 — supervision target
            "prev_emb":    prev_emb,     # [32, 4096] fp32
            "prefix_kv":   prefix_kv,    # [2, 32, 32, 128] fp16
            "mask_tgt":    mask_tgt,     # [32] bool — KL evaluated only here
            "i_ref":       i_ref,        # for binning
            "i_target":    i_tgt,        # for binning
            "reveal_frac": reveal_tgt.float().mean().item(),  # for binning
        }
```

**Train / val split** at the *sample* level (not pair level), so that
no sample appears in both: e.g., samples [0..4499] → train,
[4500..4999] → val. `index_filter` enforces this.

**Token embedding source:** lift from the loaded LLaDA model
(`model.model.embed_tokens.weight`) at training start, copy weights
into a frozen `nn.Embedding`, ship to CPU. This keeps the heavy
backbone out of training-time memory. Re-tying isn't necessary —
the mask-token embedding is also just a row of this same table.

**Worker count:** start with `num_workers=4`. The `__getitem__` is
mostly an mmap + small tensor slice; CPU-bound but cheap.

---

## 5. models/variant_c.py

2-stream sequence concat (M1 default — simplest of the three architectures
in `scoping.md`). Total params ≈ 100-150M depending on `delta_layers`
and `n_heads`.

```python
class VariantC(nn.Module):
    """
    Inputs (per batch element):
        h_ref     : [B, 32, d_model]                     — last-layer hidden at i_ref
        prev_emb  : [B, 32, d_model]                     — input embedding at i_target
        prefix_kv : [B, 2, n_kv_heads, 32, d_head]       — last-32 prefix KV
    Output:
        delta_h   : [B, 32, d_model]                     — predicted Δh
        c_pred    : [B] in [0,1]                          — confidence scalar
    """
    def __init__(self, d_model=4096, n_heads=32, n_layers=2,
                 d_ff=4 * 4096, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        # Two learned segment embeddings to distinguish h_ref from prev_emb.
        self.seg_emb = nn.Embedding(2, d_model)
        # Sinusoidal positional embedding over 2*32=64 positions.
        self.pos_emb = nn.Embedding(2 * BLOCK_LENGTH, d_model)
        self.layers = nn.ModuleList([
            VariantCBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        # Δh head — zero-init so step 0 = "h_ref reuse".
        self.delta_head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        # Conf head — pooled feature → scalar in [0,1].
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, h_ref, prev_emb, prefix_kv):
        B = h_ref.shape[0]
        # [B, 64, d]
        x = torch.cat([h_ref, prev_emb], dim=1)
        seg = torch.cat([
            self.seg_emb(torch.zeros(BLOCK_LENGTH, dtype=torch.long, device=x.device)),
            self.seg_emb(torch.ones (BLOCK_LENGTH, dtype=torch.long, device=x.device)),
        ], dim=0).unsqueeze(0).expand(B, -1, -1)
        pos = self.pos_emb(torch.arange(2 * BLOCK_LENGTH, device=x.device))\
                          .unsqueeze(0).expand(B, -1, -1)
        x = x + seg + pos

        for layer in self.layers:
            x = layer(x, prefix_kv)

        x = self.final_norm(x)
        # Use the second half (positions of prev_emb / target block) as features.
        feats = x[:, BLOCK_LENGTH:, :]                          # [B, 32, d]
        delta_h = self.delta_head(feats)                         # [B, 32, d]
        pooled  = feats.mean(dim=1)                              # [B, d]
        c_pred  = torch.sigmoid(self.conf_head(pooled).squeeze(-1))  # [B]
        return delta_h, c_pred


class VariantCBlock(nn.Module):
    """One self-attn + cross-attn-into-prefix + FFN layer."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                 dropout=dropout, batch_first=True,
                                                 kdim=d_model, vdim=d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, prefix_kv):
        # prefix_kv: [B, 2, n_kv_heads, 32, d_head] → reshape to [B, 32, d_model]
        B = prefix_kv.shape[0]
        K = prefix_kv[:, 0].reshape(B, BLOCK_LENGTH, -1)        # [B, 32, d_model]
        V = prefix_kv[:, 1].reshape(B, BLOCK_LENGTH, -1)
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               need_weights=False)[0]
        x = x + self.cross_attn(self.norm2(x), K, V,
                                 need_weights=False)[0]
        x = x + self.ffn(self.norm3(x))
        return x
```

**Variant A / B hooks**: heads.py and the loss are shared. Only
`forward` differs. M2 adds `variant_a.py` (cross-attn anchor with
`h_ref` as a frozen K/V bank, gated tanh-α init=0) and
`variant_b.py` (AdaLN-Zero modulation). Both should accept the same
`(h_ref, prev_emb, prefix_kv)` signature and return
`(delta_h, c_pred)`.

---

## 6. losses.py

```python
def composite_loss(
    delta_h_pred: Tensor,           # [B, 32, d_model]
    c_pred:       Tensor,           # [B]
    h_ref:        Tensor,           # [B, 32, d_model]   from the dataset
    h_target:     Tensor,           # [B, 32, d_model]
    mask_tgt:     Tensor,           # [B, 32]            bool
    final_norm:   nn.Module,        # backbone's final RMSNorm/LayerNorm
    lm_head:      nn.Linear,        # backbone's LM head
    *,
    lambda_mse:  float = 1.0,
    lambda_kl:   float = 0.1,
    lambda_conf: float = 0.1,
) -> dict:
    delta_h_target = h_target - h_ref                              # [B, 32, d]
    mse = F.mse_loss(delta_h_pred, delta_h_target)                  # scalar

    # KL at mask positions only
    h_pred = h_ref + delta_h_pred
    with torch.no_grad():
        logits_actual = lm_head(final_norm(h_target))               # [B, 32, V]
        log_p_actual  = F.log_softmax(logits_actual, dim=-1)
    logits_pred = lm_head(final_norm(h_pred))                       # [B, 32, V]
    log_p_pred  = F.log_softmax(logits_pred, dim=-1)
    kl_per_pos  = (log_p_actual.exp() * (log_p_actual - log_p_pred)).sum(-1)  # [B, 32]
    kl = (kl_per_pos * mask_tgt).sum() / mask_tgt.sum().clamp_min(1)

    # BCE on conf head; soft-label = mean shared mass at mask positions
    with torch.no_grad():
        p_a = log_p_actual.exp()
        p_p = log_p_pred.exp()
        shared = torch.minimum(p_a, p_p).sum(-1)                    # [B, 32]
        c_label = (shared * mask_tgt).sum(-1) / mask_tgt.sum(-1).clamp_min(1)
    bce = F.binary_cross_entropy(c_pred, c_label.detach())

    total = lambda_mse * mse + lambda_kl * kl + lambda_conf * bce
    return {"loss": total, "mse": mse.detach(), "kl": kl.detach(),
            "bce": bce.detach(), "c_label_mean": c_label.mean().detach()}
```

**Frozen `final_norm` and `lm_head`** are loaded once at training
start from the LLaDA model and moved to the training device. Their
parameters are excluded from the optimizer (`requires_grad=False`).
This is a one-time GPU cost of ≈ 0.5 GB (just the projection layers).

---

## 7. train.py

**CLI:**

```
python -m delta_model.train \
    --config delta_model/configs/m1_llada_variant_c.yaml \
    --resume_from <ckpt_path>     # optional
```

**Default config** (`configs/m1_llada_variant_c.yaml`):

```yaml
run_name:          m1_llada_variant_c_seed42
seed:              42
cache_root:        cache_v1/llada
fast_dllm_path:    external/Fast-dLLM/v1
backbone:
  model:           llada
  load_for:        [token_embed, final_norm, lm_head]
data:
  cache_root:      cache_v1/llada
  n_train:         5000
  n_test:          800            # stable across n_train changes (hash-partitioned)
  subset_ratios:   {chat: 0.35, math: 0.35, code: 0.20, stem: 0.10}
  val_frac:        0.10           # within train, last 10% goes to validation
  shuffle_seed:    42
  num_workers:     4
  batch_size:      256
  pin_memory:      true
model:
  variant:         C
  d_model:         4096
  n_heads:         32
  n_layers:        2
  d_ff:            16384
  dropout:         0.0
optim:
  optimizer:       adamw
  lr:              1.0e-4
  weight_decay:    0.01
  betas:           [0.9, 0.95]
  warmup_steps:    500
  max_steps:       20000
  scheduler:       cosine
  grad_clip:       1.0
  precision:       bf16
loss:
  lambda_mse:      1.0
  lambda_kl:       0.1
  lambda_conf:     0.1
log:
  framework:       wandb
  project:         tata-delta-model
  group:           M1-llada
  log_every:       50
  val_every:       500
  gsm8k_every:     5000
  gsm8k_subset:    50
checkpoint:
  every:           2000
  keep_last:       3
  out_dir:         ckpts/m1_llada_variant_c
```

**Loop pseudocode:**

```python
def main(cfg):
    set_seed(cfg.seed)
    backbone_loader = load_llada_components(cfg)   # token_embed, final_norm, lm_head
    train_ds = TataDeltaDataset(..., index_filter=range_filter(*cfg.data.train_samples))
    val_ds   = TataDeltaDataset(..., index_filter=range_filter(*cfg.data.val_samples))
    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                          shuffle=True,  num_workers=cfg.data.num_workers,
                          pin_memory=cfg.data.pin_memory, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.data.batch_size,
                          shuffle=False, num_workers=cfg.data.num_workers)

    model = VariantC(...).to("cuda", dtype=torch.bfloat16)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg.optim.lr, betas=cfg.optim.betas,
                               weight_decay=cfg.optim.weight_decay)
    sched = WarmupCosine(opt, warmup=cfg.optim.warmup_steps,
                         total=cfg.optim.max_steps)

    wandb_init(cfg)                               # see §9
    step = 0
    for epoch in itertools.count():
        for batch in train_dl:
            batch = move_to_cuda(batch, dtype=torch.bfloat16)
            delta_h, c_pred = model(batch["h_ref"], batch["prev_emb"],
                                     batch["prefix_kv"])
            loss_dict = composite_loss(
                delta_h, c_pred,
                batch["h_ref"], batch["h_target"], batch["mask_tgt"],
                final_norm, lm_head,
                lambda_mse=cfg.loss.lambda_mse, ...)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step(); sched.step(); opt.zero_grad()

            if step % cfg.log.log_every == 0:
                wandb.log({"train/" + k: float(v) for k, v in loss_dict.items()},
                          step=step)
                wandb.log({"lr": sched.get_last_lr()[0]}, step=step)
            if step % cfg.log.val_every == 0:
                val_metrics = run_val(model, val_dl, final_norm, lm_head)
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            if step % cfg.log.gsm8k_every == 0 and step > 0:
                gsm8k = run_gsm8k_eval(model, cfg, n_problems=cfg.log.gsm8k_subset)
                wandb.log({f"gsm8k/{k}": v for k, v in gsm8k.items()}, step=step)
            if step % cfg.checkpoint.every == 0 and step > 0:
                save_ckpt(model, opt, sched, step, cfg)

            step += 1
            if step >= cfg.optim.max_steps:
                return
```

**Validation metrics** (`run_val`): compute the full loss components,
plus per-bin breakdowns:
- by `gap = i_target - i_ref` (1..5)
- by `reveal_frac` bucket (5 quantiles 0/0.2/0.4/0.6/0.8/1.0)

Log mean MSE / KL / shared_mass per bin. This is the mandatory
"never average alone" rule from `scoping.md`.

**Step 0 sanity check:** with delta head zero-init, `delta_h = 0`,
so `h_pred = h_ref` and the model collapses to the "h_ref reuse"
baseline. KL at step 0 should equal `KL(p(h_target) || p(h_ref))`,
which is a useful baseline for "how much improvement does training
buy us." Log this at step 0 explicitly.

**Wallclock estimate:** 20k steps × batch 256 × ~150ms/step on A100
≈ 50 minutes pure training, plus validation overhead. Easily a
3-hour run including GSM8K subset evals.

---

## 8. inference/hybrid_runner.py

The thing the delta model is for. Replaces the in-block forward of
`generate_with_prefix_cache` (`Fast-dLLM/v1/llada/generate.py:132`)
with a delta-model call, with a confidence-based rollback to the
backbone.

**Signature:**

```python
@torch.no_grad()
def generate_with_delta(
    model,                       # full LLaDA backbone (unwrapped)
    delta_model: nn.Module,
    prompt: LongTensor,          # [1, S]
    *,
    steps: int = 128,            # not used directly in dual-cache style; informational
    gen_length: int = 256,
    block_length: int = 32,
    max_iter_per_block: int = 6,
    conf_threshold: float = 0.85,    # below → rollback
    mask_id: int = 126336,
    temperature: float = 0.0,
    record_rollback_stats: bool = True,
) -> tuple[LongTensor, dict]:
    """Returns (final_token_ids, stats). Mirrors generate_with_prefix_cache
    but swaps in the delta model for passes 2..N within each block."""
```

**Loop sketch:**

```
for each block b in 0..7:
    # Pass 1: full backbone forward — same as Fast-dLLM prefix_cache mode.
    out_full = model(x, use_cache=True)
    h_ref = capture_last_layer_block_hidden(out_full, block_b)          # [32, d]
    prefix_kv_last32 = slice_last_32(out_full.past_key_values[-1])       # [2, ...]
    do_one_token_transfer(out_full.logits, ...)                          # initial reveal

    pass_idx = 1                                # we're now at pass 2 in 1-indexed
    while pass_idx < max_iter_per_block and not block_b_fully_revealed():
        # Build prev_emb from current reveal pattern.
        prev_emb = build_prev_emb(x[:, b_start:b_end])
        # Delta forward.
        delta_h, c_pred = delta_model(
            h_ref.unsqueeze(0), prev_emb.unsqueeze(0),
            prefix_kv_last32.unsqueeze(0),
        )
        if c_pred.item() < conf_threshold:
            # Rollback: full backbone forward, refresh h_ref.
            out_roll = model(x, use_cache=True)
            h_ref = capture_last_layer_block_hidden(out_roll, block_b)
            prefix_kv_last32 = slice_last_32(out_roll.past_key_values[-1])
            logits = lm_head(final_norm(h_ref)).unsqueeze(0)
            stats["rollbacks"] += 1
        else:
            logits = lm_head(final_norm(h_ref + delta_h.squeeze(0))).unsqueeze(0)

        do_one_token_transfer(logits, ...)
        pass_idx += 1
```

**Key delta-vs-Fast-dLLM differences:**
1. `h_ref` is *latched* at the most recent backbone forward — pass 1
   of the block, then refreshed only on rollback.
2. Block-region KV is **never updated** by the delta model. This is
   only safe under prefix-cache mode (no block-region cache); under
   dual-cache mode it would corrupt the next pass's attention.
3. The transfer/reveal logic from Fast-dLLM is reused unchanged
   (`get_transfer_index`).

**Config knob to tune** at M1: `conf_threshold`. Sweep over
{0.80, 0.85, 0.90, 0.95} and log the (acc, speedup) pair per
threshold. Higher threshold → more rollbacks → closer to vanilla
Fast-dLLM accuracy at lower speedup.

---

## 9. eval/gsm8k_e2e.py

Loads GSM8K test set, runs the hybrid generator, compares against
gold answers using the same metric as `probe_runner` /
Fast-dLLM eval scripts (regex-extract final number, exact match).

```python
def run_gsm8k_eval(
    backbone_model, delta_model, tokenizer,
    *, n_problems: int = 1319, conf_threshold: float = 0.85,
    seed: int = 42,
) -> dict:
    ds = load_dataset("gsm8k", "main", split="test")
    if n_problems < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_problems))
    rollback_counts = []
    correct = 0
    for problem in ds:
        prompt_ids = format_prompt_llada(tokenizer, problem["question"])
        out_ids, stats = generate_with_delta(
            backbone_model, delta_model, prompt_ids,
            conf_threshold=conf_threshold,
        )
        rollback_counts.append(stats["rollbacks"])
        gen_text = tokenizer.decode(out_ids[0, prompt_ids.shape[1]:],
                                     skip_special_tokens=True)
        correct += int(extract_final_number(gen_text) ==
                       extract_final_number(problem["answer"]))
    return {
        "accuracy": correct / len(ds),
        "mean_rollbacks_per_sample": float(np.mean(rollback_counts)),
        "n_problems":               len(ds),
    }
```

Always run the same `generate_with_prefix_cache` baseline
(no delta model) on the same `ds.shuffle(seed)` slice so the two
runs are paired. Log accuracy delta and speedup ratio.

---

## 10. Logging — wandb (chosen)

Picked **wandb** over tensorboard for these reasons specific to this
project:
- We will be sweeping {variant, conf_threshold, seed}. wandb sweeps
  + parallel-coords plots make the comparison trivial.
- Run histories are kept across restarts; we'll be doing several
  collect → train cycles and want the timeline.
- System metrics (GPU mem, throughput) auto-logged for free, useful
  for catching memory regressions when we later try variant A's
  cross-attn anchor (more params).

**Setup:**

```bash
pip install wandb
wandb login                      # one-time, on the training machine
```

Init at start of `train.py`:

```python
import wandb
def wandb_init(cfg):
    wandb.init(
        project=cfg.log.project,
        group=cfg.log.group,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.model.variant, cfg.backbone.model, "M1"],
    )
```

**Logged scalars** (per `log_every` / `val_every` / `gsm8k_every`):

| key | when | meaning |
|---|---|---|
| `train/loss`, `train/mse`, `train/kl`, `train/bce` | every step | composite + components |
| `train/c_label_mean` | every step | dataset-level conf-target distribution |
| `lr` | every step | scheduler value |
| `val/loss`, `val/mse`, `val/kl`, `val/bce` | val_every | held-out 500 samples |
| `val/mse_by_gap_{1..5}` | val_every | binned by `(i_target - i_ref)` |
| `val/mse_by_reveal_frac_{0..4}` | val_every | binned by reveal-fraction quantile |
| `val/shared_mass` | val_every | mean overlap on mask positions |
| `gsm8k/accuracy_delta` | gsm8k_every | hybrid acc − vanilla prefix-cache acc |
| `gsm8k/speedup_ratio` | gsm8k_every | wall_time(vanilla) / wall_time(hybrid) |
| `gsm8k/mean_rollbacks` | gsm8k_every | per-problem rollback count |

**Artifacts:**
- Final model checkpoint (`ckpts/m1_llada_variant_c/final.pt`) saved
  as a `wandb.Artifact` of type "model" tagged with the run id.
- Validation pair-bin breakdowns saved as a parquet file artifact at
  end of training.

---

## 11. M0 — offline characterization

Before any training, run `M0_characterize.py` (one short script,
not in the file list above — write it ad-hoc):
- Load `cache_v1/llada/sample_0000.pt..0099.pt` (100 samples).
- For every `(i_ref, i_target)` pair with `i_ref < i_target ≤ 5`:
  - Compute `‖h_target − h_ref‖_2 / ‖h_ref‖_2` per position.
  - Bin by `gap` and by reveal-fraction.
- Plot:
  - heatmap: gap × reveal_frac → mean rel-Δh.
  - histogram: distribution of Δh per gap.
- Decision gate: if any bin shows mean rel-Δh > 0.5, flag and
  consider a per-bin loss reweighting before training. Otherwise
  proceed to M1 train.

This is a 1-hour analysis on 100 samples — do **after** collect,
**before** train.

---

## 12. Test plan / sanity checks

| When | Test | Pass criterion |
|---|---|---|
| post-collect | `sanity/test_collect_roundtrip.py` | shapes, dtypes, monotone reveal |
| pre-train | `sanity/test_zero_init.py` | with delta_head zero-init, loss = "h_ref reuse" baseline |
| step 100 | wandb `train/loss` decreasing? | yes |
| step 1000 | val `mse` < step-0 baseline? | yes |
| step 5000 | first GSM8K subset eval | accuracy ≥ 0.5 × vanilla baseline |
| step 20000 | full GSM8K eval | accuracy ≥ vanilla − 0.05 (target: within 5pp) |

If step 5000 accuracy is < 0.3 × vanilla, halt and revisit the loss
weights / model size before sinking another 15k steps.

---

## 13. M2 hooks (sketch only)

These are deferred but should be obvious extensions:
- `variant_a.py`, `variant_b.py` — same forward signature; train
  routes via `cfg.model.variant`.
- Multi-layer fusion — extend `data/schema.py` to add
  `h_per_pass_fusion: HalfTensor[6, 3, 32, 4096]` and gate via
  `cfg.data.use_fusion`. Storage 3×.
- Full prefix KV — drop the last-32 slice in collect, store
  `prefix_kv_per_block` as length-variable (use h5 ragged groups
  rather than .pt). Storage 5×.
- Dual-cache compatibility — extend the delta model to also output
  per-position last-layer K/V increments; collect.py needs to also
  capture the actual block-region KV as supervision target.
- Dream-7B — copy `collect_llada.py` → `collect_dream.py`, swap the
  loader; train.py is base-model-agnostic via `cfg.backbone.model`.

---

## Build order (read this if you're the agent generating code)

1. `data/schema.py`  (constants — needed everywhere)
2. `data/collect_llada.py`  (smoke-test on n_samples=2)
3. `sanity/test_collect_roundtrip.py`  (run on the 2 samples)
4. `data/dataset.py`  (verify it returns shapes per §4)
5. `models/variant_c.py` + `models/heads.py`
6. `losses.py`  (unit-test on a random batch — KL ≥ 0, BCE ≥ 0)
7. `sanity/test_zero_init.py`  (verify step-0 = "h_ref reuse" baseline)
8. `train.py` + `configs/m1_llada_variant_c.yaml`
   — overfit on 16 samples first; train loss should hit ~0
9. `inference/hybrid_runner.py`
10. `eval/gsm8k_e2e.py`
11. End-to-end run on the full 5000-sample cache.

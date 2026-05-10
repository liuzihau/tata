"""Diagnostic — print the LLaDA modeling code paths we need to plan the §3.1 fix.

After T4 confirmed that neither `LLaDAModelLM.forward()` nor the inner
`LLaDAModel.forward()` accept a `position_ids` kwarg in this Fast-dLLM v1
build, we need to look at the actual modeling code to understand:

  - what kwargs the forwards do accept
  - how RoPE is wired (is `position_ids` plumbed through internally
    even if not exposed at the top? does the attention layer compute
    position offsets from `past_key_values` length?)
  - which exact module to monkey-patch if we go that route

This script loads the model and prints:

  1. `LLaDAModelLM.forward` signature
  2. `LLaDAModel.forward` signature
  3. `RotaryEmbedding.forward` full source
  4. attention block (`LLaDABlock`/`LLaDASequentialBlock`/`LLaDALlamaBlock`)
     `forward` full source — where RoPE is actually called
  5. on-disk path of the modeling file (for grep-based follow-up)

Output is meant to be pasted back into the chat so we can write a precise
patch.

Run (GPU + Fast-dLLM required, but no training data needed):
    python -m delta_model.sanity.inspect_llada_modeling \\
        --fast_dllm_path external/Fast-dLLM/v1 > inspect.txt
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from ..llada_runtime import load_llada


def _find_first_module(model, name_keyword: str):
    for m in model.modules():
        if name_keyword in type(m).__name__:
            return m
    return None


def _print_sig(fn, label: str) -> None:
    print(f"\n=== {label} signature ===")
    try:
        print(f"{label}{inspect.signature(fn)}")
    except (ValueError, TypeError) as e:
        print(f"  could not introspect: {e}")


def _print_source(fn, label: str, max_lines: int = 120) -> None:
    print(f"\n=== {label} source ===")
    try:
        src = inspect.getsource(fn)
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if i >= max_lines:
                print(f"  ... (truncated at line {max_lines}/{len(lines)})")
                break
            print(line)
    except (OSError, TypeError) as e:
        print(f"  could not get source: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast_dllm_path", type=str, default=None)
    args = ap.parse_args()

    print("[inspect] loading LLaDA …", flush=True)
    model, _ = load_llada(fast_dllm_path=args.fast_dllm_path)

    # ----- 1, 2. Forward signatures -----
    _print_sig(model.forward,        "LLaDAModelLM.forward")
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "forward"):
        _print_sig(inner.forward,    "LLaDAModel.forward (model.model.forward)")

    # ----- 3. RotaryEmbedding -----
    rope = _find_first_module(model, "Rotary")
    if rope is None:
        print("\n!! No RotaryEmbedding-like module found "
              "(searched for class names containing 'Rotary').")
    else:
        print(f"\n[inspect] RoPE class: {type(rope).__name__}")
        _print_source(type(rope).forward,
                      f"{type(rope).__name__}.forward", max_lines=120)
        # Also print other methods that might be relevant.
        for attr in ("get_rotary_embedding", "apply_rotary_pos_emb",
                     "rotate_half"):
            if hasattr(type(rope), attr):
                _print_source(getattr(type(rope), attr),
                              f"{type(rope).__name__}.{attr}", max_lines=40)

    # ----- 4. Attention block -----
    block = None
    for cls_name in ("LLaDASequentialBlock", "LLaDALlamaBlock", "LLaDABlock"):
        block = _find_first_module(model, cls_name)
        if block is not None:
            break
    if block is None:
        print("\n!! No LLaDABlock-like module found.")
    else:
        print(f"\n[inspect] block class: {type(block).__name__}")
        _print_source(type(block).forward,
                      f"{type(block).__name__}.forward", max_lines=200)

        # 4b. Attention call site. `block.attention` here is a bound method
        # on LLaDALlamaBlock (not a submodule), so we print the method itself.
        attn = getattr(block, "attention", None)
        if attn is not None:
            print(f"\n[inspect] block.attention type: {type(attn).__name__}")
            try:
                if callable(attn) and hasattr(attn, "__func__"):
                    # Bound method: source the underlying function.
                    _print_source(attn.__func__,
                                  f"{type(block).__name__}.attention",
                                  max_lines=200)
                elif callable(attn):
                    # Plain submodule: source its forward.
                    _print_source(type(attn).forward,
                                  f"{type(attn).__name__}.forward",
                                  max_lines=200)
            except Exception as e:
                print(f"  could not source block.attention: {e}")

    # ----- 5. Modeling file path + targeted greps -----
    print("\n=== modeling file path ===")
    modeling_path: Path | None = None
    try:
        modeling_path = Path(inspect.getfile(type(model)))
        print(modeling_path)
        print(f"size: {modeling_path.stat().st_size} bytes")
    except (TypeError, OSError) as e:
        print(f"  could not locate: {e}")

    if modeling_path is not None and modeling_path.exists():
        print("\n=== grep results: where the RoPE/cache plumbing happens ===")
        text = modeling_path.read_text(encoding="utf-8", errors="replace")
        keywords = [
            "block_end_index",      # the RoPE kwarg we discovered
            "replace_position",     # the model-level kwarg discovered in forward signatures
            "self.rotary_emb",      # callers of RoPE
            "rotary_emb(",
            "past_key_values",
            "past_length",
            "layer_past",
        ]
        for kw in keywords:
            print(f"\n--- {kw!r} ---")
            for i, line in enumerate(text.splitlines(), start=1):
                if kw in line:
                    print(f"  {i:5d}: {line.rstrip()}")


if __name__ == "__main__":
    main()

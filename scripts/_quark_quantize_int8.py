"""Quark W8A8 INT8 quantization of Gemma 4 26B A4B-it (proven recipe).

Recipe adapted from nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8 (HF),
which achieved −0.08pp on GSM8K vs BF16 (essentially lossless) using
AMD Quark 0.11/0.12 with the same Gemma4ForConditionalGeneration
architecture. Key facts from that recipe:

- Scheme: W8A8 INT8 — per-channel weight (ch_axis=0, symmetric,
  static) + per-token activation (ch_axis=1, symmetric, dynamic).
- NO calibration data needed: dynamic activation scales are computed
  at inference time (dataloader=None).
- Excluded from quantization (stay BF16): lm_head, *embed_tokens*,
  *vision_tower*, *embed_vision*.
- Export: pack_method='order', weight_format='real_quantized',
  custom_mode='quark' → real INT8 weights + BF16 scales, vLLM-
  compatible quantization_config in config.json.

Runs inside the container. Source model path comes from
/tmp/bf16_path.txt (written by _find_bf16.py).
"""
from __future__ import annotations

import json
import os
import sys
import time

MODEL_OUT = os.environ.get("QUARK_OUT", "/models/gemma-4-26B-A4B-it-int8")


def main() -> None:
    with open("/tmp/bf16_path.txt") as f:
        model_in = f.read().strip()
    if not model_in or not os.path.isdir(model_in):
        print(f"FATAL: BF16 source dir not found: {model_in!r}", file=sys.stderr)
        return 1

    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from quark.torch import ModelQuantizer
    from quark.torch.quantization.config.config import (
        QConfig,
        QLayerConfig,
        QTensorConfig,
        Dtype,
    )
    from quark.torch.quantization.config.type import (
        RoundType,
        ScaleType,
        QSchemeType,
    )
    from quark.torch.quantization.observer import PerChannelMinMaxObserver

    print(f"Loading BF16 model from {model_in} (device_map=auto on MI300X)")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_in, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_in,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    weight_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        is_dynamic=False,
        qscheme=QSchemeType.per_channel,
        ch_axis=0,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    input_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        is_dynamic=True,
        qscheme=QSchemeType.per_channel,
        ch_axis=1,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )

    # Quark 0.12 renamed 0.11's Config/QuantizationConfig to QConfig/QLayerConfig.
    # MoE-aware exclusions: keep the router + experts + shared experts BF16.
    # The 31B-dense recipe (no MoE) quantized everything; on our 26B MoE the
    # router/expert scale handling corrupts output (verified empirically).
    q_cfg = QConfig(
        global_quant_config=QLayerConfig(
            input_tensors=input_spec,
            weight=weight_spec,
        ),
        exclude=[
            "lm_head",
            "*embed_tokens*",
            "*vision_tower*",
            "*embed_vision*",
            "*router*",
            "*experts*",
            "*shared_experts*",
            "*moe*",
        ],
    )

    print("Quantizing (W8A8 INT8, no calibration data needed)...")
    quantizer = ModelQuantizer(q_cfg, multi_device=True)
    model = quantizer.quantize_model(model, dataloader=None)
    quantizer.freeze(model)

    os.makedirs(MODEL_OUT, exist_ok=True)
    print(f"Exporting to {MODEL_OUT} (pack_method=order, real_quantized, quark mode)...")
    # Quark 0.12 moved export_model -> quark.torch.export_safetensors
    from quark.torch import export_safetensors
    export_safetensors(
        model,
        MODEL_OUT,
        pack_method="order",
        weight_format="real_quantized",
        custom_mode="quark",
    )
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Done in {time.time() - t0:.1f}s → {MODEL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

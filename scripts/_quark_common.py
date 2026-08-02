"""Shared helpers for the Quark W8A8 INT8 recipes.

Extracted from the dense and MoE quantizers. Only what both recipes share
belongs here: model load, QTensorConfig construction, quantize+freeze, export.
Architecture-specific logic (exclude lists, expert rewrite, key rename)
stays in the recipe scripts.

Recipes using this:
- _quark_quantize_dense.py  (Gemma 4 12B dense, excludes tuned for 31B-dense
                              baseline — nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8)
- _quark_quantize_moe.py    (Gemma 4 26B A4B MoE — expert rewrite +
                              post-export key rename per
                              nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Sequence


# ─── I/O ─────────────────────────────────────────────────────────────────────


def read_bf16_path(path: str = "/tmp/bf16_path.txt") -> str:
    """Read the BF16 snapshot path written by _find_bf16.py.

    Exits with a clear message if the file is missing or empty.
    """
    if not os.path.isfile(path):
        print(f"FATAL: {path} missing — run _find_bf16.py inside the container first", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        model_in = f.read().strip()
    if not model_in or not os.path.isdir(model_in):
        print(f"FATAL: BF16 source dir not found: {model_in!r}", file=sys.stderr)
        sys.exit(1)
    return model_in


# ─── Model load ──────────────────────────────────────────────────────────────


def load_bf16(model_in: str, model_class):
    """Load BF16 model + tokenizer on MI300X.

    `model_class` is the transformers class to instantiate (e.g.
    Gemma4UnifiedForConditionalGeneration for 12B Unified, or
    Gemma4ForConditionalGeneration for the older 26B MoE class).
    The caller passes the class because the class name varies by
    Gemma 4 generation and the older class is being phased out.

    Imports are inside the function because transformers + quark are
    heavy and only the container has them installed.
    """
    import torch
    from transformers import AutoTokenizer

    print(f"Loading BF16 model from {model_in} (device_map=auto on MI300X) via {model_class.__name__}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_in, trust_remote_code=True)
    model = model_class.from_pretrained(
        model_in,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


# ─── Quantization specs ──────────────────────────────────────────────────────


def build_w8a8_specs():
    """Return (weight_spec, input_spec) for W8A8 INT8 — proven recipe.

    Scheme: per-channel INT8 weights (ch_axis=0, symmetric, static) +
    per-token INT8 activations (ch_axis=1, symmetric, dynamic).
    Same scheme used by both the 31B-dense baseline and the MoE rewrite.
    """
    # Imports inside: quark is container-only.
    from quark.torch.quantization.config.config import (
        QTensorConfig,
        Dtype,
    )
    from quark.torch.quantization.config.type import (
        RoundType,
        ScaleType,
        QSchemeType,
    )
    from quark.torch.quantization.observer import PerChannelMinMaxObserver

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
    return weight_spec, input_spec


def build_qconfig(exclude: Sequence[str]):
    """Build a Quark 0.12 QConfig from the W8A8 specs + a per-recipe exclude list.

    Quark 0.12 renamed Config/QuantizationConfig -> QConfig/QLayerConfig
    (from the 0.11 recipe in the upstream HF repos).
    """
    from quark.torch.quantization.config.config import QConfig, QLayerConfig

    weight_spec, input_spec = build_w8a8_specs()
    return QConfig(
        global_quant_config=QLayerConfig(
            input_tensors=input_spec,
            weight=weight_spec,
        ),
        exclude=list(exclude),
    )


# ─── Quantize + export ───────────────────────────────────────────────────────


def quantize_and_export(model, tokenizer, q_cfg, model_out: str) -> int:
    """Quantize (no calibration), freeze, export to safetensors.

    Returns 0 on success, 1 on failure (caller can sys.exit()).
    """
    from quark.torch import ModelQuantizer, export_safetensors

    print("Quantizing (W8A8 INT8, no calibration data needed)...")
    quantizer = ModelQuantizer(q_cfg, multi_device=True)
    model = quantizer.quantize_model(model, dataloader=None)
    quantizer.freeze(model)

    os.makedirs(model_out, exist_ok=True)
    print(f"Exporting to {model_out} (pack_method=order, real_quantized, quark mode)...")
    # Quark 0.12 moved export_model -> quark.torch.export_safetensors
    export_safetensors(
        model,
        model_out,
        pack_method="order",
        weight_format="real_quantized",
        custom_mode="quark",
    )
    tokenizer.save_pretrained(model_out)
    return 0

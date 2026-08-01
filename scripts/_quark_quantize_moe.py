"""Quark W8A8 INT8 quantization of Gemma 4 26B A4B MoE (MoE-aware recipe).

Recipe: nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 (HF) — a *MoE* model
(256 experts, fused gate_up_proj) quantized with AMD Quark, measured
+0.00pp vs BF16 on GSM8K and +30-78% decode throughput. Two structural
steps make MoE quantization correct; the 31B-dense recipe skipped both:

1. PRE-QUANTIZATION REWRITE — Gemma4TextExperts stores expert weights as
   fused 3D tensors (gate_up_proj [E, 2I, H], down_proj [E, H, I]).
   Quark must see each expert as a standard nn.Linear, and vLLM's
   FusedMoE loader expects per-expert params. We replace each layer's
   experts module with a ModuleList[128] of (gate_proj, up_proj,
   down_proj) nn.Linear triplets, copying weights from the 3D tensors.
   See https://huggingface.co/nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8
   ("Pre-quantization rewrite" section).

2. POST-EXPORT RENAME — Quark's custom_mode='quark' export emits
   '*_quantizer.scale'/'*_quantizer.zero_point' keys. vLLM/HF expects
   '*_scale' (symmetric, no zero_point) with weight_scale squeezed
   [out,1] -> [out]. rename_keys() below does this.

Excluded layers (kept BF16):
  lm_head, *mlp.gate* (MoE router), *shared_expert_gate*,
  *visual* (vision tower), *embed_tokens*.

The quantization scheme itself (per-channel weight, per-token dynamic
activation) is unchanged from the dense recipe — that part was correct.

Quark 0.12 API drift vs the 0.11 recipes:
  Config -> QConfig, QuantizationConfig -> QLayerConfig,
  ModelQuantizer.export_model() -> quark.torch.export_safetensors().
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn as nn

MODEL_OUT = os.environ.get("QUARK_OUT", "/models/gemma-4-26B-A4B-it-int8-moe2")


def split_fused_experts(model: nn.Module) -> int:
    """Replace each layer's fused 3D-tensor experts with per-expert Linears.

    Returns the number of layers rewritten.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    rewritten = 0
    for layer in model.model.language_model.layers:
        if not getattr(layer, "enable_moe_block", False):
            continue
        experts: Gemma4TextExperts = layer.experts
        if not isinstance(experts, Gemma4TextExperts) or hasattr(experts, "gate_proj"):
            # already rewritten (or not fused)
            continue

        num_experts = experts.num_experts
        hidden = experts.hidden_dim
        inter = experts.intermediate_dim
        gate_up = experts.gate_up_proj.detach()  # [E, 2I, H]
        down = experts.down_proj.detach()        # [E, H, I]

        expert_list = nn.ModuleList()
        for e in range(num_experts):
            gu = gate_up[e]  # [2I, H]
            g, u = gu.chunk(2, dim=0)  # [I, H] each
            expert_list.append(
                nn.ModuleList(
                    [
                        nn.Linear(hidden, inter, bias=False),
                        nn.Linear(hidden, inter, bias=False),
                        nn.Linear(inter, hidden, bias=False),
                    ]
                )
            )
            expert_list[-1][0].weight.data.copy_(g)
            expert_list[-1][1].weight.data.copy_(u)
            expert_list[-1][2].weight.data.copy_(down[e])

        # Build a thin replacement module that exposes .gate_proj/.up_proj/.down_proj
        # naming per expert for Quark's observer to see standard nn.Linear names.
        replacement = nn.Module()
        replacement.gate_proj = nn.ModuleList(
            [expert_list[e][0] for e in range(num_experts)]
        )
        replacement.up_proj = nn.ModuleList(
            [expert_list[e][1] for e in range(num_experts)]
        )
        replacement.down_proj = nn.ModuleList(
            [expert_list[e][2] for e in range(num_experts)]
        )
        replacement.num_experts = num_experts

        # Forward only runs if a calibration dataloader is used; with
        # dataloader=None Quark never calls it. Keep a functional
        # equivalent (SwiGLU via per-expert linears) so any accidental
        # forward pass doesn't crash. Weight layout mirrors the fused
        # version: gate_up = [gate; up] concatenated along hidden dim.
        def _forward(self, hidden_states, top_k_index, top_k_weights):
            out = torch.zeros_like(hidden_states)
            for e in range(num_experts):
                mask = top_k_index == e
                if not mask.any():
                    continue
                x = hidden_states[mask]
                gate = self.gate_proj[e](x)
                up = self.up_proj[e](x)
                w = top_k_weights[mask].unsqueeze(-1)
                out[mask] += (torch.nn.functional.silu(gate) * up) * w
            return out

        replacement.forward = _forward.__get__(replacement, nn.Module)

        layer.experts = replacement
        rewritten += 1
    return rewritten


def rename_keys(model_dir: str) -> None:
    """Convert Quark '*_quantizer.scale' keys to vLLM/HF '*_scale' layout.

    - '*_quantizer.scale' -> '*_scale'
    - '*_quantizer.zero_point' -> dropped (symmetric quant)
    - weight_scale squeezed from [out, 1] to [out]
    """
    import glob
    from safetensors import safe_open
    from safetensors.torch import save_file

    for st_path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        tensors = {}
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if key.endswith("_quantizer.zero_point"):
                    continue  # drop
                if key.endswith("_quantizer.scale"):
                    new_key = key.replace("_quantizer.scale", "_scale")
                else:
                    new_key = key
                if new_key.endswith("_scale") and t.dim() == 2 and t.shape[1] == 1:
                    t = t.squeeze(1)  # [out, 1] -> [out]
                tensors[new_key] = t.contiguous()
        save_file(tensors, st_path)
        print(f"  renamed keys in {os.path.basename(st_path)} ({len(tensors)} tensors)")


def main() -> None:
    with open("/tmp/bf16_path.txt") as f:
        model_in = f.read().strip()
    if not model_in or not os.path.isdir(model_in):
        print(f"FATAL: BF16 source dir not found: {model_in!r}", file=sys.stderr)
        return 1

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

    print(f"Loading BF16 model from {model_in} (device_map=auto)")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_in, trust_remote_code=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_in,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Step 1: split fused expert tensors into per-expert nn.Linear triplets.
    n = split_fused_experts(model)
    print(f"Split fused experts in {n} layers (128 experts each)")

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

    q_cfg = QConfig(
        global_quant_config=QLayerConfig(
            input_tensors=input_spec,
            weight=weight_spec,
        ),
        exclude=[
            "lm_head",
            "*router*",            # MoE router (router.proj, router.scale)
            "*shared_expert_gate*",  # per-layer gate (if present)
            "*vision_tower*",      # vision tower (NOT "*visual*" — the
            "*embed_vision*",      #   actual prefixes are vision_tower /
            "*visual*",            #   embed_vision)
            "*embed_tokens*",
            # NOTE: do NOT use "*mlp.gate*" here — Gemma4's dense MLP is
            # fused as gate_up_proj in vLLM; excluding gate_proj but not
            # up_proj splits the fused shards into different schemes and
            # vLLM's quark loader raises ValueError.
        ],
    )

    print("Quantizing (W8A8 INT8, MoE-aware, no calibration data needed)...")
    quantizer = ModelQuantizer(q_cfg, multi_device=True)
    model = quantizer.quantize_model(model, dataloader=None)
    quantizer.freeze(model)

    os.makedirs(MODEL_OUT, exist_ok=True)
    print(f"Exporting to {MODEL_OUT} (pack_method=order, real_quantized, quark mode)...")
    from quark.torch import export_safetensors
    export_safetensors(
        model,
        MODEL_OUT,
        pack_method="order",
        weight_format="real_quantized",
        custom_mode="quark",
    )

    # Step 2: rename Quark keys to vLLM/HF layout.
    print("Post-export key rename (quantizer.scale -> _scale, drop zero_point)...")
    rename_keys(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Done in {time.time() - t0:.1f}s -> {MODEL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

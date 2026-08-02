"""Quark W8A8 INT8 quantization - Gemma 4 26B A4B MoE (MoE-aware recipe).

Recipe: nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8 (HF) - a *MoE* model
(256 experts, fused gate_up_proj) quantized with AMD Quark, measured
+0.00pp vs BF16 on GSM8K and +30-78% decode throughput. Two structural
steps make MoE quantization correct; the 31B-dense recipe skipped both:

1. PRE-QUANTIZATION REWRITE - Gemma4TextExperts stores expert weights as
   fused 3D tensors (gate_up_proj [E, 2I, H], down_proj [E, H, I]).
   Quark must see each expert as a standard nn.Linear, and vLLM's
   FusedMoE loader expects per-expert params. We replace each layer's
   experts module with a ModuleList[128] of (gate_proj, up_proj,
   down_proj) nn.Linear triplets, copying weights from the 3D tensors.
   See https://huggingface.co/nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8
   ("Pre-quantization rewrite" section).

2. POST-EXPORT RENAME - Quark's custom_mode='quark' export emits
   '*_quantizer.scale'/'*_quantizer.zero_point' keys. vLLM/HF expects
   '*_scale' (symmetric, no zero_point) with weight_scale squeezed
   [out,1] -> [out]. rename_keys() below does this.

Excluded layers (kept BF16):
  lm_head, *router* (MoE router), *shared_expert_gate*,
  *vision_tower* / *embed_vision* / *visual*, *embed_tokens*.

NOTE: do NOT use "*mlp.gate*" here - Gemma4's dense MLP is fused as
gate_up_proj in vLLM; excluding gate_proj but not up_proj splits the
fused shards into different schemes and vLLM's quark loader raises
ValueError.

The quantization scheme itself (per-channel weight, per-token dynamic
activation) is unchanged from the dense recipe - that part was correct.
It is provided by _quark_common.build_qconfig.

KNOWN LIMITATION: this recipe loads on the 26B A4B MoE but produces
garbage (4 attempts documented in DEPLOYMENT_JOURNAL.md Issues 11-13).
The 26B ships as BF16; only the dense path is production-claimable.

Use via quantize_int8.sh:
    bash quantize_int8.sh --kind moe --model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations

import glob
import os
import sys
import time

import torch
import torch.nn as nn

# Ensure scripts/ is on sys.path so _quark_common is importable when this
# script is copied into /tmp inside the container.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _quark_common import (  # noqa: E402
    build_qconfig,
    load_bf16,
    quantize_and_export,
    read_bf16_path,
)


# ─── MoE-only: structural rewrite + post-export key fixup ─────────────────────


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


# MoE-tuned exclude list. Differs from DENSE_EXCLUDE: no *experts*/*moe*
# (already rewritten into per-expert Linears that ARE quantized), but
# includes *router* and *shared_expert_gate*.
MOE_EXCLUDE = [
    "lm_head",
    "*router*",            # MoE router (router.proj, router.scale)
    "*shared_expert_gate*",  # per-layer gate (if present)
    "*vision_tower*",      # vision tower (NOT "*visual*" - the
    "*embed_vision*",      #   actual prefixes are vision_tower /
    "*visual*",            #   embed_vision)
    "*embed_tokens*",
]


def main() -> int:
    # 26B A4B MoE uses the older Gemma4ForConditionalGeneration class
    # (verified from google/gemma-4-26B-A4B-it/config.json: model_type=gemma4,
    # architectures=[Gemma4ForConditionalGeneration]). Import is inside
    # main() so a transformers-version mismatch surfaces at run time.
    from transformers import Gemma4ForConditionalGeneration

    model_in = read_bf16_path()
    model, tokenizer = load_bf16(model_in, Gemma4ForConditionalGeneration)

    # Step 1: split fused expert tensors into per-expert nn.Linear triplets.
    n = split_fused_experts(model)
    print(f"Split fused experts in {n} layers (128 experts each)")

    q_cfg = build_qconfig(MOE_EXCLUDE)
    model_out = os.environ.get("QUARK_OUT", "/models/gemma-4-26B-A4B-it-int8-moe2")

    t0 = time.time()
    rc = quantize_and_export(model, tokenizer, q_cfg, model_out)

    # Step 2: rename Quark keys to vLLM/HF layout.
    if rc == 0:
        print("Post-export key rename (quantizer.scale -> _scale, drop zero_point)...")
        rename_keys(model_out)
        tokenizer.save_pretrained(model_out)
        print(f"Done in {time.time() - t0:.1f}s -> {model_out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

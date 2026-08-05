---
license: apache-2.0
library_name: transformers
language:
  - en
pipeline_tag: image-text-to-text
base_model: google/gemma-4-12B-it
tags:
  - gemma
  - gemma4
  - multimodal
  - vision-language
  - quantized
  - int8
  - w8a8
  - quark
  - vllm
  - conversational
  - text-generation-inference
  - image-text-to-text
---

# Gemma-4-12B-it-Quark-W8A8-INT8

W8A8 INT8 quantized version of [google/gemma-4-12b-it](https://huggingface.co/google/gemma-4-12b-it) using [AMD Quark](https://github.com/amd/quark), produced as part of the [RetailConcierge](https://github.com/rajasingh012/retail-concierge) AMD AI DevMaster 2026 submission.

## Model Details

|                |                                                                                |
|----------------|--------------------------------------------------------------------------------|
| Base Model     | `google/gemma-4-12b-it`                                                        |
| Architecture   | `Gemma4UnifiedForConditionalGeneration` (multimodal: text + vision + audio)    |
| Parameters     | 12 B text decoder (quantized) + vision/audio embedders kept in BF16            |
| Quantization   | W8A8 INT8 (per-channel weight + per-token dynamic activation)                  |
| Quantizer      | AMD Quark `0.12.post1`                                                         |
| Model Size     | ~13 GB (safetensors shards)                                                    |
| Original Size  | ~23.9 GB (BF16)                                                                |
| Compression    | ~1.8× size reduction                                                           |

### Quantization Scheme

| Component   | dtype | Granularity                | Mode               |
|-------------|-------|----------------------------|--------------------|
| Weight      | INT8  | per-channel (`ch_axis=0`)  | symmetric, static  |
| Activation  | INT8  | per-token (`ch_axis=1`)    | symmetric, dynamic |
| `model.vision_embedder.*` | BF16 | — | unquantized (multimodal preserved) |
| `model.embed_vision.embedding_projection` | BF16 | — | unquantized |
| `model.embed_audio.embedding_projection` | BF16 | — | unquantized (12B Unified audio modality) |
| `lm_head` (tied) | BF16 | — | unquantized via `embed_tokens` tie |

## How to Use

### With vLLM (Recommended)

```bash
# Start the server (single AMD MI300X is enough)
vllm serve rajasingh012/gemma-4-12b-it-quark-w8a8-int8 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --quantization quark \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill

# Chat completion
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "rajasingh012/gemma-4-12b-it-quark-w8a8-int8",
  "messages": [{"role": "user", "content": "Recommend a pair of wireless earbuds under 5000 rupees."}],
  "max_tokens": 512,
  "temperature": 0.7
}'
```

Requires vLLM >= 0.26 (the `--quantization quark` loader). Tested on vLLM 0.26.0+rocm723 (AMD ROCm wheel) with `--tool-call-parser gemma4`.

### Hardware Requirements

- **Minimum**: 1× GPU with ≥48 GB VRAM (e.g., AMD MI300X / MI350X, NVIDIA A100-80G).
- Quantized weights measure ~12.5 GiB on GPU, leaving ample KV cache headroom.

## Quantization Details

This model was quantized using AMD Quark's per-token per-channel INT8 scheme (W8A8):

- **Weight quantization**: INT8 per-channel (one scale per output channel), symmetric, static.
- **Activation quantization**: INT8 per-token (one scale per token), symmetric, dynamic (computed at inference time, so no calibration data needed).
- **Excluded layers**: `model.vision_embedder.*`, `model.embed_vision.embedding_projection`, `model.embed_audio.embedding_projection` (vision/audio modalities preserved in BF16). `lm_head` shares storage with `embed_tokens` (`tie_word_embeddings: True`).
- **Export**: real INT8 weights with BF16 scales (no fake-quant, no zero-point), Quark 0.12 `export_safetensors`.
- **Post-export fixup (required)**: Quark 0.12 exports some vision keys under names vLLM's `gemma4_unified` loader does not map (`embed_vision.multimodal_embedder.*`, `embed_vision.patch_*`). The fixup renames them to the expected layout and copies `chat_template.jinja` into the output (vLLM 400s chat requests without it). The fixup is automated in the RetailConcierge repo (`scripts/_quark_fix_vllm_keys.py`) and is required for any Gemma 4 Unified checkpoint.

### Reproduce Quantization

Full reproducible recipe in the [RetailConcierge repo](https://github.com/rajasingh012/retail-concierge), `scripts/` folder:

```bash
# On the AMD GPU droplet (see scripts/README.md for full lifecycle)
bash /root/quantize_int8.sh --model google/gemma-4-12b-it
```

The script handles: preflight (Quark version, transformers >= 5.10.1 for `Gemma4UnifiedForConditionalGeneration`, BF16 source location), the Quark W8A8 quantize, the post-export key fixup, and output verification. Recipe provenance: `nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8` (−0.08pp GSM8K on the 31B dense class); the 12B Unified run is a fresh quantization of the same scheme.

## Accuracy

**Caveat:** the −0.08pp GSM8K figure published on the 31B dense baseline was measured on the older `Gemma4ForConditionalGeneration` class. The 12B Unified class uses a different multimodal embedding layout; accuracy for this checkpoint is validated via the RetailConcierge tool-call accuracy gate (extract_brief → search_catalog → finalize_recommendations pass rate vs BF16), not yet via GSM8K. Treat as a fresh quantization until independently benchmarked.

## Citation

If you use this model, please cite the original Gemma 4 release:

```bibtex
@misc{google2026gemma4,
  title  = {Gemma 4},
  author = {Google DeepMind},
  year   = {2026},
  url    = {https://huggingface.co/google/gemma-4-12b-it}
}
```

## License

This model is released under the **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**, following the [Gemma 4 license](https://ai.google.dev/gemma/apache_2) under which the upstream `google/gemma-4-12b-it` weights are distributed by Google DeepMind.

This is a quantized derivative of `google/gemma-4-12b-it`. Per Apache 2.0 §4:

- Modified files (the INT8-quantized `model.safetensors` and the appended `quantization_config` block in `config.json`) carry this notice as part of the model card.
- Original copyright and attribution notices from the base model are preserved (see `NOTICE`).
- A copy of the Apache 2.0 license text is included as `LICENSE`.

Original weights © Google DeepMind. Quantization performed by the model author; no warranty of any kind is provided (see `LICENSE` §7–8).

# AMD AI DevMaster Hackathon — Track 2
## Submission Checklist & Poster

**Team**: Rajasingh (Solo)
**Project**: RetailConcierge — Conversational Shopping Agent on AMD MI300X
**Track**: Track 2 — Development & Local Deployment of Private AI Agents

---

## Submission Checklist

| # | Deliverable | Status | File/Link |
|---|------------|--------|-----------|
| 1 | Project Specification Document | ✅ | `docs/hackathon_spec_document.md` |
| 2 | Project Source Code | ✅ | https://github.com/rajasingh012/retail-concierge (AGPL-3.0, public) |
| 3 | Demo Video | 🎬 In production | per `docs/demo_video_script.md` (3:40 target, real terminal) |
| 5 | Published model (extra evidence) | ✅ | https://huggingface.co/rajasingh012/gemma-4-12b-it-quark-w8a8-int8 — first AMD Quark W8A8 INT8 of Gemma 4 12B, public |
| 4 | PPT / Poster | ✅ Below | Key slides in this document |

---

## Poster / Key Slides

### Slide 1: Title

```
╔══════════════════════════════════════════════════════════╗
║                    RetailConcierge                        ║
║        Conversational Shopping Agent — 100% on AMD        ║
║                                                          ║
║     Gemma 4 12B INT8 · AMD Quark · vLLM 0.26 · MI300X    ║
║     145,615 products · 576 types · offline, audited      ║
╚══════════════════════════════════════════════════════════╝
```

### Slide 2: What it does

- Conversational product search over an offline Amazon Berkeley Objects catalog
- Clarification-first: asks ONE precise question when the request is ambiguous
- Multi-turn memory: refinement chips continue the same session
- Evidence-backed recommendations: no invented prices, no fake availability
- Hash-chained audit log on every decision

### Slide 3: The AMD stack (the 40-point story)

```
  AMD Quark 0.12          vLLM 0.26+rocm723        AMD Instinct MI300X
  W8A8 INT8 quantize  →   --quantization quark  →   192 GB HBM3 · ROCm 7.2.3
  23.9 GB → 13 GB         prefix caching            AITER attention
  (1.8×)                  chunked prefill           local inference only
```

- vLLM upgraded 0.23 → 0.26 on the AMD ROCm wheel index (ABI fixes automated)
- Quark W8A8 INT8: 12.54 GiB weight footprint on GPU
- Two Quark→vLLM incompatibilities debugged live and automated (`_quark_fix_vllm_keys.py`)

### Slide 4: Performance (measured on MI300X)

| Metric | Value |
|--------|------:|
| Output throughput (single stream) | **49.8 tok/s** |
| Peak output throughput | **51.0 tok/s** |
| Median TTFT | ~55 ms |
| TPOT | ~19.8 ms |
| Benchmark | 10,240 input → 1,280 tokens in 25.7 s |

### Slide 5: Why it's different

1. The **AMD-native quantization pipeline actually ships** — Quark INT8 → vLLM quark loader, end-to-end, reproducible from repo scripts, and the checkpoint is **published on Hugging Face**: [rajasingh012/gemma-4-12b-it-quark-w8a8-int8](https://huggingface.co/rajasingh012/gemma-4-12b-it-quark-w8a8-int8) (first AMD Quark W8A8 INT8 of Gemma 4 12B)
2. **Deterministic trust**: LLM retrieves, application code decides — screening/ranking are deterministic, provenance-tracked, audit-logged
3. **Honest engineering**: the 26B MoE quantization failure trail is documented openly (DEPLOYMENT_JOURNAL.md), not hidden

---

## Demo Video Shot List (from docs/demo_video_script.md)

| Act | Time | Content |
|-----|------|---------|
| 1. Hook | 0:25 | App boots + `amd-smi` shows MI300X |
| 2. Agent works | 1:15 | 3 queries: full flow, clarification, refinement chips |
| 3a. Live GPU | 0:30 | `amd-smi` utilization spiking during inference |
| 3b. Speed | 0:40 | tok/s table, TTFT, concurrency sweep |
| 3c. Quantization | 0:20 | Quark INT8: 23.9 → 13 GB, served with `--quantization quark` |
| 4. Close | 0:30 | Repo URL + reproducibility |

Total: ~3:40 (hard cap 5:00)

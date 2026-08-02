"""Quark W8A8 INT8 quantization - Gemma 4 dense recipe (12B).

Recipe adapted from nameistoken/Gemma-4-31B-it-Quark-W8A8-INT8 (HF),
which achieved -0.08pp on GSM8K vs BF16 (essentially lossless) on the
older Gemma4ForConditionalGeneration class. The 12B variant uses
Gemma4UnifiedForConditionalGeneration (verified from
google/gemma-4-12b-it/config.json, Aug 2 2026) which has a different
vision structure: only `vision_embedder.*` (patch + pos embed + LN),
NO vision_tower transformer encoder. The 31B recipe's 189-entry
vision_tower exclude list is therefore not applicable - it is replaced
by a single `model.vision_embedder.*` glob.

Scheme (unchanged from 31B recipe):
- W8A8 INT8 - per-channel weight (ch_axis=0, symmetric, static)
  + per-token activation (ch_axis=1, symmetric, dynamic).
- NO calibration data needed (dataloader=None).
- Excluded (stay BF16): model.vision_embedder.* (patch embed + LN),
  model.embed_vision.embedding_projection, model.embed_audio.embedding_projection.
- embed_tokens IS quantized (NOT excluded) - matches the proven 31B
  recipe which also quantizes embed_tokens. The 12B variant ties
  lm_head to embed_tokens (tie_word_embeddings=True), so excluding
  embed_tokens would also force lm_head BF16.
- lm_head is not separately listed - it shares storage with
  embed_tokens on this architecture.

CAVEAT (read before claiming quantization bonus):
The 31B accuracy figure (-0.08pp GSM8K) was measured on the
Gemma4ForConditionalGeneration class, not the Unified class used
by 12B. The recipe's exclusion strategy here is the closest analog,
NOT a measured result on Unified. Treat the first 12B run as a
fresh quantization with the 31B recipe as prior art, not a proven
drop-in. Verify accuracy via the same tool-call accuracy gate used
for 31B before claiming the bonus.

Runs inside the container. Source model path comes from
/tmp/bf16_path.txt (written by _find_bf16.py based on the --model
arg passed by quantize_int8.sh).

Use via quantize_int8.sh:
    bash quantize_int8.sh --kind dense --model google/gemma-4-12b-it
"""
from __future__ import annotations

import os
import sys

# Ensure scripts/ is on sys.path so _quark_common is importable when this
# script is copied into /tmp inside the container.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _quark_common import (  # noqa: E402
    build_qconfig,
    load_bf16,
    quantize_and_export,
    read_bf16_path,
)


# 12B-specific exclude list. Verified against google/gemma-4-12b-it
# safetensors header (Aug 2 2026): the only vision-related weights
# are model.vision_embedder.* and model.embed_vision.embedding_projection.
# No vision_tower.* paths exist on this architecture.
#
# IMPORTANT (Quark rename quirk, hit live on the droplet 2026-08-02):
# Quark 0.12.post1 renames vision_embedder -> embed_vision.patch_* /
# embed_vision.pos_* internally BEFORE matching the exclude patterns.
# Patterns written against the HF names (model.vision_embedder.*,
# model.embed_vision.embedding_projection) match NOTHING at quantize
# time, so those modules get quantized and vLLM's Gemma4Unified loader
# rejects them (Gemma4MultimodalEmbedder has no quant support). The
# exclude list must use the names Quark sees: model.embed_vision.*
# covers both the (renamed) vision embedder and the projection.
#
# lm_head is tied to embed_tokens on 12B (tie_word_embeddings=True);
# Quark un-ties it during export, so it must be excluded explicitly.
DENSE_EXCLUDE_12B = [
    "model.embed_vision.*",
    "model.embed_audio.*",
    "lm_head",
]


def main() -> int:
    # 12B uses the Unified architecture (verified from
    # google/gemma-4-12b-it/config.json). Import is inside main() so
    # the class-availability error surfaces at run time, not at module
    # import time (the container's transformers version is what matters).
    from transformers import Gemma4UnifiedForConditionalGeneration

    model_in = read_bf16_path()
    model, tokenizer = load_bf16(model_in, Gemma4UnifiedForConditionalGeneration)

    q_cfg = build_qconfig(DENSE_EXCLUDE_12B)
    model_out = os.environ.get("QUARK_OUT", "/models/gemma-4-12b-it-int8")

    import time  # noqa: PLC0415
    t0 = time.time()
    rc = quantize_and_export(model, tokenizer, q_cfg, model_out)
    if rc == 0:
        print(f"Done in {time.time() - t0:.1f}s -> {model_out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""Post-quantize fixups so a Quark 0.12 INT8 output loads in vLLM 0.26.

Quark 0.12.post1's safetensors export for Gemma4UnifiedForConditionalGeneration
(Gemma 4 12B) has two deviations from what vLLM's gemma4_unified loader maps:

1.  model.embed_vision.multimodal_embedder.embedding_projection.*  (Quark)
    -> model.embed_vision.embedding_projection.*                   (vLLM)

2.  model.embed_vision.patch_dense.* / patch_ln1.* / patch_ln2.* /
    pos_embedding / pos_norm.*                                      (Quark)
    -> model.vision_embedder.patch_dense.* / ...                    (vLLM)

Without the rename, vLLM raises:
    ValueError: There is no module or parameter named
    'embed_vision.multimodal_embedder' in Gemma4UnifiedForConditionalGeneration

Also copies chat_template.jinja from the BF16 source snapshot (or HF) into
the output — Quark does not export it, and vLLM 0.26 refuses requests with
HTTP 400 ("default chat template is no longer allowed") when it is missing.

Idempotent: already-fixed keys pass through unchanged; missing template is a
no-op if present. Safe to re-run.

Run inside the container (the quantize_int8.sh preflight already copies
_find_bf16.py / _quark_common.py / the recipe into /tmp and writes
/tmp/bf16_path.txt):

    python3 /tmp/_quark_fix_vllm_keys.py /models/gemma-4-12b-it-int8
"""
from __future__ import annotations

import glob
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/models/gemma-4-12b-it-int8"
BF16_PATH_FILE = "/tmp/bf16_path.txt"  # written by quantize_int8.sh preflight


def fix_key(k: str) -> str:
    """Map a Quark-exported key to the vLLM gemma4_unified layout.

    Returns the key unchanged when it does not need fixing (idempotent).
    """
    # 1. un-nest multimodal_embedder
    if k.startswith("model.embed_vision.multimodal_embedder."):
        return k.replace("model.embed_vision.multimodal_embedder.", "model.embed_vision.", 1)
    # 2. move vision embedder weights under vision_embedder
    if k.startswith("model.embed_vision.patch_") or k.startswith("model.embed_vision.pos_"):
        return k.replace("model.embed_vision.", "model.vision_embedder.", 1)
    return k  # unchanged


def rename_weights(model_dir: str) -> int:
    """Rename keys in every *.safetensors under model_dir. Returns total renamed."""
    total = 0
    for st_path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        tensors = {}
        renamed = 0
        with safe_open(st_path, framework="pt") as f:
            for key in f.keys():
                new_key = fix_key(key)
                if new_key != key:
                    renamed += 1
                tensors[new_key] = f.get_tensor(key).contiguous()
        if renamed:
            save_file(tensors, st_path)
            print(f"  {os.path.basename(st_path)}: {renamed} keys renamed, {len(tensors)} total")
        else:
            print(f"  {os.path.basename(st_path)}: no renames needed ({len(tensors)} tensors)")
        total += renamed
    return total


def copy_chat_template(model_dir: str) -> None:
    """Ensure chat_template.jinja exists in the output.

    Tries, in order: the output dir itself, the BF16 source snapshot
    (/tmp/bf16_path.txt), then HF download. No-op if already present.
    """
    out_tmpl = os.path.join(model_dir, "chat_template.jinja")
    if os.path.isfile(out_tmpl):
        print(f"  chat_template.jinja: already present in {model_dir}")
        return

    candidates = []
    if os.path.isfile(BF16_PATH_FILE):
        with open(BF16_PATH_FILE) as f:
            src = f.read().strip()
        if src:
            candidates.append(os.path.join(src, "chat_template.jinja"))
    for cand in candidates:
        if os.path.isfile(cand):
            import shutil
            shutil.copy2(cand, out_tmpl)
            print(f"  chat_template.jinja: copied from BF16 source ({cand})")
            return

    # Fallback: download from HF. The source snapshot may not have it if the
    # download used allow_patterns that excluded *.jinja.
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id="google/gemma-4-12b-it", filename="chat_template.jinja")
        import shutil
        shutil.copy2(p, out_tmpl)
        print(f"  chat_template.jinja: downloaded from HF ({p})")
    except Exception as e:  # noqa: BLE001 - report and continue; vLLM will 400 without it
        print(f"  WARN: chat_template.jinja not available ({type(e).__name__}: {e})")
        print("        vLLM will reject chat requests with HTTP 400 until it is present.")


def main() -> int:
    if not os.path.isdir(MODEL_DIR):
        print(f"FATAL: {MODEL_DIR} does not exist", file=sys.stderr)
        return 1
    renamed = rename_weights(MODEL_DIR)
    copy_chat_template(MODEL_DIR)
    print(f"done ({renamed} keys renamed total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

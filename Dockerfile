# RetailConcierge — AMD Radeon ROCm demo image
# Base: AMD 1-Click vLLM image (Ubuntu 24.04 + ROCm 7.2.4 + Docker).
# Source: https://www.amd.com (1-Click vLLM droplet description).
#
# This Dockerfile is a fallback / portable image. The 1-Click image on the
# AMD Developer Cloud already runs vLLM 0.23.0 in a Docker container
# with JupyterLab — use that directly. Build this image only if you're
# running outside the 1-Click environment (e.g. local ROCm dev box).

FROM rocm/dev-ubuntu-24.04:7.2.4

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RETAIL_PROVIDER=vllm \
    RETAIL_MODEL=google/gemma-3-27b-it

WORKDIR /app

# 1. Minimal host utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. PyTorch ROCm wheel + vLLM (official selector; do not build from source)
RUN pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm7.0 \
    && pip install --no-cache-dir vllm

# 3. App deps
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv export --frozen --no-dev --no-emit-project --output-file /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip uninstall --quiet -y uv \
    && rm -f /tmp/requirements.txt

# 4. App code
COPY . .

# 5. Convenience entrypoints
RUN chmod +x scripts/*.sh

# Run (portable fallback image, NOT the 1-Click image):
#   docker run --rm -it \
#     --device /dev/kfd --device /dev/dri \
#     -p 8000:8000 \
#     retail-concierge:latest \
#     bash scripts/serve-vllm-rocm.sh

EXPOSE 8000
CMD ["bash"]
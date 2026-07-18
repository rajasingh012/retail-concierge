# RetailConcierge — AMD Radeon ROCm demo image
# Base: AMD 1-Click vLLM image (Ubuntu 24.04 + ROCm 7.2.4 + Docker).
# Source: https://www.amd.com (1-Click vLLM droplet description).
#
# This Dockerfile is a fallback / portable image. The 1-Click image on the
# AMD Developer Cloud already runs vLLM 0.23.0 in a Docker container
# with JupyterLab — use that directly. Build this image only if you're
# running outside the 1-Click environment (e.g. local ROCm dev box).

FROM rocm/dev-ubuntu-24.04:7.14.0

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RETAIL_PROVIDER=vllm \
    RETAIL_MODEL=google/gemma-3-27b-it

WORKDIR /app

# 1. System deps (Playwright libs; Chromium fetched at run time)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git curl ca-certificates \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
        libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

# 2. PyTorch ROCm wheel + vLLM (official selector; do not build from source)
RUN pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm7.0 \
    && pip install --no-cache-dir vllm

# 3. App deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
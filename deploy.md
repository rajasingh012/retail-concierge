# AMD Developer Cloud — Verified Bring-up Notes

Verified against an actual AMD Developer Cloud dashboard screenshot
captured July 2026. Numbers below are real, not ballpark.

## Your credit (as of screenshot)

| Field | Value |
|---|---|
| Provider | AMD Developer Cloud (DigitalOcean-powered "GPU Droplets") |
| GPU | MI300X, 192 GB VRAM |
| $/hr | **$1.99/GPU/hr** |
| Credit balance | **$50.00** |
| Credit expiry | 639 days from capture |
| Coverage | **GPU only** — non-GPU services billed to your payment method |
| Auto-billing | Yes — after credits, charges your payment method |

## Budget math

**$50 ÷ $1.99/hr ≈ 25 hours of MI300X.**

### Suggested budget allocation

| Phase | Hours | $ spent |
|---|---|---|
| Bring-up + ROCm verify (vLLM is pre-installed) | 0.25 | $0.50 |
| Pull Gemma 3 27B BF16 (~54 GB download) | 0.5 | $1.00 |
| Iterative agent dev (5-8 sessions) | 6 | $11.94 |
| Bench run (final evidence JSON) | 1 | $1.99 |
| Demo rehearsals (3 takes) | 3 | $5.97 |
| **Live demo** | **0.5** | **$1.00** |
| Buffer for retries | 4 | $7.96 |
| **Total** | **15.25** | **$30.36** |
| **Buffer remaining** | — | **~$19** |

## CRITICAL gotchas

1. **Set a hard spending limit on the DO dashboard BEFORE spinning up.**
   The screenshot explicitly says: *"After the credits have been used, it will
   charge your payment method."* A runaway agent loop on MI300X can burn
   real money fast.

2. **AMD credit only covers GPU access.** All other services (managed DB,
   extra block storage beyond included scratch, etc.) hit your card. Stick
   to the GPU droplet only — local SQLite catalog, local file cache.

3. **July capacity is reduced** for an AMD event. Plan around it — don't try
   to spin up at peak times.

4. **Lock the droplet down with a Cloud Firewall before exposing any port.**
   See "Network lockdown" below — without this, anyone on the internet can
   hit vLLM's API and burn your GPU credit on random requests.

5. **If vLLM misbehaves, destroy + recreate.** Cheaper than debugging
   on a $50 / 25-hour budget. The pre-installed image is reproducible.

## Default model on MI300X

**Gemma 3 27B at BF16** (or Q8_0 if you want to be conservative).
Gemma 4 31B at BF16 also fits (~62 GB) if HF hosts it.

| Model | BF16 VRAM | Q4_K_M VRAM | TTFT on MI300X (est.) |
|---|---|---|---|
| Gemma 3 27B BF16 | ~54 GB | ~14 GB | Sub-second with vLLM |
| Gemma 4 31B BF16 | ~62 GB | ~16 GB | Sub-second |
| Llama 3.1 70B Q4_K_M | ~40 GB | — | Sub-second |
| Llama 3.1 70B BF16 | ~140 GB | — | 1-2s |

## Bring-up (verified steps — AMD 1-Click image)

The image ships vLLM 0.23.0 **running inside a Docker container**, with
JupyterLab also running. amd-smi / rocminfo are on the host. The MOTD
printed at SSH login tells you the container name and JupyterLab URL.

```bash
# 1. From the AMD Developer Cloud dashboard, click "Create GPU Droplet"
#    - Pick the vLLM 1-Click image (NOT bare ROCm — this one has vLLM pre-installed)
#    - Plan: MI300X x1 ($1.99/hr)
#    - Add your SSH key
#    - Set a hard spending limit of $50 on the billing page FIRST

# 2. SSH in — read the MOTD
ssh root@<droplet-ip>
# MOTD will show:
#   - JupyterLab URL + token
#   - docker exec command to drop into the vLLM container
#   - example notebooks path
# WRITE THESE DOWN.

# 3. Verify the GPU (host-side tooling works because Docker passes --device /dev/kfd)
amd-smi --showproductname --showuse --showmeminfo vram
# (or `rocm-smi` — both work)

# 4. Verify the vLLM container is up
docker ps
# You should see one container named something like 'vllm' or 'inference'.

# 5. Stop the default vLLM and re-launch it with rubric-winning flags.
#    This is what ./scripts/serve-vllm-rocm.sh does for you.
./scripts/serve-vllm-rocm.sh
# (You'll be prompted to optionally tail logs.)
# Wait for: "INFO:     The server is fired up and ready to roll!"

# 6. From your LAPTOP (where the agent code lives), point at the cloud:
RETAIL_PROVIDER=vllm \
RETAIL_BASE_URL=http://<droplet-ip>:8000/v1 \
RETAIL_MODEL=google/gemma-3-27b-it \
    PYTHONPATH=. python main.py
```

### What the 1-Click image gives you for free

- ROCm 7.2.4 driver stack (host)
- vLLM 0.23.0 + PyTorch ROCm (inside Docker container)
- amd-smi, rocminfo (host)
- JupyterLab + example notebooks (inside container, URL in MOTD)
- amd-smi can read GPU util from the host shell, even though vLLM runs in the container

### What we add on top

- `--enable-prefix-caching` — multi-turn speedup
- `--enable-chunked-prefill` — prevents head-of-line blocking
- `--kv-cache-dtype fp8` — doubles context on MI300X
- `--speculative-config ngram` — free inter-token win

Default vLLM in the 1-Click image likely runs WITHOUT these flags.
Our launcher stops it and re-launches with all of them on.

### When vLLM is unreachable

```bash
# Check the container is still running
docker ps

# Tail vLLM logs (inside the container)
docker exec <ctr-name> tail -f /tmp/vllm.log

# If totally broken: destroy + recreate droplet
# DO dashboard -> Droplet -> Destroy -> Create new from same 1-Click image
```

## Network lockdown (do this BEFORE vLLM serves anything)

Without a firewall, port 8000 is open to the whole internet and anyone
who finds it can hammer your GPU and burn your credit. Two layers:

### Layer 1 — DigitalOcean Cloud Firewall (network-level, free)

In the DO dashboard, **before** spinning up the droplet:

```
Networking → Firewalls → Create Firewall
  Name:  retail-concierge-demo
  Inbound rules:
    TCP 22   from <your-public-ip>/32     (SSH)
    TCP 8000 from <your-public-ip>/32     (vLLM OpenAI API)
  Outbound: allow all
  Apply to: (attach when you create the GPU Droplet)
```

Your public IP today: `106.222.220.132` (check before each session if your
ISP gives you a dynamic IP — `curl https://api.ipify.org`).

Note: if you also want to use JupyterLab in the browser, open TCP 8888
from your IP. Otherwise leave it closed.

### Layer 2 — droplet iptables (host-level, belt-and-suspenders)

On the droplet itself:

```bash
# Auto-detects your SSH-ing-in IP and locks port 22 + 8000 to it
sudo bash scripts/harden-droplet.sh
# or pass the IP explicitly:
sudo bash scripts/harden-droplet.sh 106.222.220.132
```

Verify from your laptop:

```bash
curl http://<droplet-ip>:8000/health         # should work
ssh <droplet-ip>                              # should work
# From any other IP / port-scan tools: timeout
```

## Deadline alignment

Hackathon ends **Aug 6, 2026** (your timezone). Credit expires **639 days
from July 2026 ≈ May 2028** — no overlap concern. Spend the credit on the
demo, not on idle dev.

## Provider choice

| Mode | Backend | When |
|---|---|---|
| **Judge demo** | vLLM on MI300X | Primary. Pre-installed, AMD-validated, exposes prefix-cache metric. |
| **Local dev** | DeepSeek API | No GPU droplet needed. Same OpenAI endpoint shape — agent code is identical. Just need `DEEPSEEK_API_KEY`. |
| **Last resort** | vLLM via Docker locally | If DeepSeek is down and you can't reach AMD cloud. |

**Local-dev with DeepSeek** (saves your $50 entirely):

```bash
export RETAIL_PROVIDER=deepseek
export RETAIL_MODEL=deepseek-chat
export DEEPSEEK_API_KEY=sk-...
PYTHONPATH=. python main.py
```

The bench script works the same way (just no vLLM prefix-cache metric — it'll show `available: false` in the JSON, which is honest).
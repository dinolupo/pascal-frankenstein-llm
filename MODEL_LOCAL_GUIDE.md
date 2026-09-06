# Running Qwen 35B-Class MoE Models Locally

Operational guide and experiment diary for running Qwen3.6-35B-A3B and
similar 35B-class GGUF models on a Linux Mint workstation with two Pascal
GPUs. Complete chronological measurements remain in
[BASELINE_LOG.md](BASELINE_LOG.md).

## Project status

### Verified hardware

- Intel i7-4790K, 32 GB RAM.
- GTX 1080 Ti, 11 GB, CUDA0.
- GTX 1070, 8 GB, CUDA1.
- Native Linux Mint, NVIDIA driver 580.173.02.
- CUDA 12 release binaries, Pascal `sm_61`.
- CUDA P2P must remain disabled.

### Current model asset

The current test asset is a Qwen3.6-35B-A3B GGUF with native MTP tensors. Set
`MODEL` to the exact local file being tested:

```text
MODEL=/path/to/qwen35b-mtp.gguf
```

The filename above identifies the current local asset.

### Binary and routing profile

```text
/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server
/home/dino/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv
```

The release requires:

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
```

## Verified 128k capacity command

This configuration processed 120,021 effective prompt tokens without OOM or
context shifting:

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
MODEL=/path/to/qwen35b-mtp.gguf

/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  --model "$MODEL" \
  --host 127.0.0.1 --port 8001 --alias qwen36-35b-a3b --parallel 1 \
  -c 131072 -n 32768 --no-context-shift \
  -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
  --moe-cache-profile /home/dino/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --jinja --no-webui
```

For remote use, replace `--host 127.0.0.1` with `--host 0.0.0.0` and add a
long, private key:

```bash
--api-key 'replace-with-a-long-random-key'
```

`0.0.0.0` is a listen address, not the address to enter in a browser. Remote
clients use the server's Tailscale address.

### Same profile with vision enabled (`mmproj`)

To add image understanding to this profile, load the matching projector with
`--mmproj`. The projector for this model
(`Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf`)
is confirmed present locally; see "Verified and candidate models" below.

**GPU offload (`--mmproj-offload`, the default) fails at this cache/context
size.** With `-ncmoe 33`, cache `160,108`, and 64k context (`-ctk/-ctv q8_0`),
GPU0 (11 GB) is already nearly full before the projector tries to allocate;
the load aborts with `cudaMalloc failed: out of memory` while allocating the
~861 MiB projector buffer, even at a reduced 16k context. This is a VRAM
budget conflict with the hot-expert cache and KV, not a broken projector.

**Verified working command: CPU-side projector (`--no-mmproj-offload`).**
This loads and answers correctly, at the cost of much slower prompt
processing during vision requests (projector compute runs on CPU):

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
MODEL=/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
MMPROJ=/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf

/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  --model "$MODEL" --mmproj "$MMPROJ" --no-mmproj-offload --image-min-tokens 1024 \
  --host 0.0.0.0 --port 8001 --alias qwen36-heretic --parallel 1 \
  --api-key 'replace-with-a-long-random-key' -c 16384 -n 4096 \
  -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
  --moe-cache-profile /home/dino/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --jinja
```

`--image-min-tokens 1024` silences a startup warning specific to this
Qwen-VL projector about grounding-task accuracy; it does not change whether
the model loads.

Verified on 6 September 2026 with the OCR image from `~/pascal-vision-samples`
(`01-street-sign-ocr.jpg`, a stop-sign photo with "WHOA" and "PARKER RANCH
CENTER" text): the model correctly read both text elements in its reasoning
before the 200-token limit cut off the final answer. Result at 16k, one run:

| Measure | Result |
| --- | ---: |
| HTTP code | 200 |
| Prompt tokens | 1546 |
| Prompt processing | 85.47 s; **18.09 t/s** |
| Generation | 200 tokens; **10.81 t/s** |
| MTP acceptance | 119/160 = 74.4% |
| GPU0 free VRAM after load | 470 MiB |
| GPU1 free VRAM after load | 1387 MiB |

The 18 t/s prompt speed reflects CPU-side image encoding, not the text-only
prefill speed; a WebUI-driven test (uploading the same image through the
built-in browser interface instead of a raw `curl` request) has not yet been
tried. GPU-offloaded projector performance at this cache size remains
unverified; it would require either a smaller hot-expert cache, a shorter
context, or freeing more of GPU0's ~2 GiB non-reclaimable baseline usage
before it can be attempted.

### Confirmed working at 64k

The same CPU-side-projector command also loads and answers correctly at the
full 64k context, matching the remote profile originally used to start a
session (`-c 65536 -n 32768 --no-context-shift`, `--host 0.0.0.0`):

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
MODEL=/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
MMPROJ=/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf

/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  --model "$MODEL" --mmproj "$MMPROJ" --no-mmproj-offload --image-min-tokens 1024 \
  --host 0.0.0.0 --port 8001 --alias qwen36-heretic --parallel 1 \
  --api-key 'replace-with-a-long-random-key' -c 65536 -n 32768 --no-context-shift \
  -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
  --moe-cache-profile /home/dino/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --jinja
```

Verified on 6 September 2026 with the counting image from
`~/pascal-vision-samples` (`04-apples-counting.jpg`, three apples): the
model loaded in about 15 seconds and correctly counted three apples. Result
at 64k, one run:

| Measure | Result |
| --- | ---: |
| HTTP code | 200 |
| Prompt tokens | 1103 |
| Prompt processing | 58.22 s; **18.94 t/s** |
| Generation | 150 tokens; **12.41 t/s** |
| MTP acceptance | 97/103 = 94.2% |
| GPU0 used / free VRAM after load | 10846 MiB / 320 MiB |
| GPU1 used / free VRAM after load | 7185 MiB / 923 MiB |

Both GPUs remain within their VRAM budget at 65,536-token capacity with the
CPU-side projector, leaving 320–923 MiB free per device. GPU-offloaded
projector is still expected to fail at 64k for the same reason it fails at
16k (see above); it has not been retried at 64k because it already failed at
the smaller context.

### GPU utilization is asymmetric during vision requests

An `nvidia-smi` sample taken during a 16k vision request (bar-chart image,
`02-bar-chart.png`) showed the hot-expert cache split is not being exercised
evenly while the CPU-side projector runs:

| Phase | GPU0 (1080 Ti) avg. compute utilization | GPU1 (1070) avg. compute utilization |
| --- | ---: | ---: |
| Image encoding + prompt prefill (~0–35s) | ~91% | ~0% |
| Text generation with MoE routing + MTP (~35–70s) | ~95–99% | ~5–12% |

VRAM usage on GPU1 stays at its expected steady level throughout (its share
of the model layers and hot-expert cache is resident, as designed by
`-ts 10,7` and `--moe-cache-slots 160,108`); what is low is *compute*
utilization, not memory occupancy. This means GPU1 holds its assigned data
but receives comparatively little computational work during a vision
request, unlike the more balanced split expected from text-only requests.
The cause has not yet been isolated: it could be specific to how the `mtmd`
multimodal pipeline schedules the image-conditioned prefill, or it could
already be present with text-only prompts under this cache profile. A
controlled text-only comparison at the same settings is open work before
concluding whether this is a vision-specific regression or a pre-existing
property of this cache/split configuration.

### GPU-offloaded projector at 64k: reducing `--moe-cache-slots`

The GPU-offload failure above was reproduced at both 16k and 64k with the
default hot-expert cache split (`--moe-cache-slots 160,108`): the projector's
~861 MiB CUDA buffer does not fit on GPU0 because the MoE expert cache and KV
cache already consume nearly all of the 1080 Ti's 11 GiB at 64k context.

Five quick screening trials (health-check timeout, no full generation) were
run on 6 September 2026 to see whether trimming the GPU0 share of
`--moe-cache-slots` frees enough VRAM for `--mmproj-offload` to succeed at the
full 64k context:

| `--moe-cache-slots` | Result | GPU0 free VRAM after load |
| --- | --- | ---: |
| 160,108 (default) | ❌ OOM (see above) | — |
| 150,100 | ❌ OOM | — |
| 150,108 | ❌ OOM (abort/core dump) | — |
| 145,108 | ❌ OOM (abort/core dump) | — |
| **140,108** | ✅ loads | 189 MiB (very tight) |
| **110,108** | ✅ loads | 1278 MiB |

Findings:

- The GPU0 expert-cache slot count is what matters; GPU1's slot count did not
  need to change (108 works in every successful trial) because GPU1 already
  has headroom under `-ts 10,7`.
- The break-even point on GPU0 is between 140 and 145 slots: 140 succeeds
  with only ~189 MiB free (too risky for production — a larger image or a
  near-full context could still OOM), 145 already fails.
- **Recommended configuration for GPU-offloaded vision at 64k:
  `--moe-cache-slots 130,108`** — a safer margin below the 140-slot cliff
  edge while still caching most of the default 160-slot budget on GPU0.
  This has not yet been verified end-to-end with an actual image request
  (only a fast health-check load was tested); before relying on it, run a
  real vision request and confirm output quality and generation speed are
  acceptable, since reducing cached experts on GPU0 will lower the hot-expert
  hit rate for text/MoE routing and may slow generation compared to the
  default 160,108 split.

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
MODEL=/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
MMPROJ=/home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf

/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  --model "$MODEL" --mmproj "$MMPROJ" --mmproj-offload --image-min-tokens 1024 \
  --host 0.0.0.0 --port 8001 --alias qwen36-heretic --parallel 1 \
  --api-key 'replace-with-a-long-random-key' -c 65536 -n 32768 --no-context-shift \
  -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
  --moe-cache-profile /home/dino/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 130,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --jinja
```

Open work: verify this command end-to-end with a real image request (VRAM
under load, generation throughput, MTP acceptance) and compare its
performance to the CPU-side-projector baseline above to decide which path is
actually preferable for day-to-day vision use at 64k.

### Qualitative comparison: GPU-offloaded vs. CPU-side projector at 64k

Quick summary of the trade-off observed so far between the two working 64k
configurations (numeric benchmark still pending, see "Open work" above):

- **`--mmproj-offload` + `--moe-cache-slots 130,108`** — image encoding runs
  on GPU0, so reasoning/output starts noticeably sooner (faster prefill).
  Cost: fewer cached hot experts on GPU0 (130 vs. the default 160) means a
  lower cache-hit rate during text generation, so reasoning and token
  generation feel slower once they start.
- **`--no-mmproj-offload` + `--moe-cache-slots 160,108`** (documented above
  under "Confirmed working at 64k") — image encoding runs on CPU, so there is
  a longer wait before reasoning starts (~18-19 t/s prompt processing during
  vision requests). Cost is paid up front instead of during generation; the
  full default expert cache stays available for text generation.

In short: GPU-offload shifts the cost from prefill to generation, it does
not remove it. Which is preferable depends on the use case (short prompts
with long generations favor CPU-side; long/heavy image prompts with short
answers may favor GPU-offload). A full numeric comparison (prompt t/s,
generation t/s, MTP acceptance %) on the same test image is still open work.

### Idea to revisit: moving the Linux display to the motherboard's iGPU

The board is an MSI Z97S SLI Krait Edition (MS-7922, Z97 chipset) paired
with an Intel Core i7-4790K, which has an integrated GPU (Intel HD Graphics
4600) wired to the motherboard's own video output. Today the desktop
session's rendering/display overhead sits on one of the two NVIDIA cards
(likely explaining the ~1.8-2.4 GiB of GPU0 VRAM that could not be reclaimed
during earlier OOM troubleshooting in this document). Moving the display
output to the iGPU output on the motherboard would free that VRAM on the
1080 Ti/1070 for `llama-server`, which could be enough on its own to fit the
GPU-offloaded `mmproj` buffer without reducing `--moe-cache-slots` at all.
Not yet tested — planned as a follow-up experiment.

## Measured results

| Configuration | Result | Status |
| --- | ---: | --- |
| Qwen 35B-class variant, 64k, F16 KV, short prompt | 46.13 t/s | three repetitions |
| Qwen 35B-class variant, 64k, Q8 KV, short prompt | 45.24 t/s | three repetitions |
| 120,021 real tokens, 128k, Q8 KV | 153.77 prompt t/s; 23.62 generation t/s | one long run |
| Mobile client over Tailscale, 64k | 48.61 t/s | preliminary, different workload |

The 128k result demonstrates context capacity, not peak decode speed. Short
context measurements demonstrate that native MTP is active and effective.

## Local verification

```bash
curl http://127.0.0.1:8001/health
nvidia-smi
```

Expected health response:

```json
{"status":"ok"}
```

This warning is not an error:

```text
tensor overrides to CPU are used with mmap enabled -
consider using --no-mmap for better performance
```

Keep `mmap` enabled for the baseline. Test `--no-mmap` separately because it
changes load time, RAM use, and page-cache behavior.

## Tailscale remote access

Tailscale creates an encrypted private network between devices. The owner uses
their own account; a friend must use a separate invited account or a
machine-specific share.

On the server:

```bash
sudo tailscale up
tailscale status
tailscale ip -4
```

On a phone or remote PC:

1. Install the Tailscale app.
2. Sign in with the invited account.
3. Enable the VPN.
4. Open `http://TAILSCALE_IP:8001`.

For authenticated API requests, the client must send:

```text
Authorization: Bearer API_KEY
```

Do not use router port forwarding. If UFW is enabled, allow the server only
through Tailscale:

```bash
sudo ufw allow in on tailscale0 to any port 8001 proto tcp
```

Rotate an API key if it appears in chat, screenshots, or shared logs.

## Open WebUI

Open WebUI is a browser frontend for chat, conversation history, documents,
knowledge bases, and configurable tools. It is not a replacement for a coding
agent.

Configure an OpenAI-compatible provider:

```text
URL:     http://TAILSCALE_IP:8001/v1
API key: the server API key
Model:   qwen36-35b-a3b
```

Validate the model directly through `llama-server` first. Then connect Open
WebUI to the same endpoint. A coding-agent setup can use:

```text
Pi Coding Agent -> llama-server
Open WebUI -> llama-server or an OpenAI-compatible agent gateway
```

## Pi Coding Agent

Pi can use an OpenAI-compatible local provider. A cautious starting
configuration is:

```json
{
  "providers": {
    "local": {
      "baseUrl": "http://127.0.0.1:8001/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "qwen36-35b-a3b",
          "name": "Qwen 35B-class MTP",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32768,
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false
          }
        }
      ]
    }
  }
}
```

Validate these behaviors with real tasks:

- structured tool calls;
- file reads and edits;
- controlled shell commands;
- preserved reasoning across turns;
- compliance with a `plan-first` skill;
- final answer separated from reasoning.

The `plan-first` skill is workflow policy, not a model-speed optimization. Test
it in a temporary workspace before using it on important repositories.

## Verified and candidate models

This table tracks which GGUF assets have been run on this hardware and which
are only planned. "Verified" means a local run produced the throughput or
capacity numbers in [BASELINE_LOG.md](BASELINE_LOG.md) or
[BENCHMARKS_AND_QUALITY.md](BENCHMARKS_AND_QUALITY.md); "candidate" means the
file is identified and pinned but has not yet been downloaded and run here.

The portable release installs `pascal-download-model.sh`, a thin wrapper
around [`scripts/download-model.sh`](scripts/download-model.sh) that fetches a
single file from a Hugging Face repository over plain `curl`, pins a
revision, and checks a SHA-256 when one is supplied:

```bash
pascal-download-model.sh REPOSITORY FILENAME OUTPUT [REVISION] [SHA256]
```

It works for any file hosted on a public (non-gated) Hugging Face repository,
which covers every entry below. `MODEL_DIR` is the external model directory
used throughout this guide.

| Model | Role | Status | Repository | Revision | Download command |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-Q4_K_M | Dense chat/coding baseline | Verified (16k, full offload) | `lmstudio-community/Qwen3.8-27B-GGUF` | local copy, not pinned | `pascal-download-model.sh lmstudio-community/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf "$MODEL_DIR/Qwen3.8-27B-Q4_K_M.gguf"` |
| Qwen3.6-35B-A3B-MTP-UD-Q4_K_M | Fast MoE + native MTP | Verified (64k capacity, short-prompt and 60k-token runs) | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` | `pascal-download-model.sh unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d 0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` |
| Qwen3.6-35B-A3B-Q4_K_M | Standard non-MTP MoE | Candidate (replicated `r=1` screening only) | `Infatoshi/Qwen3.6-35B-A3B-GGUF` | `22880479d6dd057b3fab8654afaaf34648fb524e` | `pascal-download-model.sh Infatoshi/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-Q4_K_M.gguf" 22880479d6dd057b3fab8654afaaf34648fb524e 439fcb8266f37a035d2192d0fa773e59b177f379bd2bf90976419cea8c7dbb58` |
| Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M | Uncensored MoE, no MTP | Verified (64k smoke test, routing profile borrowed from MTP-UD) | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF` | `f70be2db155a4192a59c559ece01572f3cd508ab` | `pascal-download-model.sh llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf" f70be2db155a4192a59c559ece01572f3cd508ab b82f9f2155eb9c07139d48cc3a37880cf9d0edcca345ef6ad829b62941ccbb82` |
| Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M | Uncensored MoE + native MTP; current local test asset | **Verified, locally present** (64k short-prompt, 120k-token 128k capacity test) | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` | `fa96dc904fe4dafb415ca707afab30aee2d2e703` | `pascal-download-model.sh llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf" fa96dc904fe4dafb415ca707afab30aee2d2e703 fc89d92377b27fe0f80eb683a5105d0921234c24f7c1d70ecc2356ddf994d781` |
| `mmproj-BF16.gguf` (Unsloth MTP projector) | Vision projector for MTP-UD | Candidate, untested | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` | `pascal-download-model.sh unsloth/Qwen3.6-35B-A3B-MTP-GGUF mmproj-BF16.gguf "$MODEL_DIR/mmproj-BF16.gguf" 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d da63cb47a76763c712393f8a017070188a304fa39f8aeea6edc629ed7b975cfa` |
| `Qwen3.6-35B-A3B-mmproj-BF16.gguf` (standard projector) | Vision projector for the standard model | Candidate, untested | `Infatoshi/Qwen3.6-35B-A3B-GGUF` | `22880479d6dd057b3fab8654afaaf34648fb524e` | `pascal-download-model.sh Infatoshi/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-mmproj-BF16.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-mmproj-BF16.gguf" 22880479d6dd057b3fab8654afaaf34648fb524e 37904b50d3453905e7ef5a4ada6a148bc87ff3923fe1d6c39ae028e53742ff13` |
| `Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf` | Vision projector for Heretic (no MTP) | Candidate, untested | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF` | `f70be2db155a4192a59c559ece01572f3cd508ab` | `pascal-download-model.sh llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf" f70be2db155a4192a59c559ece01572f3cd508ab 1c625f05cd52e90abc76a5f756226c3a5fe279593379c22f6c6846c970a0cd18` |
| `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf` | Vision projector for the current local test asset | **Verified, locally present** (downloaded and hash-checked; not yet performance-tested) | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF` | `fa96dc904fe4dafb415ca707afab30aee2d2e703` | `pascal-download-model.sh llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf" fa96dc904fe4dafb415ca707afab30aee2d2e703 d6050bb82a0187e1b0655f1c5daef60c9479273fbd7402ff25d3137df2e071ce` |

Expected SHA-256 hashes match [BENCHMARKS_AND_QUALITY.md](BENCHMARKS_AND_QUALITY.md);
`pascal-download-model.sh` checks them automatically when given as the fifth
argument, so a mismatch aborts before the file is kept. The Qwen3.8-27B dense
baseline has no pinned revision or published hash in this repository yet; add
them once the exact upstream source is confirmed. The `hf` CLI documented in
[BENCHMARKS_AND_QUALITY.md](BENCHMARKS_AND_QUALITY.md) remains the recommended
path when several files must be fetched together in one call; the table above
uses the single-file `pascal-download-model.sh` fallback bundled with the
release for hosts without the `hf` tool installed.

### Local presence check (Heretic MTP-preserved test asset)

The model file and its projector are now confirmed present under
`/home/dino/proj/genAI/models/` and hash-verified against the values above:

```bash
sha256sum \
  /home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf \
  /home/dino/proj/genAI/models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf
```

The projector was missing locally before this check and was fetched with the
command from the table above, then verified against
`d6050bb82a0187e1b0655f1c5daef60c9479273fbd7402ff25d3137df2e071ce`.

## Vision use case: image smoke test

This is the first demonstrated multimodal use case on this hardware: loading
the current local test model (Heretic Native-MTP-Preserved) with its
projector at 16k context and asking it about a small, fixed set of test
images. It exercises the GPU-offloaded projector path end to end, not only
text generation.

A small five-image set for exactly this purpose lives outside the repository
at `~/pascal-vision-samples` (four public-domain/CC photos from Wikimedia
Commons covering OCR, chart reading, object/spatial description, and
counting, plus one synthetic UI screenshot), with its own `README.md`
describing the source, license, and expected answer for each image. This is a
functional smoke test, not a scored benchmark; standard vision benchmarks for
a fuller evaluation include OCRBench and TextVQA (OCR), ChartQA and DocVQA
(charts/documents), MME and MMBench (general perception), POPE
(hallucination), MathVista (visual math reasoning), and RealWorldQA (natural
photos). None of these are run automatically here; they would each require
downloading their own dataset and scoring harness.

```bash
export LD_LIBRARY_PATH=/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
MODEL_DIR=/home/dino/proj/genAI/models
MODEL="$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf"
MMPROJ="$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf"

/home/dino/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin/llama-server \
  --model "$MODEL" --mmproj "$MMPROJ" --mmproj-offload \
  --host 0.0.0.0 --port 8002 --alias qwen36-35b-a3b-vision --parallel 1 \
  -c 16384 -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk f16 -ctv f16 -b 512 -ub 512 \
  --moe-cache-profile moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning off --temp 0 --seed 123 \
  --jinja
```

Send each image with an OpenAI-compatible chat request, using the matching
question from `~/pascal-vision-samples/README.md`. Example with the OCR image:

```bash
IMAGE_B64=$(base64 -w0 ~/pascal-vision-samples/01-street-sign-ocr.jpg)
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen36-35b-a3b-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What text is written on the sign in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMAGE_B64"'"}}
      ]
    }],
    "max_tokens": 256
  }'
```

Repeat for the chart, cat, apples, and synthetic-screenshot images, swapping
the MIME type (`image/png` for the two PNGs) and question accordingly.

Record, separately from text-only measurements: image-encoding/prompt time,
generation throughput, peak VRAM on both GPUs, MTP acceptance, and whether
each answer matches the expected answer noted in the image set's `README.md`.
This use case is a smoke test, not yet the controlled vision A/B matrix
described below; that matrix (Unsloth MTP, standard, Heretic, and
uncensored-MTP pairs on the same fixed images) remains open work.

## Future test matrix

### Correctness

1. Deterministic answers at temperature zero.
2. Valid tool-call JSON.
3. Tool call -> tool result -> final answer.
4. Code generation with tests and assertions.
5. Retrieval of facts inserted at 32k, 64k, and 120k.
6. Reasoning preservation across turns.

### Performance

1. Identical prompt and output length from Firefox and mobile.
2. Qwen 35B-class MTP variants with identical workloads.
3. Short context, 64k allocation, and 120k real population.
4. `mmap` versus `--no-mmap`.
5. `n-max=1,2,3` with cache and tensor split unchanged.

Every result should record model, hash, command, populated context, KV type,
cache, MTP acceptance, throughput, VRAM, RAM, swap, temperatures, and
correctness. Use `r=1` for screening and at least three repetitions for a
baseline.

## Graphical diagnostics

```bash
nvtop
btop
watch -n 1 nvidia-smi
watch -n 1 sensors
```

Do not run multiple CUDA workloads concurrently. A desktop GUI is compatible
with the server, but browsers and Electron applications can consume the VRAM
margin needed by 128k compute buffers.

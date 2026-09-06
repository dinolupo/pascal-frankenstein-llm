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

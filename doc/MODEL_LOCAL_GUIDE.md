# Running Qwen 35B-Class MoE Models Locally

Operational quick reference for the dual Pascal GPU setup.
Complete experiment log and methodology: [BASELINE_LOG.md](BASELINE_LOG.md).

## Verified setup

| Component | Value |
| --- | --- |
| Model | Qwen3.6-35B-A3B Heretic Native-MTP-Preserved Q4_K_M |
| Binaries | `~/.local/share/pascal-frankenstein-llm/bin/` (installed via release archive) |
| Routing profile | `~/.local/share/pascal-frankenstein-llm/moe-traces/qwen36-35b-mtp-merged.csv` |
| INI preset | `~/.local/share/pascal-frankenstein-llm/config/llama-models.ini` |
| Startup script | `genai-llamacpp.sh` (in `~/.local/bin`) |
| Monitoring | `genai-start.sh` — opens a tmux session with btop, nvtop, nvidia-smi, and the server |

## Starting the server

The fastest path:

```bash
genai-llamacpp.sh
```

What it runs:

```bash
export LLAMA_IP=0.0.0.0
export LLAMA_PORT=8001
export LLAMA_API_KEY=changethisverylongboringkey
install_dir="$HOME/.local/share/pascal-frankenstein-llm"

llama-server \
  --models-preset "$install_dir/config/llama-models.ini" \
  --host "$LLAMA_IP" \
  --port "$LLAMA_PORT" \
  --api-key "$LLAMA_API_KEY" \
  --models-max 1
```

The INI preset handles all model parameters. Two sections are defined:

### Text / coding profile (128k context)

Key parameters in `[Qwen3.6-35B-heretic]`:

```
c = 131072          ; 128k token context
ncmoe = 33          ; hybrid residency: layers 0-32 cold in RAM, 33-39 fully on GPU
ts = 10,7           ; asymmetric tensor split: 1080 Ti / 1070
moe-cache-slots = 160,108   ; hot expert slots per GPU
moe-cache-profile = .../qwen36-35b-mtp-merged.csv
spec-type = draft-mtp
spec-draft-n-max = 2
ctk = q8_0
ctv = q8_0
```

### Vision profile (64k context)

Key differences in `[Qwen3.6-35B-vision]` vs the text profile:

```
mmproj = MMPROJ_PATH        ; BF16 projector file
mmproj-offload = true       ; GPU offload for the projector
c = 65536                   ; 64k context (reduced to leave VRAM for the projector)
moe-cache-slots = 130,108   ; smaller CUDA0 cache to accommodate the projector
```

To switch to the vision model, set `"model": "Qwen3.6-35B-vision"` (or the
configured alias) in the API request. The text model must be unloaded first
(or use `load-on-startup = false` on the vision section, which is the default).

## Chat template override (workaround for reasoning loops)

If a client agent enters a reasoning loop, activate the fixed Jinja template
from [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates).
Copy the `.jinja` file to `INSTALL_DIR/config/chat_template.jinja`, then
uncomment these two lines in the active INI preset section:

```ini
chat-template-file = INSTALL_DIR/config/chat_template.jinja
reasoning-format   = deepseek
```

Both lines are present but commented in `config/llama-models.ini.example`.

## Verifying the installation

```bash
# Check binaries and GPU discovery
~/.local/share/pascal-frankenstein-llm/scripts/verify-install.sh

# Smoke test (no GPU required, uses stub executables)
~/.local/share/pascal-frankenstein-llm/scripts/test-installer.sh

# Quick API check after server start
curl http://127.0.0.1:8001/v1/models
```

## Generating a replacement routing profile

Use `llama-moe-trace` on representative prompts, then concatenate and validate:

```bash
install_dir="$HOME/.local/share/pascal-frankenstein-llm"
MOE_TRACE_OUT=trace-a.csv "$install_dir/bin/llama-moe-trace" \
  -m /path/to/model.gguf -ngl 99 -ncmoe 33 -fa on \
  -p "representative prompt" -n 512
```

A profile is model-specific: do not use the Qwen3.6 profile with Qwen3.8 or
another architecture. Concatenate multiple traces (omitting duplicate headers)
and validate before replacing the packaged profile.

## Remote access via Tailscale

The server binds to `0.0.0.0:8001`. On the remote machine, point your client
at `http://<tailscale-ip>:8001`. No extra configuration needed; Tailscale
handles the encrypted tunnel. A preliminary remote test recorded 48.6 t/s with
73.1% MTP acceptance (observations only, not a controlled baseline).

## Open WebUI

```bash
open-webui-start.sh   # or: uvx --python 3.12 open-webui serve
```

Default URL: `http://chat.dinuxnode.com` (local LAN alias).
Set the OpenAI-compatible endpoint to `http://127.0.0.1:8001/v1` in Open WebUI settings.

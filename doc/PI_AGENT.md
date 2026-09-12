# Configuring PI-Agent to work with my configuration

## Installation and Setup

[PI Agent](https://pi.dev/)

[Reddit topic for qwen3.6-35B](https://www.reddit.com/r/LocalLLaMA/comments/1stjwg5/been_using_pi_coding_agent_with_local_qwen36_35b/)

[Pi.dev llama.cpp configuration](https://pi.dev/docs/latest/llama-cpp)

[nono.sh Sandbox](https://nono.sh/)

## Run

> env variables

```sh
export LLAMA_IP=0.0.0.0
export LLAMA_PORT=8001
export LLAMA_API_KEY=changethisverylongboringkey
export install_dir="$HOME/.local/share/pascal-frankenstein-llm"
```

> Run with INI configuration (SANDBOX with `nono.sh` current directory)

```sh
nono run --profile nolabs-ai/pi --allow . -- pi
```

The official `nolabs-ai/pi` profile assumes Pi's binary is in a standard
location (system Node or `~/.nvm`). If `pi.dev/install.sh` bootstrapped its
own Node because none was present, Pi lives under
`~/.local/share/pi-node/...` instead, a path the profile doesn't know. The
sandbox then fails with exit code 127 ("directory is not readable"). Fix once
per machine:

```sh
nono profile init pi --extends nolabs-ai/pi --full
```

Add the standalone Node dir to `~/.config/nono/profiles/pi.json`, in
`filesystem.read`:

```json
"filesystem": {
  "read": ["$HOME/.local/share/pi-node"]
}
```

Installed extensions (e.g. `pi-token-speed`, `pi-search-on-your-browser`) are
compiled on the fly by `jiti` into `/tmp/jiti`, another path the base profile
doesn't grant. Without it, `pi` starts but silently fails to load every
extension (`EACCES ... /tmp/jiti/...`). Add it too, with read+write since
jiti writes the compiled file then reads it back:

```json
"filesystem": {
  "allow": ["/tmp/jiti"],
  "read": ["$HOME/.local/share/pi-node"]
}
```

Validate and run with the custom profile:

```sh
nono profile validate pi
nono run --profile pi --allow-cwd -- pi
```

Skip this if `command -v pi` resolves outside `~/.local/share/pi-node` on
your machine.

> Run with INI configuration (DANGEROUS FULL SYSTEM CONTROL)

```sh
llama-server \
  --models-preset "$install_dir/config/llama-models.ini" \
  --host $LLAMA_IP \
  --port $LLAMA_PORT \
  --api-key "$LLAMA_API_KEY"
```

> sample INI

```ini
; Router preset for the Pascal dual-GPU Qwen3.6-35B-A3B profile.

version = 1

[Qwen3.6-35B-heretic]
model = /home/dino/proj/genAI/models/Qwen3.6-35B-her/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
alias = Qwen36 her
load-on-startup = true
parallel = 1
c = 131072
n = 32768
no-context-shift = true
ngl = 99
ncmoe = 33
ts = 10,7
moe-cache-slots = 160,108
moe-cache-profile = /home/dino/.local/share/pascal-frankenstein-llm/moe-traces/qwen36-35b-mtp-merged.csv
fa = on
ctk = q8_0
ctv = q8_0
b = 512
ub = 512
spec-type = draft-mtp
spec-draft-n-max = 2
reasoning = on
reasoning-preserve = true
temp = 0.6
top-p = 0.95
top-k = 20
repeat-penalty = 1.0
presence-penalty = 0.0
jinja = true
no-ui = true

[Qwen3.6-35B-vision]
model = /home/dino/proj/genAI/models/Qwen3.6-35B-her/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
mmproj = /home/dino/proj/genAI/models/Qwen3.6-35B-her/mmproj-BF16.gguf
alias = Qwen36 her vision
load-on-startup = false
parallel = 1
c = 65536
ngl = 99
ncmoe = 33
ts = 10,7
moe-cache-slots = 130,108
moe-cache-profile = /home/dino/.local/share/pascal-frankenstein-llm/moe-traces/qwen36-35b-mtp-merged.csv
mmproj-offload = true
image-min-tokens = 1024
fa = on
ctk = q8_0
ctv = q8_0
b = 512
ub = 512
jinja = true
no-ui = true
```

# Configuring PI-Agent to work with my configuration

## Installation and Setup

[PI Agent](https://pi.dev/)

[Reddit topic for qwen3.6-35B](https://www.reddit.com/r/LocalLLaMA/comments/1stjwg5/been_using_pi_coding_agent_with_local_qwen36_35b/)

[Pi.dev llama.cpp configuration](https://pi.dev/docs/latest/llama-cpp)



## Run

> env variables

```sh
export LD_LIBRARY_PATH=$HOME/proj/pascal-frankenstein-llm-v0.1.0-linux-x86_64-cuda12-sm61/bin
export MODELS_DIR=$HOME/proj/genAI/models
export MODEL=$MODELS_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf
export LLAMA_IP=127.0.0.1
export LLAMA_PORT=8001
export LLAMA_BASE_URL=http://$LLAMA_IP:$LLAMA_PORT
export LLAMA_API_KEY=changethisverylongboringkey
```

> Run with INI configuration

```sh
$LD_LIBRARY_PATH/llama-server \
  --host $LLAMA_IP \
  --port $LLAMA_PORT \
  --api-key "$LLAMA_API_KEY" \
  --moe-cache-slots 160,108 \
  -ts 10,7 \
  --models-preset $HOME/proj/genAI/llama-models.ini
```

> Run in router mode

$LD_LIBRARY_PATH/llama-server \
  --models-dir "$MODELS_DIR" \
  --parallel 1 \
  --host $LLAMA_IP \
  --port $LLAMA_PORT \
  --api-key "$LLAMA_API_KEY" \
  -c 131072 -n 32768 \
  --no-context-shift \
  -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk q8_0 -ctv q8_0 -b 512 -ub 512 \
  --moe-cache-profile $HOME/proj/pascal-frankenstein-llm.worktrees/progetto-situazione-attuale/moe-traces/qwen36-35b-mtp-merged.csv \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning on --reasoning-preserve \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --jinja \
  --no-ui
```

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
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": true
          }
        }
      ]
    }
  }
}
```

        "name": "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M",


Validate these behaviors with real tasks:

- structured tool calls;
- file reads and edits;
- controlled shell commands;
- preserved reasoning across turns;
- compliance with a `plan-first` skill;
- final answer separated from reasoning.

The `plan-first` skill is workflow policy, not a model-speed optimization. Test
it in a temporary workspace before using it on important repositories.

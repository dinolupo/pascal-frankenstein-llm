#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
config_file=${QWEN_CONFIG:-"${XDG_CONFIG_HOME:-"${HOME}/.config"}/pascal-frankenstein-llm/qwen.env"}

for required in llama-cli llama-server llama-bench llama-moe-trace; do
    [[ -x "$root_dir/bin/$required" ]] || {
        printf 'missing executable: %s\n' "$root_dir/bin/$required" >&2
        exit 1
    }
done

[[ -f "$config_file" ]] || {
    printf 'missing configuration: %s\n' "$config_file" >&2
    exit 1
}
# shellcheck disable=SC1090
source "$config_file"

if [[ -n "${MODEL:-}" && ! -f "$MODEL" ]]; then
    printf 'configured MODEL does not exist: %s\n' "$MODEL" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    printf 'warning: nvidia-smi not found; GPU checks were skipped\n' >&2
fi

printf 'Installation looks complete: %s\n' "$root_dir"

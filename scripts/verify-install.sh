#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)

for required in llama-cli llama-server llama-bench llama-moe-trace; do
    [[ -x "$root_dir/bin/$required" ]] || {
        printf 'missing executable: %s\n' "$root_dir/bin/$required" >&2
        exit 1
    }
done

[[ -f "$root_dir/config/llama-models.ini.example" ]] || {
    printf 'missing router preset template: %s\n' "$root_dir/config/llama-models.ini.example" >&2
    exit 1
}

[[ -f "$root_dir/moe-traces/qwen36-35b-mtp-merged.csv" ]] || {
    printf 'missing default routing profile: %s\n' "$root_dir/moe-traces/qwen36-35b-mtp-merged.csv" >&2
    exit 1
}

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    printf 'warning: nvidia-smi not found; GPU checks were skipped\n' >&2
fi

printf 'Installation looks complete: %s\n' "$root_dir"
printf 'Edit the router preset before starting llama-server: %s\n' \
    "$root_dir/config/llama-models.ini.example"

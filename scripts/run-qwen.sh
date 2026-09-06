#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
config_file=${QWEN_CONFIG:-"${XDG_CONFIG_HOME:-"${HOME}/.config"}/pascal-frankenstein-llm/qwen.env"}

[[ -f "$config_file" ]] || {
    printf 'missing configuration: %s\n' "$config_file" >&2
    exit 1
}
# The file is user-owned local configuration, not repository input.
# shellcheck disable=SC1090
source "$config_file"

mode=${1:-64k}
case "$mode" in
    64k) context_size=${CONTEXT_SIZE:-65536} ;;
    128k) context_size=131072 ;;
    remote) context_size=${CONTEXT_SIZE:-65536}; HOST=${REMOTE_HOST:-0.0.0.0} ;;
    *) printf 'mode must be 64k, 128k, or remote\n' >&2; exit 2 ;;
esac

[[ -n "${MODEL:-}" && -f "$MODEL" ]] || {
    printf 'set MODEL to an existing GGUF file in %s\n' "$config_file" >&2
    exit 1
}

export LD_LIBRARY_PATH="$root_dir/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
args=(
    --model "$MODEL"
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8001}"
    --alias "${MODEL_ALIAS:-qwen35b-local}" --parallel 1
    -c "$context_size" -n "${MAX_TOKENS:-32768}" --no-context-shift
    -ngl "${GPU_LAYERS:-99}" -ncmoe "${MOE_CPU_LAYERS:-33}"
    -ts "${TENSOR_SPLIT:-10,7}" -fa on
    -ctk "${KV_TYPE:-q8_0}" -ctv "${KV_TYPE:-q8_0}"
    -b "${BATCH_SIZE:-512}" -ub "${UBATCH_SIZE:-512}"
    --moe-cache-slots "${MOE_CACHE_SLOTS:-160,108}"
    --spec-type draft-mtp --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-2}"
    --reasoning "${REASONING:-on}"
    --temp "${TEMPERATURE:-0.6}" --top-p "${TOP_P:-0.95}" --top-k "${TOP_K:-20}"
    --repeat-penalty 1.0 --presence-penalty 0.0
)

if [[ -n "${MOE_CACHE_PROFILE:-}" ]]; then
    args+=(--moe-cache-profile "$MOE_CACHE_PROFILE")
fi
if [[ "${REASONING_PRESERVE:-0}" == 1 ]]; then
    args+=(--reasoning-preserve)
fi
if [[ "${JINJA:-0}" == 1 ]]; then
    args+=(--jinja)
fi
if [[ -n "${API_KEY:-}" ]]; then
    args+=(--api-key "$API_KEY")
fi

exec "$root_dir/bin/llama-server" "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s PACKAGE_DIR [INSTALL_DIR]\n' "$0" >&2
    printf '       %s ARCHIVE.tar.gz [INSTALL_DIR]\n' "$0" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

source_path=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
install_dir=${2:-"${HOME}/.local/share/pascal-frankenstein-llm"}
config_dir=${XDG_CONFIG_HOME:-"${HOME}/.config"}/pascal-frankenstein-llm
bin_dir=${HOME}/.local/bin
temporary_dir=

cleanup() {
    if [[ -n "$temporary_dir" ]]; then
        rm -rf "$temporary_dir"
    fi
}
trap cleanup EXIT

if [[ -f "$source_path" ]]; then
    case "$source_path" in
        *.tar.gz|*.tgz)
            temporary_dir=$(mktemp -d)
            tar -xzf "$source_path" -C "$temporary_dir"
            source_path=$(find "$temporary_dir" -mindepth 1 -maxdepth 1 -type d -print -quit)
            ;;
        *)
            printf 'unsupported package: %s\n' "$source_path" >&2
            exit 1
            ;;
    esac
fi

if [[ ! -d "$source_path/bin" || ! -x "$source_path/bin/llama-server" ]]; then
    printf 'package must contain an executable bin/llama-server: %s\n' "$source_path" >&2
    exit 1
fi

profile="$source_path/moe-traces/qwen36-35b-mtp-merged.csv"
if [[ ! -f "$profile" ]]; then
    printf 'package is missing the default MoE routing profile: %s\n' "$profile" >&2
    exit 1
fi

mkdir -p "$install_dir" "$config_dir" "$bin_dir"
cp -a "$source_path"/. "$install_dir"/

if [[ ! -e "$config_dir/qwen.env" ]]; then
    cp "$install_dir/config/qwen.env.example" "$config_dir/qwen.env"
    sed -i "s|^MOE_CACHE_PROFILE=.*|MOE_CACHE_PROFILE=${install_dir}/moe-traces/qwen36-35b-mtp-merged.csv|" \
        "$config_dir/qwen.env"
fi

if [[ ! -e "$install_dir/config/llama-models.ini" ]]; then
    cp "$install_dir/config/llama-models.ini.example" \
        "$install_dir/config/llama-models.ini"
fi

for launcher in run-qwen.sh verify-install.sh download-model.sh; do
    ln -sfn "$install_dir/scripts/$launcher" "$bin_dir/pascal-$launcher"
done

printf 'Installed to %s\n' "$install_dir"
printf 'Configuration: %s/qwen.env\n' "$config_dir"
printf 'Launch with: pascal-run-qwen.sh 64k\n'
printf 'Ensure %s is in PATH.\n' "$bin_dir"

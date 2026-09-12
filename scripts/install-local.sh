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

mkdir -p "$install_dir"
cp -a "$source_path"/. "$install_dir"/

printf 'Installed to %s\n' "$install_dir"
printf 'Router template: %s/config/llama-models.ini.example\n' "$install_dir"
printf 'Routing profile: %s/moe-traces/qwen36-35b-mtp-merged.csv\n' "$install_dir"
printf 'Binary: %s/bin/llama-server\n' "$install_dir"

#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s REPOSITORY FILENAME OUTPUT [REVISION] [SHA256]\n' "$0" >&2
}

if [[ $# -lt 3 || $# -gt 5 ]]; then
    usage
    exit 2
fi

repository=$1
filename=$2
output=$3
revision=${4:-main}
expected_sha256=${5:-}

command -v curl >/dev/null 2>&1 || {
    printf 'curl is required\n' >&2
    exit 1
}

mkdir -p "$(dirname "$output")"
url="https://huggingface.co/${repository}/resolve/${revision}/${filename}?download=true"
temporary="${output}.part"
trap 'rm -f "$temporary"' EXIT

printf 'Downloading %s@%s/%s\n' "$repository" "$revision" "$filename"
curl --fail --location --progress-bar "$url" --output "$temporary"

if [[ -n "$expected_sha256" ]]; then
    actual_sha256=$(sha256sum "$temporary" | awk '{print $1}')
    [[ "$actual_sha256" == "$expected_sha256" ]] || {
        printf 'SHA-256 mismatch: expected %s, got %s\n' "$expected_sha256" "$actual_sha256" >&2
        exit 1
    }
fi

mv "$temporary" "$output"
printf 'Saved %s\n' "$output"

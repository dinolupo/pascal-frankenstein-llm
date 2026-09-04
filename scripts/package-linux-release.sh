#!/usr/bin/env bash
# Package an already-built Pascal CUDA release. This script does not compile.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s BUILD_DIR [VERSION]\n' "$0" >&2
    exit 2
fi

build_dir=$(cd "$1" && pwd)
version=${2:-v0.1.0}
repo_root=$(cd "$(dirname "$0")/.." && pwd)
source_root=$(cd "$build_dir/.." && pwd)
bin_dir="$build_dir/bin"
dist_dir="$repo_root/dist"
package_name="pascal-frankenstein-llm-${version}-linux-x86_64-cuda12-sm61"
archive="$dist_dir/${package_name}.tar.gz"

for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    if [[ ! -x "$bin_dir/$executable" ]]; then
        printf 'missing executable: %s\n' "$bin_dir/$executable" >&2
        exit 1
    fi
done

if ! git -C "$source_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'BUILD_DIR must be inside a Git checkout of the fork\n' >&2
    exit 1
fi

mkdir -p "$dist_dir"
stage_dir=$(mktemp -d "$dist_dir/.package.XXXXXX")
trap 'rm -rf "$stage_dir"' EXIT
package_dir="$stage_dir/$package_name"
mkdir -p "$package_dir/bin"

cp -a "$bin_dir"/llama-cli "$bin_dir"/llama-server \
    "$bin_dir"/llama-bench "$bin_dir"/llama-moe-trace "$package_dir/bin/"
cp -a "$bin_dir"/lib*.so* "$package_dir/bin/"

fork_commit=$(git -C "$source_root" rev-parse HEAD)
fork_branch=$(git -C "$source_root" branch --show-current)
cuda_architecture=$(sed -nE 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=(.*)$/\1/p' \
    "$build_dir/CMakeCache.txt" | head -n 1)
cuda_compiler=$(sed -nE 's/^CMAKE_CUDA_COMPILER:[^=]*=(.*)$/\1/p' \
    "$build_dir/CMakeCache.txt" | head -n 1)

{
    printf 'Project: Pascal Frankenstein LLM\n'
    printf 'Fork: dinolupo/llama.cpp\n'
    printf 'Fork branch: %s\n' "$fork_branch"
    printf 'Fork commit: %s\n' "$fork_commit"
    printf 'Upstream base: thecodacus/llama.cpp perf d927e7dc1\n'
    printf 'Build type: Release\n'
    printf 'CUDA compiler: %s\n' "$cuda_compiler"
    printf 'CUDA architecture: sm_%s\n' "$cuda_architecture"
    printf 'Platform: Linux x86_64\n'
} > "$package_dir/BUILD_INFO.txt"

{
    printf 'Requirements:\n'
    printf '%s\n' '  - Linux x86_64 and an NVIDIA driver compatible with CUDA 12.'
    printf '%s\n' '  - CUDA 12 runtime libraries available on the host.'
    printf '%s\n' '  - NVIDIA Pascal sm_61 target; no GGUF model weights are included.'
    printf '\nUsage after extraction:\n'
    printf '%s\n' '  export LD_LIBRARY_PATH="$PWD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
    printf '%s\n' '  ./bin/llama-server --help'
} > "$package_dir/README.txt"

for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    if LD_LIBRARY_PATH="$package_dir/bin" ldd "$package_dir/bin/$executable" | grep -q 'not found'; then
        printf 'unresolved dependency in %s\n' "$executable" >&2
        exit 1
    fi
done

(
    cd "$package_dir"
    find bin -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum BUILD_INFO.txt README.txt >> SHA256SUMS
)

tar -C "$stage_dir" -czf "$archive" "$package_name"
(
    cd "$dist_dir"
    sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)
printf 'created %s\n' "$archive"
printf 'created %s\n' "${archive}.sha256"

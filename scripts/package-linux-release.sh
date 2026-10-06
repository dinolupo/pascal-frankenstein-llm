#!/usr/bin/env bash
# Package an already-built Pascal CUDA release. This script does not compile.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s BUILD_DIR [VERSION]\n' "$0" >&2
    exit 2
fi

build_dir=$(cd "$1" && pwd)
version=${2:-v0.1.0}
if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]]; then
    printf 'invalid release version: %s\n' "$version" >&2
    exit 1
fi
repo_root=$(cd "$(dirname "$0")/.." && pwd)
source_root=${FORK_SOURCE_ROOT:-$(cd "$build_dir/.." && pwd)}
bin_dir="$build_dir/bin"
dist_dir="$repo_root/dist"
package_name="pascal-frankenstein-llm-${version}-linux-x86_64-cuda12-sm61"
archive="$dist_dir/${package_name}.tar.gz"
release_notes="$repo_root/doc/RELEASE_NOTES_${version}.md"
if [[ ! -f "$release_notes" ]]; then
    printf 'missing release notes: %s\n' "$release_notes" >&2
    exit 1
fi
if [[ -e "$archive" || -e "${archive}.sha256" ]]; then
    printf 'release artifacts already exist: %s\n' "$archive" >&2
    exit 1
fi

for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    if [[ ! -x "$bin_dir/$executable" ]]; then
        printf 'missing executable: %s\n' "$bin_dir/$executable" >&2
        exit 1
    fi
done

if ! git -C "$source_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'set FORK_SOURCE_ROOT to the llama.cpp fork checkout\n' >&2
    exit 1
fi
if [[ -n "$(git -C "$source_root" status --porcelain)" ]]; then
    printf 'fork checkout must be clean before packaging\n' >&2
    exit 1
fi
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
    printf 'project checkout must be clean before packaging\n' >&2
    exit 1
fi
fork_commit=$(git -C "$source_root" rev-parse HEAD)
for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    binary_version=$(env -u GGML_CUDA_P2P LD_LIBRARY_PATH="$bin_dir" \
        "$bin_dir/$executable" --version 2>&1)
    embedded_commit=$(printf '%s\n' "$binary_version" |
        sed -nE 's/^version:.*commit ([0-9a-f]{7,40})\).*/\1/p')
    if [[ -z "$embedded_commit" ]] ||
       [[ "$(git -C "$source_root" rev-parse "${embedded_commit}^{commit}")" != "$fork_commit" ]]; then
        printf 'binary/source mismatch: %s\n%s\n' "$executable" "$binary_version" >&2
        exit 1
    fi
done

mkdir -p "$dist_dir"
stage_dir=$(mktemp -d "$dist_dir/.package.XXXXXX")
trap 'rm -rf "$stage_dir"' EXIT
package_dir="$stage_dir/$package_name"
mkdir -p "$package_dir/bin"
mkdir -p "$package_dir/config" "$package_dir/scripts" "$package_dir/moe-traces"
mkdir -p "$package_dir/doc" "$package_dir/licenses"

cp -a "$bin_dir"/llama-cli "$bin_dir"/llama-server \
    "$bin_dir"/llama-bench "$bin_dir"/llama-moe-trace "$package_dir/bin/"
cp -a "$bin_dir"/lib*.so* "$package_dir/bin/"
cp -a "$repo_root/config/llama-models.ini.example" "$package_dir/config/"
cp -a "$repo_root/scripts/download-model.sh" \
    "$repo_root/scripts/install-local.sh" \
    "$repo_root/scripts/verify-install.sh" "$package_dir/scripts/"
cp -a "$repo_root/scripts"/{benchmark_stress.py,run-integration-gate.py,nvidia_power.py,test_benchmark_stress.py,test_nvidia_power.py,test-installer.sh,PERFORMANCE.md} \
    "$package_dir/scripts/"
cp -a "$repo_root/README.md" "$package_dir/"
cp -a "$repo_root/LICENSE" "$package_dir/licenses/LICENSE-project"
cp -a "$source_root/LICENSE" "$package_dir/licenses/LICENSE-llama.cpp"
cp -a "$source_root/README.md" "$package_dir/doc/LLAMA_FORK_README.md"
cp -a "$repo_root/doc/"*.md "$package_dir/doc/"
cp -a "$repo_root/moe-traces"/qwen36-35b-merged.csv \
    "$repo_root/moe-traces"/qwen36-35b-mtp-merged.csv "$package_dir/moe-traces/"

fork_branch=$(git -C "$source_root" branch --show-current)
cuda_architecture=unknown
cuda_compiler=unknown
if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    cuda_architecture=$(sed -nE 's/^CMAKE_CUDA_ARCHITECTURES:[^=]*=(.*)$/\1/p' \
        "$build_dir/CMakeCache.txt" | head -n 1)
    cuda_compiler=$(sed -nE 's/^CMAKE_CUDA_COMPILER:[^=]*=(.*)$/\1/p' \
        "$build_dir/CMakeCache.txt" | head -n 1)
fi
if [[ "$cuda_architecture" != 61 || "$cuda_compiler" != */cuda-12.9/* ]]; then
    printf 'expected CUDA Toolkit 12.9 and architecture 61, got %s / %s\n' \
        "$cuda_compiler" "$cuda_architecture" >&2
    exit 1
fi

{
    printf 'Project: Pascal Frankenstein LLM\n'
    printf 'Fork: dinolupo/llama.cpp\n'
    printf 'Fork branch: %s\n' "$fork_branch"
    printf 'Fork commit: %s\n' "$fork_commit"
    printf 'Project commit: %s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'Original feature lineage: thecodacus/llama.cpp perf d927e7dc1\n'
    if git -C "$source_root" rev-parse --verify 'upstream-v0.6.0^{commit}' >/dev/null 2>&1 &&
       git -C "$source_root" merge-base --is-ancestor upstream-v0.6.0 "$fork_commit"; then
        printf 'Official upstream base: llama.cpp v0.6.0 %s\n' \
            "$(git -C "$source_root" rev-parse 'upstream-v0.6.0^{commit}')"
    fi
    sed -nE 's/^CMAKE_BUILD_TYPE:[^=]*=(.*)$/Build type: \1/p' "$build_dir/CMakeCache.txt"
    printf 'CUDA compiler: %s\n' "$cuda_compiler"
    printf 'CUDA architecture: sm_%s\n' "$cuda_architecture"
    sed -nE 's/^(GGML_NATIVE|LLAMA_BUILD_TESTS|CMAKE_CXX_COMPILER):[^=]*=(.*)$/\1: \2/p' \
        "$build_dir/CMakeCache.txt"
    printf 'Platform: Linux x86_64\n'
} > "$package_dir/BUILD_INFO.txt"
for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    env -u GGML_CUDA_P2P LD_LIBRARY_PATH="$package_dir/bin" \
        "$package_dir/bin/$executable" --version > "$package_dir/${executable}-version.txt" 2>&1
done

{
    printf 'Requirements:\n'
    printf '%s\n' '  - Linux x86_64 and an NVIDIA driver compatible with CUDA 12.'
    printf '%s\n' '  - CUDA 12 runtime libraries available on the host.'
    printf '%s\n' '  - NVIDIA Pascal sm_61 target; no GGUF model weights are included.'
    printf '\nUsage after extraction:\n'
    printf '%s\n' '  ./scripts/install-local.sh .'
    printf '%s\n' '  cp config/llama-models.ini.example config/llama-models.ini'
    printf '%s\n' '  $EDITOR config/llama-models.ini'
    printf '%s\n' '  unset GGML_CUDA_P2P'
    printf '%s\n' '  LD_LIBRARY_PATH="$PWD/bin" ./bin/llama-server --models-preset config/llama-models.ini --host 127.0.0.1'
    printf '\nSee doc/RELEASE_NOTES_%s.md for validation and limitations.\n' "$version"
    printf '%s\n' 'The performance gate requires a Git source checkout for provenance.'
    } > "$package_dir/README.txt"

for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    if LD_LIBRARY_PATH="$package_dir/bin" ldd "$package_dir/bin/$executable" | grep -q 'not found'; then
        printf 'unresolved dependency in %s\n' "$executable" >&2
        exit 1
    fi
done

(
    cd "$package_dir"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

tar -C "$stage_dir" -czf "$archive" "$package_name"
(
    cd "$dist_dir"
    sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)
printf 'created %s\n' "$archive"
printf 'created %s\n' "${archive}.sha256"

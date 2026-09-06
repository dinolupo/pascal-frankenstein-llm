#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

package_dir="$work_dir/package"
home_dir="$work_dir/home"
install_dir="$work_dir/install"
mkdir -p "$package_dir/bin" "$package_dir/config" "$package_dir/scripts" \
    "$package_dir/moe-traces" "$home_dir"

for executable in llama-cli llama-server llama-bench llama-moe-trace; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$package_dir/bin/$executable"
    chmod +x "$package_dir/bin/$executable"
done
cp "$root_dir/config/qwen.env.example" "$package_dir/config/"
cp "$root_dir/scripts"/{download-model.sh,install-local.sh,run-qwen.sh,verify-install.sh} \
    "$package_dir/scripts/"
chmod +x "$package_dir/scripts/"*.sh
printf 'test profile\n' > "$package_dir/moe-traces/qwen36-35b-mtp-merged.csv"

HOME="$home_dir" "$package_dir/scripts/install-local.sh" "$package_dir" "$install_dir"
config_file="$home_dir/.config/pascal-frankenstein-llm/qwen.env"
test -f "$config_file"
test -L "$home_dir/.local/bin/pascal-run-qwen.sh"
HOME="$home_dir" QWEN_CONFIG="$config_file" "$install_dir/scripts/verify-install.sh"

printf 'MODEL=/missing/model.gguf\n' >> "$config_file"
if HOME="$home_dir" QWEN_CONFIG="$config_file" "$install_dir/scripts/verify-install.sh" >/dev/null 2>&1; then
    printf 'verification unexpectedly accepted a missing model\n' >&2
    exit 1
fi

printf 'installer smoke tests passed\n'

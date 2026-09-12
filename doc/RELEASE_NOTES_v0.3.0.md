# Pascal Frankenstein LLM v0.3.0

## Status

This release packages the portable local-serving workflow. It has
been validated on the native Linux Mint host with the CUDA 12 Pascal binaries.
The dependency-light installer smoke test also passes with stub executables.
A Docker/Podman run remains optional because container tooling is not installed
on the validation host.

## Included

- CUDA 12 `sm_61` release binaries and MoE routing profiles.
- Packaged Qwen3.6 router preset template and default MTP routing profile.
- Pi agent integration instructions for an OpenAI-compatible local endpoint.
- Transparent copy-only installer.
- NVIDIA/runtime verification helper.
- Hugging Face downloader with optional revision and SHA-256 verification.
- Installer smoke test that does not require model weights or a GPU.

## Not included

- GGUF model weights.
- CUDA toolkit or NVIDIA drivers.
- Docker or Dev Container definitions.
- A claim that container execution has the same performance as native Linux.

## Installation

```bash
tar -xzf pascal-frankenstein-llm-0.3.0-linux-x86_64-cuda12-sm61.tar.gz
cd pascal-frankenstein-llm-0.3.0-linux-x86_64-cuda12-sm61
./scripts/install-local.sh .
./scripts/verify-install.sh
cp ~/.local/share/pascal-frankenstein-llm/config/llama-models.ini.example \
   ~/.local/share/pascal-frankenstein-llm/config/llama-models.ini
$EDITOR ~/.local/share/pascal-frankenstein-llm/config/llama-models.ini
```

The installer only copies files and does not create shell configuration, links,
or start a server. Replace `MODEL_PATH` and `INSTALL_DIR` in the INI with
absolute paths. The packaged profile is installed at
`moe-traces/qwen36-35b-mtp-merged.csv`. INI files do not expand shell
variables such as `$HOME`.

## Validation

The native host validation covers archive checksum, extraction, installation,
configuration creation, launcher links, GPU discovery, and missing-model
failure handling. Throughput and correctness measurements remain documented in
[`BASELINE_LOG.md`](BASELINE_LOG.md); they are not release guarantees.

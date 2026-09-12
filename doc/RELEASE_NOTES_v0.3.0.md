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
- Local configuration under `~/.config/pascal-frankenstein-llm/qwen.env`.
- Launch modes for 64k, 128k, and Tailscale/remote serving.
- Installation and upgrade script that preserves user configuration.
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
./scripts/install-local.sh pascal-frankenstein-llm-v0.3.0-linux-x86_64-cuda12-sm61.tar.gz
$EDITOR ~/.config/pascal-frankenstein-llm/qwen.env
pascal-verify-install.sh
pascal-run-qwen.sh 64k
```

Set `MODEL` to an existing external GGUF file. Use `pascal-run-qwen.sh 128k`
only after confirming that the host has enough free VRAM and RAM.

For router mode, after installation copy
`config/llama-models.ini.example` to `config/llama-models.ini` and replace its
`MODEL_PATH` and `INSTALL_DIR` placeholders with absolute paths. The packaged
profile is installed at `moe-traces/qwen36-35b-mtp-merged.csv`. INI files do
not expand shell variables such as `$HOME`.

## Validation

The native host validation covers archive checksum, extraction, installation,
configuration creation, launcher links, GPU discovery, and missing-model
failure handling. Throughput and correctness measurements remain documented in
[`BASELINE_LOG.md`](BASELINE_LOG.md); they are not release guarantees.

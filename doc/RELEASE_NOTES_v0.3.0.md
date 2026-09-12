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

The complete, authoritative installation and router startup procedure is in
the repository [README.md](../README.md). It covers checksum verification,
archive extraction, the installer, the router INI, and the direct
`llama-server --models-preset` command. The release package does not create
shell configuration or start a server. It also creates non-prefixed user-local
links (`llama-server`, `llama-cli`, `llama-bench`, and `llama-moe-trace`) in
`~/.local/bin`, without overwriting conflicting existing paths.

## Validation

The native host validation covers archive checksum, extraction, installation,
router template and profile presence, GPU discovery, and router preset parsing.
Throughput and correctness measurements remain documented in
[`BASELINE_LOG.md`](BASELINE_LOG.md); they are not release guarantees.

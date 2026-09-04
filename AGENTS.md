# Agent Notes

## Project scope

Pascal Frankenstein LLM is a personal local-inference, hardware-adaptation, and
benchmarking project. It is not a new inference engine.

Dense-model runs are performance and diagnostic baselines, not a dual-GPU
optimization target. The actual adaptation target is
Qwen3.6-35B-A3B MoE: expert caching, CPU/GPU prefetch, and speculative decoding
in the `perf` branch of `thecodacus/llama.cpp`.

Read `README.md` for the current public summary, `BASELINE_LOG.md` for the
complete experiment record, and `BENCHMARKS_AND_QUALITY.md` before proposing
quality or long-context work.

## Hardware invariants

- Intel i7-4790K (AVX2), 32 GB DDR3.
- CUDA0 is a GTX 1080 Ti with 11 GB; CUDA1 is a GTX 1070 with 8 GB. Both are
  Pascal `sm_61`.
- GPU topology is `SYS`; CUDA P2P read/write is unsupported. Never enable
  `GGML_CUDA_P2P` for this setup.
- CUDA Graphs being disabled on Pascal is expected, not a failure.
- The NVIDIA Control Panel power mode must be **Prefer maximum performance**
  before drawing performance conclusions.

## Development environment

- Development runs in Ubuntu 24.04.1 under WSL2. Use CUDA Toolkit 12.9 and
  compile for architecture 61. Do not move to CUDA 13: Pascal is not a
  supported compilation target there.
- WSL memory was increased from 15 GB to 24 GB, with 4 GB swap, because the
  mmap-backed model and MoE expert weights need sufficient Linux page cache.
  At 15 GB, page-cache thrashing dominated and invalidated absolute MoE
  throughput measurements. Do not lower this allocation before testing; see
  `BASELINE_LOG.md` for the before/after evidence and `.wslconfig` setting.

## Repository boundaries

- `ds4/` and `q36/` are upstream reference checkouts. Do not modify them.
- `llama.cpp/` is the working llama.cpp fork. Its upstream baseline
  is branch `perf`, initially commit `d927e7dc1`; local changes must remain
  attributable and reproducible.
- GGUF files are external model assets. Use them directly from their existing
  location; do not copy them into the repository or add symlinks without an
  explicit request.
- Do not add model weights, build directories, generated binaries, or other
  large local artifacts to version control.

## Changes and validation

- Preserve correctness before speed. Do not retain an optimization with
  unexplained output, attention, KV-cache, or logits drift.
- Before modifying the fork, state the hypothesis, the files affected, and the
  correctness and performance checks that will validate it.
- Keep changes narrow and readable. Prefer comments beside non-obvious memory,
  cache-placement, routing, or scheduling decisions.
- Do not change CUDA architecture, CMake flags, or upstream reference code as
  part of an unrelated experiment.
- A change to the fork must ultimately be committed in a personal fork with
  the upstream license and attribution intact.

## Measurement rules

- Do not repeat a setup already recorded in `BASELINE_LOG.md` unless the purpose
  is replication or a clearly identified variable has changed.
- Change one variable at a time. Use `r=1` only for screening; normally require
  at least `r=3` before calling a result a baseline.
- Record model and fork identity, context allocated and actually populated,
  K/V type, batch/ubatch, cache/MTP settings, RAM/swap, VRAM per GPU, prompt
  throughput, generation throughput, and correctness observations.
- `pp512` is synthetic prefill and `tg128` is synthetic generation; neither is
  equivalent to a real chat request.
- `llama-cli` and `llama-server` use `-ts 10,7`. In this `llama-bench`, use
  `-ts 10/7` for one tensor-split configuration: a comma starts separate
  benchmark configurations and can cause OOM.

## Operational safety

- Do not run multiple huge model processes concurrently.
- In WSL, `cudaHostRegister` is currently unsupported. Keep host registration
  and its paired prefetch experiment disabled there; evaluate them on native
  Linux only.
- Do not request or handle the user's `sudo` password. Ask the user to run a
  privileged command when one is genuinely necessary.

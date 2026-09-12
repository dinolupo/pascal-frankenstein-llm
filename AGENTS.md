# Agent Notes

## Project scope

Pascal Frankenstein LLM is a personal local-inference, hardware-adaptation, and
benchmarking project. It is not a new inference engine.

Dense-model runs are performance and diagnostic baselines, not a dual-GPU
optimization target. The actual adaptation target is
Qwen3.6-35B-A3B MoE: expert caching, CPU/GPU prefetch, and speculative decoding
in the `perf` branch of `thecodacus/llama.cpp`.

Read `README.md` for the current public summary, `doc/BASELINE_LOG.md` for the
complete experiment record, and `doc/BENCHMARKS_AND_QUALITY.md` before proposing
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

- **Default assumption: native Linux, not WSL2.** The project moved off WSL2
  to a native Ubuntu install for this hardware. Do not assume WSL2-specific
  limitations (e.g. the `cudaHostRegister` restriction below, `.wslconfig`
  memory limits) apply unless something in the current session actually
  indicates WSL2 is in play. Don't run an environment-detection command on
  every request just to confirm this — it's not worth the overhead. Instead,
  notice it opportunistically (shell prompt, kernel string, paths already
  visible in the conversation) and only run an explicit check
  (e.g. `uname -r`, `grep microsoft /proc/version`) when a command's behavior
  genuinely depends on the distinction (e.g. `cudaHostRegister` support,
  `sudo` availability, display/GPU passthrough topology) and it isn't already
  clear from context.
- The historical WSL2 setup (Ubuntu 24.04.1, `.wslconfig` memory bump from
  15 GB to 24 GB with 4 GB swap to avoid page-cache thrashing with mmap-backed
  MoE weights) is preserved in `doc/BASELINE_LOG.md` for reference and for any
  future WSL2 work, but is not the active environment.
- Use CUDA Toolkit 12.9 and compile for architecture 61 regardless of host
  OS. Do not move to CUDA 13: Pascal is not a supported compilation target
  there.

## Repository boundaries

- if existing `ds4/` and `q36/` folders are present, those are upstream reference
  checkouts. Do not modify them.
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

- Do not repeat a setup already recorded in `doc/BASELINE_LOG.md` unless the purpose
  is replication or a clearly identified variable has changed.
- Change one variable at a time. Use `r=1` only for screening; normally require
  at least `r=3` before calling a result a baseline.
- Record model and fork identity, context allocated and actually populated,
  K/V type, batch/ubatch, cache/MTP settings, RAM/swap, VRAM per GPU, prompt
  throughput, generation throughput, and correctness observations.
- `pp512` is synthetic prefill and `tg128` is synthetic generation; neither is
  equivalent to a real chat request.

## Operational safety

- Do not run multiple huge model processes concurrently.
- `cudaHostRegister` was unsupported under the old WSL2 setup; on native
  Linux it should work, but re-verify (host registration + the paired
  prefetch experiment) before relying on it, since it hasn't been re-tested
  since the environment switch.
- Do not request or handle the user's `sudo` password. Ask the user to run a
  privileged command when one is genuinely necessary.

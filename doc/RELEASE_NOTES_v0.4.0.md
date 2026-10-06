# Pascal Frankenstein LLM v0.4.0 - prerelease

## Scope and source

This project release integrates official **llama.cpp v0.6.0**, while retaining
the attributed MoE expert-cache and asymmetric dual-Pascal adaptation.
The project version is **v0.4.0**, not llama.cpp's version.

- Official upstream base: `d81235049384534c167caea52b85a694f6103d14`.
- Personal fork: [dinolupo/llama.cpp](https://github.com/dinolupo/llama.cpp),
  branch `pascal-v0.6.0-stack`, commit
  `98851873179280d11a2f4e40d3cd453df2cd7b47`.
- Four attributed commits above the official tag: operational cache/overlap,
  optional CPU/CUDA quantization port, MTP export migration, prefetch tracing.
- Original MoE feature lineage: `thecodacus/llama.cpp` `perf`, initially
  `d927e7dc1`. Upstream license and contributor attribution remain intact.

This is a hardware-adaptation release, not a new inference engine or a
universal compatibility certification. The installed local service was not
changed by the integration work.

## Included

- Linux x86_64 CUDA 12.9 / Pascal `sm_61` binaries:
  `llama-server`, `llama-cli`, `llama-bench`, `llama-moe-trace`.
- Project shared libraries, source/build identity, per-file SHA-256 manifest,
  archive checksum, project and llama.cpp licenses.
- Router preset example, routing profiles, copy-only installer and verifier.
- Performance runner, offline tests and UUID-matched NV-CONTROL power-policy
  verifier; experiment record and sustainable upstream integration procedure.
- Per-device MoE-cache quotas and device-local hot experts, with CPU cold path.
- Validated CPU/CUDA Pascal Turbo/TQ support, preserving newer upstream Q2_0.
- Newer upstream batch, MTP, kernel and server implementations rather than
  replaying obsolete upstream patches.

No model weights, private local INI, private chat template, NVIDIA libraries,
drivers or CUDA toolkit are bundled.

## Validation on the development host

Native Linux Mint; i7-4790K AVX2, 32 GB DDR3; GTX 1080 Ti 11 GB and
GTX 1070 8 GB; CUDA Toolkit 12.9, driver 580.178.04.

- Incremental build, numerical and operational gates for each feature group.
- Nine focused CTests covering sampling, Turbo quantization, MTP export
  contracts, reasoning budget, grammar, batch allocation, chat and JSON schema.
- 23 offline measurement-tool tests and the dependency-light installer smoke.
- Real numerical optional-quantization fixtures: 128 selected cases per GPU,
  no unsupported skips; 18 CPU round trips and six model configurations.
- Synthetic Qwen35moe/Qwen4Exp CPU-reference agreement, serialization and
  mixed token/embedding batch ordering on both GPUs.
- Real-model arithmetic, EOG, prefix reuse, image, reasoning, structured JSON,
  tool call/continuation, MTP-off, cache-off and scalar-cache smoke checks.
- Native read-only host registration, full-model pinning and actual prefetch
  uploads passed. These remain opt-in, not certified speed optimizations.

### Recorded operational performance - not a speedup certificate

Latest complete chat gate: three conversations of ten turns per profile,
T=0.6, seed 123, max 384 output tokens, reasoning off and prompt reuse.
Q8 K/V, batch/ubatch 512, split 10,7, native MTP n-max 2; text cache 160,108
at allocated 128k and vision cache 130,108 at allocated 64k.
Registration/prefetch were off; sampled UUID-matched power preference was 1.

| Profile | Mean generation | Mean prefill | Mean TTFT |
| --- | ---: | ---: | ---: |
| Text | 44.56 tokens/s | 84.13 tokens/s | 0.828 s |
| Vision | 41.60 tokens/s | 77.58 tokens/s | 0.942 s |

This gate used `590b56282` plus the six-line tracing patch later committed as
`988518731`. It was not a repeat on the final clean rebuild. The gate was
accepted with an explicit TTFT-only exception; strict latency thresholds were
not all met. Historical controls lacked the same verified power policy.
**The final contemporary original/maintenance/candidate/original comparison
has not been completed. No performance improvement is certified.**

### Populated long-context screening on the clean rebuild

Single cold repetition per profile, greedy retrieval of three codes spread
through the prompt, strict JSON without answer-enforcing constants, normal
stop and verified cached continuation:

| Profile | Allocated context | Actual prompt tokens | Cold request elapsed |
| --- | ---: | ---: | ---: |
| Short control | 131,072 | 3,807 | 16.68 s |
| Vision preset, text-only input | 65,536 | 58,698 | 328.74 s |
| Text preset | 131,072 | 117,503 | 771.85 s |

All three objective retrieval and cached-continuation checks passed on the
clean `988518731` rebuild. These are **r1 screening**, not a long-context
throughput baseline, image-at-64k test or broad quality certification.
An earlier run failed on a GPU telemetry timeout despite correct 64k answers;
another run was interrupted. Failed evidence was retained. The successful
screening used bounded VRAM/power probes, not clock/temperature telemetry.

## Compatibility decisions and remaining limits

- Legacy `--mtp-shared-embd` / borrowed-target exports are not restored.
  Re-export using upstream `--mtp` without the legacy flag, or retain the full
  Native-MTP-Preserved model for native `draft-mtp`. Removing metadata does not
  recreate missing weights. Legacy sharing metadata fails with an actionable
  error; draft-only Qwen4Exp is rejected as an ordinary target.
- Real Qwen4Exp/Laguna model inference and non-CPU/non-CUDA backends are not
  runtime-certified. Synthetic architecture tests do not replace that coverage.
- Turbo Tensor-Core-only fusion was not imported for Pascal.
- One registered loading timeout remains unexplained; later early/deferred/
  unmonitored polling checks passed. No telemetry-deadlock claim is made.
- GPU/power sampling is explicitly deferred during opt-in registered loads;
  RAM sampling continues. Deferred samples are not GPU/power coverage.
- Session raw results are local evidence, not downloadable release assets.
  The versioned summaries and reproduction commands are included.

## Requirements and installation

The binaries target this AVX2 host and `sm_61`; they are not portable to every
CPU/GPU. A compatible NVIDIA driver, CUDA 12 runtime libraries, sufficient
RAM/VRAM and external GGUF/projector assets are required.

Download both release assets into the same directory, then:

```bash
sha256sum -c pascal-frankenstein-llm-v0.4.0-linux-x86_64-cuda12-sm61.tar.gz.sha256
tar -xzf pascal-frankenstein-llm-v0.4.0-linux-x86_64-cuda12-sm61.tar.gz
cd pascal-frankenstein-llm-v0.4.0-linux-x86_64-cuda12-sm61
sha256sum -c SHA256SUMS
./scripts/install-local.sh .
cp config/llama-models.ini.example config/llama-models.ini
${EDITOR:-vi} config/llama-models.ini
unset GGML_CUDA_P2P
LD_LIBRARY_PATH="$PWD/bin" ./bin/llama-server \
  --models-preset config/llama-models.ini --host 127.0.0.1 --port 8001 \
  --models-max 1
```

Replace model, projector and installation placeholders with absolute paths.
The example's optional chat-template override is not bundled. Preserve an
existing custom INI; the installer copies the example, not your active INI.
Bind to loopback unless explicitly configuring authenticated remote access.
Never set `GGML_CUDA_P2P`, even to `0`, on this SYS/no-P2P setup.
CUDA Graphs being disabled on Pascal is expected. CUDA 13 is not a supported
compiler target for these GPUs.

Before interpreting measurements verify Prefer maximum performance:
`python3 scripts/nvidia_power.py`. This requires the current X display and
X11/XNVCtrl libraries. The performance gate also needs a source checkout for
Git provenance, Python `requests`, and operational aliases matching the runner;
the packaged example uses different public aliases.

## Source and future updates

GitHub's automatic source archive does not contain the submodule checkout.
Use:

```bash
git clone --branch v0.4.0 --recurse-submodules \
  https://github.com/dinolupo/pascal-frankenstein-llm.git
```

See [UPSTREAM_INTEGRATION.md](UPSTREAM_INTEGRATION.md) for attributed patch-stack
updates, [BASELINE_LOG.md](BASELINE_LOG.md) for the experiment record, and
[PERFORMANCE.md](../scripts/PERFORMANCE.md) for gate commands and thresholds.

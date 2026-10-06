# Incremental upstream integration

## Stable installation and candidate branches

Never update the installed binaries or operational INI while integrating a
release. Use separate candidate branches in this repository and its llama.cpp
submodule. Preserve the previous validated SHAs and raw test evidence.
Only push or deploy after explicit approval.

The v0.6.0 candidate branches are `integration/llama-v0.6.0` (parent) and
`pascal-integration-v0.6.0` (fork), initially based on `befe02c` and `47b3df22d`.
The official annotated tag resolves to
`d81235049384534c167caea52b85a694f6103d14`; the release API's
`target_commitish` is not the authoritative resolved tag.

## Select the integration method from evidence

Fetch the official tag into a namespaced local ref, avoiding automatic tag
imports on subsequent updates:

```bash
git -C llama.cpp fetch --no-tags https://github.com/ggml-org/llama.cpp.git \
  refs/tags/v0.6.0:refs/tags/upstream-v0.6.0
git -C llama.cpp rev-parse 'upstream-v0.6.0^{commit}'
git -C llama.cpp merge-base HEAD upstream-v0.6.0
git -C llama.cpp cherry upstream-v0.6.0 HEAD
# Exit 1 indicates conflicts, not a modification of the checkout.
git -C llama.cpp merge-tree --write-tree HEAD upstream-v0.6.0
```

The initial v0.6.0 dry-run reported 398 conflict notifications. Its merge base
was `8f5ab832c` from July, despite September upstream integrations in the fork.
`git cherry` identified 677 patch-equivalent commits and 27 non-equivalent
commits. Non-equivalence does NOT prove a feature is missing upstream: for
example Qwen4Exp/MTP code must be reconciled with its newer upstream equivalent.

For this update, the approved approach is:

1. Validate the old source as the performance reference.
2. Backport and validate EOG, then MMQ, separately.
3. Preserve that maintenance branch and start a release patch-stack branch at
   the verified official tag.
4. Reapply attributed, coherent feature groups, not hundreds of duplicate
   upstream commits and not one opaque squash.

Classify the fork delta into expert tracing/cache, masked hot/cold routing,
CPU/GPU overlap, Pascal device placement/quotas, host registration/prefetch,
optional quantization support, and changes already represented upstream.
Do not discard optional features without a documented decision.
Retain original authors and source SHAs; use `cherry-pick -x` where applicable
and record manual adaptations. A textually clean patch is not proof of correctness.

For later releases, a clean stack based on official upstream history should
reduce artificial conflicts. Use a history-preserving merge for published
branches, or replay the attributed stack on a new candidate without rewriting
the previous published branch.

## Mandatory checkpoint gate

### Current attributed stack and unresolved contracts

On `pascal-v0.6.0-stack`, `4d0ac3bdb` contains the operational cache/overlap
group and `7760081b1` contains the validated optional CPU/CUDA Pascal port.
The parent pin is still the maintenance checkpoint, not a promoted release.

The user selected CPU and CUDA Pascal coverage; preserve other upstream backend
sources unchanged and document unverified shared HIP/MUSA code. Do not import
Tensor-Core-only Turbo MMA fusion or unused stubs into a Pascal stack.
Upstream Q2_0 is newer than the old fork patch and must remain intact.
The new attention dispatch already permits compiled mixed K/V pairs; adapt
template registration rather than reintroducing its obsolete selection API.

The original shared-embedding MTP export is **not** an upstream equivalent:
`--mtp-shared-embd`, `nextn_shared_target_tensors` and
`llama_model_params.model_shared` are absent from the official release.
Its drafts omit embeddings and borrow target-owned tensors; upstream's
self-contained drafts require those tensors. Resolve legacy compatibility
explicitly before declaring feature parity. Likewise preserve explicit rejection
of a draft-only Qwen4Exp export used as an ordinary target.
No local Qwen4Exp asset is currently runtime-certified.

Before changing the fork, state the hypothesis, files affected, correctness
checks and performance checks. Every coherent checkpoint must compile and pass
the same tests before another feature is introduced.

Build using the available generator, CUDA 12.9 and architecture 61:

```bash
cmake -S llama.cpp -B llama.cpp/build-pascal-cuda -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON
cmake --build llama.cpp/build-pascal-cuda \
  --target llama-server llama-bench test-backend-ops test-sampling -j 4
llama.cpp/build-pascal-cuda/bin/test-sampling
llama.cpp/build-pascal-cuda/bin/test-backend-ops test \
  -b CUDA0 -o RMS_NORM,SCALE,MUL_MAT_ID
llama.cpp/build-pascal-cuda/bin/test-backend-ops test \
  -b CUDA1 -o RMS_NORM,SCALE,MUL_MAT_ID
```

Do not enable P2P. CUDA Graphs disabled on Pascal and absent NCCL are expected.
Build outputs stay untracked. When changing APIs, rebuild libraries and their
consumers together; do not reuse saved session formats across incompatible versions.

Run [the checkpoint runner](../scripts/run-integration-gate.py) as documented in
[the performance protocol](../scripts/PERFORMANCE.md). It uses an owned loopback
server, the actual installed text/vision presets, sequential model loads,
functional smoke tests, three ten-turn stress repetitions per profile and
memory sampling. It never installs the candidate.

Default strict thresholds compared with matching baseline workload:

- generation throughput loss greater than 5%: stop;
- prefill throughput loss greater than 10%: stop;
- TTFT increase greater than 10%: stop;
- correctness failure, missing metrics, incomplete stream or telemetry failure:
  stop regardless of speed.

Compare both the initial baseline and previous accepted checkpoint. Investigate
borderline/noisy results; do not silently relax thresholds or adjust cache quotas
to mask a memory regression. Different output lengths/routing at positive
temperature must be considered; use a greedy control when necessary.
Keep warm-up outside measurements and use the same prompt-cache policy.

For hybrid/recurrent models, record `cache_n` and `prompt_n` when TTFT changes.
Noncanonical generated BPE tokens or template trimming can invalidate a prefix
and require checkpoint rollback even when visible text is unchanged. Do not
reuse state against different token IDs. A passing greedy control helps diagnose
a stochastic gate; it does not replace the operational measurement.

For this integration, the user explicitly approved accepting explained latency
compromises with operational presets unchanged. Opt in with
`--accept-latency-regression "Recorded diagnosis and acceptance rationale"` on
the checkpoint runner or benchmark. Only a TTFT-only breach can be accepted:
generation/prefill breaches, functional errors and incomplete measurements still
fail. Reports retain percentage deltas, absolute TTFT cost, violations, rationale
and whether the exception was used. `passed` indicates policy acceptance, not
absence of regressions; inspect `performance_thresholds_passed` in the gate and
`performance_assessment.thresholds_passed` in benchmark reports. The exception
applies to both baseline and previous-checkpoint comparisons. Strict mode remains
the default. Previously rejected evidence is never rewritten.

## Feature-specific coverage

- EOG: exact short answer and normal stop, no accepted draft tokens after EOG,
  subsequent-turn prefix reuse, MTP enabled and disabled.
- MMQ: actually exercise MMQ with many experts and a small physical batch.
  The cached hot/cold dispatcher can bypass MMQ: cached decode alone is not
  a reproduction. Preserve skipped IDs (`-1`) and their zero-output semantics.
- Cache/overlap: validate disabled, scalar and asymmetric quotas; GPU ownership,
  cold fallback, masked IDs and CPU/GPU result merging.
- Fusion: validate actual activation and numerical tolerance on both Pascal
  GPUs; upstream speedups on another GPU are not local measurements.
- MTP/batch migration: embedding/token order, rejection and rollback, greedy and
  positive-temperature sampling, prefix reuse, structured outputs.
- Vision: actual image request, not only text requests to a loaded projector.
- Long context: substantially populate the context and check an objective answer;
  allocated 128k or a short stress conversation is not a long-context quality test.
- Direct-I/O/prefetch: separate opt-in experiments, not changes hidden in an
  upstream update. Revalidate host registration on native Linux.

## Promotion and provenance

Commit only accepted checkpoints locally. Record source/tag/model identities,
template and routing-profile hashes, context allocated and populated, K/V,
batch/ubatch, cache/MTP configuration, acceptance, throughput, TTFT, RAM/swap
and VRAM per GPU in [BASELINE_LOG.md](BASELINE_LOG.md).

Raw logs, generated answers, JSON reports and telemetry remain outside Git.
Do not execute model-generated code on the host.
Pin the final tested fork SHA in the parent repository and update any branch
hint deliberately. Never advertise a blocked or untested stage as validated.
Preserve upstream licensing and credit thecodacus for the cache/overlap design;
the Pascal adaptation is not the complete expert-cache implementation.

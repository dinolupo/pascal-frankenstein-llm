# Quality, long-context, and performance protocol

Status: started on 2 September 2026. Allocating `-c 65536` or `-c 131072` is
not a successful long-context test by itself: the context must be substantially
populated and the answer must pass objective checks.

## Candidates

1. Qwen3.6-35B-A3B-UD-Q4_K_M from the MTP repository, the fast MoE
   configuration:
   `-ngl 99 -ncmoe 33 -ts 10,7 -fa on`, cache `160,108`, MTP on and off.
2. Qwen3.8-27B-Q4_K_M dense, the quality-oriented reference.

Qwen3.6 declares a 262,144-token native context. The 64k and 128k tests must
not use YaRN or another RoPE extension.

## Prepare a fresh test machine

Clone this repository with its llama.cpp submodule, then choose a separate
directory for model weights. Do not download GGUF files into the Git checkout.

```bash
git clone --recurse-submodules \
  https://github.com/dinolupo/pascal-frankenstein-llm.git
cd pascal-frankenstein-llm

REPO_DIR=$PWD
MODEL_DIR=/path/to/external/models
mkdir -p "$MODEL_DIR"
```

Build the pinned fork with the Pascal/CUDA 12 command in the project README, or
extract the matching pre-release. Set `SERVER_BIN` to the resulting executable:

```bash
# Source build:
SERVER_BIN="$REPO_DIR/llama.cpp/build-pascal-cuda/bin/llama-server"

# Pre-release alternative:
# SERVER_BIN=/path/to/extracted/bin/llama-server
# export LD_LIBRARY_PATH="$(dirname "$SERVER_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use the official `hf` CLI to download only the required files. Revisions are
pinned so another machine retrieves the same assets used by this protocol.
Each vision projector is downloaded from the same repository as its model;
similarly named projector files are not assumed to be interchangeable.

### Download the current test matrix

```bash
MODEL_DIR=/path/to/external/models

hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf mmproj-BF16.gguf \
  --revision 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d \
  --local-dir "$MODEL_DIR"

hf download Infatoshi/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-Q4_K_M.gguf Qwen3.6-35B-A3B-mmproj-BF16.gguf \
  --revision 22880479d6dd057b3fab8654afaaf34648fb524e \
  --local-dir "$MODEL_DIR"

hf download llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF \
  Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf \
  Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf \
  --revision f70be2db155a4192a59c559ece01572f3cd508ab \
  --local-dir "$MODEL_DIR"
```

Verify the downloaded files before running a benchmark:

| File | Expected SHA-256 | Role |
| --- | --- | --- |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` | Fast MTP baseline |
| `mmproj-BF16.gguf` | `da63cb47a76763c712393f8a017070188a304fa39f8aeea6edc629ed7b975cfa` | Unsloth MTP vision projector |
| `Qwen3.6-35B-A3B-Q4_K_M.gguf` | `439fcb8266f37a035d2192d0fa773e59b177f379bd2bf90976419cea8c7dbb58` | Standard non-MTP baseline |
| `Qwen3.6-35B-A3B-mmproj-BF16.gguf` | `37904b50d3453905e7ef5a4ada6a148bc87ff3923fe1d6c39ae028e53742ff13` | Standard vision projector |
| `Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf` | `b82f9f2155eb9c07139d48cc3a37880cf9d0edcca345ef6ad829b62941ccbb82` | Current uncensored, non-MTP candidate |
| `Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf` | `1c625f05cd52e90abc76a5f756226c3a5fe279593379c22f6c6846c970a0cd18` | Current Heretic vision projector |

```bash
sha256sum \
  "$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  "$MODEL_DIR/mmproj-BF16.gguf" \
  "$MODEL_DIR/Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  "$MODEL_DIR/Qwen3.6-35B-A3B-mmproj-BF16.gguf" \
  "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf" \
  "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf"
```

### Uncensored MTP candidate

[`llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF`](https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF)
is the first candidate to test. Its model card reports that the native NextN/MTP
tensors are preserved, unlike the current Heretic GGUF that failed to create an
MTP context. This is a third-party model claim, not yet a local validation.

```bash
hf download \
  llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GGUF \
  Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf \
  Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-mmproj-BF16.gguf \
  --revision fa96dc904fe4dafb415ca707afab30aee2d2e703 \
  --local-dir "$MODEL_DIR"
```

Expected SHA-256 values are
`fc89d92377b27fe0f80eb683a5105d0921234c24f7c1d70ecc2356ddf994d781`
for the Q4_K_M model and
`d6050bb82a0187e1b0655f1c5daef60c9479273fbd7402ff25d3137df2e071ce`
for its projector. Verify both before benchmarking.

## Common rules

- One server slot and no other CUDA workload.
- Separate warm-up; at least three repetitions for final throughput results.
- Record fork commit, GGUF SHA-256, allocated and actually used context, K/V
  type, batch/ubatch, expert-cache/MTP settings, RAM/swap, and VRAM per GPU.
- Use greedy sampling for repeatable comparisons; measure a realistic chat
  profile separately.
- With MTP, compare score, format, stop condition, and token count—not only
  speed. Different output is not automatically equivalent output.
- Never execute generated code on the host. HumanEval+ and Aider runs must use
  a network-isolated, unprivileged, resource-limited container or sandbox.

## Server starting points

The authoritative day-to-day commands are kept at the top of
[MODEL_LOCAL_GUIDE.md](MODEL_LOCAL_GUIDE.md): the Heretic
`Native-MTP-Preserved` text server at 128k with cache `160,108`, and the
matching vision server at 64k with GPU projector offload and cache `130,108`.
Use this file for benchmark protocol and historical comparison; do not treat
older candidate commands below as preferred launch commands unless a test
explicitly calls for that model.

All controlled benchmark commands use one slot. Bind to loopback for local
measurement; use `--host 0.0.0.0` plus a private API key only for remote
Tailscale use.

### MTP-UD operational candidate

This matches the fastest verified `-ncmoe 33`, cache `160,108`, MTP `n-max=2`
configuration. The 64k `llama-server` result still requires three controlled
replications, so the command is a candidate rather than a new baseline.

```bash
REPO_DIR=/path/to/pascal-frankenstein-llm
MODEL_DIR=/path/to/external/models
SERVER_BIN="$REPO_DIR/llama.cpp/build-pascal-cuda/bin/llama-server"

"$SERVER_BIN" \
  -m "$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  -c 65536 --parallel 1 -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk f16 -ctv f16 -b 512 -ub 512 \
  --moe-cache-profile "$REPO_DIR/moe-traces/qwen36-35b-mtp-merged.csv" \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning off --temp 0 --seed 123 \
  --host 127.0.0.1 --port 18081 --jinja --no-webui
```

### Standard non-MTP candidate

The replicated synthetic-generation result favored `-ncmoe 30` with a
distributed `112,112` cache. This exact server profile has not yet been
replicated after the WSL memory increase.

```bash
"$SERVER_BIN" \
  -m "$MODEL_DIR/Qwen3.6-35B-A3B-Q4_K_M.gguf" \
  -c 16384 --parallel 1 -ngl 99 -ncmoe 30 -ts 10,7 -fa on \
  -ctk f16 -ctv f16 -b 2048 -ub 512 \
  --moe-cache-profile "$REPO_DIR/moe-traces/qwen36-35b-merged.csv" \
  --moe-cache-slots 112,112 \
  --reasoning off --temp 0 --seed 123 \
  --host 127.0.0.1 --port 18082 --jinja --no-webui
```

### Current Heretic candidate

This reproduces the settings of the completed 64k smoke test. Its routing
profile was collected with the Unsloth MTP GGUF, not with the Heretic model, so
it remains provisional and must not be presented as an optimized Heretic
profile.

```bash
"$SERVER_BIN" \
  -m "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf" \
  -c 65536 --parallel 1 -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk f16 -ctv f16 -b 512 -ub 512 \
  --moe-cache-profile "$REPO_DIR/moe-traces/qwen36-35b-mtp-merged.csv" \
  --moe-cache-slots 160,108 \
  --reasoning off --temp 0 --seed 123 \
  --host 127.0.0.1 --port 18083 --jinja --no-webui
```

### Vision server candidate

Vision has not yet been performance-tested on this hardware. Start at 16k so
the approximately 861 MiB BF16 projector and its compute buffers have VRAM
headroom. GPU projector offload is the expected fast path and is enabled by
default; it is written explicitly below. If it does not fit, repeat with only
`--no-mmproj-offload` changed and report the performance cost.

```bash
"$SERVER_BIN" \
  -m "$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  --mmproj "$MODEL_DIR/mmproj-BF16.gguf" --mmproj-offload \
  -c 16384 --parallel 1 -ngl 99 -ncmoe 33 -ts 10,7 -fa on \
  -ctk f16 -ctv f16 -b 512 -ub 512 \
  --moe-cache-profile "$REPO_DIR/moe-traces/qwen36-35b-mtp-merged.csv" \
  --moe-cache-slots 160,108 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --reasoning off --temp 0 --seed 123 \
  --host 127.0.0.1 --port 18084 --jinja --no-webui
```

For the current Heretic vision test, start from its text-only command above,
reduce the initial context to 16k, and add:

```bash
--mmproj \
  "$MODEL_DIR/Qwen3.6-35B-A3B-uncensored-heretic-mmproj-BF16.gguf" \
--mmproj-offload
```

For the standard model, start from its own text-only command and add the
Infatoshi projector `Qwen3.6-35B-A3B-mmproj-BF16.gguf`. For the uncensored MTP
candidate, use the two `Native-MTP-Preserved` files together and enable MTP only
after that model has created a valid MTP context. Do not compare their speeds
until each pair has passed the same fixed image set.

## Next-step execution plan

Perform the following in order. A later stage must not silently change a
variable established by an earlier one.

1. **Replicate the text-only HTTP baseline.** Run the MTP-UD 64k server command
   above three times with the same warm-up and request. Record prompt and
   generation throughput, MTP acceptance, token count, output, RAM/swap, and
   VRAM per GPU. This closes the main gap between the fast CLI result and an
   operational server baseline.
2. **Test Fable's prefill path on native Linux.** With expert cache and MTP off,
   run a 2x2 A/B matrix: host registration off/on and expert prefetch off/on.
   The off cells must leave the variables unset; in this fork, merely setting
   `GGML_CUDA_REGISTER_HOST=0` still requests host registration.
   Confirm that `GGML_CUDA_REGISTER_HOST=1` logs successful pinning before
   interpreting throughput. Use the largest batch/ubatch that fits, screen with
   `r=1`, and repeat the finalists with `r>=3`. Only after that clean test,
   repeat the winning setting with the operational cache enabled. WSL is not a
   substitute because its CUDA path rejected `cudaHostRegister` even after the
   memory allocation was raised to 24 GB.
3. **Validate the uncensored MTP candidate.** First confirm that the model
   creates an MTP context. Then compare MTP off/on at `n-max=2`, using a routing
   trace collected from that exact GGUF. Compare output quality and acceptance
   as well as speed; do not reuse the current Heretic smoke-test number as its
   baseline.
4. **Run the vision matrix.** Test the Unsloth MTP, standard, current Heretic,
   and uncensored-MTP pairs on the same fixed images. For each pair compare
   text-only with projector loaded, image input with projector on GPU, and the
   CPU-projector fallback. Measure image-encoding/prompt time separately from
   generation, plus peak VRAM, RAM/swap, MTP acceptance, and correctness.
5. **Tune only verified bottlenecks.** In separate A/B tests, screen MTP
   `n-max=1,2,3,4`, batch/ubatch, `--sched-async-cpu` on/off, and nearby cache
   quotas that preserve at least 900 MiB free VRAM per GPU. Retain a setting
   only if `r>=3` improves the target workload without unexplained output or
   logits drift.
6. **Continue quality and context work.** Complete the planned automatic
   quality suites, then test a genuinely populated 128k context. Do not trade
   away the verified 64k profile merely to make 128k allocate.

The current `llama-moe-trace` tool uses the perplexity argument set and cannot
accept `--mmproj` or image input. A vision-specific routing profile therefore
requires either adding multimodal tracing support to the fork or another
instrumented capture path. Treat that as a code change justified by the vision
A/B results, not as a prerequisite for the first smoke tests.

## Routing profile: completed first pass

The multi-domain v2 routing-profile work completed on 3 September 2026. It
improves held-out coverage and non-MTP decode, but lowers MTP acceptance and
throughput on the control prompt. It therefore does not replace the v1
operational profile. Its raw trace archive is deliberately not versioned in
this lean public repository; the methods, hashes, and measured conclusion are
recorded in `BASELINE_LOG.md`.

Any future profile must use equal decode-token counts for coding, agent tool
selection, RAG, reasoning, conversation/role play, and Italian/English
instruction following. A held-out prompt set must measure hot-expert coverage,
Jaccard overlap with the earlier profile, per-domain and per-GPU coverage, and
old/new CSV A/B throughput.

## Long-context work

Completed: 64k fit tests, expert-placement screening, short replications with
F16 K/V, and a direct 64k F16 versus Q8_0/Q8_0 K/V comparison.

Open: a 128k candidate, contexts genuinely populated to roughly 90%, and at
least three stable repetitions.

An intermediate 64k run with 60,132 real tokens, `-ncmoe 33`, cache `160,108`,
and MTP `n-max=2` measured 132.8 prompt t/s and 25.5 generation t/s with F16
K/V. Q8_0/Q8_0 saved about 525 MiB but generated at 23.0 t/s. These are
single long runs; see `BASELINE_LOG.md` for all placement comparisons.

Measure real prompt tokens, ingestion time, prompt and generation throughput,
MTP acceptance, load time, RAM, swap, per-GPU VRAM, and answer correctness.
Also test prefix-cache reuse on a second request.

## Quality evaluation

| Area | Suite | Purpose |
| --- | --- | --- |
| Function coding | HumanEval+ (164 tasks) | Greedy pass@1 with extended tests |
| Repository coding | Aider Polyglot subset, then 225 tasks | Executable multi-file edits |
| Long context | RULER at 4k/32k/64k/128k | Retrieval, multi-hop, aggregation, variable tracking |
| Vision | Fixed local image set | OCR, charts, screenshots, spatial details, and hallucination control |
| Tools/agents | BFCL non-live single-turn, then multi-turn | Function choice, arguments, format |
| Reasoning | `llama-eval` GSM8K and GPQA | Accuracy and thinking/non-thinking stability |
| Instruction following | Fixed local set, optionally IFEval | Constraints, JSON, language, irrelevant-tool refusal |
| Conversation | Fixed philosophy/role-play rubrics | Coherence, persona, memory, repetition |

HumanEval+ does not replace agentic coding evaluation; RULER does not replace a
real repository task; one subjective evaluation is not a benchmark percentage.

## Minimum conclusion criteria

- 64k and 128k genuinely populated without OOM, growing swap, or a severe
  performance cliff.
- At least 20 interactive generation t/s for the selected MoE profile at 64k;
  report the 128k result even if it is slower.
- No statistically meaningful MTP on/off regression in the selected automatic
  quality scores.
- Well-formed tool calls and long-context results above the planned 4k baseline.
- Matching model/projector pairs load reliably and pass the fixed vision set;
  report image-encoding speed separately from text generation.
- A final server command replicated at least three times and documented.

## Method references

- [Hardware/benchmark discussion](https://www.reddit.com/r/LocalLLM/comments/1vm0g2b/best_local_coding_model_for_16gb_vram_64ram/)
- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [EvalPlus / HumanEval+](https://github.com/evalplus/evalplus)
- [Aider Polyglot](https://aider.chat/docs/leaderboards/)
- [RULER](https://github.com/NVIDIA/RULER)
- [BFCL](https://github.com/EnlightenedAI/BFCL)
- [Hugging Face `hf download`](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)
- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

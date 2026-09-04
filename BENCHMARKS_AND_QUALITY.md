# Quality, long-context, and performance protocol

Status: started on 2 September 2026. Allocating `-c 65536` or `-c 131072` is
not a successful long-context test by itself: the context must be substantially
populated and the answer must pass objective checks.

## Candidates

1. Qwen3.6-35B-A3B-MTP-UD-Q4_K_M, the fast MoE configuration:
   `-ngl 99 -ncmoe 33 -ts 10,7 -fa on`, cache `160,108`, MTP on and off.
2. Qwen3.8-27B-Q4_K_M dense, the quality-oriented reference.

Qwen3.6 declares a 262,144-token native context. The 64k and 128k tests must
not use YaRN or another RoPE extension.

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
- A final server command replicated at least three times and documented.

## Method references

- [Hardware/benchmark discussion](https://www.reddit.com/r/LocalLLM/comments/1vm0g2b/best_local_coding_model_for_16gb_vram_64ram/)
- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [EvalPlus / HumanEval+](https://github.com/evalplus/evalplus)
- [Aider Polyglot](https://aider.chat/docs/leaderboards/)
- [RULER](https://github.com/NVIDIA/RULER)
- [BFCL](https://github.com/EnlightenedAI/BFCL)

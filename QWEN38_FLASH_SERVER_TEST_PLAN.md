# Qwen3.8 Flash server/API tuning plan

Date: 2026-09-10

Goal: find whether Qwen3.8 Flash Next can reach a useful server/API speed on the
Pascal dual-GPU machine at 32k context. Stop each candidate quickly when API
prefill is below the useful floor.

## Safety rules

- Use `llama-server` only; do not use `llama-cli`.
- Run exactly one model process at a time.
- Keep `--load-mode mmap`; do not use `--load-mode none` for this 82 GB model on
  32 GB RAM.
- Keep `-fit off`.
- Use bounded API requests (`n_predict` 64 for screening).
- Stop a candidate immediately after the API timing sample if prompt/prefill is
  below 10 tok/s.
- Do not continue expensive sweeps unless a candidate reaches at least 10-20
  tok/s prompt/prefill.

## Baselines already observed

- Qwen3.6 Heretic known-good text command: 120k prompt at ~153.77 prompt tok/s,
  ~23.62 generation tok/s, ~72.8% MTP acceptance.
- Qwen3.8 dual-GPU `-ncmoe 44`, q4/q4 KV, cache `96,48`: ~2.8 tok/s live
  decode in Web UI.
- Qwen3.8 dual-GPU `-ncmoe 44`, q8/q8 KV, cache `80,40`: ~3.8 tok/s live
  decode in Web UI.
- Qwen3.8 dual-GPU `-ncmoe 99` with `-ts 10,7` was slower and was stopped.

## Screening candidates

| ID | Status | Purpose | Key flags | Continue condition |
| --- | --- | --- | --- | --- |
| S1 | done | Codacus-style single GPU sanity check | `CUDA_VISIBLE_DEVICES=0`, no `-ts`, `-ncmoe 99`, cache `48`, q8/q8 KV, `-b 512 -ub 256` | prompt 10.27 tok/s, generation 1.52 tok/s; not viable |
| S2 | done | Dual-GPU known-best API measurement | `-ts 10,7`, `-ncmoe 44`, cache `80,40`, q8/q8 KV, `-b 512 -ub 256` | prompt 12.28 tok/s, generation 1.76 tok/s; better than S1 but too slow |
| S3 | done | Dual-GPU more GPU0 cache if S2 is close | `-ts 10,7`, `-ncmoe 44`, cache `88,40`, q8/q8 KV, `-b 512 -ub 256` | prompt 9.09 tok/s, generation 1.78 tok/s; rejected |
| S4 | done | Dual-GPU larger batch if S2/S3 has headroom | `-ts 10,7`, `-ncmoe 44`, cache `80,40`, q8/q8 KV, `-b 1024 -ub 256` | prompt 11.54 tok/s, generation 1.81 tok/s; did not beat S2 |
| S5 | done | Codacus CPU-MoE with larger batch if S1 is close | single GPU, `-ncmoe 99`, cache `48`, q8/q8 KV, `-b 1024 -ub 512` | prompt 8.77 tok/s, generation 1.53 tok/s; rejected |
| S6 | done | Dual-GPU S2 without MTP to test draft overhead | `-ts 10,7`, `-ncmoe 44`, cache `80,40`, q8/q8 KV, no `-md`, `-b 512 -ub 256` | prompt 12.60 tok/s, generation 1.95 tok/s; best so far but still low |

## Result log

Append each completed candidate here with:

- command identity;
- load success/failure;
- idle VRAM;
- API prompt tok/s;
- API generation tok/s;
- whether the candidate was stopped for being below threshold.

### S1 - Codacus-style single GPU0

- Command identity: `CUDA_VISIBLE_DEVICES=0`, no `-ts`, `-ncmoe 99`,
  `--moe-cache-slots 48`, q8/q8 KV, `-b 512 -ub 256`, 32k context, mmap,
  MTP draft `n-max=1`.
- Load: success, but slow mmap/page-cache load.
- API: `prompt_n=2173`, `prompt_ms=211627.665`, prompt 10.27 tok/s.
- API: `predicted_n=64`, `predicted_ms=41510.292`, generation 1.52 tok/s.
- MTP: `draft_n=33`, `draft_n_accepted=30`, about 90.9% acceptance.
- Decision: not viable as optimum; prefill only barely reaches the floor and
  decode is much worse than the dual-GPU live result.

### S2 - Dual-GPU cache 80,40

- Command identity: `-ts 10,7`, `-ncmoe 44`, `--moe-cache-slots 80,40`,
  q8/q8 KV, `-b 512 -ub 256`, 32k context, mmap, MTP draft `n-max=1`.
- Load: success.
- API: `prompt_n=2173`, `prompt_ms=176888.105`, prompt 12.28 tok/s.
- API: `predicted_n=64`, `predicted_ms=36298.22`, generation 1.76 tok/s.
- MTP: `draft_n=34`, `draft_n_accepted=29`, about 85.3% acceptance.
- Decision: better than S1, but still too slow for the target.

### S3 - Dual-GPU cache 88,40

- Command identity: `-ts 10,7`, `-ncmoe 44`, `--moe-cache-slots 88,40`,
  q8/q8 KV, `-b 512 -ub 256`, 32k context, mmap, MTP draft `n-max=1`.
- Load: success.
- API: `prompt_n=2173`, `prompt_ms=239026.774`, prompt 9.09 tok/s.
- API: `predicted_n=64`, `predicted_ms=35930.01`, generation 1.78 tok/s.
- MTP: `draft_n=34`, `draft_n_accepted=29`, about 85.3% acceptance.
- Decision: rejected; more GPU0 cache worsened prompt below the floor.

### S4 - Dual-GPU cache 80,40, batch 1024

- Command identity: `-ts 10,7`, `-ncmoe 44`, `--moe-cache-slots 80,40`,
  q8/q8 KV, `-b 1024 -ub 256`, 32k context, mmap, MTP draft `n-max=1`.
- Load: success.
- API: `prompt_n=2173`, `prompt_ms=188347.233`, prompt 11.54 tok/s.
- API: `predicted_n=64`, `predicted_ms=35455.828`, generation 1.81 tok/s.
- MTP: `draft_n=34`, `draft_n_accepted=29`, about 85.3% acceptance.
- Decision: rejected; larger batch did not beat S2 prefill and only
  marginally improved generation.

### S6 - Dual-GPU cache 80,40, no MTP

- Added after S1-S4 because API generation with MTP stayed around 1.5-1.8 tok/s
  despite 85-91% draft acceptance, while earlier `llama-bench` without MTP
  showed much better synthetic generation for the same broad placement.
- Command identity: `-ts 10,7`, `-ncmoe 44`, `--moe-cache-slots 80,40`,
  q8/q8 KV, no draft model, `-b 512 -ub 256`, 32k context, mmap.
- Load: success.
- API: `prompt_n=2173`, `prompt_ms=172454.602`, prompt 12.60 tok/s.
- API: `predicted_n=64`, `predicted_ms=32899.772`, generation 1.95 tok/s.
- Decision: best API result so far, but still far below the practical target.

### S5 - Codacus-style single GPU0, larger batch

- Command identity: `CUDA_VISIBLE_DEVICES=0`, no `-ts`, `-ncmoe 99`,
  `--moe-cache-slots 48`, q8/q8 KV, `-b 1024 -ub 512`, 32k context, mmap,
  MTP draft `n-max=1`.
- Load: success.
- API: `prompt_n=2173`, `prompt_ms=247718.255`, prompt 8.77 tok/s.
- API: `predicted_n=64`, `predicted_ms=41957.671`, generation 1.53 tok/s.
- MTP: `draft_n=33`, `draft_n_accepted=30`, about 90.9% acceptance.
- Decision: rejected; larger batch on single GPU worsened prompt below the
  floor and did not fix decode.

## Final ranking

| Rank | Candidate | Prompt tok/s | Generation tok/s | Notes |
| --- | --- | ---: | ---: | --- |
| 1 | S6 dual-GPU cache `80,40`, no MTP | 12.60 | 1.95 | Best API result, still not practical |
| 2 | S2 dual-GPU cache `80,40`, MTP | 12.28 | 1.76 | Best MTP result |
| 3 | S4 dual-GPU cache `80,40`, MTP, batch 1024 | 11.54 | 1.81 | Larger batch did not help enough |
| 4 | S1 single-GPU Codacus-style, batch 512 | 10.27 | 1.52 | MTP acceptance good, base decode poor |
| 5 | S3 dual-GPU cache `88,40`, MTP | 9.09 | 1.78 | More cache worsened prefill |
| 6 | S5 single-GPU Codacus-style, batch 1024 | 8.77 | 1.53 | Rejected |

Conclusion: on this i7-4790K/DDR3/Pascal system, Qwen3.8 Flash Next UD-IQ3_XXS
does not reach the useful 10-20 tok/s API target in generation. The best prefill
barely clears 10 tok/s, while generation remains below 2 tok/s. The bottleneck is
not missing MTP: MTP acceptance was high, but draft overhead did not improve
end-to-end API speed here. The most plausible limiting factors are the huge
mmap-backed MoE working set, DDR3/old CPU memory latency and bandwidth, SSD page
traffic, and Pascal-era CUDA kernels.

## Aggressive follow-up sweep

Requested on 2026-09-10 morning: run at least 10 more attempts before declaring
the model impractical. These tests keep the same safety constraints but try
thread count, CPU scheduling, cache distribution, KV type, context size, and
tail-GPU residency changes.

Result sink:

- JSONL: `qwen38_flash_sweep_results.jsonl`
- Harness: `tools/qwen38_flash_sweep.py`

| ID | Status | Purpose | Key flags |
| --- | --- | --- | --- |
| S7 | done - prompt 13.11 tok/s, generation 2.07 tok/s | Best no-MTP plus Codacus CPU scheduling flag | dual GPU, `-ncmoe 44`, cache `80,40`, q8/q8, `--no-sched-async-cpu`, `-t 4` |
| S8 | done - prompt 13.32 tok/s, generation 1.84 tok/s | Best no-MTP with 6 threads | dual GPU, `-ncmoe 44`, cache `80,40`, q8/q8, `-t 6` |
| S9 | done - prompt 13.89 tok/s, generation 1.91 tok/s | Best no-MTP with 3 threads | dual GPU, `-ncmoe 44`, cache `80,40`, q8/q8, `-t 3` |
| S10 | done - prompt 12.50 tok/s, generation 2.08 tok/s | Lower cache to reduce pressure/page churn | dual GPU, `-ncmoe 44`, cache `64,32`, q8/q8, `-t 4` |
| S11 | done - prompt 12.30 tok/s, generation 2.09 tok/s | Move cache toward GPU0 while leaving GPU1 room | dual GPU, `-ncmoe 44`, cache `96,32`, q8/q8, `-t 4` |
| S12 | done - prompt 12.63 tok/s, generation 2.07 tok/s | Mixed KV to reduce K footprint only | dual GPU, `-ncmoe 44`, cache `80,40`, q4/q8, `-t 4` |
| S13 | done - prompt 9.79 tok/s, generation 1.84 tok/s | Mixed KV to reduce V footprint only | dual GPU, `-ncmoe 44`, cache `80,40`, q8/q4, `-t 4` |
| S14 | done - prompt 11.10 tok/s, generation 2.15 tok/s | Check whether 16k context materially helps | dual GPU, `-ncmoe 44`, cache `80,40`, q8/q8, `-c 16384`, `-t 4` |
| S15 | done - prompt 11.66 tok/s, generation 2.11 tok/s | More fully resident tail experts if it fits | dual GPU, `-ncmoe 43`, cache `72,32`, q8/q8, `-t 4` |
| S16 | done - prompt 11.01 tok/s, generation 1.71 tok/s | Single-GPU Codacus-style without MTP overhead | `CUDA_VISIBLE_DEVICES=0`, no `-ts`, `-ncmoe 99`, cache `48`, q8/q8, no MTP |

The harness marks each row in this file as `running`/`done`/`failed` and appends
the measured prompt/generation rates in the result column.

### Aggressive sweep ranking

| Rank | Candidate | Prompt tok/s | Generation tok/s | Interpretation |
| --- | --- | ---: | ---: | --- |
| 1 | S14 16k context | 11.10 | 2.15 | Best generation, but only because context allocation is smaller |
| 2 | S15 `-ncmoe 43`, cache `72,32` | 11.66 | 2.11 | Slightly more resident tail experts helps decode a little |
| 3 | S11 cache `96,32` | 12.30 | 2.09 | More GPU0 cache does not break, but gain is tiny |
| 4 | S10 cache `64,32` | 12.50 | 2.08 | Lower cache is about tied with larger cache |
| 5 | S7 `--no-sched-async-cpu` | 13.11 | 2.07 | Helps prefill, not decode enough |
| 6 | S12 q4/q8 KV | 12.63 | 2.07 | K q4 is acceptable, but not faster enough |
| 7 | S9 3 threads | 13.89 | 1.91 | Best prefill, weaker decode |
| 8 | S13 q8/q4 KV | 9.79 | 1.84 | V q4 hurts both sides |
| 9 | S8 6 threads | 13.32 | 1.84 | More threads improve prefill, hurt decode |
| 10 | S16 single GPU no MTP | 11.01 | 1.71 | Single-GPU Codacus-style remains worse |

The 10 extra attempts confirm that the generation bottleneck is not a single
flag mistake. Disabling MTP, changing CPU scheduling, changing thread count,
moving cache, mixed KV, reducing context to 16k, and changing `n-cpu-moe` all
remain around 1.7-2.15 generation tok/s.

#!/usr/bin/env python3
"""Multi-turn stress benchmark with authoritative tokens and fail-closed gates."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import requests


TURNS_PROMPTS = [
    "Ciao! Spiegami brevemente quali sono le differenze architetturali chiave tra una GPU Pascal (GTX 1080 Ti) e una moderna architettura con Tensor Core.",
    "Molto chiaro. Ora approfondisci come funziona l'operazione di Matrix Multiplication a 8-bit (INT8) rispetto a FP16 su una scheda senza Tensor Core nativi.",
    "Tenendo a mente quanto hai appena detto, cosa succede se applichiamo il Multi-Token Prediction (MTP) a un modello Mixture of Experts su questo hardware?",
    "Come influisce la cache degli esperti MoE (hot/cold caching) sul traffico PCIe tra RAM di sistema e VRAM?",
    "Fammi un riassunto dei punti chiave toccati finora in questa conversazione, evidenziando le criticità di VRAM.",
    "Se dovessimo aggiungere una seconda scheda asimmetrica (come una GTX 1070 da 8GB), come andrebbe ripartito il carico dei tensori?",
    "Scrivi un breve frammento in pseudocodice o C per spiegare la logica di smistamento dei token tra i diversi esperti.",
    "Analizza il codice appena scritto e dimmi quali sono i potenziali colli di bottiglia per quanto riguarda la latenza di memoria.",
    "Supponiamo ora che la finestra di contesto raggiunga decine di migliaia di token: come deve gestire la KV-cache l'algoritmo Flash Attention per non esaurire la memoria?",
    "Ottimo lavoro fin qui. Concludi sintetizzando l'intera discussione in una tabella comparativa finale.",
]
SYSTEM_PROMPT = "Sei un assistente AI specializzato in architetture hardware e sistemi distribuiti. Rispondi in modo tecnico e conciso."


def get_vram_stats(timeout=10):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
         "--format=csv,noheader,nounits"], text=True, timeout=timeout,
    )
    gpus = []
    for line in output.strip().splitlines():
        index, name, used, total = (part.strip() for part in line.split(","))
        gpus.append({"id": index, "name": name, "used_mb": int(used), "total_mb": int(total)})
    if not gpus:
        raise ValueError("No GPU telemetry available")
    return gpus


def read_completion(response):
    """A chunk can contain zero, one or multiple tokens; use usage instead."""
    content, reasoning = [], []
    usage = timings = finish_reason = first_at = None
    done = False
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            raise ValueError("SSE chunk is not an object")
        if chunk.get("error"):
            raise ValueError(f"Server SSE error: {chunk['error']}")
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        if chunk.get("timings") is not None:
            timings = chunk["timings"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            for key, destination in (("content", content), ("reasoning_content", reasoning)):
                value = delta.get(key)
                if value:
                    if not isinstance(value, str):
                        raise ValueError(f"Invalid {key}")
                    if first_at is None:
                        first_at = time.perf_counter()
                    destination.append(value)
    if not done or finish_reason not in ("stop", "length"):
        raise ValueError("Incomplete or failed stream: missing DONE or valid finish_reason")
    if not content:
        raise ValueError("No final assistant content (possibly reasoning budget exhausted)")
    if not isinstance(usage, dict):
        raise ValueError("Missing authoritative token usage")
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise ValueError("Invalid completion_tokens")
    if not isinstance(timings, dict):
        raise ValueError("Missing native server timings")
    for key in ("predicted_per_second", "prompt_per_second"):
        value = timings.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < float("inf"):
            raise ValueError(f"Invalid native timing: {key}")
    return {
        "content": "".join(content), "reasoning_content": "".join(reasoning),
        "usage": usage, "timings": timings, "finish_reason": finish_reason,
        "first_at": first_at,
    }


def summarize(results):
    if not results:
        raise ValueError("No measurements")
    summary = {}
    for key in ("tok_per_sec", "prefill_tps", "ttft_s"):
        values = [result[key] for result in results]
        if any(not isinstance(value, (int, float)) or not 0 < value < float("inf") for value in values):
            raise ValueError(f"Invalid measurements: {key}")
        summary[key] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
        }
    return summary


def assess_reports(baseline, candidate, latency_rationale=None):
    if latency_rationale is not None and not latency_rationale.strip():
        raise ValueError("Latency acceptance requires a nonempty rationale")
    if not baseline.get("passed") or baseline["workload"] != candidate["workload"]:
        raise ValueError("Baseline failed or workload identity mismatch")
    if baseline["workload"]["repeats"] < 3:
        raise ValueError("Gate requires at least three repetitions")
    expected = baseline["workload"]["repeats"] * baseline["workload"]["turns"]
    if len(baseline["results"]) != expected or len(candidate["results"]) != expected:
        raise ValueError("Gate requires complete repetitions")
    reference, current = summarize(baseline["results"]), summarize(candidate["results"])
    changes = {key: (current[key]["mean"] / reference[key]["mean"] - 1) * 100 for key in current}
    failures = [key for key, limit, lower in (
        ("tok_per_sec", -5, True), ("prefill_tps", -10, True), ("ttft_s", 10, False),
    ) if (changes[key] < limit - 1e-9 if lower else changes[key] > limit + 1e-9)]
    accepted_latency = failures == ["ttft_s"] and latency_rationale is not None
    return {
        "deltas_percent": changes,
        "threshold_violations": failures,
        "thresholds_passed": not failures,
        "accepted": not failures or accepted_latency,
        "latency_rationale": latency_rationale,
        "latency_exception_used": accepted_latency,
        "ttft_delta_seconds": current["ttft_s"]["mean"] - reference["ttft_s"]["mean"],
    }


def compare_reports(baseline, candidate):
    assessment = assess_reports(baseline, candidate)
    require_accepted(assessment)
    return assessment["deltas_percent"]


def require_accepted(assessment):
    if not assessment["accepted"]:
        raise ValueError(
            f"Performance gate failed: {assessment['threshold_violations']}; "
            f"deltas={assessment['deltas_percent']}"
        )


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--api-key", default=os.getenv("LLAMA_API_KEY", ""))
    parser.add_argument("--model", default="Qwen36 her vis")
    parser.add_argument("--turns", type=positive_int, default=10)
    parser.add_argument("--max-tokens", type=positive_int, default=2048)
    parser.add_argument("--timeout", type=positive_int, default=60, help="Stream inactivity timeout (seconds)")
    parser.add_argument("--stream", action="store_true", default=True, help="Compatibility option; always streaming")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="JSON results, never includes API key")
    parser.add_argument("--baseline", type=Path, help="Reference JSON for regression gate")
    parser.add_argument("--accept-latency-regression", metavar="RATIONALE",
                        help="Explicitly accept only a TTFT breach; preserve thresholds and rationale")
    parser.add_argument("--reasoning", choices=("on", "off"), default="on")
    parser.add_argument("--no-cache-prompt", action="store_true", help="Reprocess full history on every turn")
    args = parser.parse_args()
    if args.accept_latency_regression is not None and not args.accept_latency_regression.strip():
        parser.error("latency acceptance requires a nonempty rationale")
    if not 0 <= args.temperature < float("inf"):
        parser.error("temperature must be finite and nonnegative")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    workload = {
        "model": args.model, "turns": args.turns, "max_tokens": args.max_tokens,
        "seed": args.seed, "temperature": args.temperature, "reasoning": args.reasoning,
        "repeats": args.repeats, "prompts": TURNS_PROMPTS, "system": SYSTEM_PROMPT,
        "cache_prompt": not args.no_cache_prompt,
    }
    report = {"schema_version": 1, "workload": workload, "results": [], "passed": False,
              "measurements_complete": False}
    status = 1
    try:
        print(f"Target: {args.url} | Model: {args.model}", flush=True)
        print(get_vram_stats(), flush=True)
        with requests.get(f"{args.url}/v1/models", headers=headers, timeout=5) as response:
            response.raise_for_status()
            response.json()
        for repeat in range(args.repeats):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for i in range(args.turns):
                messages.append({"role": "user", "content": TURNS_PROMPTS[i % len(TURNS_PROMPTS)]})
                payload = {
                    "model": args.model, "messages": messages, "max_tokens": args.max_tokens,
                    "temperature": args.temperature, "seed": args.seed, "stream": True,
                    "stream_options": {"include_usage": True}, "timings_per_token": True,
                    "chat_template_kwargs": {"enable_thinking": args.reasoning == "on"},
                    "cache_prompt": not args.no_cache_prompt,
                }
                started = time.perf_counter()
                with requests.post(f"{args.url}/v1/chat/completions", headers=headers,
                                   json=payload, stream=True, timeout=args.timeout) as response:
                    response.raise_for_status()
                    result = read_completion(response)
                elapsed = time.perf_counter() - started
                messages.append({"role": "assistant", "content": result["content"],
                                 "reasoning_content": result["reasoning_content"]})
                first_at = result.pop("first_at")
                result.update({
                    "repeat": repeat + 1, "turn": i + 1, "ttft_s": first_at - started,
                    "elapsed_s": elapsed, "tok_per_sec": result["timings"]["predicted_per_second"],
                    "prefill_tps": result["timings"]["prompt_per_second"],
                    "total_tokens": result["usage"]["completion_tokens"], "vram": get_vram_stats(),
                })
                report["results"].append(result)
                print(f"Repeat {repeat + 1}, turn {i + 1}: {result['tok_per_sec']:.2f} t/s | "
                      f"TTFT {result['ttft_s']:.3f}s | {result['total_tokens']} tokens | "
                      f"{result['finish_reason']}", flush=True)
        report["summary"] = summarize(report["results"])
        report["measurements_complete"] = True
        if args.baseline:
            assessment = assess_reports(
                json.loads(args.baseline.read_text()), report, args.accept_latency_regression,
            )
            report["performance_assessment"] = assessment
            report["comparison"] = assessment["deltas_percent"]
            if assessment["latency_exception_used"]:
                print(f"LATENCY EXCEPTION: {json.dumps(assessment)}", file=sys.stderr, flush=True)
            require_accepted(assessment)
        report["passed"] = True
        status = 0
        print(json.dumps(report["summary"], indent=2), flush=True)
    except (requests.RequestException, OSError, ValueError, RuntimeError, KeyError, TypeError,
            subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(f"BENCHMARK FAILED: {error}", file=sys.stderr)
    finally:
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())

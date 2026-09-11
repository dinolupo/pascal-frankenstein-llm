#!/usr/bin/env python3
"""Run bounded Qwen3.8 Flash llama-server API sweeps safely."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "QWEN38_FLASH_SERVER_TEST_PLAN.md"
RESULTS = ROOT / "qwen38_flash_sweep_results.jsonl"
LOG_DIR = ROOT / "qwen38_flash_sweep_logs"

SERVER = Path("/home/dino/proj/pascal-frankenstein-llm/llama.cpp/build-pascal-qwen4exp/bin/llama-server")
MODEL = Path("/home/dino/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")
MTP = Path("/home/dino/proj/genAI/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf")
PROFILE = ROOT / "moe-traces/qwen38-flash-next-merged.csv"

HOST = "127.0.0.1"
PORT = 8080
BASE_URL = f"http://{HOST}:{PORT}"


TESTS = [
    {
        "id": "S7",
        "purpose": "Best no-MTP plus --no-sched-async-cpu",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "80,40",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": ["--no-sched-async-cpu"],
    },
    {
        "id": "S8",
        "purpose": "Best no-MTP with 6 threads",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "80,40",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "6",
        "extra": [],
    },
    {
        "id": "S9",
        "purpose": "Best no-MTP with 3 threads",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "80,40",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "3",
        "extra": [],
    },
    {
        "id": "S10",
        "purpose": "Lower cache to reduce pressure/page churn",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "64,32",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
    {
        "id": "S11",
        "purpose": "More GPU0 cache while leaving GPU1 room",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "96,32",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
    {
        "id": "S12",
        "purpose": "Mixed KV q4/q8",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "80,40",
        "ctk": "q4_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
    {
        "id": "S13",
        "purpose": "Mixed KV q8/q4",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "80,40",
        "ctk": "q8_0",
        "ctv": "q4_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
    {
        "id": "S14",
        "purpose": "16k context sanity check",
        "cuda": None,
        "mtp": False,
        "ncmoe": "44",
        "ts": "10,7",
        "cache": "80,40",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "16384",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
    {
        "id": "S15",
        "purpose": "More fully resident tail experts",
        "cuda": None,
        "mtp": False,
        "ncmoe": "43",
        "ts": "10,7",
        "cache": "72,32",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
    {
        "id": "S16",
        "purpose": "Single-GPU Codacus-style without MTP",
        "cuda": "0",
        "mtp": False,
        "ncmoe": "99",
        "ts": None,
        "cache": "48",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "ctx": "32768",
        "batch": "512",
        "ubatch": "256",
        "threads": "4",
        "extra": [],
    },
]


def request_json(path: str, payload: dict[str, object] | None = None, timeout: int = 5) -> dict[str, object] | None:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None


def update_plan(test_id: str, status: str, result: str | None = None) -> None:
    text = PLAN.read_text()
    pattern = re.compile(rf"(\| {re.escape(test_id)} \| )[^|]+(\| .*? \| .*? \|)(?:[^\n]*)", re.MULTILINE)

    def repl(match: re.Match[str]) -> str:
        suffix = f" {result} |" if result else match.group(2)
        return f"{match.group(1)}{status} {suffix}"

    PLAN.write_text(pattern.sub(repl, text))


def append_result(result: dict[str, object]) -> None:
    with RESULTS.open("a") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def command_for(test: dict[str, object]) -> list[str]:
    cmd = [
        str(SERVER),
        "-m",
        str(MODEL),
    ]
    if test["mtp"]:
        cmd += [
            "-md",
            str(MTP),
            "-ngld",
            "0",
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "1",
        ]
    cmd += [
        "-ngl",
        "99",
        "-ncmoe",
        str(test["ncmoe"]),
    ]
    if test["ts"]:
        cmd += ["-ts", str(test["ts"])]
    cmd += [
        "--load-mode",
        "mmap",
        "-fit",
        "off",
        "-fa",
        "on",
        "-ctk",
        str(test["ctk"]),
        "-ctv",
        str(test["ctv"]),
        "-c",
        str(test["ctx"]),
        "-np",
        "1",
        "--cache-reuse",
        "256",
        "-b",
        str(test["batch"]),
        "-ub",
        str(test["ubatch"]),
        "-t",
        str(test["threads"]),
        "--moe-cache-profile",
        str(PROFILE),
        "--moe-cache-slots",
        str(test["cache"]),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--jinja",
        "--no-webui",
    ]
    cmd += list(test["extra"])
    return cmd


def wait_ready(proc: subprocess.Popen[bytes], seconds: int = 540) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        health = request_json("/health", timeout=2)
        if health and health.get("status") == "ok":
            return True
        time.sleep(2)
    return False


def terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def run_api() -> dict[str, object]:
    base = "Qwen3.8 Flash Next local mmap MoE prefill screening on Pascal hardware. "
    prompt = (base * 120) + "\nAnswer in one concise paragraph: identify the most likely bottleneck."
    payload = {
        "prompt": prompt,
        "n_predict": 64,
        "temperature": 0,
        "cache_prompt": False,
        "stream": False,
    }
    started = time.time()
    data = request_json("/completion", payload=payload, timeout=900)
    wall_s = time.time() - started
    if data is None:
        return {"ok": False, "wall_s": wall_s, "error": "api request failed"}
    timings = data.get("timings", {})
    if not isinstance(timings, dict):
        timings = {}
    prompt_n = timings.get("prompt_n") or data.get("prompt_n")
    prompt_ms = timings.get("prompt_ms") or data.get("prompt_ms")
    predicted_n = timings.get("predicted_n") or data.get("predicted_n")
    predicted_ms = timings.get("predicted_ms") or data.get("predicted_ms")
    prompt_tps = prompt_n / (prompt_ms / 1000.0) if prompt_n and prompt_ms else None
    predicted_tps = predicted_n / (predicted_ms / 1000.0) if predicted_n and predicted_ms else None
    return {
        "ok": True,
        "wall_s": wall_s,
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "prompt_tps": prompt_tps,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "predicted_tps": predicted_tps,
        "timings": timings,
    }


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    env_base = os.environ.copy()
    env_base["LD_LIBRARY_PATH"] = f"{SERVER.parent}:{env_base.get('LD_LIBRARY_PATH', '')}"

    for test in TESTS:
        test_id = str(test["id"])
        update_plan(test_id, "running")
        log_path = LOG_DIR / f"{test_id}.log"
        env = env_base.copy()
        if test["cuda"] is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(test["cuda"])
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        cmd = command_for(test)
        print(f"=== {test_id} starting: {test['purpose']} ===", flush=True)
        print(" ".join(cmd), flush=True)
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        result: dict[str, object] = {
            "id": test_id,
            "purpose": test["purpose"],
            "cmd": cmd,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        try:
            if not wait_ready(proc):
                result.update({"ok": False, "error": "server did not become ready", "exit_code": proc.poll()})
                update_plan(test_id, "failed", "server did not become ready")
                append_result(result)
                print(json.dumps(result, indent=2, sort_keys=True), flush=True)
                continue
            result["ready_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            api = run_api()
            result["api"] = api
            prompt_tps = api.get("prompt_tps")
            predicted_tps = api.get("predicted_tps")
            if isinstance(prompt_tps, (int, float)) and isinstance(predicted_tps, (int, float)):
                summary = f"prompt {prompt_tps:.2f} tok/s, generation {predicted_tps:.2f} tok/s"
                update_plan(test_id, "done", summary)
            else:
                update_plan(test_id, "failed", "API timing unavailable")
            append_result(result)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        finally:
            terminate(proc)
            time.sleep(5)

    return 0


if __name__ == "__main__":
    sys.exit(main())

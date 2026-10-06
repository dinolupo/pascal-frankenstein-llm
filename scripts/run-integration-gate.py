#!/usr/bin/env python3
"""Run sequential profile gates on an owned loopback server, never the installation."""

import argparse
import base64
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import struct
import sys
import threading
import time
import zlib

import requests

from benchmark_stress import assess_reports, get_vram_stats, require_accepted
from nvidia_power import cuda_gpu_uuids, read_power_modes, require_max_performance


def parse_source_commit(version):
    match = re.search(r"^version:.*\bcommit ([0-9a-f]{7,40})\)", version, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Cannot identify the server's embedded source commit: {version}")
    return match.group(1)


def server_environment(server):
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(server.resolve().parent)
    # This fork checks presence, not value: even "0" enables peer-access setup.
    environment.pop("GGML_CUDA_P2P", None)
    environment.pop("LLAMA_API_KEY", None)
    return environment


def monitor(stop, output, errors, power_uuids=None, gpu_ready=None, gpu_lock=None, gpu_probe_timeout=10):
    try:
        if not 0 < gpu_probe_timeout < float("inf"):
            raise ValueError("GPU probe timeout must be finite and positive")
        with output.open("w") as stream:
            while not stop.is_set():
                sample = {"time": time.time()}
                sample["meminfo"] = Path("/proc/meminfo").read_text()
                with gpu_lock if gpu_lock is not None else nullcontext():
                    if gpu_ready is None or gpu_ready.is_set():
                        probe_started = time.monotonic()
                        sample["gpus"] = get_vram_stats(timeout=gpu_probe_timeout)
                        if power_uuids is not None:
                            sample["power_modes"] = read_power_modes()
                        sample["gpu_probe_elapsed_s"] = time.monotonic() - probe_started
                        sample["gpu_sample_completed_at"] = time.time()
                    else:
                        sample["gpu_sampling_deferred"] = True
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                if "power_modes" in sample:
                    require_max_performance(sample["power_modes"], power_uuids)
                stop.wait(2)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        errors.append(str(error))
        stop.set()


def red_image():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = (b"\x00" + b"\xff\x00\x00" * 64) * 64
    png = (b"\x89PNG\r\n\x1a\n" +
           chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)) +
           chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def functional_checks(url, model, vision):
    records = []
    messages = []
    for prompt, expected in (
        ("What is 17 times 19? Answer with the number only.", "323"),
        ("What was the number you just answered? Answer with the number only.", "323"),
    ):
        messages.append({"role": "user", "content": prompt})
        with requests.post(f"{url}/v1/chat/completions", json={
            "model": model, "messages": messages, "max_tokens": 128,
            "temperature": 0, "seed": 123,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=300) as response:
            response.raise_for_status()
            result = response.json()
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"]["content"].strip() != expected:
            raise RuntimeError(f"Arithmetic/EOG/prefix functional check failed: {result}")
        messages.append({"role": "assistant", "content": choice["message"]["content"]})
        records.append(result)
    if vision:
        with requests.post(f"{url}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is the dominant color? Answer with one English color word only."},
                {"type": "image_url", "image_url": {"url": red_image()}},
            ]}], "max_tokens": 128, "temperature": 0, "seed": 123,
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=300) as response:
            response.raise_for_status()
            result = response.json()
        if result["choices"][0]["message"]["content"].strip().lower().strip(".") != "red":
            raise RuntimeError(f"Vision functional check failed: {result}")
        records.append(result)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--previous", type=Path, help="Also compare previous accepted checkpoint")
    parser.add_argument("--accept-latency-regression", metavar="RATIONALE",
                        help="Explicit latency-only exception; never waive throughput or functional failures")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--reasoning", choices=("on", "off"), default="off")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--require-max-performance", action="store_true",
                        help="Verify NV-CONTROL preference for every CUDA GPU before and during the gate")
    parser.add_argument("--expected-source", metavar="SHA",
                        help="Require this embedded binary source commit, not the current checkout")
    args = parser.parse_args()
    if args.accept_latency_regression is not None and not args.accept_latency_regression.strip():
        parser.error("latency acceptance requires a nonempty rationale")
    if args.expected_source is not None and not re.fullmatch("[0-9a-f]{7,40}", args.expected_source):
        parser.error("expected source must be a hexadecimal Git commit SHA (7 to 40 characters)")
    if args.results.exists():
        parser.error("results directory already exists; preserve old evidence")
    processes = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
    if any(name.strip() in ("llama-server", "llama-bench", "llama-cli") for name in processes):
        parser.error("Another model process is active; stop it before running the gate")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    args.results.mkdir(parents=True)
    url = f"http://127.0.0.1:{args.port}"
    source_diff = subprocess.check_output(
        ["git", "-C", str(args.server.resolve().parent), "diff", "HEAD"],
    )
    (args.results / "source.diff").write_bytes(source_diff)
    metadata = {
        "server": str(args.server.resolve()),
        "preset": str(args.preset.resolve()),
        "preset_sha256": hashlib.sha256(args.preset.read_bytes()).hexdigest(),
        "model_aliases": ["Qwen36 her jin", "Qwen36 her vis"],
        "reasoning": args.reasoning,
        "temperature": args.temperature,
        "require_max_performance": args.require_max_performance,
        "source_sha": subprocess.check_output(
            ["git", "-C", str(args.server.resolve().parent), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
        "runner_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("run-integration-gate.py", "benchmark_stress.py", "nvidia_power.py")
        },
        "library_sha256": {
            library.name: hashlib.sha256(library.read_bytes()).hexdigest()
            for library in sorted(args.server.resolve().parent.glob("*.so*"))
        },
        "source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "latency_rationale": args.accept_latency_regression,
        "opt_in_environment": {
            name: os.environ.get(name)
            for name in ("GGML_CUDA_REGISTER_HOST", "GGML_SCHED_PREFETCH_EXPERTS")
        },
        "performance_assessments": {},
        "measurements_complete": False,
        "passed": False,
    }
    stop = threading.Event()
    errors = []
    environment = server_environment(args.server)
    server = None
    sampler = None
    # Keep probes out of bulk registration after an intermittent loading timeout.
    # RAM remains sampled; full GPU telemetry resumes after warm-up.
    gpu_ready = threading.Event() if "GGML_CUDA_REGISTER_HOST" in environment else None
    gpu_lock = threading.Lock() if gpu_ready is not None else None
    metadata["gpu_polling_deferred_during_registration_load"] = gpu_ready is not None
    try:
        metadata["server_version"] = subprocess.check_output(
            [str(args.server.resolve()), "--version"], env=environment,
            stderr=subprocess.STDOUT, text=True, timeout=30,
        )
        embedded_commit = parse_source_commit(metadata["server_version"])
        binary_source = subprocess.check_output(
            ["git", "-C", str(args.server.resolve().parent), "rev-parse", f"{embedded_commit}^{{commit}}"],
            text=True,
        ).strip()
        metadata["checkout_sha"] = metadata["source_sha"]
        metadata["source_sha"] = binary_source
        metadata["source_diff_applies_to_binary"] = binary_source == metadata["checkout_sha"]
        if args.expected_source and not binary_source.startswith(args.expected_source):
            raise RuntimeError(f"Binary source mismatch: expected {args.expected_source}, got {binary_source}")
        power_uuids = None
        if args.require_max_performance:
            power_uuids = cuda_gpu_uuids()
            metadata["power_modes_start"] = read_power_modes()
            require_max_performance(metadata["power_modes_start"], power_uuids)
        for reference_path in (args.baseline, args.previous):
            if reference_path:
                reference = json.loads((reference_path / "gate.json").read_text())
                if not reference.get("passed") or reference["preset_sha256"] != metadata["preset_sha256"]:
                    raise ValueError("Reference gate failed or preset changed")
        with (args.results / "server.log").open("w") as log:
            server = subprocess.Popen(
                [str(args.server.resolve()), "--models-preset", str(args.preset.resolve()),
                 "--host", "127.0.0.1", "--port", str(args.port), "--models-max", "1"],
                stdout=log, stderr=subprocess.STDOUT, env=environment,
            )
            deadline = time.monotonic() + 300
            ready = False
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}")
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    ready = response.ok
                    response.close()
                except requests.ConnectionError:
                    pass
                if ready:
                    break
                time.sleep(1)
            if not ready:
                raise RuntimeError("Server did not become healthy")
            sampler = threading.Thread(
                target=monitor, args=(stop, args.results / "memory.jsonl", errors,
                                     power_uuids, gpu_ready, gpu_lock),
            )
            sampler.start()
            for profile, model in (("text", "Qwen36 her jin"), ("vision", "Qwen36 her vis")):
                if gpu_ready is not None:
                    with gpu_lock:
                        gpu_ready.clear()
                # Warm up separately; never include it in the measured conversations.
                response = requests.post(f"{url}/v1/chat/completions", json={
                    "model": model, "messages": [{"role": "user", "content": "Rispondi solo: pronto."}],
                    "max_tokens": 128, "temperature": 0, "seed": 123,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=300)
                response.raise_for_status()
                warmup = response.json()
                response.close()
                if not warmup.get("choices"):
                    raise RuntimeError("Warm-up returned no choices")
                if gpu_ready is not None:
                    with gpu_lock:
                        gpu_ready.set()
                    if args.require_max_performance:
                        modes = read_power_modes()
                        metadata[f"power_modes_loaded_{profile}"] = modes
                        require_max_performance(modes, power_uuids)
                (args.results / f"warmup-{profile}.json").write_text(json.dumps(warmup, indent=2))
                functional = functional_checks(url, model, profile == "vision")
                (args.results / f"functional-{profile}.json").write_text(json.dumps(functional, indent=2))
                if errors:
                    raise RuntimeError(f"Memory/power monitor failed: {errors}")
                command = [
                    sys.executable, str(Path(__file__).with_name("benchmark_stress.py")),
                    "--url", url, "--model", model, "--turns", "10",
                    "--max-tokens", str(args.max_tokens), "--repeats", "3",
                    "--timeout", "180", "--seed", "123",
                    "--temperature", str(args.temperature), "--reasoning", args.reasoning,
                    "--output", str(args.results / f"{profile}.json"),
                ]
                if args.baseline:
                    command.extend(["--baseline", str(args.baseline / f"{profile}.json")])
                if args.accept_latency_regression is not None:
                    command.extend(["--accept-latency-regression", args.accept_latency_regression])
                with (args.results / f"{profile}.log").open("w") as output:
                    completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                               env=environment, check=False)
                report = json.loads((args.results / f"{profile}.json").read_text())
                if "performance_assessment" in report:
                    metadata["performance_assessments"][f"baseline_{profile}"] = report["performance_assessment"]
                if completed.returncode != 0 or not report.get("passed"):
                    raise RuntimeError(
                        f"{profile} benchmark failed (exit {completed.returncode}): "
                        f"{report.get('error', 'report not accepted')}"
                    )
                if args.previous:
                    previous = json.loads((args.previous / f"{profile}.json").read_text())
                    assessment = assess_reports(previous, report, args.accept_latency_regression)
                    metadata["performance_assessments"][f"previous_{profile}"] = assessment
                    metadata[f"previous_{profile}_comparison"] = assessment["deltas_percent"]
                    if assessment["latency_exception_used"]:
                        print(f"LATENCY EXCEPTION ({profile}, previous): {json.dumps(assessment)}",
                              file=sys.stderr, flush=True)
                    require_accepted(assessment)
                if errors:
                    raise RuntimeError(f"Memory monitor failed: {errors}")
            if args.require_max_performance:
                metadata["power_modes_end"] = read_power_modes()
                require_max_performance(metadata["power_modes_end"], power_uuids)
            metadata["measurements_complete"] = True
            metadata["performance_thresholds_passed"] = all(
                assessment["thresholds_passed"]
                for assessment in metadata["performance_assessments"].values()
            )
            metadata["passed"] = True
    except (OSError, ValueError, requests.RequestException, subprocess.SubprocessError,
            RuntimeError, KeyError, IndexError, TypeError) as error:
        metadata["error"] = str(error)
        print(f"GATE FAILED: {error}", file=sys.stderr)
    finally:
        stop.set()
        if sampler:
            sampler.join()
        if errors:
            metadata["passed"] = False
            metadata["monitor_errors"] = errors
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        (args.results / "gate.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if metadata["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

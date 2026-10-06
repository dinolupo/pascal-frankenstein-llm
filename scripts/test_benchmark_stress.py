import argparse
import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import requests

import benchmark_stress as benchmark

SPEC = importlib.util.spec_from_file_location(
    "integration_gate", Path(__file__).with_name("run-integration-gate.py"),
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class Response:
    def __init__(self, chunks):
        self.chunks = chunks

    def iter_lines(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield b"data: [DONE]" if chunk == "[DONE]" else (
                b"data: " + json.dumps(chunk).encode()
            )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": []}


def chunks():
    return [
        {"choices": [{"delta": {"reasoning_content": "Thought.", "content": "Several tokens."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 12, "prompt_tokens": 20},
         "timings": {"predicted_per_second": 45, "prompt_per_second": 150}},
        "[DONE]",
    ]


class BenchmarkTests(unittest.TestCase):
    def test_usage_not_chunks_and_reasoning_separation(self):
        result = benchmark.read_completion(Response(chunks()))
        self.assertEqual(result["usage"]["completion_tokens"], 12)
        self.assertEqual(result["content"], "Several tokens.")
        self.assertEqual(result["reasoning_content"], "Thought.")
        self.assertIsNotNone(result["first_at"])

    def test_incomplete_missing_metrics_and_server_error(self):
        cases = [
            chunks()[:-1], chunks()[:2] + ["[DONE]"],
            [{"error": {"message": "failed"}}, "[DONE]"],
            [{"choices": [{"delta": {"reasoning_content": "Only thinking"}}]},
             *chunks()[1:]],
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                benchmark.read_completion(Response(case))

    def test_timeout_propagates(self):
        with self.assertRaises(requests.exceptions.Timeout):
            benchmark.read_completion(Response([requests.exceptions.Timeout("idle")]))

    def test_malformed_sse(self):
        with patch.object(Response, "iter_lines", return_value=iter([b"data: {invalid"])):
            with self.assertRaises(ValueError):
                benchmark.read_completion(Response([]))

    def test_gate_exact_thresholds_and_failures(self):
        baseline = {
            "passed": True, "workload": {"repeats": 3, "turns": 1},
            "results": [{"tok_per_sec": 100, "prefill_tps": 100, "ttft_s": 1}] * 3,
        }
        candidate = copy.deepcopy(baseline)
        candidate["results"] = [{"tok_per_sec": 95, "prefill_tps": 90, "ttft_s": 1.1}] * 3
        benchmark.compare_reports(baseline, candidate)
        for metric, value in (("tok_per_sec", 94.9), ("prefill_tps", 89.9), ("ttft_s", 1.101)):
            failing = copy.deepcopy(candidate)
            for result in failing["results"]:
                result[metric] = value
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                benchmark.compare_reports(baseline, failing)
        candidate["results"].pop()
        with self.assertRaises(ValueError):
            benchmark.compare_reports(baseline, candidate)

    def test_bad_baseline_and_workload(self):
        baseline = {"passed": False}
        with self.assertRaises(ValueError):
            benchmark.compare_reports(baseline, {})
        baseline = {"passed": True, "workload": {"repeats": 1}}
        with self.assertRaises(ValueError):
            benchmark.compare_reports(baseline, {"workload": {"repeats": 1}})

    def test_latency_exception_preserves_breach_and_cost(self):
        baseline = {
            "passed": True, "workload": {"repeats": 3, "turns": 1},
            "results": [{"tok_per_sec": 100, "prefill_tps": 100, "ttft_s": 1}] * 3,
        }
        candidate = copy.deepcopy(baseline)
        candidate["results"] = [{"tok_per_sec": 100, "prefill_tps": 100, "ttft_s": 1.25}] * 3
        strict = benchmark.assess_reports(baseline, candidate)
        self.assertFalse(strict["accepted"])
        rationale = "Token-prefix mismatch requires hybrid checkpoint rollback."
        accepted = benchmark.assess_reports(baseline, candidate, rationale)
        self.assertTrue(accepted["accepted"])
        self.assertFalse(accepted["thresholds_passed"])
        self.assertTrue(accepted["latency_exception_used"])
        self.assertEqual(accepted["threshold_violations"], ["ttft_s"])
        self.assertEqual(accepted["latency_rationale"], rationale)
        self.assertEqual(accepted["ttft_delta_seconds"], 0.25)
        for metric in ("tok_per_sec", "prefill_tps"):
            failing = copy.deepcopy(candidate)
            for result in failing["results"]:
                result[metric] = 80
            assessment = benchmark.assess_reports(baseline, failing, rationale)
            self.assertFalse(assessment["accepted"])
            with self.assertRaises(ValueError):
                benchmark.require_accepted(assessment)
        candidate["results"].pop()
        with self.assertRaises(ValueError):
            benchmark.assess_reports(baseline, candidate, rationale)
        with self.assertRaises(ValueError):
            benchmark.assess_reports(baseline, baseline, " ")

    def test_invalid_input(self):
        for value in ("0", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark.positive_int(value)
        with self.assertRaises(ValueError):
            benchmark.summarize([{"tok_per_sec": float("nan")}])

    def test_cli_auth_and_history(self):
        with patch.object(sys, "argv", ["benchmark", "--turns", "2", "--api-key", "test-only"]), \
             patch.object(benchmark, "get_vram_stats", return_value=[{"id": "0"}]), \
             patch.object(benchmark.requests, "get", return_value=Response([])) as get, \
             patch.object(benchmark.requests, "post", side_effect=[Response(chunks()), Response(chunks())]) as post:
            self.assertEqual(benchmark.main(), 0)
            self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer test-only")
            history = post.call_args.kwargs["json"]["messages"]
            self.assertEqual(history[2]["content"], "Several tokens.")
            self.assertEqual(history[2]["reasoning_content"], "Thought.")

    def test_cli_timeout_is_failure(self):
        with patch.object(sys, "argv", ["benchmark", "--turns", "1"]), \
             patch.object(benchmark, "get_vram_stats", return_value=[]), \
             patch.object(benchmark.requests, "get", return_value=Response([])), \
             patch.object(benchmark.requests, "post", side_effect=requests.exceptions.Timeout("idle")):
            self.assertEqual(benchmark.main(), 1)

    def test_cli_latency_acceptance_and_throughput_failure(self):
        baseline = {
            "passed": True,
            "workload": {
                "model": "Qwen36 her vis", "turns": 1, "max_tokens": 2048,
                "seed": 123, "temperature": 0.6, "reasoning": "on", "repeats": 3,
                "prompts": benchmark.TURNS_PROMPTS, "system": benchmark.SYSTEM_PROMPT,
                "cache_prompt": True,
            },
            "results": [{"tok_per_sec": 45, "prefill_tps": 150, "ttft_s": 0.1}] * 3,
        }
        for exception, slow_generation, expected in (
            (False, False, 1), (True, False, 0), (True, True, 1),
        ):
            with self.subTest(exception=exception, slow_generation=slow_generation), \
                 tempfile.TemporaryDirectory() as directory:
                reference = Path(directory) / "baseline.json"
                output = Path(directory) / "result.json"
                reference.write_text(json.dumps(baseline))
                argv = ["benchmark", "--turns", "1", "--repeats", "3",
                        "--baseline", str(reference), "--output", str(output)]
                if exception:
                    argv.extend(["--accept-latency-regression", "Reproduced prefix rollback"])
                responses = chunks()
                if slow_generation:
                    responses[2]["timings"]["predicted_per_second"] = 30
                # Each request calls the clock at start, first content, and finish.
                with patch.object(sys, "argv", argv), \
                     patch.object(benchmark, "get_vram_stats", return_value=[{"id": "0"}]), \
                     patch.object(benchmark.requests, "get", return_value=Response([])), \
                     patch.object(benchmark.requests, "post", side_effect=[
                         Response(responses), Response(responses), Response(responses),
                     ]), patch.object(benchmark.time, "perf_counter",
                                      side_effect=[0, 0.2, 1, 2, 2.2, 3, 4, 4.2, 5]):
                    self.assertEqual(benchmark.main(), expected)
                report = json.loads(output.read_text())
                self.assertTrue(report["measurements_complete"])
                self.assertEqual(report["passed"], expected == 0)
                self.assertFalse(report["performance_assessment"]["thresholds_passed"])

    def test_runner_unsets_presence_based_p2p(self):
        with patch.dict(os.environ, {"GGML_CUDA_P2P": "0", "LLAMA_API_KEY": "test-only"}):
            environment = gate.server_environment(Path("/tmp/candidate/bin/llama-server"))
            self.assertNotIn("GGML_CUDA_P2P", environment)
            self.assertNotIn("LLAMA_API_KEY", environment)
            self.assertEqual(os.environ["GGML_CUDA_P2P"], "0")
            self.assertEqual(environment["LD_LIBRARY_PATH"], "/tmp/candidate/bin")

    def test_embedded_source_identity_not_checkout(self):
        self.assertEqual(gate.parse_source_commit(
            "version: 0.3.0-dev (build 10819, commit 47b3df22d)\nbuilt with GNU"),
            "47b3df22d")
        self.assertEqual(gate.parse_source_commit(
            "CUDA initialization\nversion: 0.6.0 (build 1, commit 590b56282)\n"),
            "590b56282")
        for invalid in ("", "unidentified binary", "version: 0.6.0 (commit unknown)"):
            with self.subTest(version=invalid), self.assertRaises(RuntimeError):
                gate.parse_source_commit(invalid)

    def test_image_fixture_is_png(self):
        prefix, encoded = gate.red_image().split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertTrue(base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n"))

    def test_power_monitor_records_failure_and_stops(self):
        modes = [{"nvctrl_gpu": 0, "uuid": "GPU-a", "power_mizer_mode": 2}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "memory.jsonl"
            stop, errors = threading.Event(), []
            with patch.object(gate, "get_vram_stats", return_value=[{"id": "0"}]), \
                 patch.object(gate.subprocess, "check_output", return_value="GPU state"), \
                 patch.object(gate.Path, "read_text", return_value="MemAvailable: 1000 kB"), \
                 patch.object(gate, "read_power_modes", return_value=modes):
                gate.monitor(stop, output, errors, ["GPU-a"])
            self.assertTrue(stop.is_set())
            self.assertEqual(len(errors), 1)
            self.assertIn("Prefer maximum performance", errors[0])
            self.assertEqual(json.loads(output.read_text())["power_modes"], modes)

    def test_default_monitor_does_not_require_x_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "memory.jsonl"
            stop, errors = threading.Event(), []
            with patch.object(gate, "get_vram_stats", return_value=[{"id": "0"}]), \
                 patch.object(gate.subprocess, "check_output", return_value="GPU state"), \
                 patch.object(gate.Path, "read_text", return_value="MemAvailable: 1000 kB"), \
                 patch.object(gate, "read_power_modes", side_effect=AssertionError("unexpected X query")), \
                 patch.object(stop, "wait", side_effect=lambda timeout: stop.set()):
                gate.monitor(stop, output, errors)
            self.assertEqual(errors, [])
            self.assertNotIn("power_modes", json.loads(output.read_text()))

    def test_registration_load_defers_gpu_probes_then_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "memory.jsonl"
            stop, ready, errors = threading.Event(), threading.Event(), []
            modes = [{"nvctrl_gpu": 0, "uuid": "GPU-a", "power_mizer_mode": 1}]

            def advance(timeout):
                if ready.is_set():
                    stop.set()
                else:
                    ready.set()

            with patch.object(gate, "get_vram_stats", return_value=[{"id": "0"}]) as vram, \
                 patch.object(gate.subprocess, "check_output", return_value="GPU state"), \
                 patch.object(gate.Path, "read_text", return_value="MemAvailable: 1000 kB"), \
                 patch.object(gate, "read_power_modes", return_value=modes) as policy, \
                 patch.object(stop, "wait", side_effect=advance):
                gate.monitor(stop, output, errors, ["GPU-a"], ready, threading.Lock())
            samples = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(errors, [])
            self.assertTrue(samples[0]["gpu_sampling_deferred"])
            self.assertNotIn("gpus", samples[0])
            self.assertNotIn("power_modes", samples[0])
            self.assertEqual(samples[1]["power_modes"], modes)
            self.assertNotIn("gpu_sampling_deferred", samples[1])
            vram.assert_called_once()
            policy.assert_called_once()


if __name__ == "__main__":
    unittest.main()

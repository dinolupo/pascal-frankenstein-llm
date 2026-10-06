import argparse
import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
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

    def test_runner_unsets_presence_based_p2p(self):
        with patch.dict(os.environ, {"GGML_CUDA_P2P": "0", "LLAMA_API_KEY": "test-only"}):
            environment = gate.server_environment(Path("/tmp/candidate/bin/llama-server"))
            self.assertNotIn("GGML_CUDA_P2P", environment)
            self.assertNotIn("LLAMA_API_KEY", environment)
            self.assertEqual(os.environ["GGML_CUDA_P2P"], "0")
            self.assertEqual(environment["LD_LIBRARY_PATH"], "/tmp/candidate/bin")

    def test_image_fixture_is_png(self):
        prefix, encoded = gate.red_image().split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertTrue(base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()

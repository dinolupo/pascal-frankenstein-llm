import copy
import unittest
from unittest.mock import patch

import nvidia_power as power


class PowerTests(unittest.TestCase):
    def setUp(self):
        self.modes = [
            {"nvctrl_gpu": 0, "uuid": "GPU-a", "power_mizer_mode": 1},
            {"nvctrl_gpu": 1, "uuid": "GPU-b", "power_mizer_mode": 1},
        ]

    def test_maximum_and_uuid_mapping_independent_of_order(self):
        power.require_max_performance(self.modes, ["GPU-b", "GPU-a"])

    def test_all_nonmaximum_modes_rejected(self):
        for mode in (0, 2, 3, -1):
            modes = copy.deepcopy(self.modes)
            modes[1]["power_mizer_mode"] = mode
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                power.require_max_performance(modes, ["GPU-a", "GPU-b"])

    def test_empty_duplicate_and_incomplete_coverage_rejected(self):
        for modes in ([], [self.modes[0]] * 2, [self.modes[0]]):
            with self.subTest(modes=modes), self.assertRaises(RuntimeError):
                power.require_max_performance(modes, ["GPU-a", "GPU-b"])

    def test_cuda_identity_query_and_empty_result(self):
        with patch.object(power.subprocess, "check_output", return_value="GPU-b\nGPU-a\n"):
            self.assertEqual(power.cuda_gpu_uuids(), ["GPU-b", "GPU-a"])
        with patch.object(power.subprocess, "check_output", return_value=""):
            with self.assertRaises(RuntimeError):
                power.cuda_gpu_uuids()

    def test_missing_library_is_not_a_successful_fallback(self):
        with patch.object(power.c, "CDLL", side_effect=OSError("library unavailable")):
            with self.assertRaises(OSError):
                power.read_power_modes()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from omega_v2_2a_d3_diagnostic.metrics import (
    diagnostic_metrics,
    local_ulp_scalar,
    old_comparator_argmax_details,
    sum_u4_variants,
    tensor_raw_sha256,
)


class D3DiagnosticCpuTests(unittest.TestCase):
    def test_local_ulp_zero_sign_and_normal(self) -> None:
        smallest_subnormal = torch.nextafter(torch.tensor(0.0), torch.tensor(float("inf"))).item()
        one_ulp = (torch.nextafter(torch.tensor(1.0), torch.tensor(float("inf"))) - 1.0).item()
        self.assertEqual(local_ulp_scalar(0.0), smallest_subnormal)
        self.assertEqual(local_ulp_scalar(1.0), one_ulp)
        self.assertEqual(local_ulp_scalar(-1.0), one_ulp)
        self.assertIsNone(local_ulp_scalar(float("nan")))

    def test_known_old_metrics_and_argmax_indices(self) -> None:
        g_r = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)
        sum_g_u = torch.tensor([0.5, 0.5, -1.0], dtype=torch.float32)
        metrics = diagnostic_metrics(g_r, sum_g_u)
        self.assertEqual(metrics["max_abs"], 1.0)
        self.assertEqual(metrics["max_rel_old"], 1.0)
        self.assertAlmostEqual(metrics["E_L2"], math.sqrt(1.5) / math.sqrt(5.0))
        self.assertEqual(metrics["E_inf"], 0.5)
        details = old_comparator_argmax_details(g_r, sum_g_u, metrics)
        self.assertEqual(details["argmax_max_rel_old"]["flat_index"], 0)
        self.assertEqual(details["argmax_max_rel_old"]["multi_index_row_major"], [0])
        self.assertEqual(details["argmax_max_rel_old"]["old_denominator"], 0.5)
        self.assertFalse(details["argmax_max_rel_old"]["floor_1e_6_active"])
        self.assertEqual(details["argmax_max_abs"]["flat_index"], 2)

    def test_floor_activity_and_tensor_norm_ulp_context(self) -> None:
        g_r = torch.tensor([0.0, 2e-7], dtype=torch.float32)
        sum_g_u = torch.tensor([5e-7, 3e-7], dtype=torch.float32)
        metrics = diagnostic_metrics(g_r, sum_g_u)
        details = old_comparator_argmax_details(g_r, sum_g_u, metrics)
        self.assertTrue(details["argmax_max_rel_old"]["floor_1e_6_active"])
        self.assertEqual(details["argmax_max_rel_old"]["old_denominator"], torch.tensor(1e-6).item())
        self.assertIn("ulp_norm_S_stack_inf", details["tensor_magnitude_context"])
        self.assertIn("max_abs_over_ulp_norm_S_stack_inf", details["tensor_magnitude_context"])

    def test_five_summation_orders_known_result(self) -> None:
        grads = [torch.tensor([float(i), float(2 * i)], dtype=torch.float32) for i in (1, 2, 3, 4)]
        sums = sum_u4_variants(grads)
        expected32 = torch.tensor([10.0, 20.0], dtype=torch.float32)
        expected64 = torch.tensor([10.0, 20.0], dtype=torch.float64)
        for name in ("S_forward", "S_reverse", "S_pairwise", "S_stack"):
            self.assertTrue(torch.equal(sums[name], expected32), name)
        self.assertTrue(torch.equal(sums["S_fp64"], expected64))
        with self.assertRaises(ValueError):
            sum_u4_variants(grads[:3])

    def test_raw_tensor_sha_matches_name_dtype_shape_bytes_recipe(self) -> None:
        name = "run_01/W_Q/gR"
        tensor = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        payload = (
            name.encode("utf-8") + b"\0" + str(tensor.dtype).encode("ascii")
            + json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            + tensor.contiguous().numpy().tobytes(order="C")
        )
        self.assertEqual(tensor_raw_sha256(name, tensor), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()

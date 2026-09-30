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

from omega_v2_2a_d3q.config import CALIBRATION_SEED, make_seed_plan
from omega_v2_2a_d3q.metrics import (
    GATE_A_LIMIT,
    GATE_B_LIMIT,
    ORACLE_INF_LIMIT,
    ORACLE_L2_LIMIT,
    evaluate_oracle_gate,
    evaluate_primary_gates,
    inclusive_threshold_pass,
    local_ulp_scalar,
    old_max_rel_diagnostic,
    scale_aware_gate_c_pass,
    sum_u4_diagnostics,
    tensor_raw_sha256,
    to_fp32_round_nearest_even,
)


class D3QMetricUnitTests(unittest.TestCase):
    def test_calibration_seed_derivation_and_mode_guard(self) -> None:
        plan = make_seed_plan(CALIBRATION_SEED, mode="qa")
        self.assertEqual(plan.weight_seed, CALIBRATION_SEED)
        self.assertEqual(plan.input_seed, 20261938)
        self.assertEqual(plan.loss_w_seed, 20262938)
        with self.assertRaisesRegex(ValueError, "QA_SEED_REJECTED"):
            make_seed_plan(None, mode="qa")
        with self.assertRaisesRegex(ValueError, "CALIBRATION_SEED_FORBIDDEN"):
            make_seed_plan(CALIBRATION_SEED, mode="official")

    def test_fp32_ulp_rule_and_round_to_nearest_even(self) -> None:
        zero = torch.tensor(0.0, dtype=torch.float32)
        inf = torch.tensor(float("inf"), dtype=torch.float32)
        self.assertEqual(local_ulp_scalar(0.0), float(torch.nextafter(zero, inf).item()))
        one_ulp = float((torch.nextafter(torch.tensor(1.0), inf) - 1.0).item())
        self.assertEqual(local_ulp_scalar(1.0), one_ulp)
        self.assertEqual(local_ulp_scalar(-1.0), one_ulp)
        largest = torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32)
        toward_zero = torch.tensor(0.0, dtype=torch.float32)
        self.assertEqual(local_ulp_scalar(largest), float((largest - torch.nextafter(largest, toward_zero)).item()))
        self.assertIsNone(local_ulp_scalar(float("nan")))
        halfway = 1.0 + 2.0**-24
        self.assertEqual(float(to_fp32_round_nearest_even(halfway).item()), 1.0)

    def test_gate_a_b_and_oracle_synthetic_pass_fail(self) -> None:
        one_thousand = torch.ones(1000, dtype=torch.float64)
        gate_a_pass_sum = one_thousand.clone()
        gate_a_pass_sum[0] += 1.9e-7
        gate_a_pass = evaluate_primary_gates(one_thousand, gate_a_pass_sum)
        self.assertTrue(gate_a_pass["gates"]["A"]["pass"])
        self.assertTrue(gate_a_pass["gates"]["B"]["pass"])

        gate_a_fail_sum = torch.ones(1, dtype=torch.float64)
        gate_a_fail_sum[0] += 2.2e-7
        gate_a_fail = evaluate_primary_gates(torch.ones(1, dtype=torch.float64), gate_a_fail_sum)
        self.assertFalse(gate_a_fail["gates"]["A"]["pass"])
        self.assertTrue(gate_a_fail["gates"]["B"]["pass"])

        gate_b_fail_sum = one_thousand.clone()
        gate_b_fail_sum[0] += 6e-7
        gate_b_fail = evaluate_primary_gates(one_thousand, gate_b_fail_sum)
        self.assertTrue(gate_b_fail["gates"]["A"]["pass"])
        self.assertFalse(gate_b_fail["gates"]["B"]["pass"])
        self.assertTrue(gate_b_fail["gates"]["C"]["pass"])

        oracle_pass = evaluate_oracle_gate(torch.zeros(1, dtype=torch.float64), torch.tensor([1e-27], dtype=torch.float64))
        self.assertTrue(oracle_pass["pass"])
        self.assertIsNone(oracle_pass["max_abs_FP64_threshold"])
        oracle_fail = evaluate_oracle_gate(torch.zeros(1, dtype=torch.float64), torch.tensor([1e-25], dtype=torch.float64))
        self.assertFalse(oracle_fail["pass"])
        self.assertFalse(oracle_fail["gates"]["E_inf_FP64"]["pass"])

    def test_inclusive_gate_boundaries(self) -> None:
        self.assertTrue(inclusive_threshold_pass(GATE_A_LIMIT, GATE_A_LIMIT))
        self.assertFalse(inclusive_threshold_pass(math.nextafter(GATE_A_LIMIT, math.inf), GATE_A_LIMIT))
        self.assertTrue(inclusive_threshold_pass(GATE_B_LIMIT, GATE_B_LIMIT))
        self.assertFalse(inclusive_threshold_pass(math.nextafter(GATE_B_LIMIT, math.inf), GATE_B_LIMIT))
        self.assertTrue(inclusive_threshold_pass(ORACLE_L2_LIMIT, ORACLE_L2_LIMIT))
        self.assertFalse(inclusive_threshold_pass(math.nextafter(ORACLE_L2_LIMIT, math.inf), ORACLE_L2_LIMIT))
        self.assertTrue(inclusive_threshold_pass(ORACLE_INF_LIMIT, ORACLE_INF_LIMIT))
        self.assertFalse(inclusive_threshold_pass(math.nextafter(ORACLE_INF_LIMIT, math.inf), ORACLE_INF_LIMIT))
        eight_ulps = 8.0 * local_ulp_scalar(1.0)
        self.assertTrue(scale_aware_gate_c_pass(eight_ulps, 1.0)[0])
        self.assertFalse(scale_aware_gate_c_pass(math.nextafter(eight_ulps, math.inf), 1.0)[0])

    def test_gate_c_scale_rounding_and_nonfinite(self) -> None:
        g_r = torch.tensor([1.0, 1.0], dtype=torch.float64)
        step = 2.0**-23
        exactly_eight_ulps = torch.tensor([1.0 + 8 * step, 1.0], dtype=torch.float64)
        result = evaluate_primary_gates(g_r, exactly_eight_ulps)
        self.assertTrue(result["gates"]["C"]["pass"])
        over_eight_ulps = torch.tensor([1.0 + 9 * step, 1.0], dtype=torch.float64)
        result_over = evaluate_primary_gates(g_r, over_eight_ulps)
        self.assertFalse(result_over["gates"]["C"]["pass"])
        nonfinite = evaluate_primary_gates(torch.tensor([float("nan")], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
        self.assertFalse(nonfinite["finite"])
        self.assertFalse(any(gate["pass"] for gate in nonfinite["gates"].values()))

    def test_fp32_summations_and_old_max_rel_are_diagnostic_only(self) -> None:
        grads = [torch.tensor([float(i)], dtype=torch.float32) for i in (1, 2, 3, 4)]
        sums = sum_u4_diagnostics(grads)
        for name in ("S_forward", "S_reverse", "S_pairwise", "S_stack"):
            self.assertTrue(torch.equal(sums[name], torch.tensor([10.0], dtype=torch.float32)))
        diagnostic = old_max_rel_diagnostic(torch.tensor([10.0]), sums["S_stack"])
        self.assertEqual(diagnostic["max_rel_old"], 0.0)
        self.assertNotIn("S_reverse_equal_gR_NON_GATE", diagnostic)
        self.assertTrue(torch.equal(sums["S_reverse"], torch.tensor([10.0], dtype=torch.float32)))
        with self.assertRaises(ValueError):
            sum_u4_diagnostics(grads[:3])

    def test_tensor_raw_hash_recipe(self) -> None:
        name = "dryrun_slot/W_Q/gR_cuda_fp32"
        tensor = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        payload = (
            name.encode("utf-8")
            + b"\0"
            + str(tensor.dtype).encode("ascii")
            + json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
            + tensor.contiguous().numpy().tobytes(order="C")
        )
        self.assertEqual(tensor_raw_sha256(name, tensor), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()

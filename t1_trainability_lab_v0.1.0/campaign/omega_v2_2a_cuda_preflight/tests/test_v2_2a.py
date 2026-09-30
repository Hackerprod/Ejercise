
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent / "omega_v2_0_conformance"))
sys.path.insert(0, str(PACKAGE_ROOT.parent))

import torch

from omega_v2.core import ContractualCoreBlock
from omega_v2.ledger import state_dict_sha256
from omega_v2_2a_cuda_preflight.checks import (
    compare_cuda_to_cpu,
    iso_flop_gate,
    parameter_storage_gate,
    schema_sha256,
)
from omega_v2_2a_cuda_preflight import runner as v2_2a_runner
from omega_v2_2a_cuda_preflight.runner import PACKAGE_ROOT as RUNNER_PACKAGE_ROOT
from omega_v2_2a_cuda_preflight.variants import build_initial_variants


class V22ACpuUnitTests(unittest.TestCase):
    def test_v2_0_seeded_r4_u4_counts_and_initial_clone(self) -> None:
        pair = build_initial_variants(device="cpu")
        report = parameter_storage_gate(pair["r4_cpu"], pair["u4_cpu"])
        self.assertTrue(pair["initial_clone_report"]["initial_values_bitwise_equal"])
        self.assertTrue(report["pass"])
        self.assertEqual(report["R4_unique_parameters"], 1_048_576)
        self.assertEqual(report["U4_unique_parameters"], 4_194_304)

    def test_v2_0_flop_ledger_exact_d256_m8_b8_k4(self) -> None:
        pair = build_initial_variants(device="cpu")
        report = iso_flop_gate(pair["r4_cpu"], pair["u4_cpu"])
        self.assertTrue(report["pass"])
        self.assertEqual(report["per_round_b1_flops"], 16_842_752)
        self.assertEqual(report["k4_b1_flops"], 67_371_008)
        self.assertEqual(report["k4_b8_R4_forward_flops"], 538_968_064)
        self.assertEqual(report["k4_b8_U4_forward_flops"], 538_968_064)
        self.assertEqual(report["R4_non_gemm_counts_total"], report["U4_non_gemm_counts_total"])

    def test_state_schema_and_v2_0_value_hash_are_stable(self) -> None:
        left = ContractualCoreBlock(256, dtype=torch.float32, seed=20260930)
        right = ContractualCoreBlock(256, dtype=torch.float32, seed=20260930)
        self.assertEqual(schema_sha256(left), schema_sha256(right))
        self.assertEqual(state_dict_sha256(left), state_dict_sha256(right))

    def test_cuda_cpu_comparator_reports_md318_metrics(self) -> None:
        reference = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)
        candidate = reference.clone()
        report = compare_cuda_to_cpu(reference, candidate)
        self.assertEqual(report["max_abs"], 0.0)
        self.assertEqual(report["max_scaled_error"], 0.0)
        self.assertEqual(report["L2_relative_error"], 0.0)
        self.assertTrue(report["pass"])

    def test_package_import_smoke_does_not_start_preflight(self) -> None:
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(RUNNER_PACKAGE_ROOT, PACKAGE_ROOT)
        self.assertTrue(callable(v2_2a_runner.main))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

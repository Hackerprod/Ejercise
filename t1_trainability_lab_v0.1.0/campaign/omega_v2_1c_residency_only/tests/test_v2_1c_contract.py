from __future__ import annotations

from pathlib import Path
import unittest

UNIT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = UNIT_ROOT.parent
KQ2_ROOT = CAMPAIGN_ROOT / "omega_v2_1b_candidate_02"

TEST_NAMES = [
    "test_v2_1c_candidate02_kernel_frozen_hash",
    "test_v2_1c_attempt02_preserved",
    "test_v2_1c_attempt03_same_72_cells",
    "test_v2_1c_schedule_affinity_and_qpc_match_attempt02",
    "test_v2_1c_candidate02_correctness",
    "test_v2_1c_no_kernel_candidate_or_tuning",
    "test_v2_1c_residency_gate_d512_m4_k8",
    "test_v2_1c_causal_control_delta_cb",
    "test_v2_1c_c_cold_control_gt_a",
    "test_v2_1c_matrixization_gate",
    "test_v2_1c_stability_gate_nine_primary_cells",
    "test_v2_1c_m1_residency_diagnostic_non_gate",
    "test_v2_1c_attempt02_vs_v2_1c_comparison_report",
    "test_v2_1c_conformance_block",
]


class V21cContractTests(unittest.TestCase):
    def test_approved_new_test_inventory_is_unique(self) -> None:
        self.assertEqual(len(TEST_NAMES), 14)
        self.assertEqual(len(set(TEST_NAMES)), len(TEST_NAMES))

    def test_72_cell_contract_cardinality(self) -> None:
        self.assertEqual(len((512, 640)) * len((1, 4, 8, 16)) * len((1, 4, 8)) * len(("A", "B", "C")), 72)

    def test_primary_stability_contract_has_nine_cells(self) -> None:
        cells = [(512, 4, k, variant) for variant in "ABC" for k in (1, 8)]
        cells.extend((512, m, 8, "A") for m in (1, 8, 16))
        self.assertEqual(len(cells), 9)

    def test_physical_harness_links_candidate02_kernel_tus(self) -> None:
        cmake = (UNIT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("${KQ2_SOURCE_DIR}/q4_kernel_candidate2.cpp", cmake)
        self.assertIn("${KQ2_SOURCE_DIR}/full_block_candidate2.cpp", cmake)
        self.assertIn("omega_v2_1b_candidate_02/src", cmake)
        self.assertIn("select_p_cores_by_h0=select_p_cores_by_h0_v2_1c_frozen", cmake)
        self.assertIn("measure_h0_cores=measure_h0_cores_v2_1c_frozen", cmake)
        self.assertNotIn("q4_kernel_candidate1.cpp", cmake)
        self.assertNotIn("${V2_1_SOURCE_DIR}/v2_full_block.cpp", cmake)

    def test_sweep_consumes_sealed_selection_without_remeasurement(self) -> None:
        selector = (UNIT_ROOT / "src" / "select_v2_1c_workers.cpp").read_text(encoding="utf-8")
        wrapper = (UNIT_ROOT / "src" / "main_v2_1c.cpp").read_text(encoding="utf-8")
        self.assertIn("OMEGA_V2_1C_SELECTED_CPU_SET_IDS", selector)
        self.assertIn("OMEGA_V2_1C_SELECTED_V_I", selector)
        self.assertIn("return {};", selector)
        self.assertIn("--core-selection-preflight", wrapper)
        self.assertIn("monotonic_qpc_test(error)", wrapper)
        self.assertIn("measure_qpc_overhead_ns()", wrapper)

    def test_preflight_cli_enforces_md313_order_and_explicit_sweep_go(self) -> None:
        phases = (UNIT_ROOT / "scripts" / "v2_1c_phases.py").read_text(encoding="utf-8")
        self.assertLess(phases.index("def binding_stage"), phases.index("def core_selection_stage"))
        self.assertLess(phases.index("def core_selection_stage"), phases.index("def correctness_stage"))
        self.assertIn("--seal-preflight-only", phases)
        self.assertIn("--go-medicion", phases)
        self.assertIn("requires the judge's explicit GO medicion", phases)
        self.assertIn("DIAGNOSTIC_ONLY / NOT_PAIRED_KERNEL_COMPARISON", phases)

    def test_no_native_speed_gate_is_declared_for_v2_1c(self) -> None:
        spec = (UNIT_ROOT / "OMEGA_V2_1C_RESIDENCY_ONLY_SPEC.md").read_text(encoding="utf-8")
        self.assertIn("There is no PyTorch S_native gate", spec)
        self.assertIn("S_native", spec)
        self.assertIn("S_native is not a gate in this unit", spec)


if __name__ == "__main__":
    unittest.main()

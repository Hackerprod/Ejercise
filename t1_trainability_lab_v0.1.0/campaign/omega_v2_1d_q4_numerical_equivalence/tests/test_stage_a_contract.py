from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

UNIT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = UNIT_ROOT.parent
sys.path.insert(0, str(UNIT_ROOT / "scripts"))

import generate_calibration_states as states
import run_v2_1d_stage_a as stage_a_launcher


class StageAContractTests(unittest.TestCase):
    def test_frozen_spec_status_and_stage_a_stops(self) -> None:
        spec = (UNIT_ROOT / "OMEGA_V2_1D_STAGE_A_SPEC_FROZEN.md").read_text(encoding="utf-8")
        self.assertIn("FROZEN — MD/330 + MD/331 incorporated", spec)
        self.assertIn("OMEGA-V2-1D-D640-Q4-CANONICAL-V1", spec)
        self.assertIn("historical_pre_Q4_FP32_bytes_available = false", spec)
        self.assertIn("K            = {1,4}", spec)
        self.assertIn("Do not add K8 to the scientific grid", spec)
        self.assertIn("do not execute scientific cells", spec.lower())

    def test_grid_cardinality_and_only_qa_cli_modes(self) -> None:
        count = len((512,640)) * len((1,4,8,16)) * len((1,4)) * len(states.FAMILIES)
        self.assertEqual(count, 32)
        runner = (UNIT_ROOT / "scripts" / "run_v2_1d_stage_a_qa.py").read_text(encoding="utf-8")
        self.assertIn("--dry-run", runner)
        self.assertIn("--companion-binding-only", runner)
        self.assertIn("--d640-binding-only", runner)
        self.assertNotIn('"--run-stage-a"', runner)
        self.assertNotIn('"--launch-scientific"', runner)

    def test_official_launcher_is_guarded_and_not_used_by_qa_runner(self) -> None:
        launcher=(UNIT_ROOT/"scripts"/"run_v2_1d_stage_a.py").read_text(encoding="utf-8")
        qa=(UNIT_ROOT/"scripts"/"run_v2_1d_stage_a_qa.py").read_text(encoding="utf-8")
        for required in ("COMPANION_BINDING_PASS","D640_Q4_BINDING_PASS","working tree is not clean","source seal","STAGE_A_ONE_CALIBRATION_GO","--run-calibration","wall_clock_operational_only_ns","EXPECTED_UNIT_TEST_COUNT","unit_tests_report.json","collect_artifacts","git_head_commit","merge-base","stage_a_one_shot_ids"):
            self.assertIn(required,launcher)
        self.assertNotIn("run_v2_1d_stage_a.py",qa.split("def main()",1)[-1])
        self.assertNotIn("perf_"+"counter",launcher)
        self.assertEqual(launcher.count("import create_source_seal"),1)
        self.assertEqual(launcher.count("same_sha256_hex(")-1,4)
        postseal=(UNIT_ROOT/"tests"/"verify_preconditions_real_postseal.py").read_text(encoding="utf-8")
        self.assertIn("verify_preconditions",postseal)
        self.assertIn("--run-calibration",postseal)
        seal_script=(UNIT_ROOT/"scripts"/"create_source_seal.py").read_text(encoding="utf-8")
        for required in ("supersedes_source_seal_commit","supersedes_source_seal_sha256","launcher_sha256_case_normalization_pre_science","PRE_SCIENTIFIC_ABORT_00.json"):
            self.assertIn(required,seal_script)

    def test_sha256_hex_case_comparison(self) -> None:
        same="A1B2C3D4"*8
        self.assertTrue(stage_a_launcher.same_sha256_hex(same,same.lower()))
        self.assertFalse(stage_a_launcher.same_sha256_hex(same,("a1b2c3d4"*7)+"00000000"))

    def test_historical_formula_state_bytes(self) -> None:
        d, m = 512, 1
        raw = states.historical_state_bytes(d,m)
        self.assertEqual(len(raw),d*m*4)
        decoded=[row[0] for row in struct.iter_unpack("<f",raw)]
        expected=[(((i*37+d+m)%127)-63)/256.0 for i in range(d*m)]
        self.assertEqual(decoded,expected)

    def test_cal_randn_seed_shape_dtype_and_reproducibility(self) -> None:
        d,m=512,4
        first,seed=states.cal_randn_state_bytes(d,m)
        second,seed2=states.cal_randn_state_bytes(d,m)
        self.assertEqual(seed,20260930+d+m)
        self.assertEqual(seed2,seed)
        self.assertEqual(first,second)
        self.assertEqual(len(first),d*m*4)

    def test_native_state_path_does_not_regenerate_torch_rng(self) -> None:
        native=(UNIT_ROOT/"src"/"stage_a_main.cpp").read_text(encoding="utf-8")
        self.assertNotIn("torch::",native)
        self.assertNotIn("torch.randn",native)
        self.assertIn("load_float_file",native)

    def test_companion_links_frozen_candidate_translation_units(self) -> None:
        cmake=(UNIT_ROOT/"CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("${KQ2_SOURCE_DIR}/q4_kernel_candidate2.cpp",cmake)
        self.assertIn("${KQ2_SOURCE_DIR}/full_block_candidate2.cpp",cmake)
        self.assertNotIn("candidate2.cpp\n  src/",cmake)
        self.assertIn("/arch:AVX2",cmake)
        self.assertIn("/fp:precise",cmake)

    def test_d640_canonical_encoding_contract(self) -> None:
        native=(UNIT_ROOT/"src"/"stage_a_execution.cpp").read_text(encoding="utf-8")
        expected_order=("W_Q","W_K","W_V","W_O","W_gate","W_up","W_down")
        positions=[native.index(f'"{name}"') for name in expected_order]
        self.assertEqual(positions,sorted(positions))
        for contract in ("OMEGA-V2-1D-D640-Q4-CANONICAL-V1","u32(2)","u64(static_cast<std::uint64_t>(matrix.packed.logical_bytes))","packed.logical_bytes","scales_fp16.logical_bytes"):
            self.assertIn(contract,native)
        self.assertIn("15453147065333665836",native)

    def test_candidate_scalar_trace_mirror_and_checkpoints_exist(self) -> None:
        trace=(UNIT_ROOT/"src"/"stage_a_trace.cpp").read_text(encoding="utf-8")
        execution=(UNIT_ROOT/"src"/"stage_a_execution.cpp").read_text(encoding="utf-8")
        for symbol in ("scalar_q4_round_trace","capture_candidate_round","attention_logits","mlp_normalized","attention_probabilities"):
            self.assertIn(symbol,trace)
        self.assertIn("candidate_provenance",execution)
        self.assertIn("scalar_provenance",execution)
        for invalid in ("/".join(("FROZEN_DIRECT","VALIDATED_MIRROR")),"/".join(("RECONSTRUCTED_DIAGNOSTIC","VALIDATED_MIRROR")),"/".join(("FROZEN_DIRECT","FROZEN_DIRECT"))):
            self.assertNotIn(invalid,execution)
        main=(UNIT_ROOT/"src"/"stage_a_main.cpp").read_text(encoding="utf-8")
        self.assertIn("bitwise_equal(mirror_state, frozen_scalar.state)",main)

    def test_primary_metric_definitions_are_symmetric(self) -> None:
        metrics=(UNIT_ROOT/"src"/"stage_a_metrics.cpp").read_text(encoding="utf-8")
        self.assertIn("(std::max)(result.candidate_l2, result.reference_l2)",metrics)
        self.assertIn("(std::max)(result.candidate_inf, result.reference_inf)",metrics)
        self.assertIn("std::max)(std::fabs(c), std::fabs(r)), scale_floor",metrics)
        self.assertIn("HISTORICAL_COMPATIBILITY_DIAGNOSTIC",metrics)
        self.assertIn("1e-12",metrics)

    def test_native_execution_path_contains_no_qpc_calls(self) -> None:
        for name in ("stage_a_main.cpp","stage_a_execution.cpp"):
            source=(UNIT_ROOT/"src"/name).read_text(encoding="utf-8").lower()
            self.assertNotIn("qpc_",source)
            self.assertNotIn("query_hardware",source)

    def test_dry_run_reports_plan_without_native_invocation(self) -> None:
        process=subprocess.run([sys.executable,str(UNIT_ROOT/"scripts"/"run_v2_1d_stage_a_qa.py"),"--dry-run"],capture_output=True,text=True,check=True)
        report=json.loads(process.stdout)
        self.assertEqual(report["scientific_cells_planned"],32)
        self.assertFalse(report["native_executable_invoked"])
        self.assertFalse(report["official_launcher_invoked"])


if __name__ == "__main__":
    unittest.main()

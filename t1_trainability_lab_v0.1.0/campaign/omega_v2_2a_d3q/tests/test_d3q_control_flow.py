from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
import symtable
import sys
import tempfile
import unittest
from unittest import mock

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from omega_v2_2a_d3q.config import DryRunSeedSlot
from omega_v2_2a_d3q.core import run_calibration_seed_cpu_qa
from omega_v2_2a_d3q import runner


def _unresolved_globals(path: Path) -> list[str]:
    table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    module_bound = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()
    }
    builtin_names = set(dir(builtins)) | {
        "__name__", "__file__", "__package__", "__spec__", "__loader__", "__cached__", "__builtins__"
    }
    missing: set[str] = set()

    def visit(scope) -> None:
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in module_bound and symbol.get_name() not in builtin_names:
                missing.add(symbol.get_name())
        for child in scope.get_children():
            visit(child)

    visit(table)
    return sorted(missing)


class D3QControlFlowTests(unittest.TestCase):
    def test_cpu_calibration_qa_only_does_not_query_cuda(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA availability query")), \
             mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA synchronize")), \
             mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("CUDA memory reset")):
            result = run_calibration_seed_cpu_qa(20260930)
        self.assertTrue(result["pass"])
        self.assertEqual(result["seed_plan"]["master_seed"], 20260930)
        self.assertFalse(result["cuda_used"])
        self.assertIsNone(result["D3Q_verdict"])

    def test_pre_cuda_official_dry_run_and_launch_plan_do_not_create_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "official_slot"
            dry_run = runner.pre_cuda_official_path_dry_run(result_root)
            launch_plan = runner.prepare_official_launch(result_root)
            self.assertEqual(dry_run["phase"], "PRE_CUDA_OFFICIAL_PATH_DRY_RUN")
            self.assertFalse(dry_run["result_slot_created"])
            self.assertFalse(launch_plan["slot_created"])
            self.assertFalse(result_root.exists())

    def test_official_path_rejects_no_go_and_existing_slot_before_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / "absent_slot"
            with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA query before GO")):
                with self.assertRaisesRegex(RuntimeError, "explicit --go-d3q-official"):
                    runner.run_official_d3q(go_d3q_official=False, result_root=absent)
            self.assertFalse(absent.exists())

            existing = Path(temporary) / "existing_slot"
            existing.mkdir()
            with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA query for existing slot")):
                with self.assertRaises(FileExistsError):
                    runner.run_official_d3q(go_d3q_official=True, result_root=existing)
            self.assertTrue(existing.is_dir())

    def test_full_official_control_flow_dry_run_five_symbolic_slots_and_artifacts(self) -> None:
        calls: list[str] = []

        def stub(case: DryRunSeedSlot, mark_boundary):
            self.assertIsInstance(case, DryRunSeedSlot)
            self.assertFalse(hasattr(case, "master_seed"))
            calls.append(case.case_id)
            return runner._stub_seed_executor(case, mark_boundary)

        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "stubbed_official_results"
            self.assertFalse(result_root.exists())
            with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA availability query")), \
                 mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA synchronize")), \
                 mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("CUDA memory reset")), \
                 mock.patch.object(torch.cuda, "max_memory_allocated", side_effect=AssertionError("CUDA memory query")), \
                 mock.patch.object(torch.cuda, "max_memory_reserved", side_effect=AssertionError("CUDA memory query")):
                result = runner.full_official_control_flow_dry_run(result_root, seed_executor=stub)
            self.assertEqual(len(calls), 5)
            self.assertEqual(result["terminal_status"], "OMEGA_V2_2A_D3Q_PASS")
            self.assertEqual(result["completed_seed_slots"], 5)
            self.assertTrue(result["artifact_seal"]["verified"])
            self.assertTrue((result_root / "d3q_metrics.json").is_file())
            self.assertTrue((result_root / "OMEGA_V2_2A_D3Q_REPORT.md").is_file())
            self.assertTrue((result_root / "artifact_hashes.json").is_file())
            verified = json.loads((result_root / "artifact_hashes_verified.json").read_text(encoding="utf-8"))
            self.assertTrue(verified["verified"])
            self.assertEqual(len(list(result_root.glob("*_raw_gradients.pt"))), 5)

    def test_full_control_flow_runs_remaining_slots_after_seed_failure(self) -> None:
        calls: list[str] = []

        def one_seed_fails(case: DryRunSeedSlot, mark_boundary):
            calls.append(case.case_id)
            result = runner._stub_seed_executor(case, mark_boundary)
            if case.case_id == "dryrun_slot_02":
                result["seed_pass"] = False
                result["terminal_status"] = "SEED_FAIL"
                result["families"]["W_Q"]["seed_family_pass"] = False
            return result

        with tempfile.TemporaryDirectory() as temporary:
            result = runner.full_official_control_flow_dry_run(
                Path(temporary) / "seed_failure_stub",
                seed_executor=one_seed_fails,
            )
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["terminal_status"], "OMEGA_V2_2A_D3Q_FAIL")
        self.assertFalse(result["seed_results"][1]["seed_pass"])

    def test_package_compiles_imports_and_static_unresolved_name_audit_is_zero(self) -> None:
        py_files = sorted(PACKAGE_ROOT.rglob("*.py"))
        for path in py_files:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        for module_name in (
            "omega_v2_2a_d3q.config",
            "omega_v2_2a_d3q.metrics",
            "omega_v2_2a_d3q.core",
            "omega_v2_2a_d3q.runner",
        ):
            self.assertIsNotNone(importlib.import_module(module_name))
        unresolved = {path.relative_to(PACKAGE_ROOT).as_posix(): _unresolved_globals(path) for path in py_files}
        self.assertEqual({path: names for path, names in unresolved.items() if names}, {})

    def test_only_one_strict_source_seal_verifier_checks_references(self) -> None:
        source_text = (PACKAGE_ROOT / "runner.py").read_text(encoding="utf-8")
        self.assertEqual(source_text.count("def _verify_source_seal("), 1)
        seal = {
            "spec_sha256": runner.sha256_file(runner.SPEC_PATH),
            "source_sha256": runner.source_hashes(),
            "reference_sha256": runner.reference_hashes(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            seal_path = Path(temporary) / "SOURCE_SEAL.json"
            seal_path.write_text(json.dumps(seal), encoding="utf-8")
            with mock.patch.object(runner, "SOURCE_SEAL_PATH", seal_path):
                self.assertEqual(runner._verify_source_seal()["reference_sha256"], seal["reference_sha256"])
                seal["reference_sha256"] = {"stale_reference": "0"}
                seal_path.write_text(json.dumps(seal), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "D3Q_REFERENCE_SEAL_MISMATCH"):
                    runner._verify_source_seal()

    def test_smoke_snapshots_final_source_and_rejects_stale_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            reference_path = temp_root / "calibration_reference.pt"
            stub_result = runner._stub_seed_executor(DryRunSeedSlot("calibration_reference_stub"), lambda: None)
            source_names = ("gR", "gU0", "gU1", "gU2", "gU3")
            bundle_families = {
                family: {
                    source_name: stub_result["tensor_bundle"]["cuda_fp32"][family][f"{source_name}_cuda_fp32"]
                    for source_name in source_names
                }
                for family in stub_result["tensor_bundle"]["cuda_fp32"]
            }
            torch.save({"schema": "omega-v2-2a-d3-gradient-bundle-v1", "run": "run_01", "families": bundle_families}, reference_path)
            v22a_seal_path = temp_root / "V22A_SOURCE_SEAL.json"
            v22a_seal_path.write_text(json.dumps({"environment": {
                "torch_version": str(torch.__version__),
                "torch_cuda_runtime": torch.version.cuda,
            }}), encoding="utf-8")
            smoke_root = temp_root / "calibration_smoke"
            source_seal_path = temp_root / "SOURCE_SEAL.json"

            def smoke_stub(plan, mark_boundary):
                self.assertEqual(plan.master_seed, 20260930)
                return runner._stub_seed_executor(DryRunSeedSlot("calibration_smoke_stub"), mark_boundary)

            with mock.patch.object(runner, "CALIBRATION_SMOKE_ROOT", smoke_root), \
                 mock.patch.object(runner, "D3_CALIBRATION_GRADIENT_BUNDLE", reference_path), \
                 mock.patch.object(runner, "D3_CALIBRATION_GRADIENT_BUNDLE_SHA256", runner.sha256_file(reference_path)), \
                 mock.patch.object(runner, "V22A_SOURCE_SEAL", v22a_seal_path), \
                 mock.patch.object(runner, "SOURCE_SEAL_PATH", source_seal_path), \
                 mock.patch.object(runner, "configure_d3q_execution"), \
                 mock.patch.object(torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("CUDA memory reset")), \
                 mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA synchronize")):
                result = runner.run_calibration_seed_smoke(
                    go_calibration_smoke=True,
                    result_root=smoke_root,
                    seed_executor=smoke_stub,
                )
                smoke_metrics = json.loads((smoke_root / "calibration_smoke.json").read_text(encoding="utf-8"))
                self.assertEqual(result["terminal_status"], "CALIBRATION_SMOKE_COMPLETE")
                self.assertEqual(smoke_metrics["source_sha256"], runner.source_hashes())
                self.assertEqual(smoke_metrics["spec_sha256"], runner.sha256_file(runner.SPEC_PATH))
                crosscheck = smoke_metrics["calibration_smoke"]["D3_diagnostic_crosscheck_NON_GATE"]
                self.assertTrue(crosscheck["all_bitwise_equal"])
                self.assertFalse(crosscheck["affects_smoke_terminal_status"])
                self.assertIsNone(smoke_metrics["D3Q_verdict"])

                mismatch_families = {
                    family: {name: tensor.clone() for name, tensor in family_tensors.items()}
                    for family, family_tensors in bundle_families.items()
                }
                mismatch_families["W_Q"]["gU0"].view(-1)[0] += 0.125
                mismatch_reference = temp_root / "calibration_reference_mismatch.pt"
                torch.save({"schema": "omega-v2-2a-d3-gradient-bundle-v1", "run": "run_01", "families": mismatch_families}, mismatch_reference)
                mismatch_smoke_root = temp_root / "calibration_smoke_mismatch"
                with mock.patch.object(runner, "CALIBRATION_SMOKE_ROOT", mismatch_smoke_root), \
                     mock.patch.object(runner, "D3_CALIBRATION_GRADIENT_BUNDLE", mismatch_reference), \
                     mock.patch.object(runner, "D3_CALIBRATION_GRADIENT_BUNDLE_SHA256", runner.sha256_file(mismatch_reference)), \
                     mock.patch.object(runner, "V22A_SOURCE_SEAL", v22a_seal_path), \
                     mock.patch.object(runner, "configure_d3q_execution"), \
                     mock.patch.object(torch.cuda, "is_available", return_value=True), \
                     mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("CUDA memory reset")), \
                     mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA synchronize")):
                    mismatch_result = runner.run_calibration_seed_smoke(
                        go_calibration_smoke=True,
                        result_root=mismatch_smoke_root,
                        seed_executor=smoke_stub,
                    )
                mismatch_metrics = json.loads((mismatch_smoke_root / "calibration_smoke.json").read_text(encoding="utf-8"))
                self.assertEqual(mismatch_result["terminal_status"], "CALIBRATION_SMOKE_COMPLETE")
                self.assertEqual(mismatch_metrics["terminal_status"], "CALIBRATION_SMOKE_COMPLETE")
                self.assertFalse(mismatch_metrics["D3_diagnostic_crosscheck_NON_GATE"]["all_bitwise_equal"])
                self.assertFalse(mismatch_metrics["D3_diagnostic_crosscheck_NON_GATE"]["affects_smoke_terminal_status"])

                with mock.patch.object(runner, "source_hashes", return_value={"runner.py": "stale"}):
                    with self.assertRaisesRegex(RuntimeError, "D3Q_SOURCE_SEAL_STOP: calibration smoke spec/source snapshot is stale"):
                        runner.create_source_seal()
                self.assertFalse(source_seal_path.exists())

    def test_smoke_runtime_mismatch_aborts_before_cuda_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            reference_path = temp_root / "calibration_reference.pt"
            stub_result = runner._stub_seed_executor(DryRunSeedSlot("reference"), lambda: None)
            bundle_families = {
                family: {key: value for key, value in tensors.items() if key in ("gR_cuda_fp32", "gU0_cuda_fp32", "gU1_cuda_fp32", "gU2_cuda_fp32", "gU3_cuda_fp32")}
                for family, tensors in stub_result["tensor_bundle"]["cuda_fp32"].items()
            }
            renamed = {
                family: {
                    source_name: family_tensors[f"{source_name}_cuda_fp32"]
                    for source_name in ("gR", "gU0", "gU1", "gU2", "gU3")
                }
                for family, family_tensors in bundle_families.items()
            }
            torch.save({"schema": "omega-v2-2a-d3-gradient-bundle-v1", "run": "run_01", "families": renamed}, reference_path)
            seal_path = temp_root / "V22A_SOURCE_SEAL.json"
            result_root = temp_root / "must-not-exist"
            env_mismatches = (
                {"torch_version": "deliberately-mismatched"},
                {"torch_version": str(torch.__version__), "torch_cuda_runtime": "deliberately-mismatched"},
            )
            for environment in env_mismatches:
                seal_path.write_text(json.dumps({"environment": environment}), encoding="utf-8")
                with mock.patch.object(runner, "D3_CALIBRATION_GRADIENT_BUNDLE", reference_path), \
                     mock.patch.object(runner, "D3_CALIBRATION_GRADIENT_BUNDLE_SHA256", runner.sha256_file(reference_path)), \
                     mock.patch.object(runner, "V22A_SOURCE_SEAL", seal_path), \
                     mock.patch.object(runner, "configure_d3q_execution", side_effect=AssertionError("configure must not run")), \
                     mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA query must not run")):
                    with self.assertRaisesRegex(RuntimeError, "D3Q_ENVIRONMENT_MISMATCH"):
                        runner.run_calibration_seed_smoke(go_calibration_smoke=True, result_root=result_root)
                self.assertFalse(result_root.exists())

    def test_calibration_reference_hash_is_checked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference_path = Path(temporary) / "bad_reference.pt"
            torch.save({"schema": "not-read"}, reference_path)
            with mock.patch.object(torch, "load", side_effect=AssertionError("bundle must not be loaded before hash match")):
                with self.assertRaisesRegex(RuntimeError, "D3Q_CALIBRATION_REFERENCE_SHA256_MISMATCH"):
                    runner.load_calibration_diagnostic_reference(reference_path, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()

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

from omega_v2_2a_d3_diagnostic import runner
from omega_v2.core import MATRIX_FAMILIES
from omega_v2_2a_d3_diagnostic.metrics import sum_u4_variants, tensor_raw_sha256


def _fake_cuda_run(reference_run: dict | None = None) -> dict:
    families = {}
    for index, family in enumerate(MATRIX_FAMILIES, start=1):
        g_r = torch.tensor([4.0 * index, 8.0 * index], dtype=torch.float32)
        g_u = torch.tensor([float(index), 2.0 * index], dtype=torch.float32)
        families[family] = {"gR": g_r, "gU0": g_u.clone(), "gU1": g_u.clone(), "gU2": g_u.clone(), "gU3": g_u.clone(), **sum_u4_variants([g_u.clone() for _ in range(4)])}
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    w = torch.tensor([3.0, 4.0], dtype=torch.float32)
    tensor_bundle = {family: {name: tensor.clone() for name, tensor in values.items()} for family, values in families.items()}
    tensor_bundle["__inputs__"] = {"x_fp32_cpu": x.clone(), "w_fp32_cpu": w.clone()}
    return {
        "families_cuda": families,
        "tensor_bundle_cpu": tensor_bundle,
        "x_cpu": x,
        "w_cpu": w,
        "input_copy_bitwise_equal": True,
        "loss_weight_copy_bitwise_equal": True,
        "weight_copy_report": {"all_bitwise_equal": True},
        "initial_clone_report": {"initial_values_bitwise_equal": True},
        "initial_comparison_to_run_01": None if reference_run is None else {
            "R4_initial_weights_bitwise_equal": True,
            "U4_initial_weights_bitwise_equal": True,
            "x_bitwise_equal": True,
            "w_bitwise_equal": True,
            "all_bitwise_equal": True,
        },
        "initial_weight_state_dict_sha256": "fixed-v2-0-seed-hash",
        "fixed_seeds": {"WEIGHT_SEED": 20260930, "INPUT_SEED": 20261938, "LOSS_W_SEED": 20262938},
    }


def _fake_fp64(x_fp32: torch.Tensor, w_fp32: torch.Tensor) -> dict:
    families = {}
    for index, family in enumerate(MATRIX_FAMILIES, start=1):
        g_r = torch.tensor([4.0 * index, 8.0 * index], dtype=torch.float64)
        g_u = torch.tensor([float(index), 2.0 * index], dtype=torch.float64)
        families[family] = {
            "gR64": g_r,
            "gU0_64": g_u.clone(), "gU1_64": g_u.clone(), "gU2_64": g_u.clone(), "gU3_64": g_u.clone(),
            "S_fp64_cpu": g_u + g_u + g_u + g_u,
        }
    families["__inputs__"] = {"x_fp64": x_fp32.double(), "w_fp64": w_fp32.double()}
    return {
        "families": families,
        "x_fp64": x_fp32.double(), "w_fp64": w_fp32.double(),
        "fp32_to_fp64_initial_weights_bitwise_value_preserved": True,
        "fp32_to_fp64_x_exact_value_preserved": True,
        "fp32_to_fp64_w_exact_value_preserved": True,
    }


def _unresolved_globals(path: Path) -> list[str]:
    table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    module_bound = {symbol.get_name() for symbol in table.get_symbols() if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()}
    builtin_names = set(dir(builtins)) | {"__name__", "__file__", "__package__", "__spec__", "__loader__", "__cached__", "__builtins__"}
    missing: set[str] = set()
    def visit(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in module_bound and symbol.get_name() not in builtin_names:
                missing.add(symbol.get_name())
        for child in scope.get_children():
            visit(child)
    visit(table)
    return sorted(missing)


class D3DiagnosticControlFlowDryRunTests(unittest.TestCase):
    def test_package_compiles_imports_and_has_no_unresolved_global_names(self) -> None:
        py_files = sorted(PACKAGE_ROOT.rglob("*.py"))
        for path in py_files:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        for name in (
            "omega_v2_2a_d3_diagnostic.core",
            "omega_v2_2a_d3_diagnostic.metrics",
            "omega_v2_2a_d3_diagnostic.diagnostic",
            "omega_v2_2a_d3_diagnostic.runner",
            "omega_v2_2a_d3_diagnostic.tests.test_d3_diagnostic",
        ):
            self.assertIsNotNone(importlib.import_module(name))
        with mock.patch.object(runner, "main", return_value=0):
            try:
                importlib.import_module("omega_v2_2a_d3_diagnostic.__main__")
            except SystemExit as error:
                self.assertEqual(error.code, 0)
        issues = {path.relative_to(PACKAGE_ROOT).as_posix(): _unresolved_globals(path) for path in py_files}
        self.assertEqual({path: names for path, names in issues.items() if names}, {})
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_diagnostic_control_flow_stubs_tensor_persistence_and_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            result_root = temp_root / "diagnostic_control_flow_only"
            source_seal_path = temp_root / "SOURCE_SEAL.json"
            source_seal = {
                "spec_sha256": runner.sha256_file(runner.SPEC_PATH),
                "python_source_sha256": runner.source_hashes(),
                "environment": {"qa_mode": "CPU-stub-only"},
                "classification": "CALIBRATION_DIAGNOSTIC_ONLY",
            }
            source_seal_path.write_text(json.dumps(source_seal), encoding="utf-8")
            with mock.patch.object(runner, "SOURCE_SEAL_PATH", source_seal_path), \
                 mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA runtime query")), \
                 mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA synchronize")), \
                 mock.patch.object(torch.cuda, "empty_cache", side_effect=AssertionError("CUDA cache call")):
                result = runner.execute_diagnostic(explicit_go=True, result_root=result_root, cuda_run_fn=_fake_cuda_run, fp64_fn=_fake_fp64)
            self.assertEqual(result["terminal_classification"], "DIAGNOSTIC_COMPLETE", result)
            self.assertEqual(result["classification"], "CALIBRATION_DIAGNOSTIC_ONLY")
            self.assertFalse(result["may_rescue_V2_2A"])
            self.assertIsNone(result["architectural_verdict"])
            self.assertEqual(result["held_out_D3Q_seeds_touched"], [])
            self.assertTrue(result["reproducibility"]["all_bitwise_equal"])
            self.assertTrue(result["run_02"]["initial_comparison_to_run_01"]["all_bitwise_equal"])
            self.assertTrue(result["artifact_seal"]["verified"])
            for name in ("cuda_gradients_run_01.pt", "cuda_gradients_run_02.pt", "cpu_fp64_oracle.pt", "d3_diagnostic_metrics.json", "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md", "artifact_hashes.json", "artifact_hashes_verified.json"):
                self.assertTrue((result_root / name).is_file(), name)
            bundle = torch.load(result_root / "cuda_gradients_run_01.pt", map_location="cpu", weights_only=True)
            self.assertIn("S_stack", bundle["families"]["W_Q"])
            self.assertIn("S_fp64", bundle["families"]["W_down"])
            manifest = json.loads((result_root / "artifact_hashes.json").read_text(encoding="utf-8"))
            for path_text, item in manifest["artifacts"].items():
                path = Path(path_text)
                self.assertEqual(path.stat().st_size, item["size_bytes"])
                self.assertEqual(runner.sha256_file(path), item["sha256"])
            self.assertTrue((result_root / "OMEGA_V2_2A_D3_DIAGNOSTIC_REPORT.md.sha256").is_file())
            self.assertEqual(
                result["tensor_bundles"]["cuda_run_01"]["file_sha256"],
                runner.sha256_file(result_root / "cuda_gradients_run_01.pt"),
            )
            for family, tensors in bundle["families"].items():
                for tensor_name, tensor in tensors.items():
                    logical_name = f"run_01/{family}/{tensor_name}"
                    self.assertEqual(
                        result["tensor_bundles"]["cuda_run_01"]["tensor_raw_sha256"][logical_name],
                        tensor_raw_sha256(logical_name, tensor),
                    )
                    metadata = result["tensor_bundles"]["cuda_run_01"]["tensor_metadata"][logical_name]
                    self.assertEqual(metadata["shape"], list(tensor.shape))
                    self.assertEqual(metadata["dtype"], str(tensor.dtype))
                    self.assertEqual(metadata["raw_sha256"], tensor_raw_sha256(logical_name, tensor))
            for family in MATRIX_FAMILIES:
                raw_hashes = result["run_01_metrics"][family]["raw_tensor_sha256"]
                metadata = result["run_01_metrics"][family]["tensor_metadata"]
                for tensor_name, tensor in bundle["families"][family].items():
                    logical_name = f"run/{family}/{tensor_name}"
                    self.assertEqual(raw_hashes[logical_name], tensor_raw_sha256(logical_name, tensor))
                    self.assertEqual(metadata[logical_name]["shape"], list(tensor.shape))
                    self.assertEqual(metadata[logical_name]["dtype"], str(tensor.dtype))
                    self.assertEqual(metadata[logical_name]["raw_sha256"], raw_hashes[logical_name])

    def test_direct_execution_requires_explicit_go(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "explicit diagnostic GO"):
            runner.execute_diagnostic()


if __name__ == "__main__":
    unittest.main()

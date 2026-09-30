from __future__ import annotations

import builtins
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import symtable
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from omega_v2_2a_cuda_preflight import runner, variants


class PreCudaFrontierReached(Exception):
    pass


def _write_temporary_source_seal(path: Path, environment: dict) -> None:
    runner.write_json(path, {
        "schema": "offline-test-source-seal",
        "spec_sha256": runner.sha256_file(PACKAGE_ROOT / "OMEGA_V2_2A_SPEC.md"),
        "python_source_sha256": runner.source_hashes(),
        "environment": environment,
        "environment_matches_md317": True,
    })


def _undefined_globals_for_python_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    table = symtable.symtable(source, str(path), "exec")
    module_bound = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()
    }
    builtin_names = set(dir(builtins)) | {"__name__", "__file__", "__package__", "__spec__", "__loader__", "__cached__", "__builtins__"}
    unresolved: set[str] = set()

    def visit(scope: symtable.SymbolTable) -> None:
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if symbol.is_referenced() and symbol.is_global() and name not in module_bound and name not in builtin_names:
                unresolved.add(name)
        for child in scope.get_children():
            visit(child)

    visit(table)
    return sorted(unresolved)


class V22AOfflineHarnessQATests(unittest.TestCase):
    def test_all_package_python_sources_compile(self) -> None:
        python_files = sorted(PACKAGE_ROOT.rglob("*.py"))
        self.assertTrue(python_files)
        for path in python_files:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_all_library_modules_and_cli_entrypoint_import_smoke(self) -> None:
        module_names = (
            "omega_v2_2a_cuda_preflight.core",
            "omega_v2_2a_cuda_preflight.variants",
            "omega_v2_2a_cuda_preflight.checks",
            "omega_v2_2a_cuda_preflight.runner",
            "omega_v2_2a_cuda_preflight.tests.test_v2_2a",
        )
        for name in module_names:
            self.assertIsNotNone(importlib.import_module(name))
        with mock.patch.object(runner, "main", return_value=0):
            try:
                importlib.import_module("omega_v2_2a_cuda_preflight.__main__")
            except SystemExit as error:
                self.assertEqual(error.code, 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_unresolved_global_names_in_package(self) -> None:
        python_files = sorted(PACKAGE_ROOT.rglob("*.py"))
        unresolved = {
            path.relative_to(PACKAGE_ROOT).as_posix(): _undefined_globals_for_python_file(path)
            for path in python_files
        }
        unresolved = {path: names for path, names in unresolved.items() if names}
        self.assertEqual(unresolved, {})

    def test_make_fixed_inputs_export_identity(self) -> None:
        self.assertIs(runner.make_fixed_inputs, variants.make_fixed_inputs)

    def test_pre_cuda_official_path_dry_run(self) -> None:
        class EnvironmentSeal(dict):
            pass

        environment = EnvironmentSeal({"dry_run": "pre_cuda_frontier"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seal_path = root / "SOURCE_SEAL.json"
            result_root = root / "official_results_must_not_exist"
            _write_temporary_source_seal(seal_path, environment)
            frontier_events: list[str] = []
            cuda_calls: list[str] = []

            def stop_before_first_cuda_copy(cpu_tensor: torch.Tensor):
                self.assertEqual(cpu_tensor.device.type, "cpu")
                self.assertEqual(tuple(cpu_tensor.shape), (8, 8, 256))
                frontier_events.append("fixed_cpu_input_built_before_cuda_copy")
                raise PreCudaFrontierReached

            def forbidden_cuda_call(*_args, **_kwargs):
                cuda_calls.append("unexpected_cuda_call")
                raise AssertionError("pre-CUDA dry run attempted a CUDA runtime call")

            patches = [
                mock.patch.object(runner, "SOURCE_SEAL_PATH", seal_path),
                mock.patch.object(runner, "RESULTS_ROOT", result_root),
                mock.patch.object(runner, "environment_record", return_value=environment),
                mock.patch.object(variants, "copy_fixed_tensor_to_cuda", side_effect=stop_before_first_cuda_copy),
                mock.patch.object(torch.cuda, "is_available", side_effect=forbidden_cuda_call),
                mock.patch.object(torch.cuda, "synchronize", side_effect=forbidden_cuda_call),
                mock.patch.object(torch.cuda, "empty_cache", side_effect=forbidden_cuda_call),
                mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=forbidden_cuda_call),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaises(PreCudaFrontierReached):
                    runner._official_run()
            self.assertEqual(frontier_events, ["fixed_cpu_input_built_before_cuda_copy"])
            self.assertEqual(cuda_calls, [])
            self.assertFalse(result_root.exists())
            dry_result = {"classification": "PRE_CUDA_DRY_RUN_PASS", "cuda_scientific_cells_started": len(cuda_calls)}
            self.assertEqual(dry_result["classification"], "PRE_CUDA_DRY_RUN_PASS")
            self.assertEqual(dry_result["cuda_scientific_cells_started"], 0)

    def test_full_official_control_flow_dry_run(self) -> None:
        environment = {"dry_run": "full_official_control_flow"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seal_path = root / "SOURCE_SEAL.json"
            result_root = root / "temporary_official_results"
            _write_temporary_source_seal(seal_path, environment)
            dry_block = object()
            r4 = SimpleNamespace(recurrent=SimpleNamespace(block=dry_block), zero_grad=lambda **_kw: None, parameters=lambda: [])
            u4 = SimpleNamespace(blocks=[object() for _ in range(4)], zero_grad=lambda **_kw: None, parameters=lambda: [])
            cpu_pair = {"r4_cpu": r4, "u4_cpu": u4}
            tensors = {
                "x_cpu": object(), "x_cuda": object(), "loss_weights_cpu": object(), "loss_weights_cuda": object(),
                "target_cpu": object(), "target_cuda": object(),
                "cpu_to_cuda_copy_bitwise": {"input": True, "loss_weights": True, "target": True},
            }
            cell_names: list[str] = []
            cuda_calls: list[str] = []

            def fake_measure_cell(name, action, _started):
                cell_names.append(name)
                value = action()
                return value, {"cell": name, "status": "OK", "peak_memory_allocated_bytes": 0, "peak_memory_reserved_bytes": 0, "within_3_gib_allocated_budget": True, "wall_seconds": 0.0}

            def forbidden_cuda_call(*_args, **_kwargs):
                cuda_calls.append("unexpected_cuda_call")
                raise AssertionError("full-flow dry run attempted a CUDA runtime operation")

            fake_pass = lambda **kwargs: {"pass": True, **kwargs}
            patches = [
                mock.patch.object(runner, "SOURCE_SEAL_PATH", seal_path),
                mock.patch.object(runner, "RESULTS_ROOT", result_root),
                mock.patch.object(runner, "environment_record", return_value=environment),
                mock.patch.object(runner, "build_initial_variants", return_value=cpu_pair),
                mock.patch.object(runner, "parameter_storage_gate", return_value={"pass": True, "R4_unique_parameters": 1_048_576, "U4_unique_parameters": 4_194_304}),
                mock.patch.object(runner, "make_fixed_inputs", return_value=tensors),
                mock.patch.object(runner, "_new_cuda_copy", side_effect=lambda module: module),
                mock.patch.object(runner, "weight_copy_report", return_value={"all_bitwise_equal": True}),
                mock.patch.object(runner, "cuda_correctness_gate", side_effect=lambda *a, k_values, **kw: {"pass": True, "cells": [{"K": k, "pass": True} for k in k_values]}),
                mock.patch.object(runner, "initialization_and_trace_gate", return_value={"pass": True, "fallback_used": False}),
                mock.patch.object(runner, "gradient_sharing_gate", return_value={"pass": True, "families": []}),
                mock.patch.object(runner, "iso_flop_gate", return_value={"pass": True, "R4_forward_flops": 538_968_064, "U4_forward_flops": 538_968_064}),
                mock.patch.object(runner, "k_flex_forward_cell", side_effect=lambda _b, _x, k: {"output_finite": True, "schema_unchanged": True, "value_unchanged": True, "parameter_count_unchanged": True, "K": k}),
                mock.patch.object(runner, "k_flex_backward_cell", side_effect=lambda _b, _x, _w, k: {"finite": True, "schema_unchanged": True, "value_unchanged": True, "parameter_count_unchanged": True, "K": k}),
                mock.patch.object(runner, "schema_sha256", return_value="schema-hash"),
                mock.patch.object(runner, "state_dict_sha256", return_value="value-hash"),
                mock.patch.object(runner, "parameter_count", return_value=1_048_576),
                mock.patch.object(runner, "optimizer_smoke", side_effect=lambda _v, _x, _t, *, label: {"pass": True, "variant": label, "updates": 20, "L0": 1.0, "L20": 0.98}),
                mock.patch.object(runner, "_measure_cell", side_effect=fake_measure_cell),
                mock.patch.object(torch.cuda, "is_available", side_effect=forbidden_cuda_call),
                mock.patch.object(torch.cuda, "synchronize", side_effect=forbidden_cuda_call),
                mock.patch.object(torch.cuda, "empty_cache", side_effect=forbidden_cuda_call),
                mock.patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=forbidden_cuda_call),
            ]
            with ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                result = runner._official_run()
                seal = runner._seal_official_result(result)

            self.assertEqual(result["terminal_status"], "OMEGA_V2_2A_LOCAL_PREFLIGHT_PASS")
            self.assertTrue(seal["verified"])
            self.assertEqual(len(cell_names), 15)
            self.assertEqual(cell_names[:4], ["D1_R4_K1", "D1_R4_K4", "D2_R4_U4_K4_PARITY", "D3_GRADIENT_IDENTITY_K4"])
            self.assertEqual(cell_names[-2:], ["D7_R4_SMOKE_20_UPDATES", "D7_U4_SMOKE_20_UPDATES"])
            self.assertEqual(cuda_calls, [])
            self.assertTrue(result_root.is_dir())
            self.assertIn("vram_runtime", result["gates"])
            self.assertEqual(result["gate_results"][-1]["gate"], "D8_vram_wall_contract")
            qa_summary = {
                "classification": "HARNESS_QA_ONLY",
                "cuda_kernel_launches": len(cuda_calls),
                "cuda_scientific_cells_started": len(cuda_calls),
                "scientific_evidence": "NONE",
                "control_flow_cell_stubs_exercised": len(cell_names),
                "result_files_written_under_temporary_directory": True,
            }
            self.assertEqual(qa_summary["classification"], "HARNESS_QA_ONLY")
            self.assertEqual(qa_summary["cuda_kernel_launches"], 0)
            self.assertEqual(qa_summary["cuda_scientific_cells_started"], 0)
            self.assertEqual(qa_summary["scientific_evidence"], "NONE")

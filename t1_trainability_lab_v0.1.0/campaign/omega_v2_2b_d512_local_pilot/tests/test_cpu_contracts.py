from __future__ import annotations

import tempfile
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from omega_v2_2a_d3q.metrics import evaluate_primary_gates, sum_u4_diagnostics
from omega_v2_2b_d512_local_pilot.config import (
    CALIBRATION_SEED,
    D7_BETAS,
    D7_EPS,
    D7_LR,
    D7_UPDATES,
    D7_WEIGHT_DECAY,
    OFFICIAL_MASTER_SEEDS,
    make_seed_plan,
)
from omega_v2_2b_d512_local_pilot.core import d7_optimizer_loop
from omega_v2_2b_d512_local_pilot.ledger import recompute_d512_ledger
from omega_v2_2b_d512_local_pilot.metrics import d1_correctness_metrics, d3_d512_gates, d6_invariance_record, d6_invariance_snapshot
from omega_v2_2b_d512_local_pilot import runner
from omega_v2_2b_d512_local_pilot.runner import (
    PilotContext,
    TERMINAL_CAPACITY_HOLD,
    TERMINAL_CALIBRATION_QA_CAPACITY_HOLD,
    TERMINAL_CALIBRATION_QA_HARNESS_HOLD,
    TERMINAL_CALIBRATION_QA_SCIENTIFIC_HOLD,
    TERMINAL_CALIBRATION_SMOKE_COMPLETE,
    TERMINAL_FAIL,
    TERMINAL_PASS,
    _terminal_decision,
    _finalize_wall_in_process,
    _calibration_smoke_terminal,
    finalize_external_launch_logs,
    full_mocked_control_flow_dry_run,
    pre_cuda_official_dry_run,
    _render_smoke_report,
    from_import_resolution_audit,
    unresolved_name_audit,
    package_import_sweep,
)


class SeedGuardTests(unittest.TestCase):
    def test_official_seeds_are_rejected_for_qa_and_smoke(self) -> None:
        for seed in OFFICIAL_MASTER_SEEDS:
            for mode in ("qa", "smoke"):
                with self.subTest(seed=seed, mode=mode), self.assertRaises(ValueError):
                    make_seed_plan(seed, mode=mode)

    def test_calibration_seed_is_rejected_for_official(self) -> None:
        with self.assertRaisesRegex(ValueError, "CALIBRATION_SEED_FORBIDDEN"):
            make_seed_plan(CALIBRATION_SEED, mode="official")

    def test_only_calibration_seed_is_accepted_for_qa(self) -> None:
        plan = make_seed_plan(CALIBRATION_SEED, mode="qa")
        self.assertEqual(plan.master_seed, CALIBRATION_SEED)
        self.assertEqual(plan.mode, "qa")


class LedgerTests(unittest.TestCase):
    def test_frozen_d512_flop_integers(self) -> None:
        ledger = recompute_d512_ledger()
        self.assertEqual(
            ledger["actual_flops"],
            {
                "per_round_B1": 67_239_936,
                "K4_B1": 268_959_744,
                "K4_B8": 2_151_677_952,
                "K4_B128": 34_426_847_232,
            },
        )
        self.assertTrue(ledger["exact_match"])


class MetricTests(unittest.TestCase):
    def test_d1_runs_elementwise_and_normwise_gates_independently(self) -> None:
        cpu = torch.ones(16_384, dtype=torch.float32)
        elementwise_only_failure = cpu.clone()
        elementwise_only_failure[0] += 1e-3
        result = d1_correctness_metrics(elementwise_only_failure, cpu)
        self.assertFalse(result["elementwise_gate"]["pass"])
        self.assertTrue(result["normwise_gate"]["pass"])

        normwise_only_failure = cpu + 5e-5
        result = d1_correctness_metrics(normwise_only_failure, cpu)
        self.assertTrue(result["elementwise_gate"]["pass"])
        self.assertFalse(result["normwise_gate"]["pass"])

    def test_d3_primary_A_B_C_gates_on_synthetic_gradients(self) -> None:
        reference = torch.ones(8, dtype=torch.float64)
        exact = evaluate_primary_gates(reference, reference.clone())
        self.assertTrue(exact["scientific_gates_pass"])
        self.assertEqual(set(exact["gates"]), {"A", "B", "C"})

        perturbed = evaluate_primary_gates(reference, reference + 1e-4)
        self.assertFalse(perturbed["scientific_gates_pass"])
        self.assertTrue(all(perturbed["gates"][name]["pass"] is False for name in ("A", "B", "C")))

    def test_d3_d512_gates_wires_real_D3Q_functions(self) -> None:
        g_r = torch.full((2, 3), 4.0, dtype=torch.float32)
        g_u = [torch.ones((2, 3), dtype=torch.float32) for _ in range(4)]
        sums = sum_u4_diagnostics(g_u)
        s64 = (((g_u[0].double() + g_u[1].double()) + g_u[2].double()) + g_u[3].double())
        primary = evaluate_primary_gates(g_r.double(), s64)
        result = d3_d512_gates(g_r, g_u)

        self.assertTrue(primary["scientific_gates_pass"])
        self.assertTrue(result["pass"])
        self.assertEqual(result["primary_S64"]["gates"], primary["gates"])
        self.assertEqual(set(result["sums_fp32"]), {"S_forward", "S_reverse", "S_pairwise", "S_stack"})
        self.assertEqual(result["diagnostics_NON_GATE"]["old_max_rel"]["S_reverse_equal_gR_NON_GATE"], True)

    def test_real_d3q_import_resolution_audit_includes_deferred_imports(self) -> None:
        audit = from_import_resolution_audit()
        self.assertTrue(audit["pass"], audit["missing"])
        deferred = [row for row in audit["imports"] if row["deferred_or_local"]]
        self.assertTrue(any(row["symbol"] == "sum_u4_diagnostics" for row in deferred))

    def test_all_package_modules_import_without_running_main(self) -> None:
        result = package_import_sweep()
        self.assertTrue(result["pass"], result["errors"])
        self.assertIn("omega_v2_2b_d512_local_pilot.__main__", result["modules"])


class D6D7Tests(unittest.TestCase):
    def test_d6_schema_value_and_unique_parameter_count_before_after(self) -> None:
        before = torch.nn.Linear(3, 2, bias=False)
        after = torch.nn.Linear(3, 2, bias=False)
        after.load_state_dict(before.state_dict())
        before_snapshot = d6_invariance_snapshot(before)
        after(torch.ones(1, 3)).sum().backward()
        after_snapshot = d6_invariance_snapshot(after)
        record = d6_invariance_record(before_snapshot, after_snapshot)
        self.assertTrue(record["pass"])
        for key in (
            "SCHEMA_SHA256_before", "SCHEMA_SHA256_after",
            "VALUE_SHA256_before", "VALUE_SHA256_after",
            "parameter_count_before", "parameter_count_after",
        ):
            self.assertIn(key, record)
        with torch.no_grad():
            after.weight[0, 0] += 1
        self.assertFalse(d6_invariance_record(before_snapshot, d6_invariance_snapshot(after))["pass"])

    def test_d7_loop_has_21_forwards_and_20_adamw_updates(self) -> None:
        torch.manual_seed(17)
        model = torch.nn.Linear(2, 1, bias=False)
        x = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        target = torch.tensor([[0.25], [-0.5]])
        initial_loss = torch.mean((model(x).detach() - target) ** 2).item()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=D7_LR,
            betas=D7_BETAS,
            eps=D7_EPS,
            weight_decay=D7_WEIGHT_DECAY,
        )
        calls: list[bool] = []

        def forward() -> torch.Tensor:
            calls.append(torch.is_grad_enabled())
            return model(x)

        result = d7_optimizer_loop(forward, target, tuple(model.parameters()), optimizer)
        self.assertEqual(D7_UPDATES, 20)
        self.assertEqual(result["forward_count"], 21)
        self.assertEqual(result["optimizer_step_count"], 20)
        self.assertEqual(len(calls), 21)
        self.assertTrue(all(calls[:20]))
        self.assertFalse(calls[20])
        self.assertAlmostEqual(result["L0"], initial_loss, places=7)
        self.assertEqual(int(optimizer.state[next(iter(model.parameters()))]["step"].item()), 20)


class RunnerContracts(unittest.TestCase):
    def test_smoke_terminal_classification_covers_complete_capacity_science_and_harness(self) -> None:
        def seed_result(failures=None):
            return {
                "gate_results": {name: {"pass": True} for name in ("D1", "D2", "D3", "D4", "D5", "D6", "D7")},
                "scientific_failures": failures or [],
                "seed_pass": True,
            }

        with tempfile.TemporaryDirectory(prefix="omega_v2b_smoke_terminal_test_") as temporary:
            context = PilotContext(result_root=Path(temporary) / "complete", official=False, wall_start=0.0)
            context.cell_records = [{"status": "PASS"} for _ in range(15)]
            self.assertEqual(
                _calibration_smoke_terminal(seed_result(), context, None, required_d8_cells=15),
                TERMINAL_CALIBRATION_SMOKE_COMPLETE,
            )
            self.assertEqual(
                _calibration_smoke_terminal(seed_result(), context, None, required_d8_cells=15, wall_gate_seconds=1800.01),
                TERMINAL_CALIBRATION_QA_CAPACITY_HOLD,
            )
            context.capacity_issues.append({"reason": "OOM"})
            self.assertEqual(
                _calibration_smoke_terminal(seed_result(), context, None, required_d8_cells=15),
                TERMINAL_CALIBRATION_QA_CAPACITY_HOLD,
            )
            context.capacity_issues.clear()
            self.assertEqual(
                _calibration_smoke_terminal(seed_result([{"gate": "D3"}]), context, None, required_d8_cells=15),
                TERMINAL_CALIBRATION_QA_SCIENTIFIC_HOLD,
            )
            self.assertEqual(
                _calibration_smoke_terminal(seed_result(), context, {"kind": "PilotHardStop", "error": "runtime crash"}, required_d8_cells=15),
                TERMINAL_CALIBRATION_QA_HARNESS_HOLD,
            )

    def test_terminal_priority_is_scientific_fail_then_capacity_then_pass(self) -> None:
        terminal, capacity, reason = _terminal_decision(
            [{"gate": "D3"}], [{"reason": "OOM"}], {"kind": "runtime"}
        )
        self.assertEqual(terminal, TERMINAL_FAIL)
        self.assertTrue(capacity)
        self.assertEqual(reason, "OOM")

        terminal, capacity, reason = _terminal_decision(
            [], [{"reason": "VRAM_BUDGET"}], {"kind": "PilotHardStop", "error": "CUDA runtime crash"}
        )
        self.assertEqual(terminal, TERMINAL_FAIL)
        self.assertTrue(capacity)
        self.assertEqual(reason, "VRAM_BUDGET")

        terminal, capacity, reason = _terminal_decision([], [{"reason": "VRAM_BUDGET"}], None)
        self.assertEqual((terminal, capacity, reason), (TERMINAL_CAPACITY_HOLD, True, "VRAM_BUDGET"))

        terminal, capacity, reason = _terminal_decision([], [], None)
        self.assertEqual((terminal, capacity, reason), (TERMINAL_PASS, False, None))

    def test_total_wall_is_decided_in_process_after_artifact_persistence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_wall_end_test_") as temporary:
            root = Path(temporary) / "result"
            context = PilotContext(result_root=root, official=True, wall_start=time.perf_counter())
            result = {
                "terminal_status": TERMINAL_PASS,
                "scientific_failures": [],
                "capacity_issue": False,
                "capacity_reason": None,
                "seed_results": [],
            }
            with patch("omega_v2_2b_d512_local_pilot.runner.D8_WALL_LIMIT_SECONDS", -1):
                _finalize_wall_in_process(result, context, root)
            persisted = json.loads((root / "OFFICIAL_RESULT.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["terminal_status"], TERMINAL_CAPACITY_HOLD)
            self.assertEqual(persisted["capacity_reason"], "WALL_TIME")
            self.assertGreater(persisted["official_wall_gate_seconds"], -1)

    def test_external_log_finalizer_does_not_recompute_wall_or_terminal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_log_finalize_test_") as temporary:
            root = Path(temporary) / "official"
            logs = Path(temporary) / "logs"
            root.mkdir()
            logs.mkdir()
            for name in ("command.txt", "stdout.log", "stderr.log"):
                (logs / name).write_text(name, encoding="utf-8")
            (Path(temporary) / "seal.json").write_text("{}\n", encoding="utf-8")
            (root / "OFFICIAL_RESULT.json").write_text(
                '{"official_id":"OMEGA-V2-2B-LOCAL-PILOT-01","terminal_status":"OMEGA_V2_2B_LOCAL_PILOT_PASS","scientific_failures":[],"official_wall_start":1.0,"official_wall_gate_end":13.5,"official_wall_gate_seconds":12.5,"seed_results":[],"capacity_issue":false,"capacity_reason":null}',
                encoding="utf-8",
            )
            initial_result_sha = runner.sha256_file(root / "OFFICIAL_RESULT.json")
            with (
                patch("omega_v2_2b_d512_local_pilot.runner.OFFICIAL_RESULTS_ROOT", root),
                patch("omega_v2_2b_d512_local_pilot.runner.OFFICIAL_LAUNCH_LOG_ROOT", logs),
            ):
                finalized = finalize_external_launch_logs(smoke=False)
            persisted = json.loads((root / "OFFICIAL_RESULT.json").read_text(encoding="utf-8"))
            final_result_sha = runner.sha256_file(root / "OFFICIAL_RESULT.json")
            external_manifest = json.loads((root / "external_launch_logs_artifact_hashes.json").read_text(encoding="utf-8"))
        self.assertEqual(finalized["gate_wall_seconds_unchanged"], True)
        self.assertEqual(persisted["official_wall_gate_seconds"], 12.5)
        self.assertEqual(persisted["terminal_status"], TERMINAL_PASS)
        self.assertEqual(initial_result_sha, final_result_sha)
        self.assertTrue(external_manifest["verified"])

    def test_smoke_log_finalizer_preserves_gate_result_and_wall(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_smoke_log_finalize_test_") as temporary:
            root = Path(temporary) / "smoke"
            logs = Path(temporary) / "logs"
            root.mkdir()
            logs.mkdir()
            for name in ("command.txt", "stdout.log", "stderr.log"):
                (logs / name).write_text(name, encoding="utf-8")
            result_path = root / "CALIBRATION_SMOKE_RESULT.json"
            result_path.write_text(
                '{"terminal_status":"V2_2B_CALIBRATION_QA_HARNESS_HOLD","classification":"CALIBRATION_QA_ONLY","V2_2B_verdict":null,"wall_gate_start":1.0,"wall_gate_end":2.2,"wall_gate_seconds":1.2}',
                encoding="utf-8",
            )
            initial_hash = runner.sha256_file(result_path)
            with (
                patch("omega_v2_2b_d512_local_pilot.runner.CALIBRATION_SMOKE_ROOT", root),
                patch("omega_v2_2b_d512_local_pilot.runner.CALIBRATION_SMOKE_LOG_ROOT", logs),
            ):
                finalized = finalize_external_launch_logs(smoke=True)
            external_manifest = json.loads((root / "external_launch_logs_artifact_hashes.json").read_text(encoding="utf-8"))
            final_hash = runner.sha256_file(result_path)
        self.assertTrue(finalized["gate_wall_seconds_unchanged"])
        self.assertEqual(initial_hash, final_hash)
        self.assertEqual(external_manifest["gate_wall_seconds_frozen"], 1.2)
        self.assertTrue(external_manifest["verified"])

    def test_smoke_report_uses_the_persisted_D3_A_B_C_field(self) -> None:
        smoke = {
            "terminal_status": "CALIBRATION_QA_ONLY",
            "seed_result": {
                "seed_plan": {"master_seed": CALIBRATION_SEED},
                "gate_results": {},
            },
            "structural_gates_pass": True,
            "D3_A_B_C_pass": True,
            "cell_records": [],
        }
        report = _render_smoke_report(smoke)
        self.assertIn("D3 A/B/C gates: `True`", report)

    def test_smoke_artifact_manifest_includes_boundary_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_smoke_manifest_test_") as temporary:
            root = Path(temporary) / "smoke"
            root.mkdir()
            boundary = root / "QA_SMOKE_BOUNDARY.json"
            boundary.write_text('{"QA_ONLY":true}\n', encoding="utf-8")
            smoke = {
                "terminal_status": "V2_2B_CALIBRATION_QA_HOLD",
                "V2_2B_verdict": None,
                "seed_result": {
                    "seed_plan": {"master_seed": CALIBRATION_SEED},
                    "gate_results": {},
                },
                "structural_gates_pass": False,
                "D3_A_B_C_pass": False,
            }
            with patch("omega_v2_2b_d512_local_pilot.runner.CALIBRATION_SMOKE_ROOT", root):
                result = runner._persist_smoke_artifacts(smoke)
            manifest = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
        self.assertTrue(result["verified"])
        self.assertIn(str(boundary.resolve()), manifest["artifacts"])

    def test_from_import_resolution_and_module_sweep_pass(self) -> None:
        resolution = from_import_resolution_audit()
        sweep = package_import_sweep()
        self.assertTrue(resolution["pass"], resolution["missing"])
        self.assertTrue(any(
            row["file"] == "metrics.py" and row["symbol"] == "sum_u4_diagnostics" and row["deferred_or_local"] and row["exists"]
            for row in resolution["imports"]
        ))
        self.assertTrue(sweep["pass"], sweep["errors"])

    def test_calibration_source_snapshot_is_bound_to_current_spec_and_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_snapshot_test_") as temporary:
            snapshot_path = Path(temporary) / "CALIBRATION_SOURCE_SNAPSHOT.json"
            qa_path = Path(temporary) / "QA_REPORT.json"
            runner.write_json(qa_path, {"status": "PASS", "source_sha256": runner.source_hashes()})
            with (
                patch("omega_v2_2b_d512_local_pilot.runner.CALIBRATION_SOURCE_SNAPSHOT_PATH", snapshot_path),
                patch("omega_v2_2b_d512_local_pilot.runner.QA_REPORT_PATH", qa_path),
            ):
                created = runner.write_calibration_source_snapshot()
                verified = runner.verify_calibration_source_snapshot()
        self.assertTrue(verified["verified"])
        self.assertEqual(created["snapshot_sha256"], verified["snapshot_sha256"])

    def test_source_audit_detects_injected_undefined_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_audit_test_") as temporary:
            root = Path(temporary)
            (root / "broken.py").write_text("def value():\n    return missing_audit_probe\n", encoding="utf-8")
            issues = unresolved_name_audit(root)
        self.assertEqual(issues, [{"path": "broken.py", "unresolved": ["missing_audit_probe"]}])

    def test_pre_cuda_dry_run_does_not_create_official_slot(self) -> None:
        result = pre_cuda_official_dry_run()
        self.assertFalse(result["slot_created"])
        self.assertFalse(result["official_seed_plans_materialized"])
        self.assertEqual(result["cuda_kernels_launched"], 0)

    def test_result_slot_is_lazy_until_boundary_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_boundary_test_") as temporary:
            slot = Path(temporary) / "symbolic_slot"
            context = PilotContext(result_root=slot, official=True, wall_start=0.0)
            self.assertFalse(slot.exists())
            context.mark_boundary(master_seed="symbolic_slot_01", cell_id="D1_symbolic_slot_01_K1")
            self.assertTrue(slot.is_dir())
            marker = slot / "EXECUTION_BOUNDARY.json"
            self.assertTrue(marker.is_file())
            self.assertTrue(context.boundary_crossed)

    def test_mocked_full_control_flow_uses_real_orchestrator_and_persists_cases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="omega_v2b_control_flow_test_") as temporary:
            result = full_mocked_control_flow_dry_run(Path(temporary) / "mocked")
        self.assertTrue(result["all_scenarios_pass"])
        self.assertTrue(result["artifact_hashes_verified"])
        self.assertEqual(result["cuda_kernels_launched"], 0)
        self.assertFalse(result["seed_values_materialized"])
        self.assertTrue(result["cases"]["HARD_STOP"]["partial_seed_persisted"])
        self.assertEqual(result["cases"]["HARD_STOP"]["not_run_seed_count"], 2)


if __name__ == "__main__":
    unittest.main()

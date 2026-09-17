"""Synthetic unit tests for Phase 2 orchestration and gate helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_er32_cost_gate as runner  # noqa: E402


def _training_result(config: str, rounds: int, *, status: str = "completed", seconds: float = 1.0, updates: int = 6) -> dict[str, object]:
    measured = [{"phase": "measured", "timing": {"total_seconds": seconds}} for _ in range(4)]
    return {
        "config": config,
        "rounds": rounds,
        "status": status,
        "updates_completed": updates,
        "updates": measured,
        "parameter_count": 100 if config == "F" else 100 - runner.PARAMETER_REDUCTION,
        "tensor_inventory": {
            "parameter_bytes": 1000 if config == "F" else 1000 - runner.PARAMETER_BYTE_REDUCTION,
            "gradient_bytes": 1000 if config == "F" else 1000 - runner.PARAMETER_BYTE_REDUCTION,
            "optimizer_exp_avg_bytes": 1000 if config == "F" else 1000 - runner.PARAMETER_BYTE_REDUCTION,
            "optimizer_exp_avg_sq_bytes": 1000 if config == "F" else 1000 - runner.PARAMETER_BYTE_REDUCTION,
            "optimizer_other_tensor_bytes": 0,
            "persistent_train_state_bytes": 4000,
        },
    }


def _inference_result(
    config: str,
    rounds: int,
    mode_a_tps: float = 100.0,
    mode_b_tps: float = 100.0,
    rss: int = 1000,
    mode_a_ms: float = 10.0,
    mode_b_seconds: float = 2.56,
) -> dict[str, object]:
    return {
        "config": config,
        "rounds": rounds,
        "mode_a": {"tokens_per_second": mode_a_tps, "ms_per_token": mode_a_ms},
        "mode_b": {"mean_tokens_per_second": mode_b_tps, "mean_seconds_per_window": mode_b_seconds},
        "steady_state_rss_bytes": rss,
    }


def test_schedule_and_measured_update_selection() -> None:
    assert runner.WINDOW_SCHEDULE == (0, 1, 0, 1, 0, 1)
    result = _training_result("F", 1)
    result["updates"] = [
        {"phase": "warmup", "timing": {"total_seconds": 99.0}},
        {"phase": "warmup", "timing": {"total_seconds": 99.0}},
        *result["updates"],  # type: ignore[list-item]
    ]
    aggregate = runner.aggregate_training_metrics([result, _training_result("F", 4), _training_result("ER32", 1), _training_result("ER32", 4)])
    assert aggregate["by_combination"]["F_K1"]["measured_updates"] == 4
    assert aggregate["by_combination"]["F_K1"]["measured_total_seconds"] == 4.0


def test_fresh_process_ordering_and_no_reuse(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_spawn(command: list[str], token: str) -> dict[str, object]:
        calls.append((command[command.index("--config") + 1], command[command.index("--rounds") + 1]))
        return {"config": calls[-1][0], "rounds": int(calls[-1][1]), "status": "completed"}

    results = runner.run_training_children(tmp_path, tmp_path / "ledger.jsonl", spawn=fake_spawn)
    assert [(config, str(rounds)) for config, rounds in runner.TRAINING_COMBINATIONS] == calls
    assert len(results) == 4


def test_spawn_child_is_mocked_and_decodes_only_child_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0
        stdout = "diagnostic line\n{\"status\": \"completed\"}\n"
        stderr = ""

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Completed())
    assert runner._spawn_child(["synthetic-child"], "token")["status"] == "completed"


def test_tensor_inventory_counts_real_tensor_categories() -> None:
    model = runner.nn.Linear(2, 2)
    optimizer = runner.make_adamw(model)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    inventory = runner.tensor_inventory(model, optimizer)
    assert inventory["parameter_bytes"] == 24
    assert inventory["gradient_bytes"] == 24
    assert inventory["optimizer_exp_avg_bytes"] == 24
    assert inventory["optimizer_exp_avg_sq_bytes"] == 24
    assert inventory["persistent_train_state_bytes"] >= 96


def test_snapshot_fields_and_guard() -> None:
    snapshots: list[dict[str, int | None | str]] = []
    snapshot = runner.capture_snapshot(snapshots, "before_teacher", snapshot_fn=lambda: {"rss_bytes": 1, "available_system_bytes": runner.MIN_AVAILABLE_BYTES, "os_peak_working_set_bytes": 2})
    assert {"rss_bytes", "available_system_bytes", "os_peak_working_set_bytes"}.issubset(snapshot)
    with pytest.raises(runner.ResourceGuard):
        runner.capture_snapshot(snapshots, "after_teacher", snapshot_fn=lambda: {"rss_bytes": 1, "available_system_bytes": runner.MIN_AVAILABLE_BYTES - 1, "os_peak_working_set_bytes": 2})


def test_self_hash_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    digest, file_hash = runner.write_self_hashed_json(path, {"schema": "er32-cost-gate-v1", "path": path})
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["artifact_self_hash"] == digest
    assert file_hash == runner.sha256_file(path)
    with pytest.raises(FileExistsError):
        runner.write_self_hashed_json(path, {"status": "mutated"})


def test_gate_classifications_and_thresholds() -> None:
    assert runner.classify_training_cost({"R_train": 1.25})["classification"] == "PASS"
    assert runner.classify_training_cost({"R_train": 1.2501})["classification"] == "COST_REGRESSION"
    metrics = runner.aggregate_inference_metrics([
        _inference_result("F", 1), _inference_result("F", 4), _inference_result("ER32", 1, 80, 80), _inference_result("ER32", 4, 80, 80)
    ])
    assert runner.classify_inference_cost(metrics)["classification"] == "PASS"
    metrics["ER32_over_F_time_cost"]["K1_mode_a"] = 2.0
    assert runner.classify_inference_cost(metrics)["classification"] == "INFERENCE_COST_REGRESSION"


def test_training_cost_ratio_canaries_use_er32_over_f_time() -> None:
    faster = runner.aggregate_training_metrics([
        _training_result("F", 1, seconds=2.5),
        _training_result("F", 4, seconds=2.5),
        _training_result("ER32", 1, seconds=1.75),
        _training_result("ER32", 4, seconds=1.75),
    ])
    slower = runner.aggregate_training_metrics([
        _training_result("F", 1, seconds=2.5),
        _training_result("F", 4, seconds=2.5),
        _training_result("ER32", 1, seconds=3.25),
        _training_result("ER32", 4, seconds=3.25),
    ])
    assert faster["R_train"] == pytest.approx(0.70)
    assert runner.classify_training_cost(faster)["classification"] == "PASS"
    assert slower["R_train"] == pytest.approx(1.30)
    assert runner.classify_training_cost(slower)["classification"] == "COST_REGRESSION"


def test_inference_cost_ratio_canaries_use_time_fields_not_throughput() -> None:
    faster = runner.aggregate_inference_metrics([
        _inference_result("F", 1, mode_a_tps=100.0, mode_b_tps=100.0, mode_a_ms=10.0, mode_b_seconds=2.0),
        _inference_result("F", 4, mode_a_tps=100.0, mode_b_tps=100.0, mode_a_ms=10.0, mode_b_seconds=2.0),
        _inference_result("ER32", 1, mode_a_tps=2000.0, mode_b_tps=1.0, mode_a_ms=7.0, mode_b_seconds=1.4),
        _inference_result("ER32", 4, mode_a_tps=2000.0, mode_b_tps=1.0, mode_a_ms=7.0, mode_b_seconds=1.4),
    ])
    slower = runner.aggregate_inference_metrics([
        _inference_result("F", 1, mode_a_tps=100.0, mode_b_tps=100.0, mode_a_ms=10.0, mode_b_seconds=2.0),
        _inference_result("F", 4, mode_a_tps=100.0, mode_b_tps=100.0, mode_a_ms=10.0, mode_b_seconds=2.0),
        _inference_result("ER32", 1, mode_a_tps=1.0, mode_b_tps=2000.0, mode_a_ms=13.0, mode_b_seconds=2.6),
        _inference_result("ER32", 4, mode_a_tps=1.0, mode_b_tps=2000.0, mode_a_ms=13.0, mode_b_seconds=2.6),
    ])
    assert faster["ER32_over_F_time_cost"]["K1_mode_a"] == pytest.approx(0.70)
    assert faster["ER32_over_F_time_cost"]["K1_mode_b"] == pytest.approx(0.70)
    assert runner.classify_inference_cost(faster)["classification"] == "PASS"
    assert slower["ER32_over_F_time_cost"]["K1_mode_a"] == pytest.approx(1.30)
    assert slower["ER32_over_F_time_cost"]["K1_mode_b"] == pytest.approx(1.30)
    assert runner.classify_inference_cost(slower)["classification"] == "INFERENCE_COST_REGRESSION"


def test_memory_safety_guard_is_inconclusive_not_er32_fail() -> None:
    results = [_training_result(config, rounds) for config, rounds in runner.TRAINING_COMBINATIONS]
    results[1]["status"] = "INCONCLUSIVE_RESOURCE"
    assert runner.classify_memory_safety(results)["classification"] == "INCONCLUSIVE_RESOURCE"


def test_persistent_footprint_requires_exact_reduction() -> None:
    results = [_training_result(config, rounds) for config, rounds in runner.TRAINING_COMBINATIONS]
    assert runner.classify_persistent_footprint(results)["classification"] == "PASS"
    results[1]["parameter_count"] = 99
    assert runner.classify_persistent_footprint(results)["classification"] == "FAIL"


def test_inference_metric_aggregation_and_memory_anomaly() -> None:
    results = [_inference_result("F", 1, rss=1000), _inference_result("F", 4, rss=1000), _inference_result("ER32", 1, rss=1000), _inference_result("ER32", 4, rss=1000)]
    metrics = runner.aggregate_inference_metrics(results)
    assert metrics["ER32_over_F_time_cost"]["K1_mode_a"] == 1.0
    results[-1]["steady_state_rss_bytes"] = 1000 + runner.RSS_ANOMALY_BYTES + 1
    assert runner.classify_inference_cost(runner.aggregate_inference_metrics(results))["classification"] == "MEMORY_RUNTIME_ANOMALY"


def test_analysis_repair_derives_values_once_and_preserves_sources(tmp_path: Path) -> None:
    training = runner.aggregate_training_metrics([
        _training_result("F", 1, seconds=2.5),
        _training_result("F", 4, seconds=2.5),
        _training_result("ER32", 1, seconds=1.75),
        _training_result("ER32", 4, seconds=1.75),
    ])
    inference = runner.aggregate_inference_metrics([
        _inference_result("F", 1, mode_a_ms=10.0, mode_b_seconds=2.0),
        _inference_result("F", 4, mode_a_ms=10.0, mode_b_seconds=2.0),
        _inference_result("ER32", 1, mode_a_ms=7.0, mode_b_seconds=1.4),
        _inference_result("ER32", 4, mode_a_ms=7.0, mode_b_seconds=1.4),
    ])
    cost_path = tmp_path / "cost_report.json"
    inference_path = tmp_path / "inference_report.json"
    integration_path = tmp_path / "integration_report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    runner.write_self_hashed_json(cost_path, {"artifact": "cost_report", "run_id": "synthetic", "training": training, "gates": {"TRAINING_COST": {"classification": "PASS"}}})
    runner.write_self_hashed_json(inference_path, {"artifact": "inference_report", "run_id": "synthetic", "inference": inference, "gates": {"INFERENCE_COST": {"classification": "PASS"}}})
    integration_path.write_text("integration\n", encoding="utf-8")
    ledger_path.write_text("ledger\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in (cost_path, inference_path, integration_path, ledger_path)}

    artifact_hash, file_hash = runner.write_analysis_repair(tmp_path)
    repair_path = tmp_path / "analysis_repair.json"
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    assert repair["artifact_self_hash"] == artifact_hash
    assert file_hash == runner.sha256_file(repair_path)
    unsigned = dict(repair)
    unsigned["artifact_self_hash"] = runner.SELF_HASH_PLACEHOLDER
    assert runner.sha256_bytes(runner.canonical_json(unsigned)) == artifact_hash
    assert repair["benchmark_rerun"] is False
    assert repair["raw_measurements_unchanged"] is True
    assert repair["gate_V_training_cost"]["corrected_metric"] == pytest.approx(0.70)
    assert repair["gate_VI_inference_cost"]["corrected_metrics"]["K1_mode_a"] == pytest.approx(0.70)
    assert repair["recalculated_gate_classifications"]["TRAINING_COST"]["classification"] == "PASS"
    assert repair["recalculated_gate_classifications"]["INFERENCE_COST"]["classification"] == "PASS"
    assert {path.name: path.read_bytes() for path in (cost_path, inference_path, integration_path, ledger_path)} == before
    with pytest.raises(FileExistsError):
        runner.write_analysis_repair(tmp_path)


def test_inference_report_global_status_uses_full_six_gate_map() -> None:
    full_gates = {key: {"classification": "PASS"} for key in ("I", "II", "III", "IV", "V", "VI")}
    report = runner.build_inference_report("synthetic-run", {}, {}, full_gates)
    assert set(report["gates"]) == {"INTEGRATION_CORRESPONDENCE", "EXPLICIT_EFFICIENT_EQUIVALENCE", "INFERENCE_COST"}
    assert report["global_status"] == "PASS"


def test_cli_refuses_without_execute(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main([]) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "refused"
    assert "--execute" in output["reason"]


def test_child_requires_parent_token(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv(runner.CHILD_TOKEN_ENV, raising=False)
    assert runner.main(["--execute", "--child-training", "--child-token", "wrong", "--config", "F", "--rounds", "1", "--run-dir", ".", "--ledger", "ledger.jsonl"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "refused"

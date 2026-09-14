"""CPU-only tests for OMEGA update-efficiency preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from run_efficiency_gate import (
    CONFIGS,
    GATE_ID,
    run_gate,
    validate_gate_options,
)
from omega_nominal_microbatch_runner import SUPPORTED_PHYSICAL_BATCHES


def test_config_validation_supports_only_a_b_c() -> None:
    assert SUPPORTED_PHYSICAL_BATCHES == (2, 4, 8)
    assert [physical_batch for _, physical_batch in CONFIGS] == [2, 4, 8]
    for invalid in (1, 2.0, 3, 16, True):
        with pytest.raises(ValueError, match="physical_batch"):
            from omega_nominal_microbatch_runner import microbatch_count_for_physical_batch

            microbatch_count_for_physical_batch(invalid)


def test_cpu_dry_run_executes_a_b_c_and_writes_append_only_schema(tmp_path: Path) -> None:
    result = run_gate(device_name="cpu", output_dir=tmp_path, dry_run=True, torch_compile=False, authorize_gpu_benchmark=False)
    assert result["gate"] == GATE_ID
    assert result["mode"] == "cpu_dry_run"
    assert result["performance_measured"] is False
    assert [(item["config"], item["physical_batch"], item["microbatches"]) for item in result["results"]] == [("A", 2, 4), ("B", 4, 2), ("C", 8, 1)]
    assert all(item["valid_tokens"] == 2048 and item["state_shape"] == [8, 1, 2] for item in result["results"])
    assert all(item["seconds_per_update"] is None and item["gpu_reserved_bytes"] is None for item in result["results"])
    ledger_lines = (tmp_path / "efficiency_gate_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    report_lines = (tmp_path / "efficiency_gate_report.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 3
    assert len(report_lines) == 1
    assert all(json.loads(line)["record_type"] == "config_result" for line in ledger_lines)
    assert json.loads(report_lines[0])["schema_version"] == 1


def test_compile_flag_rejected_on_cpu() -> None:
    with pytest.raises(ValueError, match="requires --device cuda"):
        validate_gate_options(device_name="cpu", dry_run=True, torch_compile=True, authorize_gpu_benchmark=False)


def test_cuda_request_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        validate_gate_options(device_name="cuda", dry_run=False, torch_compile=False, authorize_gpu_benchmark=True)


def test_static_report_and_runner_policy_are_fp32_only() -> None:
    report = json.loads((Path(__file__).with_name("efficiency_gate_report.json")).read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["status"] == "CPU_PREPARATION_ONLY"
    assert report["precision_policy"] == {"dtype": "float32", "mixed_precision": False, "precision_reduction": False}
    assert report["no_scientific_result"] is True
    assert report["no_15_gpu_campaign_decision"] is True
    source = Path(__file__).with_name("run_efficiency_gate.py").read_text(encoding="utf-8")
    assert "torch.float32" in source
    assert "autocast" not in source.lower()
    assert "bfloat16" not in source.lower()

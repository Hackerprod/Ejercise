from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_omega_readout_cache_survival_audit as runner  # noqa: E402


def test_validated_topology_has_one_logical_per_physical_core() -> None:
    targets = runner.load_validated_topology()
    assert [(item.architecture, item.physical_core, item.logical_processor) for item in targets] == [("Zen5", 0, 0), ("Zen5c", 1, 2), ("Zen5c", 2, 4), ("Zen5c", 3, 6)]


def test_block_order_is_deterministic_and_balanced() -> None:
    first = runner.balanced_block_order(7)
    second = runner.balanced_block_order(7)
    assert first == second
    assert len(first) == 4 * runner.BLOCK_COUNT
    assert all(first.count((architecture, core, block)) == 1 for architecture, core, block in first)


def test_ratio_is_ratio_of_medians_and_bootstraps_complete_blocks() -> None:
    result = runner.ratio_of_medians([2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [0.01] * 4, bootstrap_rounds=50)
    assert result["R_eviction"] == pytest.approx(0.5)
    assert result["S_survival"] == pytest.approx(0.5)
    assert result["classification"] == "MEASUREMENT_COMPLETE"


def test_unresolved_denominator_requires_all_three_numeric_conditions() -> None:
    result = runner.ratio_of_medians([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], bootstrap_rounds=50)
    assert result["classification"] == "UNRESOLVED_DENOMINATOR"
    assert result["R_eviction"] is None
    noisy_negative = runner.ratio_of_medians([1.0, 1.0, 1.0], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0], bootstrap_rounds=50)
    assert noisy_negative["classification"] == "UNRESOLVED_DENOMINATOR"


def test_states_are_allowlisted_without_architectural_gate() -> None:
    assert runner.STATE_ALLOWLIST == {"MEASUREMENT_COMPLETE", "UNRESOLVED_DENOMINATOR", "MEASUREMENT_RESOLUTION_INSUFFICIENT", "TECHNICAL_FAIL"}
    result = runner.ratio_of_medians([1.0, 1.0], [1.0, 1.0], [0.0, 0.0], bootstrap_rounds=20)
    assert result["classification"] in runner.STATE_ALLOWLIST


def test_exact_warmup_constant_and_single_core_contract() -> None:
    assert runner.WARMUP_CALLS == 16
    assert runner.TOKEN_BATCH == 1
    assert runner.TOKEN_COUNT == 1
    assert runner.load_validated_topology()[0].architecture == "Zen5"
    assert sum(item.architecture == "Zen5c" for item in runner.load_validated_topology()) == 3


def test_cold_buffer_meets_positive_control_size() -> None:
    assert runner.POSITIVE_CONTROL_BYTES >= 64 * 1024 * 1024
    assert runner.POSITIVE_CONTROL_BYTES >= 4 * runner.LLC_BYTES


def test_condition_call_contract_has_head_only_for_posthead() -> None:
    class SpyModel:
        def __init__(self) -> None:
            self.head_calls = 0
            self.probe_calls = 0

        def logits_from_states(self, _state: object) -> object:
            self.head_calls += 1
            return __import__("torch").zeros(1, 1, 3)

        def recur_states(self, _tokens: object, _state: object) -> tuple[None, None, None, None]:
            self.probe_calls += 1
            return None, None, None, None

    model = SpyModel()
    runner.block_measurement(model, object(), lambda: object(), cold_bytes=64)
    assert model.head_calls == 1
    assert model.probe_calls == 1 + runner.WARMUP_CALLS + 1 + 1


def test_resolution_control_precedes_main_data_contract() -> None:
    assert runner.choose_resolution_state([1e-8], 1e-3) == "MEASUREMENT_RESOLUTION_INSUFFICIENT"
    assert runner.choose_resolution_state([1e-3], 1e-3) == "MEASUREMENT_COMPLETE"


def test_self_hash_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    digest = runner.write_self_hashed_json(path, {"status": "MEASUREMENT_COMPLETE"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact_self_hash"] == digest
    with pytest.raises(FileExistsError):
        runner.write_self_hashed_json(path, {"status": "TECHNICAL_FAIL"})


def test_real_cli_requires_explicit_flags_and_positive_control(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.main([])
    with pytest.raises(SystemExit):
        runner.main(["--positive-control"])
    with pytest.raises(SystemExit):
        runner.main(["--full", "--confirm-omega-readout-cache-survival-audit", "--output-dir", str(tmp_path)])


def test_smoke_does_not_load_models_or_run_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert runner.main(["--smoke"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {"blocks": 0, "real_data_loaded": False, "status": "MEASUREMENT_COMPLETE"}

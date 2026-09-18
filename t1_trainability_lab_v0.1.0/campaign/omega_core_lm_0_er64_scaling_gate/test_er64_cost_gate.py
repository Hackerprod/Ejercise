from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_er64_cost_gate as runner  # noqa: E402


def _training(config: str, rounds: int, seconds: float) -> dict[str, object]:
    return {"config": config, "rounds": rounds, "updates": [{"phase": "warmup", "timing": {"total_seconds": seconds}}, {"phase": "measured", "timing": {"total_seconds": seconds}}]}


def _inference(config: str, rounds: int, mode_a_ms: float, mode_b_seconds: float, rss: int = 1000) -> dict[str, object]:
    return {"config": config, "rounds": rounds, "mode_a": {"ms_per_token": mode_a_ms}, "mode_b": {"mean_seconds_per_window": mode_b_seconds}, "steady_state_rss_bytes": rss}


def test_er64_factory_has_rank64_and_copies_f_nonlexical_state() -> None:
    f = runner.make_f_reference(1, 20260913, vocab_size=31, dimension=8, slots=2)
    candidate = runner.make_candidate("ER64", 1, f, experimental_seed=20260913)
    assert candidate.embedding.C.shape == (31, 64)
    assert candidate.embedding.U.shape == (64, 8)
    assert not any(tuple(parameter.shape) == (31, 8) for parameter in candidate.parameters())
    for name, value in f.state_dict().items():
        if name != "embedding.weight":
            assert torch.equal(value, candidate.state_dict()[name]), name


def test_training_ratio_is_er64_over_f_and_canaries() -> None:
    faster = runner.aggregate_training_metrics([_training("F", 1, 1.0), _training("F", 4, 1.0), _training("ER64", 1, 0.7), _training("ER64", 4, 0.7)])
    slower = runner.aggregate_training_metrics([_training("F", 1, 1.0), _training("F", 4, 1.0), _training("ER64", 1, 1.3), _training("ER64", 4, 1.3)])
    assert faster["R_train"] == pytest.approx(0.7)
    assert runner.classify_training_cost(faster)["classification"] == "PASS"
    assert slower["R_train"] == pytest.approx(1.3)
    assert runner.classify_training_cost(slower)["classification"] == "COST_REGRESSION"


def test_inference_ratio_uses_time_fields_not_throughput() -> None:
    faster = runner.aggregate_inference_metrics([_inference("F", 1, 10.0, 2.0), _inference("F", 4, 10.0, 2.0), _inference("ER64", 1, 7.0, 1.4), _inference("ER64", 4, 7.0, 1.4)])
    slower = runner.aggregate_inference_metrics([_inference("F", 1, 10.0, 2.0), _inference("F", 4, 10.0, 2.0), _inference("ER64", 1, 13.0, 2.6), _inference("ER64", 4, 13.0, 2.6)])
    assert faster["ER64_over_F_time_cost"]["K1_mode_a"] == pytest.approx(0.7)
    assert faster["ER64_over_F_time_cost"]["K4_mode_b"] == pytest.approx(0.7)
    assert runner.classify_inference_cost(faster)["classification"] == "PASS"
    assert slower["ER64_over_F_time_cost"]["K1_mode_a"] == pytest.approx(1.3)
    assert runner.classify_inference_cost(slower)["classification"] == "INFERENCE_COST_REGRESSION"


def test_rss_anomaly_is_separate_from_timing_threshold() -> None:
    results = [_inference("F", 1, 10.0, 2.0, 1000), _inference("F", 4, 10.0, 2.0, 1000), _inference("ER64", 1, 10.0, 2.0, 1000 + runner.RSS_ANOMALY_BYTES + 1), _inference("ER64", 4, 10.0, 2.0, 1000)]
    assert runner.classify_inference_cost(runner.aggregate_inference_metrics(results))["classification"] == "MEMORY_RUNTIME_ANOMALY"


def test_self_hash_cost_report_is_write_once_and_reproducible(tmp_path: Path) -> None:
    training = [_training("F", 1, 1.0), _training("F", 4, 1.0), _training("ER64", 1, 1.0), _training("ER64", 4, 1.0)]
    inference = [_inference("F", 1, 1.0, 1.0), _inference("F", 4, 1.0, 1.0), _inference("ER64", 1, 1.0, 1.0), _inference("ER64", 4, 1.0, 1.0)]
    path = tmp_path / "cost_report.json"
    digest, file_digest = runner.write_self_hashed_json(path, runner.build_cost_report(training, inference))
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["artifact_self_hash"] == digest
    assert file_digest == runner.sha256_file(path)
    assert parsed["benchmark_rerun"] is False
    with pytest.raises(FileExistsError):
        runner.write_self_hashed_json(path, runner.build_cost_report(training, inference))

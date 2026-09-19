from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_omega_ce_only_baseline as runner  # noqa: E402


def tiny_model(seed: int = 11, k: int = 1) -> runner.OmegaCoreLMFast:
    return runner.fresh_model(seed, k, vocab_size=17, dimension=4, slots=1)


def test_ce_uses_full_coefficient_and_no_kl() -> None:
    model = tiny_model()
    states = torch.randn(2, 5, 1, 4)
    targets = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    result = runner.ce_only_loss(model, states, targets)
    projected = model.project(states).reshape(-1, 4)
    expected = F.cross_entropy(model.logits_from_projected(projected), targets.reshape(-1))
    assert torch.equal(result["ce"], expected)
    assert torch.equal(result["total"], expected)
    assert "kl" not in result


def test_ce_chunks_vocab_projection_and_never_builds_full_batch_vocab() -> None:
    model = tiny_model()
    calls: list[tuple[int, ...]] = []
    original = model.logits_from_projected

    def intercepted(projected: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(projected.shape))
        assert projected.shape[0] <= runner.CHUNK_SIZE
        return original(projected)

    model.logits_from_projected = intercepted  # type: ignore[method-assign]
    states = torch.randn(1, runner.CHUNK_SIZE * 2 + 1, 1, 4)
    targets = torch.randint(0, 17, (1, runner.CHUNK_SIZE * 2 + 1))
    result = runner.ce_only_loss(model, states, targets)
    assert result["chunks"] == 3
    assert [shape[0] for shape in calls] == [512, 512, 1]
    assert all(len(shape) == 2 for shape in calls)


def test_ce_forward_uses_recur_states_not_forward_window() -> None:
    model = tiny_model()
    model.forward_window = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("forbidden"))  # type: ignore[method-assign]
    inputs = torch.randint(0, 17, (2, 5))
    targets = torch.randint(0, 17, (2, 5))
    state = model.initial_state(2, device=torch.device("cpu"))
    _, readout, losses = runner.ce_only_forward_loss(model, inputs, targets, state)
    assert readout.shape == (2, 5, 1, 4)
    assert torch.isfinite(losses["total"])


def test_phase0_worker_behavior_separates_ce_and_distillation() -> None:
    ce_row = runner._phase0_worker("CE-only-K1", smoke=True)
    assert ce_row["teacher_imported"] is False
    assert ce_row["teacher_forward_calls"] == 0
    assert ce_row["kl_calculations"] == 0

    distill_row = runner._phase0_worker("DISTILL-K1", smoke=True)
    assert distill_row["teacher_imported"] is True
    assert distill_row["teacher_forward_calls"] == runner.PHASE0_MEASURED_UPDATES
    assert distill_row["kl_calculations"] == runner.PHASE0_MEASURED_UPDATES


def test_cpu_runtime_uses_shared_r1_authority_and_is_idempotent() -> None:
    assert runner.r1.CPU_INTRAOP_THREADS == runner.CPU_INTRAOP_THREADS == 4
    assert runner.r1.CPU_INTEROP_THREADS == runner.CPU_INTEROP_THREADS == 1
    assert not hasattr(runner, "_CPU_RUNTIME_CONFIGURED")
    assert not hasattr(runner, "configure_cpu_runtime")
    runner.r1.configure_cpu_runtime()
    runner.validate_policy()
    runner.r1.configure_cpu_runtime()
    assert torch.get_num_threads() == runner.CPU_INTRAOP_THREADS


def test_ce_measurement_contract_has_no_teacher_or_kl_work() -> None:
    report = runner.phase0_measurement("CE-only-K1", student_forward_seconds=1, vocab_loss_seconds=1, backward_seconds=1, clip_seconds=1, adamw_seconds=1, total_seconds=5, rss_bytes_peak=1)
    assert report["teacher_loaded"] is False
    assert report["teacher_forward_calls"] == 0
    assert report["kl_calculations"] == 0


def test_initialization_gate_accepts_exact_state_and_rejects_mismatch(tmp_path: Path) -> None:
    paths: dict[tuple[int, int], Path] = {}
    for seed in runner.SEEDS:
        for k in runner.KS:
            model = tiny_model(seed, k)
            path = tmp_path / f"{seed}_{k}.pt"
            torch.save({"update": 0, "model": model.state_dict()}, path)
            paths[(seed, k)] = path
    evidence = runner.initialization_gate(oracle_paths=paths, model_factory=lambda seed, k: tiny_model(seed, k))
    assert evidence["passed"] is True
    altered = torch.load(paths[(runner.SEEDS[0], 1)], weights_only=False)
    altered["model"]["embedding.weight"][0, 0] += 1
    torch.save(altered, paths[(runner.SEEDS[0], 1)])
    with pytest.raises(runner.InitializationGateError, match="differs"):
        runner.initialization_gate(oracle_paths=paths, model_factory=lambda seed, k: tiny_model(seed, k))


def test_frozen_manifest_paths_are_direct_r1_references() -> None:
    paths = runner.frozen_manifest_paths()
    assert paths["train"].name == "train_manifest.json"
    assert "omega_core_lm_0_r1_scientific_scoping_a" in paths["train"].as_posix()
    assert runner.load_frozen_manifests()["direct_reuse"] is True
    assert runner.load_frozen_manifests()["reconstructed"] is False


def test_phase0_formula_even_boundary_and_self_hash() -> None:
    report = runner.synthetic_phase0_report()
    expected_q = (8.0 + 10.0) / (3.0 + 4.0)
    assert report["q_cost"] == pytest.approx(expected_q)
    assert report["U_equal_cost"] == 2 * int((2000 / expected_q) // 2)
    assert report["protocol"]["total_updates"] == 24
    assert runner.verify_self_hash(report)
    assert report["measurements"]["CE-only-K4"]["teacher_loaded"] is False


def test_phase0_subprocess_workers_run_exact_synthetic_protocol(tmp_path: Path) -> None:
    report = runner.run_phase0_subprocesses(tmp_path, confirm_real_execution=False, smoke=True)
    assert len(report["measurements"]) == 4
    assert len({row["worker_pid"] for row in report["measurements"].values()}) == 4
    for name, row in report["measurements"].items():
        assert row["updates_executed"] == 6
        assert row["warmup_updates"] == 2
        assert row["measured_updates"] == 4
        assert row["total_seconds"] > 0
        assert row["rss_bytes_peak"] > 0
        if name.startswith("CE-only"):
            assert row["teacher_imported"] is False
            assert row["teacher_forward_calls"] == 0
            assert row["kl_calculations"] == 0
        else:
            assert row["teacher_imported"] is True
            assert row["teacher_forward_calls"] == 4
            assert row["kl_calculations"] == 4
    assert runner.verify_self_hash(report)
    assert (tmp_path / "phase0_report.json").is_file()


def _curve(value: float) -> list[dict[str, float]]:
    return [{"update": update, "nll": value} for update in runner.BOUNDARIES]


def test_phase_a_classification_and_automatic_equal_cost_plan() -> None:
    baselines = {(seed, k): _curve(10.0) for seed in runner.SEEDS for k in runner.KS}
    results = [
        {"seed": seed, "K": k, "validation_curve": _curve(10.05 if k == 1 else 10.06)}
        for seed in runner.SEEDS for k in runner.KS
    ]
    report = runner.build_phase_a_comparison(results, baselines, u_equal_cost=1400)
    assert report["classification_at_2000"] == "NONINFERIOR"
    assert report["equal_cost_continuation"]["required"] is False
    results[0]["validation_curve"] = _curve(10.2)
    report = runner.build_phase_a_comparison(results, baselines, u_equal_cost=1400)
    assert report["classification_at_2000"] == "MIXED"
    assert report["equal_cost_continuation"] == {
        "required": True,
        "scheduled": True,
        "executed": False,
        "updates": 1400,
        "reason": "automatic continuation when Phase A is not NONINFERIOR",
    }


def test_phase_a_delta_ce_is_informational_k1_minus_k4() -> None:
    baselines = {(seed, k): _curve(10.0) for seed in runner.SEEDS for k in runner.KS}
    results = [
        {"seed": seed, "K": k, "validation_curve": _curve(10.1 if k == 1 else 10.0)}
        for seed in runner.SEEDS for k in runner.KS
    ]
    report = runner.build_phase_a_comparison(results, baselines, u_equal_cost=1000)
    assert report["informational_Delta_CE_K1_minus_K4"][0]["boundaries"][-1]["delta_ce_k1_minus_k4"] == pytest.approx(0.1)


def test_equal_cost_classifications_and_executable_continuation_contract() -> None:
    baselines = {(seed, k): _curve(10.0) for seed in runner.SEEDS for k in runner.KS}
    noninferior = [{"seed": seed, "K": k, "validation_curve": [{"update": 2400, "nll": 10.05}]} for seed in runner.SEEDS for k in runner.KS]
    advantage = [{"seed": seed, "K": k, "validation_curve": [{"update": 2400, "nll": 10.2}]} for seed in runner.SEEDS for k in runner.KS]
    mixed = advantage[:]
    mixed[0] = {"seed": runner.SEEDS[0], "K": 1, "validation_curve": [{"update": 2400, "nll": 10.05}]}
    assert runner.build_equal_cost_comparison(noninferior, baselines, u_equal_cost=2400)["classification"] == "CE-ONLY-COST-NONINFERIOR"
    assert runner.build_equal_cost_comparison(advantage, baselines, u_equal_cost=2400)["classification"] == "DISTILLATION-COST-ADVANTAGE"
    assert runner.build_equal_cost_comparison(mixed, baselines, u_equal_cost=2400)["classification"] == "COST-MIXED"
    calls: list[tuple[int, int, int]] = []
    phase_a = [{"seed": seed, "K": k} for seed in runner.SEEDS for k in runner.KS]

    def continuation(result: dict[str, int], target: int) -> dict[str, object]:
        calls.append((result["seed"], result["K"], target))
        return {**result, "validation_curve": [{"update": target, "nll": 10.0}]}

    continued = runner.run_equal_cost_continuation(phase_a, u_equal_cost=2400, continuation_runner=continuation)
    assert len(continued) == 4
    assert calls == [(seed, k, 2400) for seed in runner.SEEDS for k in runner.KS]


def test_load_r1_baseline_curves_reads_existing_list_artifacts() -> None:
    curves = runner.load_r1_baseline_curves()
    assert set(curves) == {(seed, k) for seed in runner.SEEDS for k in runner.KS}
    assert all([int(point["update"]) for point in curve] == list(runner.BOUNDARIES) for curve in curves.values())


def test_generation_audit_reads_historical_32_r1_outputs_without_rerun() -> None:
    report = runner.build_generation_audit_report()
    assert report["generation_source_commit"] == "ca8f4ad"
    assert report["protocol"]["generation_count"] == 32
    assert report["training_performed"] is False
    assert runner.verify_self_hash(report)


def test_ce_generation_metrics_and_historical_comparison_are_documentary() -> None:
    tokens = [1, 1, 2, 1, 1, 2]
    metrics = runner.generation_metrics(tokens, None)
    assert metrics["distinct_1"] == pytest.approx(2 / 6)
    assert metrics["distinct_2"] == pytest.approx(3 / 5)
    assert metrics["max_same_token_run"] == 2
    assert 3 in metrics["exact_cycle_lengths_1_to_8"]
    assert metrics["eos_position"] is None
    historical_report = runner.build_generation_audit_report()
    ce_records = [
        {
            "K": row["K"],
            "seed": row["seed"],
            "prompt_document_id": row["prompt_document_id"],
            "generated_token_ids": [7],
            "metrics": runner.generation_metrics([7], None),
        }
        for row in historical_report["historical_generations"]
    ]
    report = runner.build_generation_audit_report(ce_generation_report={"generations": ce_records})
    assert report["ce_vs_historical_comparison"]["comparison_count"] == 32
    assert report["ce_vs_historical_comparison"]["quality_gate"] is None
    assert report["documentary_only"] is True
    assert runner.verify_self_hash(report)


def test_real_entrypoints_require_authorization_and_default_does_not_execute() -> None:
    with pytest.raises(SystemExit):
        runner.main([])
    with pytest.raises(runner.RealExecutionAuthorizationError):
        runner.run_phase_a_real(Path("unused"), confirm_real_execution=False)


def test_paths_and_contract_are_isolated() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert runner.HERE.name == "omega_ce_only_baseline"
    assert "omega_core_lm_0_r1_scientific_scoping_a" in source
    assert "write" not in source[source.index("R1_DIR"):source.index("def initialization_gate")]
    assert "checkpoint_00000.pt" in source

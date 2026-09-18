from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_er32_quality_scoping_a as runner  # noqa: E402


def _baselines(updates: tuple[int, ...] = (0, 2, 4)) -> dict[int, dict[str, object]]:
    return {
        seed: {
            "K1": {"path": "synthetic", "sha256": "k1", "curve": [{"update": update, "nll": 4.0 - update * 0.01, "tokens": 8} for update in updates]},
            "K4": {"path": "synthetic", "sha256": "k4", "curve": [{"update": update, "nll": 3.9 - update * 0.01, "tokens": 8} for update in updates]},
            "delta_R1": [{"update": update, "delta": 0.1} for update in updates],
        }
        for seed in runner.SEEDS
    }


def test_contract_is_cpu_eager_and_real_run_requires_explicit_confirmation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "torch.cuda" not in source
    assert "torch.compile" not in source
    with pytest.raises(SystemExit):
        runner.main([])


def test_resume_cli_routes_one_combination_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[tuple[Path, Path, int, str]] = []

    def fake_resume(output_dir: Path, checkpoint: Path, seed: int, variant: str) -> dict[str, object]:
        calls.append((output_dir, checkpoint, seed, variant))
        return {"mode": "authorized_resume", "seed": seed, "variant": variant}

    monkeypatch.setattr(runner, "run_resume", fake_resume)
    checkpoint = tmp_path / "checkpoint_01500.pt"
    assert runner.main([
        "--full",
        "--confirm-quality-scoping-a",
        "--resume-checkpoint",
        str(checkpoint),
        "--seed",
        "20260913",
        "--variant",
        "ER32-K4",
        "--output-dir",
        str(tmp_path / "output"),
    ]) == 0
    assert calls == [(tmp_path / "output", checkpoint, 20260913, "ER32-K4")]
    assert json.loads(capsys.readouterr().out)["mode"] == "authorized_resume"


def test_fresh_single_cli_routes_one_new_combination_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[tuple[Path, int, str]] = []

    def fake_fresh(output_dir: Path, seed: int, variant: str) -> dict[str, object]:
        calls.append((output_dir, seed, variant))
        return {"mode": "authorized_fresh_single", "seed": seed, "variant": variant}

    monkeypatch.setattr(runner, "run_fresh_single", fake_fresh)
    assert runner.main([
        "--full",
        "--confirm-quality-scoping-a",
        "--seed",
        "20260914",
        "--variant",
        "ER32-K1",
        "--output-dir",
        str(tmp_path / "output"),
    ]) == 0
    assert calls == [(tmp_path / "output", 20260914, "ER32-K1")]
    assert json.loads(capsys.readouterr().out)["mode"] == "authorized_fresh_single"


def test_single_selector_requires_both_seed_and_variant(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        runner.main(["--full", "--confirm-quality-scoping-a", "--seed", "20260914"])
    assert "requires both --seed and --variant" in capsys.readouterr().err


def test_schedule_and_boundaries_are_frozen() -> None:
    assert runner.schedule(5) == [
        {"update": 0, "window": 0, "pair": 0},
        {"update": 1, "window": 1, "pair": 0},
        {"update": 2, "window": 0, "pair": 1},
        {"update": 3, "window": 1, "pair": 1},
        {"update": 4, "window": 0, "pair": 2},
    ]
    assert runner.checkpoint_boundaries(2000, 500) == (0, 500, 1000, 1500, 2000)
    with pytest.raises(ValueError, match="boundary"):
        runner.validate_resume_boundary(501)


def test_frozen_baseline_loader_reads_exact_updates_and_delta(tmp_path: Path) -> None:
    for seed in runner.SEEDS:
        for variant, values in (("shared_K1", [4.0, 3.0, 2.0]), ("shared_K4", [3.5, 2.5, 1.5])):
            path = tmp_path / f"{variant}_seed_{seed}" / "validation_curve.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps([{"update": update, "nll": nll, "tokens": 8} for update, nll in zip((0, 2, 4), values)]), encoding="utf-8")
    baselines = runner.load_frozen_r1_baselines(tmp_path, updates=(0, 2, 4))
    assert baselines[20260913]["K1"]["curve"][1]["nll"] == 3.0
    assert baselines[20260913]["delta_R1"] == [
        {"update": 0, "delta": 0.5},
        {"update": 2, "delta": 0.5},
        {"update": 4, "delta": 0.5},
    ]


def test_frozen_baseline_loader_selects_required_prefix_from_full_r1_curve(tmp_path: Path) -> None:
    full_updates = tuple(range(0, 5001, 500))
    for seed in runner.SEEDS:
        for variant in ("shared_K1", "shared_K4"):
            path = tmp_path / f"{variant}_seed_{seed}" / "validation_curve.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps([{"update": update, "nll": float(update), "tokens": 4096} for update in full_updates]),
                encoding="utf-8",
            )
    baselines = runner.load_frozen_r1_baselines(tmp_path)
    assert baselines[20260913]["K1"]["selected_updates"] == [0, 500, 1000, 1500, 2000]
    assert baselines[20260913]["K1"]["ignored_extra_updates"] == [2500, 3000, 3500, 4000, 4500, 5000]
    assert [point["update"] for point in baselines[20260914]["K4"]["curve"]] == [0, 500, 1000, 1500, 2000]


def test_baseline_loader_rejects_duplicate_or_misaligned_points(tmp_path: Path) -> None:
    for seed in runner.SEEDS:
        for variant in ("shared_K1", "shared_K4"):
            path = tmp_path / f"{variant}_seed_{seed}" / "validation_curve.json"
            path.parent.mkdir(parents=True)
            updates = [0, 2, 2] if variant == "shared_K1" else [0, 2, 4]
            path.write_text(json.dumps([{"update": update, "nll": 1.0} for update in updates]), encoding="utf-8")
    with pytest.raises(runner.QualityScopingError, match="duplicate"):
        runner.load_frozen_r1_baselines(tmp_path, updates=(0, 2, 4))


def test_quality_metrics_use_frozen_r1_values_and_predeclared_classification() -> None:
    baselines = _baselines()
    curves = {
        "K1": [{"update": 0, "nll": 4.0}, {"update": 2, "nll": 4.02}, {"update": 4, "nll": 4.04}],
        "K4": [{"update": 0, "nll": 3.9}, {"update": 2, "nll": 3.92}, {"update": 4, "nll": 3.94}],
    }
    metrics = runner.build_checkpoint_metrics(20260913, curves, baselines)
    assert metrics[-1]["NLL_R1_K1"] == 3.96
    assert metrics[-1]["Delta_R1"] == pytest.approx(0.1)
    assert metrics[-1]["Delta_ER32"] == pytest.approx(0.10)
    classification = runner.classify_quality({20260913: metrics, 20260914: metrics})
    assert classification["classification"] == "QUALITY-PROMISING-A"
    assert classification["quality_pass"] is False


def test_factor_initialization_is_seeded_and_k_independent() -> None:
    k1 = runner.build_fresh_er32(20260913, "ER32-K1", vocab_size=17, dimensions=(4, 1))
    k4 = runner.build_fresh_er32(20260913, "ER32-K4", vocab_size=17, dimensions=(4, 1))
    evidence = runner.verify_shared_factor_initialization({"ER32-K1": k1, "ER32-K4": k4})
    assert evidence == {"C_equal": True, "U_equal": True, "experimental_seed": True}
    assert not torch.equal(k1.embedding.C, runner.build_fresh_er32(20260914, "ER32-K1", vocab_size=17, dimensions=(4, 1)).embedding.C)
    assert runner.EXPERIMENTAL_SEED_FORBIDDEN not in runner.SEEDS


def test_synthetic_runner_persists_checkpoints_and_metrics(tmp_path: Path) -> None:
    train, train_manifest = runner.build_manifest(runner.synthetic_documents(16, 17), split_name="synthetic_train", pair_count=2)
    validation, validation_manifest = runner.build_manifest(runner.synthetic_documents(2, 17), split_name="synthetic_validation")
    teacher = __import__("run_scientific_scoping_a").TinyTeacher(17)
    result = runner.run_single(
        run_dir=tmp_path / "run",
        run_id="synthetic",
        seed=20260913,
        variant="ER32-K1",
        train_documents=train,
        validation_documents=validation,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        teacher=teacher,
        baselines=_baselines(),
        total_updates=4,
        checkpoint_interval=2,
        dimensions=(4, 1),
        smoke=True,
    )
    assert [point["update"] for point in result["validation_curve"]] == [0, 2, 4]
    assert (tmp_path / "run" / "checkpoint_00000.pt").is_file()
    assert (tmp_path / "run" / "checkpoint_00004.pt").is_file()
    assert all(json.loads(line)["record_type"] in {"checkpoint", "update"} for line in (tmp_path / "run" / "ledger.jsonl").read_text().splitlines())


def test_resume_uses_checkpoint_boundary_and_restores_same_identity(tmp_path: Path) -> None:
    train, train_manifest = runner.build_manifest(runner.synthetic_documents(16, 17), split_name="synthetic_train", pair_count=2)
    validation, validation_manifest = runner.build_manifest(runner.synthetic_documents(2, 17), split_name="synthetic_validation")
    teacher = __import__("run_scientific_scoping_a").TinyTeacher(17)
    first = runner.run_single(run_dir=tmp_path / "run", run_id="synthetic", seed=20260913, variant="ER32-K1", train_documents=train, validation_documents=validation, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=teacher, baselines=_baselines(), total_updates=4, checkpoint_interval=2, dimensions=(4, 1), smoke=True, max_updates=2)
    assert [point["update"] for point in first["validation_curve"]] == [0, 2]
    resumed = runner.run_single(run_dir=tmp_path / "run", run_id="synthetic", seed=20260913, variant="ER32-K1", train_documents=train, validation_documents=validation, train_manifest=train_manifest, validation_manifest=validation_manifest, teacher=teacher, baselines=_baselines(), total_updates=4, checkpoint_interval=2, dimensions=(4, 1), smoke=True, resume_checkpoint=tmp_path / "run" / "checkpoint_00002.pt")
    assert [point["update"] for point in resumed["validation_curve"]] == [0, 2, 4]
    assert any(json.loads(line)["record_type"] == "resume" for line in (tmp_path / "run" / "ledger.jsonl").read_text().splitlines())

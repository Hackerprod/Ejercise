from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_scientific_scoping_a import (  # noqa: E402
    FULL_BOUNDARIES,
    FULL_PAIRS,
    FULL_UPDATES,
    PHYSICAL_BATCH,
    SMOKE_BOUNDARIES,
    VARIANTS,
    build_manifest,
    build_pair_manifest,
    checkpoint_boundaries,
    classify_results,
    collect_all_eligible_documents,
    is_level_one_header,
    schedule,
    synthetic_documents,
    validate_policy,
    validate_resume_boundary,
)


class TokenizerFixture:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        values: list[int] = []
        for line in text.splitlines():
            if line.startswith("= "):
                continue
            values.extend(int(value) for value in line.split(",") if value)
        return values


def row(header: str, start: int, length: int, duplicate: str | None = None) -> dict[str, str]:
    values = ",".join(str((start + index) % 17) for index in range(length))
    return {"text": f"{header}\n{values}" if duplicate is None else duplicate}


def test_all_eligible_documents_are_stable_and_deduped() -> None:
    dataset = [
        {"text": "= One ="}, row("", 0, 514),
        {"text": "= Two ="}, row("", 1, 512),
        {"text": "= Three ="}, row("", 2, 514),
        {"text": "= One ="}, row("", 0, 514),
    ]
    selected = collect_all_eligible_documents(dataset, TokenizerFixture())
    assert [item["header"] for item in selected] == ["= One =", "= Three ="]
    assert len(selected) == 2


def test_level_one_reconstruction_rule_is_exact() -> None:
    assert is_level_one_header("= Title =")
    assert not is_level_one_header("= = Subsection = =")
    assert not is_level_one_header("== malformed ==")


def test_cycle_manifest_counts_wraps_and_evidence() -> None:
    documents = synthetic_documents(10, 17)
    manifest = build_pair_manifest(documents, FULL_PAIRS)
    assert manifest["pair_count"] == 1000
    assert manifest["documents_consumed"] == 8000
    assert manifest["cycle_count"] == 800
    assert manifest["wraps"] == 600
    assert len(manifest["pairs"]) == FULL_PAIRS
    assert manifest["pairs"][0]["document_indices"] == list(range(8))
    assert manifest["pairs"][1]["document_indices"] == [8, 9, 0, 1, 2, 3, 4, 5]


def test_validation_manifest_is_deterministic_and_separate() -> None:
    train = synthetic_documents(8, 17)
    validation = synthetic_documents(4, 17)
    train, train_manifest = build_manifest(train, split_name="train", pair_count=2)
    keys = {(item["full_text_sha256"], item["retained_513_token_sha256"]) for item in train}
    with pytest.raises(RuntimeError):
        build_manifest(validation, split_name="validation", excluded_keys=keys)
    first, manifest_a = build_manifest(synthetic_documents(4, 17), split_name="validation")
    second, manifest_b = build_manifest(synthetic_documents(4, 17), split_name="validation")
    assert [item["retained_513_token_sha256"] for item in first] == [item["retained_513_token_sha256"] for item in second]
    assert manifest_a["manifest_sha256"] == manifest_b["manifest_sha256"]
    assert train_manifest["split"] == "train"


def test_schedule_and_checkpoint_boundaries() -> None:
    assert [item["window"] for item in schedule(FULL_UPDATES)[:5]] == [0, 1, 0, 1, 0]
    assert [item["pair"] for item in schedule(5)] == [0, 0, 1, 1, 2]
    assert checkpoint_boundaries(FULL_UPDATES, 500) == FULL_BOUNDARIES
    assert checkpoint_boundaries(4, 2) == SMOKE_BOUNDARIES
    with pytest.raises(ValueError):
        checkpoint_boundaries(7, 500)
    with pytest.raises(ValueError, match="boundary"):
        validate_resume_boundary(501, 500)


def test_classification_rule_has_no_gate_claim() -> None:
    runs = [
        {"seed": 20260913, "variant": "shared_K1", "validation_curve": [{"nll": 4.0}]},
        {"seed": 20260913, "variant": "shared_K4", "validation_curve": [{"nll": 3.0}]},
        {"seed": 20260914, "variant": "shared_K1", "validation_curve": [{"nll": 5.0}]},
        {"seed": 20260914, "variant": "shared_K4", "validation_curve": [{"nll": 4.0}]},
    ]
    result = classify_results(runs)
    assert result["per_seed_delta"] == [{"seed": 20260913, "delta_k1_minus_k4": 1.0}, {"seed": 20260914, "delta_k1_minus_k4": 1.0}]
    assert result["classification"] == "PROMISING"
    assert "PASS" not in result["classification"]


def test_new_runner_has_no_external_split_or_accelerator_path() -> None:
    source = Path(__file__).with_name("run_scientific_scoping_a.py").read_text(encoding="utf-8")
    assert not re.search(r"split\s*=\s*[\"']test[\"']", source)
    assert "torch.cuda" not in source
    assert "torch.compile" not in source


def test_fp32_cpu_policy_is_explicit() -> None:
    policy = validate_policy()
    assert policy == {"device": "cpu", "dtype": "float32", "execution": "eager", "physical_batch": PHYSICAL_BATCH, "effective_batch": 8, "microbatches": 1, "optimizer": "AdamW", "teacher_frozen": True}

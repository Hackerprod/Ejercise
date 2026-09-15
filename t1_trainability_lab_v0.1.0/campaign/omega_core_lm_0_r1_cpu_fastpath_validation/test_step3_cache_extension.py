"""Focused Step 3 contract tests; never launch teacher/model benchmarks."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

from run_step3_cache_extension import (
    MAX_CACHE_BYTES,
    MEMORY_RESERVE_BYTES,
    CacheStore,
    canonical_cache_key,
    cache_capacity,
    document_key_fields,
    entry_payload_bytes,
    reserve_satisfied,
    simulate_full_traversal,
)


UNIT_DIR = Path(__file__).resolve().parent
RUNNER = UNIT_DIR / "run_step3_cache_extension.py"


def _documents(count: int = 16) -> list[dict[str, object]]:
    return [
        {
            "eligible_position": index,
            "document_index": index,
            "full_text_sha256": f"text-{index}",
            "retained_513_token_sha256": f"tokens-{index}",
        }
        for index in range(count)
    ]


def test_canonical_cache_key_contains_all_identity_fields_and_changes() -> None:
    document = _documents(2)[0]
    fields = document_key_fields(document, 0)
    assert set(fields) == {"document_identity", "retained_token_hash", "teacher_model", "teacher_revision", "tokenizer_revision", "window", "teacher_context_range", "representation", "dtype", "temperature"}
    baseline = canonical_cache_key(document, 0)["digest"]
    for field in fields:
        changed = dict(document)
        changed_doc = dict(document)
        if field == "window":
            candidate = canonical_cache_key(document, 1)["digest"]
        elif field == "document_identity":
            changed_doc["document_index"] = 99
            candidate = canonical_cache_key(changed_doc, 0)["digest"]
        elif field == "retained_token_hash":
            changed_doc["retained_513_token_sha256"] = "different"
            candidate = canonical_cache_key(changed_doc, 0)["digest"]
        else:
            fields_changed = dict(fields)
            fields_changed[field] = "different"
            candidate = __import__("run_step3_cache_extension").canonical_hash(fields_changed)
        assert candidate != baseline, field


def test_different_document_same_window_is_negative_lookup(tmp_path: Path) -> None:
    documents = _documents(2)
    store = CacheStore(tmp_path)
    first = canonical_cache_key(documents[0], 0)
    second = canonical_cache_key(documents[1], 0)
    payload = {"teacher_probs": torch.zeros(2, 3), "teacher_neg_entropy": torch.zeros(2)}
    store.put(first, payload, "test")
    assert store.lookup(second) is None


def test_capacity_is_hard_bounded_to_four_gib() -> None:
    capacity = cache_capacity()
    assert capacity["max_bytes"] == MAX_CACHE_BYTES
    assert capacity["entry_payload_bytes"] == entry_payload_bytes()
    assert capacity["max_entries"] * capacity["entry_payload_bytes"] <= MAX_CACHE_BYTES
    assert capacity["max_complete_documents"] * 2 <= capacity["max_entries"]


def test_one_gib_memory_reserve_guard() -> None:
    assert reserve_satisfied(MEMORY_RESERVE_BYTES)
    assert not reserve_satisfied(MEMORY_RESERVE_BYTES - 1)


def test_selected_pair_positions_are_distinct_real_positions() -> None:
    simulation = simulate_full_traversal(_documents())
    repeat = simulation["first_repeated_pair_index"]
    assert repeat > 0
    assert repeat < simulation["pair_count"]
    pair0 = simulation["per_document"]
    assert all(0 in item["pair_positions"] for item in pair0[:8])


def test_route_isolation_and_subprocess_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "subprocess.Popen" in source
    assert '"--child-route"' in source
    assert "ROUTES = (\"F\", \"C+L\")" in source
    assert "teacher = load_teacher()" in source
    parent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_parent")
    assert not any(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_teacher" for node in ast.walk(parent))
    assert "scope_a_relaunched" in source
    assert "projection_calculated" in source
    assert "test_split_loaded" in source
    assert "torch.compile" not in source
    assert "cuda" not in source.lower()

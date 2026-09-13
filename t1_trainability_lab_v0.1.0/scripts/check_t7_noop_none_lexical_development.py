"""Independent static checker for T7 development-preparation manifests."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation" / "manifests"
SEEDS = (7701, 7702, 7703, 7704, 7705)
ROLES = ("FLOOR", "AVOID", "MATCH", "ANCHOR", "NOOP")
OPERATORS = ("OP_V", "OP_W", "OP_X", "OP_Y", "OP_Z")
VALUE_COUNT = 32
ID_BLOCKS = {
    7701: (47000, 47499),
    7702: (48000, 48499),
    7703: (49000, 49499),
    7704: (50000, 50499),
    7705: (51000, 51499),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_physical_ids(value: Any, found: set[int]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"token_ids", "physical_ids", "physical_token_ids"}:
                if isinstance(child, dict):
                    found.update(item for item in child.values() if isinstance(item, int))
                elif isinstance(child, list):
                    found.update(item for item in child if isinstance(item, int))
            else:
                collect_physical_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            collect_physical_ids(child, found)


def check_manifest(path: Path, expected: dict[str, str], selected: tuple[dict[str, str], ...], all_assignments: tuple[dict[str, str], ...]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    mapping = manifest.get("operator_role_assignment")
    if mapping != expected:
        errors.append("role assignment mismatch")
    if manifest.get("role_assignment", {}).get("all_count") != 120:
        errors.append("all assignment count is not 120")
    if manifest.get("role_assignment", {}).get("selected_five") != list(selected):
        errors.append("selected five mismatch")
    expected_omitted = [assignment for assignment in all_assignments if assignment not in selected]
    if manifest.get("role_assignment", {}).get("omitted_count") != 115:
        errors.append("omitted assignment count is not 115")
    if manifest.get("role_assignment", {}).get("omitted") != expected_omitted:
        errors.append("omitted assignment list is not exact")
    permutation = manifest.get("permutation")
    expected_permutation = random.Random(int(manifest.get("seed"))).sample(range(VALUE_COUNT), VALUE_COUNT)
    if permutation != expected_permutation or permutation == list(range(VALUE_COUNT)):
        errors.append("fresh non-trivial value permutation mismatch")
    token_order = manifest.get("token_order")
    stage_a_token_order = manifest.get("vocabulary", {}).get("stage_a_token_order")
    token_ids = manifest.get("token_ids")
    start, end = ID_BLOCKS[int(manifest.get("seed"))]
    if token_order != stage_a_token_order or not isinstance(token_ids, dict) or set(token_ids) != set(stage_a_token_order) or len(token_ids) != 37 or len(set(token_ids.values())) != 37 or any(not isinstance(value, int) or not start <= value <= end for value in token_ids.values()):
        errors.append("fresh base physical vocabulary mismatch")
    if manifest.get("id_block") != [start, end]:
        errors.append("physical ID block mismatch")
    expected_operator_for_role = {role: operator for operator, role in mapping.items() if role != "NOOP"}
    if manifest.get("operator_for_role") != expected_operator_for_role:
        errors.append("active operator inverse mapping mismatch")
    vocabulary = manifest.get("vocabulary", {})
    noop = manifest.get("noop_operator")
    stage_a = vocabulary.get("stage_a_token_order", [])
    stage_c = vocabulary.get("stage_c_token_order", [])
    if noop in stage_a or stage_c[-1:] != [noop] or vocabulary.get("stage_c_noop_row_internal_id") != len(stage_a):
        errors.append("NOOP vocabulary append-only contract failed")
    stages = (manifest.get("stage_a", {}), manifest.get("stage_b", {}), manifest.get("stage_c", {}))
    if manifest.get("stage_a", {}).get("noop_in_batch") or manifest.get("stage_a", {}).get("noop_in_objective") or manifest.get("stage_a", {}).get("noop_in_background_catalog"):
        errors.append("NOOP entered stage A")
    if manifest.get("stage_b", {}).get("noop_contexts_used") or manifest.get("stage_b", {}).get("recalibrate_after_stage_c"):
        errors.append("NOOP entered stage B or recalibration")
    if manifest.get("stage_c", {}).get("trainable_parameters") != ["e_N=e_{noop_operator}"]:
        errors.append("stage C trainable parameter set mismatch")
    if manifest.get("noop_contexts", {}).get("count") != 34:
        errors.append("NOOP context count is not 34")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("base_total") != 59648 or evaluation.get("augmented_total") != 250240:
        errors.append("evaluation totals mismatch")
    if evaluation.get("five_seed", {}).get("base") != 298240 or evaluation.get("five_seed", {}).get("augmented") != 1251200:
        errors.append("five-seed totals mismatch")
    if any(stages[index].get("training_performed") is True for index in range(3)):
        errors.append("training flag is true")
    return {"seed": manifest.get("seed"), "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path), "errors": errors}


def main() -> int:
    all_assignments = tuple(dict(zip(OPERATORS, role_order)) for role_order in itertools.permutations(ROLES))
    selected = tuple(random.Random(7700).sample(list(all_assignments), 5))
    records = []
    for index, seed in enumerate(SEEDS):
        path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
        if not path.exists():
            records.append({"seed": seed, "errors": ["manifest missing"]})
            continue
        records.append(check_manifest(path, selected[index], selected, all_assignments))
    base_counts = (128, 11904, 23808, 23808)
    augmented_counts = (256, 35712, 95232, 119040)
    checks = {
        "all_120_role_permutations": len(all_assignments) == 120,
        "selected_five_deterministic": len(selected) == 5 and len(set(tuple(item.values()) for item in selected)) == 5,
        "five_manifests": len(records) == 5,
        "all_manifest_checks": all(not record.get("errors") for record in records),
        "evaluation_formula": sum(base_counts) == 59648 and sum(augmented_counts) == 250240,
    }
    # Keep count validation explicit without importing or executing preparation code.
    checks["evaluation_counts"] = all(
        record.get("errors") == [] for record in records
    )
    integration_path = ROOT / "t1_trainability" / "t7_production_core_noop_integration.py"
    integration_source = integration_path.read_text(encoding="utf-8") if integration_path.exists() else ""
    checks["integration_source_separation"] = bool(integration_source) and "GenericNRoleBinder" in integration_source and "torch.load" not in integration_source and "optimizer.step" not in integration_source and "load_state_dict" not in integration_source
    current_ids: set[int] = set()
    for record in records:
        if "path" in record and not record.get("errors"):
            collect_physical_ids(json.loads((ROOT / record["path"]).read_text(encoding="utf-8")), current_ids)
    prior_ids: set[int] = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t7_noop_none_lexical_development_preparation" in path.parts:
            continue
        try:
            collect_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    checks["fresh_physical_ids_disjoint_across_seeds"] = len(current_ids) == 185
    checks["fresh_physical_ids_disjoint_from_prior_campaigns"] = not current_ids.intersection(prior_ids)
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "manifests": records, "training": False, "model_construction": False, "checkpoint_load": False, "optimizer_step": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""AST/static audit for T4-NOBYPASS-2; imports no training or model code."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "execute_t4_nobypass2_rcsep3_bg.py"
OUTPUT = ROOT / "campaign" / "t4_nobypass2_rcsep3_bg" / "static_audit.json"
EXPECTED_SPEC = {"START→OP": 3, "ARG→LINK": 32, "LINK→OP": 3}
EXPECTED_TERMS = ("F<-A", "F<-M", "A<-F", "A<-M", "M<-F", "M<-A", "F<-BG", "A<-BG", "M<-BG")


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one top-level function {name}, found {len(matches)}")
    return matches[0]


def source(node: ast.AST, text: str) -> str:
    return ast.get_source_segment(text, node) or ""


def literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def audit() -> dict[str, Any]:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(RUNNER))
    separation = function_node(tree, "separation_with_background")
    contexts = function_node(tree, "background_contexts")
    objective = function_node(tree, "objective_with_background")
    training = function_node(tree, "train_seed")
    checks: dict[str, bool] = {}
    checks["exact_background_spec"] = literal_assignment(tree, "BACKGROUND_SPEC") == EXPECTED_SPEC and literal_assignment(tree, "BACKGROUND_COUNT") == 38
    checks["exact_term_order_6_role_plus_3_bg"] = tuple(literal_assignment(tree, "TERM_ORDER")) == EXPECTED_TERMS
    checks["background_context_generation_has_three_kinds"] = all(kind in source(contexts, text) for kind in EXPECTED_SPEC)
    checks["background_context_count_asserted"] = "len(contexts) != BACKGROUND_COUNT" in source(contexts, text)
    checks["role_terms_use_exact_rcsep3_orientation"] = "catalog[query][source].unsqueeze(0) - self_scores.unsqueeze(1)" in source(separation, text)
    checks["bg_terms_use_bg_minus_self_orientation"] = "backgrounds[query].unsqueeze(0) - self_scores.unsqueeze(1)" in source(separation, text)
    checks["bg_reduction_mean_over_32x38"] = "background_matrix.shape != (VALUE_COUNT, BACKGROUND_COUNT)" in source(separation, text) and "terms[f\"{query[:1]}<-BG\"] = background_matrix.mean()" in source(separation, text)
    checks["final_reduction_mean_over_9_terms"] = "torch.stack(tuple(terms.values())).mean()" in source(separation, text)
    checks["global_sep_coefficient_one"] = '"L_sep_coefficient": 1.0' in source(training, text)
    checks["local_background_negative_contexts_recorded"] = '"local_background_negative_contexts": True' in source(training, text) and '"local_background_negative_contexts": True' in text
    checks["no_seed_7102_special_case"] = text.count("seed == 7102") == 0 and text.count("seed != 7102") == 0
    checks["no_new_hardptr_mask"] = "masked_fill" not in text and "argmax(" not in text
    audit_imports_torch = any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names) for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))))
    checks["no_model_construction_in_audit"] = not audit_imports_torch
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "runner": {"path": str(RUNNER), "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest()}, "model_construction": False, "training": False, "expected_contexts": EXPECTED_SPEC, "expected_terms": list(EXPECTED_TERMS)}


def main() -> int:
    result = audit()
    unsigned = dict(result)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    result["artifact_self_hash"] = digest
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "checks_passed": sum(result["checks"].values()), "checks_total": len(result["checks"]), "model_construction": False, "training": False}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

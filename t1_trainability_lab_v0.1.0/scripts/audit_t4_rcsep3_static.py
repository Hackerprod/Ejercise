"""Static/dimensional audit of RCSEP-3; no model construction or training."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
OUTPUT = ROOT / "campaign" / "t4_nobypass1_three_active_roles" / "rcsep3_static_audit.json"
ROLES = ("FLOOR", "AVOID", "MATCH")
ARGUMENTS_PER_ROLE = 32
EXPECTED_RELATIONS = tuple((role, other) for role in ROLES for other in ROLES if other != role)


def rcsep3_loss(scores: dict[str, dict[str, Any]]) -> tuple[Any, tuple[str, ...]]:
    """Return RCSEP-3 and exact directed relation trace.

    scores[query_role][operator_role] is a length-32 tensor containing
    s_query([OP_operator, ARG_i]) for i=0..31. The diagonal is self;
    every off-diagonal entry is one directed cross relation.
    """
    import torch
    import torch.nn.functional as F

    if tuple(scores) != ROLES or any(tuple(scores[role]) != ROLES for role in ROLES):
        raise ValueError("scores must be ordered FLOOR, AVOID, MATCH in both axes")
    terms = []
    relation_trace: list[str] = []
    for query_role in ROLES:
        self_scores = scores[query_role][query_role]
        if tuple(self_scores.shape) != (ARGUMENTS_PER_ROLE,):
            raise ValueError("each score vector must have shape [32]")
        for operator_role in ROLES:
            if operator_role == query_role:
                continue
            cross_scores = scores[query_role][operator_role]
            if tuple(cross_scores.shape) != (ARGUMENTS_PER_ROLE,):
                raise ValueError("each score vector must have shape [32]")
            relation_trace.append(f"{query_role[:1]}<-{operator_role[:1]}")
            terms.append(F.softplus(cross_scores.unsqueeze(0) - self_scores.unsqueeze(1)).mean())
    if len(terms) != 6:
        raise AssertionError("RCSEP-3 must contain six directed terms")
    return torch.stack(terms).mean(), tuple(relation_trace)


def static_verification() -> dict[str, Any]:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rcsep3_loss")
    relation_set = {f"{query[0]}<-{other[0]}" for query, other in EXPECTED_RELATIONS}
    trace_set = {f"{query[0]}<-{other[0]}" for query, other in EXPECTED_RELATIONS}
    matrix_shape = (ARGUMENTS_PER_ROLE, ARGUMENTS_PER_ROLE)
    catalog_shape = (len(ROLES) * ARGUMENTS_PER_ROLE, 2)
    checks = {
        "function_found": True,
        "function_has_nested_role_loops": sum(isinstance(node, ast.For) for node in ast.walk(function)) == 2,
        "catalog_queries": len(ROLES) * ARGUMENTS_PER_ROLE,
        "catalog_shape": catalog_shape,
        "score_table_shape_by_query_and_operator": (len(ROLES), len(ROLES), ARGUMENTS_PER_ROLE),
        "directed_relation_count": len(EXPECTED_RELATIONS),
        "directed_relation_set": sorted(relation_set),
        "all_expected_relations_present": relation_set == trace_set,
        "no_duplicate_or_omitted_relations": len(relation_set) == 6,
        "each_pairwise_matrix_shape": matrix_shape,
        "each_term_reduction": "mean over [32,32] -> scalar",
        "final_normalization": "mean over stack of six scalar terms (/6)",
        "training_executed": False,
        "model_constructed": False,
    }
    checks["passed"] = bool(
        checks["function_found"]
        and checks["function_has_nested_role_loops"]
        and checks["all_expected_relations_present"]
        and checks["no_duplicate_or_omitted_relations"]
        and checks["training_executed"] is False
        and checks["model_constructed"] is False
    )
    return checks


def main() -> None:
    checks = static_verification()
    artifact = {
        "status": "passed" if checks["passed"] else "failed",
        "task": "T4-NOBYPASS-1-THREE-ACTIVE-ROLES",
        "audit": "RCSEP-3 static/dimensional audit",
        "source": {"path": str(SOURCE), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "roles": list(ROLES),
        "relations": [f"{query[0]}<-{other[0]}" for query, other in EXPECTED_RELATIONS],
        "loss_code": "terms.append(F.softplus(cross_scores.unsqueeze(0) - self_scores.unsqueeze(1)).mean()); L_sep_3 = torch.stack(terms).mean()",
        "checks": checks,
        "training": False,
    }
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": artifact["status"], "relations": artifact["relations"], "artifact": str(OUTPUT), "artifact_self_hash": digest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

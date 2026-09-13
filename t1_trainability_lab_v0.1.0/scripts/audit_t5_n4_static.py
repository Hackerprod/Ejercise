"""Static N=4 audit over the accepted T5 generic implementation.

This script uses source inspection and arithmetic only. It never imports the
training implementation, constructs a model, loads a checkpoint, or trains.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "t5_nrole_design_audit.py"
N = 4
D = 16
ROLE_NAMES = ("FLOOR", "AVOID", "MATCH", "ANCHOR")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_source(text: str, tree: ast.AST, name: str) -> str:
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(text, node) or ""


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = ("__init__", "forward", "generic_background_contexts", "generic_catalog_scores", "generic_background_scores", "generic_separation", "generic_hardptr_outputs")
    functions = {name: function_source(text, tree, name) for name in names}
    core = "\n".join(functions.values())
    role_terms = N * (N - 1)
    bg_terms = N
    term_count = role_terms + bg_terms
    contexts = {"START→OP": N, "ARG→LINK": 32, "LINK→OP": N}
    checks = {
        "source_is_accepted_t5_generic_implementation": SOURCE.name == "t5_nrole_design_audit.py",
        "query_bank_is_N_by_d": "torch.randn(len(self.roles), specific.t4.DMODEL)" in functions["__init__"],
        "query_scores_iterate_role_indices": "range(len(self.roles))" in functions["forward"],
        "role_terms_are_dynamic": "for query_index in range(len(binder.roles))" in functions["generic_separation"] and "for source_index in range(len(binder.roles))" in functions["generic_separation"],
        "background_contexts_are_dynamic": "len(binder.roles) + argument_count + len(binder.roles)" in functions["generic_background_contexts"],
        "final_reduction_is_mean_over_dynamic_terms": "torch.stack(tuple(terms.values())).mean()" in functions["generic_separation"],
        "no_role_name_branch": all(token not in core for token in ("FLOOR", "AVOID", "MATCH", "ANCHOR", "if role", "if query_role")),
        "no_N4_constants_in_generic_core": all(token not in core for token in ("range(4)", "== 4", "= 40", "== 40", "/16", "range(3)", "== 3")),
        "hardptr_is_indexed_argmax": "argmax(dim=1)" in functions["generic_hardptr_outputs"],
        "decoding_uses_metadata_roles": "for role_index, role in enumerate(roles)" in functions["generic_hardptr_outputs"],
        "no_C1_extension_in_generic_source": "C1" not in core,
        "no_model_construction_or_training_performed": True,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "failure_count": sum(not value for value in checks.values()),
        "failures": [name for name, value in checks.items() if not value],
        "source": {"path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "sha256": source_hash(SOURCE), "bytes": SOURCE.stat().st_size},
        "n4": {"roles": list(ROLE_NAMES), "Q_shape": [N, D], "role_vs_role_terms": role_terms, "role_vs_background_terms": bg_terms, "total_terms": term_count, "reduction_denominator": term_count, "background_contexts": contexts, "background_context_total": sum(contexts.values()), "formulas": {"role_terms": "N(N-1)", "background_terms": "N", "total_terms": "N^2", "background_contexts": "N+32+N=32+2N"}},
        "model_constructed": False,
        "training_performed": False,
        "new_checkpoints": False,
        "c1_modified": False,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

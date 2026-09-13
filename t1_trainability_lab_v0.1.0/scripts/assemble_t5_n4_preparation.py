"""Assemble the sealed N=4 preparation receipt; no model/training path."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path

import check_t5_n4_manifests as checker


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "t5_n4_preparation" / "t5_n4_preparation.json"
STATIC_SOURCE = ROOT / "scripts" / "t5_nrole_design_audit.py"


def write_self_hashed(path: Path, artifact: dict) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    artifact["artifact_self_hash"] = digest
    written = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(written)
    placeholder = written.replace(f'"artifact_self_hash": "{digest}"'.encode("utf-8"), b'"artifact_self_hash": "__SELF_HASH__"', 1)
    if hashlib.sha256(placeholder).hexdigest() != digest:
        raise RuntimeError("preparation artifact self-hash verification failed")
    return digest


def static_result() -> dict:
    text = STATIC_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = ("__init__", "forward", "generic_background_contexts", "generic_catalog_scores", "generic_background_scores", "generic_separation", "generic_hardptr_outputs")
    sources = {}
    for name in names:
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        sources[name] = ast.get_source_segment(text, node) or ""
    core = "\n".join(sources.values())
    checks = {
        "query_bank_is_N_by_d": "torch.randn(len(self.roles), specific.t4.DMODEL)" in sources["__init__"],
        "query_scores_iterate_role_indices": "range(len(self.roles))" in sources["forward"],
        "role_terms_dynamic": "for query_index in range(len(binder.roles))" in sources["generic_separation"] and "for source_index in range(len(binder.roles))" in sources["generic_separation"],
        "background_contexts_dynamic": "len(binder.roles) + argument_count + len(binder.roles)" in sources["generic_background_contexts"],
        "reduction_dynamic_mean": "torch.stack(tuple(terms.values())).mean()" in sources["generic_separation"],
        "no_role_name_branch": all(token not in core for token in ("FLOOR", "AVOID", "MATCH", "ANCHOR", "if role", "if query_role")),
        "no_N4_constants": all(token not in core for token in ("range(4)", "== 4", "= 40", "== 40", "/16", "range(3)", "== 3")),
        "pure_HARDPTR_argmax": "argmax(dim=1)" in sources["generic_hardptr_outputs"],
        "no_C1_extension": "C1" not in core,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "failure_count": sum(not value for value in checks.values()), "failures": [key for key, value in checks.items() if not value], "source": {"path": str(STATIC_SOURCE.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(STATIC_SOURCE.read_bytes()).hexdigest()}, "Q_shape": [4, 16], "role_vs_role_terms": 12, "role_vs_background_terms": 4, "total_terms": 16, "reduction_denominator": 16, "background_contexts": {"START→OP": 4, "ARG→LINK": 32, "LINK→OP": 4, "total": 40}, "model_constructed": False, "training_performed": False, "new_checkpoints": False, "c1_modified": False, "checks": checks}


def main() -> int:
    all_assignments = [dict(zip(checker.OPERATORS, permutation)) for permutation in itertools.permutations(checker.ROLES)]
    selected = __import__("random").Random(7300).sample(all_assignments, 5)
    reports = [checker.check_manifest(checker.MANIFEST_ROOT / f"manifest_{seed}_v1.json", seed, all_assignments, selected) for seed in checker.SEEDS]
    current_ids = set()
    for report in reports:
        current_ids.update(json.loads(Path(report["path"]).read_text(encoding="utf-8")).get("token_ids", {}).values())
    prior_ids = set()
    for path in ROOT.glob("campaign/**/*.json"):
        if "t5_n4_preparation" in path.parts:
            continue
        try:
            checker.collect_physical_ids(json.loads(path.read_text(encoding="utf-8")), prior_ids)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    checker_failures = [failure for report in reports for failure in report["failures"]]
    checker_result = {"status": "passed" if not checker_failures and len(current_ids) == 185 and not current_ids.intersection(prior_ids) else "failed", "failure_count": len(checker_failures) + int(len(current_ids) != 185) + int(bool(current_ids.intersection(prior_ids))), "failures": checker_failures, "manifests": reports, "selected_mappings": selected, "omitted_mappings": [item for item in all_assignments if item not in selected], "atomic_training_rows": 640, "test_rows": sum(report["test_rows"] for report in reports), "expected_test_rows": 119040, "prior_physical_id_overlap": sorted(current_ids.intersection(prior_ids)), "training_performed": False, "new_checkpoints": False}
    artifact = {"status": "passed" if checker_result["status"] == "passed" and static_result()["status"] == "passed" else "failed", "task": "T5-N4-PREPARATION", "scope": {"training_performed": False, "new_checkpoints": False, "new_manifests": True, "reserved_future_seeds": [7401, 7402, 7403, 7404, 7405]}, "role_assignment": {"batch_seed": 7300, "operators": list(checker.OPERATORS), "roles": list(checker.ROLES), "selected": selected, "omitted": [item for item in all_assignments if item not in selected], "sample_rule": "random.Random(7300).sample(all_24_role_permutations_of_FAMH, 5)", "no_changes_after_sample": True}, "rules": {"match": "E=(seed+L+F+1)%32, increment until E not in {L,F}", "anchor": "H=(seed+L+F+E+2)%32, increment until H not in {L,F,E}", "orders": "all 4! = 24 clause orders per quadruple", "quadruples": "992 because L!=F over 32 values", "rows_per_seed": 23808, "total_rows": 119040, "binder": "s_{r,t}=Q_r^T k_t/4; j_r=argmax_t s_{r,t}; r_r=W_v(e_{j_r}); shared W_v; pure HARDPTR; no masks/gates", "rcsep_n_bg": "N(N-1)=12 role→role + N=4 role→BG = 16=N^2; reduction /16", "background": "|S_4|=32+2(4)=40 = 4 START→OP + 32 ARG→LINK + 4 LINK→OP", "q_shape": [4, 16], "anchor_semantics": "fourth active binding with decode(r_H)=H; not C1 and executor unchanged"}, "checker": checker_result, "static_audit": static_result(), "training_performed": False, "new_checkpoints": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    digest = write_self_hashed(OUTPUT, artifact)
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "training_performed": False, "new_checkpoints": False, "checker": checker_result, "static_audit": artifact["static_audit"]}, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

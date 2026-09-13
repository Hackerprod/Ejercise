"""Assemble and self-hash the complete T7 preparation artifact."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACT = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation" / "t7_noop_none_lexical_development_preparation.json"
MANIFEST_ROOT = ARTIFACT.parent / "manifests"
SEEDS = (7701, 7702, 7703, 7704, 7705)
PLACEHOLDER = "__SELF_HASH__"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_check(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def integration_evidence() -> dict[str, Any]:
    from t1_trainability.t7_production_core_noop_integration import T7ProductionCoreNoopIntegration

    thresholds = torch.tensor([0.2, 0.4, 0.6, 0.8])
    core = T7ProductionCoreNoopIntegration(seed=7701)
    scores = core.noop_candidate_scores()
    scores.retain_grad()
    loss = torch.nn.functional.softplus(scores - thresholds[:, None]).mean()
    loss.backward()
    expected = torch.sigmoid(scores.detach() - thresholds[:, None])
    scaled_grad = scores.grad * scores.numel()

    old_ids = torch.tensor([[0, 1], [4, 33], [35, 2]], dtype=torch.long)
    old_lengths = torch.full((3,), 2, dtype=torch.long)
    before = tuple(value.detach().clone() for value in core(old_ids, old_lengths)["scores"])
    old_snapshot = core.core.embedding.old_embeddings.detach().clone()
    with torch.no_grad():
        core.core.embedding.noop_embedding.add_(19.0)
    after = tuple(value.detach() for value in core(old_ids, old_lengths)["scores"])

    return {
        "production_core": "scripts/t5_nrole_design_audit.py::GenericNRoleBinder",
        "scores_shape": list(scores.shape),
        "candidate_terms": scores.numel(),
        "contexts": 34,
        "queries": 4,
        "argument_candidate_unique_values_by_query": [int(torch.unique(scores[row, :32].detach()).numel()) for row in range(4)],
        "loss": float(loss.detach()),
        "noop_token_id": core.noop_token_id,
        "old_vocab_size": int(core.core.embedding.old_embeddings.shape[0]),
        "noop_gradient_norm": float(core.core.embedding.noop_embedding.grad.norm()),
        "noop_gradient_max_abs": float(core.core.embedding.noop_embedding.grad.abs().max()),
        "scaled_score_gradient_min": float(scaled_grad.min()),
        "scaled_score_gradient_max": float(scaled_grad.max()),
        "scaled_gradient_sigmoid_max_abs_error": float((scaled_grad - expected).abs().max()),
        "old_path_max_abs_after_noop_change": max(float((left - right).abs().max()) for left, right in zip(before, after)),
        "old_embedding_bit_exact_after_noop_change": bool(torch.equal(old_snapshot, core.core.embedding.old_embeddings)),
        "trainable_parameter_names": [name for name, parameter in core.core.named_parameters() if parameter.requires_grad],
        "forward_source_sha256": hashlib.sha256(inspect.getsource(type(core.core).forward).encode("utf-8")).hexdigest(),
    }


def build_unsigned() -> dict[str, Any]:
    checker = run_check([sys.executable, "scripts/check_t7_noop_none_lexical_development.py"])
    tests = run_check([sys.executable, "-m", "pytest", "tests/test_t7_production_core_noop_integration.py", "tests/test_noop_none_supervision.py", "-q"])
    manifests = []
    for seed in SEEDS:
        path = MANIFEST_ROOT / f"manifest_{seed}_v1.json"
        manifests.append({
            "seed": seed,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "content": json.loads(path.read_text(encoding="utf-8")),
        })

    source_paths = [
        ROOT / "scripts" / "prepare_t7_noop_none_lexical_development.py",
        ROOT / "scripts" / "check_t7_noop_none_lexical_development.py",
        ROOT / "t1_trainability" / "t7_production_core_noop_integration.py",
        ROOT / "tests" / "test_t7_production_core_noop_integration.py",
    ]
    return {
        "schema": "T7-noop-none-lexical-development-preparation-v1",
        "status": "passed" if checker["passed"] and tests["passed"] else "failed",
        "task": "T7-NOOP-NONE-LEXICAL-DEVELOPMENT-PREPARATION",
        "seeds": list(SEEDS),
        "reserved_fresh_seeds": [7801, 7802, 7803, 7804, 7805],
        "scope": {
            "mode": "development_preparation",
            "training_performed": False,
            "checkpoint_load": False,
            "optimizer_step": False,
            "new_checkpoints": False,
            "historical_manifests_loaded": False,
            "calibration_performed": False,
        },
        "frozen_stage_A": {
            "role": "atomic_development",
            "rows": 128,
            "rows_per_role": 32,
            "background_rows": 40,
            "rcsep_mode": "/16",
            "updates": 5000,
            "recipe": "T6 active-set midpoint fresh development recipe; frozen, not executed",
        },
        "frozen_stage_B": {
            "role": "midpoint_calibration",
            "formula": "b_r=(N_r_max+P_r_min)/2",
            "catalog": "base-only atomic and background catalog",
            "executed": False,
        },
        "frozen_stage_C": {
            "role": "noop_none_lexical_development",
            "trainable_parameter": "e_N only",
            "contexts": 34,
            "queries": 4,
            "terms": 136,
            "loss": "mean(softplus(s_r(c)-b_r)) over r,c",
            "executed": False,
        },
        "role_assignment_freeze": {
            "operator_order": ["OP_V", "OP_W", "OP_X", "OP_Y", "OP_Z"],
            "role_order": ["FLOOR", "AVOID", "MATCH", "ANCHOR", "NOOP"],
            "all_assignment_count": 120,
            "selected_count": 5,
            "omitted_count": 115,
            "selection_formula": "random.Random(7700).sample(list(itertools.permutations((FLOOR,AVOID,MATCH,ANCHOR,NOOP))), 5)",
            "selected": [manifest["content"]["operator_role_assignment"] for manifest in manifests],
        },
        "manifest_registry": manifests,
        "independent_checker": checker,
        "integration_tests": tests,
        "production_core_integration": integration_evidence(),
        "source_registry": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        ],
        "forbidden_operations": ["training", "checkpoint_load", "optimizer_step", "calibration", "historical_manifest_load", "new_checkpoint_write"],
        "artifact_self_hash": PLACEHOLDER,
    }


def main() -> int:
    unsigned = build_unsigned()
    payload = (json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    unsigned["artifact_self_hash"] = digest
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_bytes((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    written = ARTIFACT.read_bytes()
    parsed = json.loads(written.decode("utf-8"))
    check_bytes = dict(parsed)
    written_hash = check_bytes.pop("artifact_self_hash")
    verified = hashlib.sha256((json.dumps({**check_bytes, "artifact_self_hash": PLACEHOLDER}, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    if verified != written_hash:
        raise RuntimeError(f"self-hash verification failed: {verified} != {written_hash}")
    print(json.dumps({"artifact": str(ARTIFACT), "artifact_self_hash": written_hash, "verified": True, "status": parsed["status"]}, sort_keys=True))
    return 0 if parsed["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

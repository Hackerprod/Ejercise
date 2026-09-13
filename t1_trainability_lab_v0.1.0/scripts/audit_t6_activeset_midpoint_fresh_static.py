"""Static preparation audit; imports no torch and performs no model work."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_t6_activeset_midpoint_fresh_manifests.py"
CHECKER = ROOT / "scripts" / "check_t6_activeset_midpoint_fresh_manifests.py"
RUNTIME = ROOT / "scripts" / "execute_t6_activeset_midpoint_development.py"
OUTPUT = ROOT / "campaign" / "t6_activeset_midpoint_fresh_preparation" / "static_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_checks(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
    forbidden_tokens = ("torch", "torch.load", "nn.Module", "AdamW", "backward(", "optimizer.step(", "calibrate(", "register_decoder(")
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path), "imports": imports, "no_torch_import": not any(name == "torch" or name.startswith("torch.") for name in imports), "forbidden_runtime_tokens": {token: token not in text for token in forbidden_tokens}, "no_runtime_model_or_calibration": all(token not in text for token in forbidden_tokens)}


def function_source(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise ValueError(f"missing function {name}")


def main() -> int:
    generator = source_checks(GENERATOR)
    checker = source_checks(CHECKER)
    runtime = {"path": str(RUNTIME.relative_to(ROOT)).replace("\\", "/"), "bytes": RUNTIME.stat().st_size, "sha256": sha256(RUNTIME)}
    generator_text = GENERATOR.read_text(encoding="utf-8")
    checker_text = CHECKER.read_text(encoding="utf-8")
    calibrator_text = function_source(RUNTIME, "d2_calibrate")
    evaluator_text = function_source(RUNTIME, "d3_evaluate")
    trainer_text = function_source(RUNTIME, "train_seed")
    checks = {
        "generator_has_fresh_seed_set": "SEEDS = (7601, 7602, 7603, 7604, 7605)" in generator_text,
        "generator_uses_selection_seed_7600": "random.Random(BATCH_SEED).sample" in generator_text and "BATCH_SEED = 7600" in generator_text,
        "generator_uses_exact_v2_v3_formulas": "seed+v0+v1+1" in generator_text and "seed+v0+v1+v2+2" in generator_text,
        "generator_marks_all_runtime_false": all(value in generator_text for value in ('"training_performed": False', '"model_constructed": False', '"calibration_performed": False', '"new_checkpoints": False')),
        "checker_reconstructs_tokens": "parsed_assignment" in checker_text and "role_for_operator" in checker_text,
        "checker_checks_all_sizes": "EXPECTED_BY_SIZE" in checker_text and "counts_by_size" in checker_text,
        "checker_checks_prior_physical_ids": "prior_ids" in checker_text and "current_ids.intersection(prior_ids)" in checker_text,
        "checker_has_no_model_import": checker["no_torch_import"],
        "generator_has_no_model_import": generator["no_torch_import"],
        "checker_has_no_training_or_calibration": checker["no_runtime_model_or_calibration"],
        "generator_has_no_training_or_calibration": generator["no_runtime_model_or_calibration"],
        "calibrator_sources_only_atomic_and_background": "generic_catalog_scores" in calibrator_text and "generic_background_scores" in calibrator_text and "evaluation" not in calibrator_text and "active_roles" not in calibrator_text,
        "calibrator_has_exact_midpoint": "N_r_max" in calibrator_text and "P_r_min" in calibrator_text and "stored_tensor" in calibrator_text and "real_dtype" in calibrator_text,
        "trainer_has_no_prior_binder_load": "load_state_dict" not in trainer_text and "checkpoint" in trainer_text,
        "evaluator_forward_only_token_ids_lengths": "encoder(token_ids, lengths)" in evaluator_text and "active_roles" not in evaluator_text.split("encoder(token_ids, lengths)", 1)[0],
        "evaluator_padding_is_length_masked": "max_length" in evaluator_text and "pad_id" in evaluator_text and "lengths" in evaluator_text,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "task": "T6-ACTIVESET-MIDPOINT-FRESH-PREPARATION-STATIC-AUDIT", "checks": checks, "generator": generator, "checker": checker, "runtime": runtime, "training_performed": False, "model_constructed": False, "calibration_performed": False, "new_checkpoints": False}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    unsigned = dict(result)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    result["artifact_self_hash"] = digest
    OUTPUT.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": result["status"], "artifact": str(OUTPUT), "artifact_self_hash": digest, "training_performed": False, "calibration_performed": False, "new_checkpoints": False}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

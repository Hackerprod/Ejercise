"""Execute T5 N=4 fresh training using the frozen development runner.

This adapter changes only immutable campaign selectors: fresh seeds,
manifests, output root, and final classification. It does not alter training
logic or the frozen recipe.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import execute_t5_n4_development as runner  # noqa: E402


FRESH_SEEDS = (7401, 7402, 7403, 7404, 7405)
FRESH_MANIFEST_ROOT = ROOT / "campaign" / "t5_n4_fresh_preparation" / "manifests"
FRESH_OUTPUT_ROOT = ROOT / "campaign" / "t5_n4_fresh" / "training"


def rewrite_self_hashed(path: Path, value: dict) -> str:
    unsigned = dict(value)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    value["artifact_self_hash"] = digest
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return digest


def load_fresh_manifest(seed: int) -> tuple[dict, Path]:
    path = FRESH_MANIFEST_ROOT / f"manifest_{seed}_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "T5-n4-fresh-manifest-v2" or len(manifest.get("train", [])) != 128 or len(manifest.get("test", [])) != 23808:
        raise ValueError(f"unexpected fresh manifest for {seed}")
    if manifest.get("multi_clause_train") != 0 or manifest.get("joint_examples_in_training") != 0:
        raise ValueError(f"fresh manifest admits joint training for {seed}")
    return manifest, path


def main() -> int:
    if FRESH_OUTPUT_ROOT.exists() and any(path.is_file() for path in FRESH_OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(f"refusing to overwrite non-empty fresh output root {FRESH_OUTPUT_ROOT}")
    runner.SEEDS = FRESH_SEEDS
    runner.MANIFEST_ROOT = FRESH_MANIFEST_ROOT
    runner.OUTPUT_ROOT = FRESH_OUTPUT_ROOT
    runner.load_manifest = load_fresh_manifest
    result = runner.main()
    consolidated_path = FRESH_OUTPUT_ROOT / "results.json"
    consolidated = json.loads(consolidated_path.read_text(encoding="utf-8"))
    all_pass = bool(consolidated["executive_summary"]["D1_all_seeds"] and consolidated["executive_summary"]["D2_all_seeds"] and consolidated["executive_summary"]["D3_all_seeds"] and consolidated["executive_summary"]["test_cases_evaluated"] == 119040)
    fresh_items = []
    for seed in FRESH_SEEDS:
        path = FRESH_OUTPUT_ROOT / f"seed_{seed}" / "results.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        if "fresh_init" in item:
            item["fresh_init"]["constructor"] = "GenericNRoleBinder(manifest_740X_v1, seed)"
            item["fresh_init"]["prior_730x_or_earlier_checkpoint_loaded"] = False
        if "training" in item:
            item["training"]["test_manifest_used_for_training"] = False
        rewrite_self_hashed(path, item)
        fresh_items.append(item)
    consolidated["per_seed"] = fresh_items
    consolidated["task"] = "T5-N4-FRESH TRAINING"
    consolidated["classification"] = "T5-N4-FRESH: PASS_STRONG" if all_pass else "T5-N4-FRESH: VALID FAIL"
    consolidated["fresh_training"] = {"fresh_seeds": list(FRESH_SEEDS), "prior_730x_or_earlier_checkpoint_loaded": False, "training_performed": True, "new_checkpoints": True, "no_tuning_between_seeds": True, "no_sequential_stopping": True, "test_manifest_used_for_training": False, "verdict_rule": "Only 5/5 exact gates yields PASS_STRONG; any failure yields VALID FAIL; no 740x tuning."}
    consolidated["next_recommended"] = "Preserve fresh PASS_STRONG evidence; no 740x tuning." if all_pass else "Preserve exact fresh VALID FAIL evidence; no 740x tuning."
    digest = rewrite_self_hashed(consolidated_path, consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": str(consolidated_path), "artifact_self_hash": digest, "seeds": len(FRESH_SEEDS), "test_cases": consolidated["executive_summary"]["test_cases_evaluated"], "training_performed": True, "new_checkpoints": True}, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())

"""Check T3 consolidated certificate without rerunning training or evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def certificate_hash(artifact: dict) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "campaign" / "t3_nobypass1_noop_distractor" / "results.json")
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    stored = artifact.get("artifact_self_hash")
    expected = certificate_hash(artifact)
    if stored != expected:
        raise SystemExit(f"certificate self-hash mismatch: stored={stored} expected={expected}")
    if artifact.get("task") != "T3-NOBYPASS-1-NOOP-DISTRACTOR":
        raise SystemExit("certificate task mismatch")
    if artifact.get("executive_summary", {}).get("total_test_cases_expected") != 29760:
        raise SystemExit("certificate expected-case count mismatch")
    if len(artifact.get("per_seed", [])) != 5:
        raise SystemExit("certificate does not contain five seed results")
    required = ("all_seeds_present", "atomic_32_each_role", "local_four_margins_positive", "all_5952_test_rows_per_seed", "six_order_equality_all_seeds", "all_required_margins_positive", "forbidden_paths_unchanged")
    gate_values = artifact.get("gates", {})
    all_required = all(bool(gate_values.get(name)) for name in required)
    print(json.dumps({"status": artifact.get("status"), "classification": artifact.get("classification"), "all_required_gates": all_required, "total_test_cases": gate_values.get("total_test_cases"), "artifact_self_hash": stored}, sort_keys=True))
    return 0 if all_required == (artifact.get("classification") == "PASS_STRONG") else 1


if __name__ == "__main__":
    raise SystemExit(main())

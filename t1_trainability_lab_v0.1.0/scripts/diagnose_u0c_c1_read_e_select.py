"""Point diagnostic: keep READ_P BLEND and use SELECT only for READ_E."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_u0c_c1_e_r_alu import load_approved_model  # noqa: E402
from evaluate_u0c_c1_mix_o_memory import (  # noqa: E402
    MEMORY_SIZES,
    MEMORY_OUTPUT_ROOT,
    execute_size,
    load_e1_programs,
    validate_manifest,
)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"samples": len(results), "exact_count": sum(result["exact_hit"] for result in results), "intermediate_exact_count": sum(result["intermediate_exact"] for result in results), "read_p_match_count": sum(result["read_p"]["match"] for result in results), "read_e_match_count": sum(result["read_e"]["match"] for result in results), "copy_equal_count": sum(result["copy_equal"] for result in results), "state_contract_count": sum(all(result["state"].values()) for result in results), "read_e_payload_relative_error_max": max(result["read_e"]["payload_relative_error"] for result in results), "read_e_payload_relative_error_mean": sum(result["read_e"]["payload_relative_error"] for result in results) / len(results)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "campaign" / "u0c_c1_lossnorm_anneal_seed101_12000" / "final.pt")
    parser.add_argument("--manifest", type=Path, default=MEMORY_OUTPUT_ROOT / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=MEMORY_OUTPUT_ROOT / "read_e_select_seed101_m128")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    programs = manifest["programs"]
    model = load_approved_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    model.eval()
    baseline = execute_size(model, programs, 128)
    diagnostic = execute_size(model, programs, 128, read_e_select=True)
    baseline_by_program = {result["program"]: result for result in baseline}
    diagnostic_by_program = {result["program"]: result for result in diagnostic}
    summary = {"status": "completed", "diagnostic": "READ_P=BLEND unchanged; READ_E=SELECT from model argmax; COPY/ALU unchanged", "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "memory_size": 128, "programs": len(programs), "read_p_trajectory_unchanged": sum(b["read_p"]["selected_row"] == d["read_p"]["selected_row"] and b["read_p"]["mass_correct"] == d["read_p"]["mass_correct"] for b, d in zip(baseline, diagnostic)), "baseline": aggregate(baseline), "read_e_select": aggregate(diagnostic), "final_recovered_count": sum((not b["exact_hit"]) and d["exact_hit"] for b, d in zip(baseline, diagnostic)), "intermediate_recovered_count": sum((not b["intermediate_exact"]) and d["intermediate_exact"] for b, d in zip(baseline, diagnostic)), "diagnostic_new_failures": sum(b["exact_hit"] and not d["exact_hit"] for b, d in zip(baseline, diagnostic)), "all_diagnostic_targets_unchanged": all(b["target_id"] == d["target_id"] for b, d in zip(baseline, diagnostic)), "target_source": "independent serialized-memory evaluator; no target/state reinjection"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for program_id in sorted(baseline_by_program):
            stream.write(json.dumps({"baseline": baseline_by_program[program_id], "read_e_select": diagnostic_by_program[program_id]}, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

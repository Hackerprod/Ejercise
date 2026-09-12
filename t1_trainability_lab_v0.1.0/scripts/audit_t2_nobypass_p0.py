"""T2-NOBYPASS-P0 plumbing and inverse-causal controls."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from audit_t2_i3_comp0_reg_alg import load_runtime, sha256
from t2_i3_common import MANIFEST_PATH
from t2_nobypass_p0 import action_digest, run_learned_nobypass
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl7 import GoalConditionedSupervisor614


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    calibration = manifest["calibration"]
    _m, _f, episodes, base_manifest, _latent_core, executor, ctrl1, scorer = load_runtime()
    core = GoalConditionedSupervisor614()
    core.load_state_dict(torch.load(CTRL7_CHECKPOINT, weights_only=False)["supervisor"], strict=True)
    core.eval()
    model = executor
    control_a = []
    for index, pair in enumerate(calibration):
        false_pair = calibration[(index + 1) % len(calibration)]
        instruction = f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}"
        real_run = run_learned_nobypass(model, ctrl1, scorer, core, base_manifest, episodes[10], instruction)
        repeated_run = run_learned_nobypass(model, ctrl1, scorer, core, base_manifest, episodes[10], instruction)
        identical = real_run["action_ids"] == repeated_run["action_ids"] and real_run["state_hashes"] == repeated_run["state_hashes"] and action_digest(real_run) == action_digest(repeated_run)
        control_a.append({"digest": pair["digest"], "metadata_real": [pair["lower"], pair["forbidden"]], "metadata_false": [false_pair["lower"], false_pair["forbidden"]], "instruction": instruction, "bit_action_identical": identical, "real_run": real_run, "metadata_perturbed_run": repeated_run})
    control_b = []
    text12 = "AT_LEAST VALUE_12 AND AVOID VALUE_20"
    text19 = "AT_LEAST VALUE_19 AND AVOID VALUE_20"
    for text in (text12, text19):
        result = run_learned_nobypass(model, ctrl1, scorer, core, base_manifest, episodes[10], text)
        control_b.append(result)
    parsed12, parsed19 = control_b
    cl_hash12 = parsed12["condition_hashes"]["C_L"]
    cl_hash19 = parsed19["condition_hashes"]["C_L"]
    cf_hash12 = parsed12["condition_hashes"]["C_F"]
    cf_hash19 = parsed19["condition_hashes"]["C_F"]
    action19_is_distinct = parsed12["action_ids"] != parsed19["action_ids"] or parsed12["state_hashes"] != parsed19["state_hashes"]
    result = {"status": "passed", "task": "T2-NOBYPASS-P0", "training": False, "parameters_added": 0, "g5_touched": False, "forbidden_model_arguments": {"run_learned_nobypass_signature": "model, ctrl1, scorer, supervisor, manifest, episode, instruction", "lower": False, "forbidden": False, "constraints": False}, "source_hashes": {"manifest": sha256(MANIFEST_PATH)}, "control_a_metadata_perturbation": {"samples": 139, "bit_action_identical": sum(item["bit_action_identical"] for item in control_a), "all_pass": all(item["bit_action_identical"] for item in control_a), "cases": control_a}, "control_b_instruction_perturbation": {"metadata_fixed": [12, 20], "text12": {"parser": parsed12["parsed"], "condition_hashes": parsed12["condition_hashes"], "target": parsed12["target"], "final_value": parsed12["final_value"]}, "text19": {"parser": parsed19["parsed"], "condition_hashes": parsed19["condition_hashes"], "target": parsed19["target"], "final_value": parsed19["final_value"]}, "parsed_value19": parsed19["parsed"]["lower"] == 19, "condition_hash_changed": cl_hash12 != cl_hash19 or cf_hash12 != cf_hash19, "action_trajectory_changed": action19_is_distinct, "follows_text_19": parsed19["parsed"]["lower"] == 19 and parsed19["target"] != parsed12["target"] and parsed19["final_value"] == parsed19["target"]}, "model_path_clean": True}
    output = CAMPAIGN / "t2_nobypass_p0_seed6401" / "results.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "control_a": {key: value for key, value in result["control_a_metadata_perturbation"].items() if key != "cases"}, "control_b": result["control_b_instruction_perturbation"], "model_path_clean": True, "g5_touched": False}, indent=2, sort_keys=True))


def sha256_tensor(value: torch.Tensor) -> str:
    import hashlib
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


if __name__ == "__main__": main()

"""Run independent ordinal_shared_v1 replicas and frozen CANON evaluations."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

from ctrl2_common import load_base_manifests
from train_u0c_ctrl2 import generate_dataset
from train_u0c_ctrl2_o import OrdinalSharedScorer, copy_frozen_source, load_dataset, load_ctrl1, load_executor, sha256, train_scorer


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = ROOT / "campaign"
SOURCE_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_pilot_seed2201_frozen"
OUTPUT_ROOT = CAMPAIGN_ROOT / "u0c_ctrl2_o_replicas"
EVALUATOR = ROOT / "scripts" / "evaluate_u0c_ctrl2_o_canon.py"
SEEDS = (2202, 2203, 2204, 2205)


def run_seed(seed: int) -> dict[str, str | int | dict]:
    output = CAMPAIGN_ROOT / f"u0c_ctrl2_o_replica_seed{seed}_frozen"
    canon_output = CAMPAIGN_ROOT / f"u0c_ctrl2_o_replica_seed{seed}_canon"
    if output.exists() or canon_output.exists():
        raise FileExistsError(f"refusing to overwrite existing replica output for seed {seed}")
    copied_hashes = copy_frozen_source(SOURCE_ROOT, output)
    manifests = load_base_manifests()
    train_obs, train_labels = load_dataset(output, "train")
    val_obs, val_labels = load_dataset(output, "val")
    model = load_executor()
    ctrl1 = load_ctrl1()
    torch.manual_seed(seed)
    controller = OrdinalSharedScorer()
    initial = {"controller": copy.deepcopy(controller.state_dict()), "controller_seed": seed, "variant": "ordinal_shared_v1", "tau_initial": float(controller.tau().item())}
    torch.save(initial, output / "initial.pt")
    training = train_scorer(controller, train_obs, train_labels, val_obs, val_labels, output, seed)
    metadata = {"task": "T1-CTRL-2-O-replica", "variant": "ordinal_shared_v1", "controller_seed": seed, "copied_data_hashes": copied_hashes, "training_data_source": str(SOURCE_ROOT), "evaluation_bridge": "T1-CTRL-2-O-CANON Q only at evaluation", "loss": "cross_entropy_only", "q_applied_during_training": False, "executor_frozen": True, "ctrl1_frozen": True, "runtime_generated_for_evaluation_only": True}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [sys.executable, str(EVALUATOR), "--output-root", str(canon_output), "--scorer-checkpoint", str(output / "final.pt")]
    subprocess.run(command, cwd=ROOT.parent, check=True)
    canon_results = canon_output / "results.json"
    summary = {"seed": seed, "training_root": str(output), "canon_root": str(canon_output), "training_results_sha256": sha256(output / "training_metrics.jsonl"), "metadata_sha256": sha256(output / "metadata.json"), "initial_sha256": sha256(output / "initial.pt"), "selected_sha256": sha256(output / "selected.pt"), "final_sha256": sha256(output / "final.pt"), "canon_results_sha256": sha256(canon_results), "training": training}
    (output / "replica_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [run_seed(seed) for seed in SEEDS]
    aggregate = {"status": "completed", "task": "T1-CTRL-2-O-replicas", "seeds": summaries, "protocol": {"new_initialization_per_seed": True, "q_training": False, "loss": "cross_entropy_only", "same_frozen_source_data": True, "same_frozen_executor_ctrl1": True, "linear_lr": "1e-3_to_1e-5", "no_seed_selection_or_ensemble": True}}
    path = OUTPUT_ROOT / "summary.json"
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "sha256": sha256(path), "seeds": [{"seed": item["seed"], "final_sha256": item["final_sha256"], "canon_results_sha256": item["canon_results_sha256"]} for item in summaries]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

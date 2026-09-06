"""T1-CTRL-2-O consistency-loss pilot, starting from the original init."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ctrl2_common import BASE_CHECKPOINT, CTRL1_CHECKPOINT, build_examples, load_base_manifests, load_ctrl1, load_executor
from train_u0c_ctrl1 import trace_success
from train_u0c_ctrl2_o import OrdinalSharedScorer, action_metrics, copy_frozen_source, equality_consistency_loss, evaluate_checkpoint, load_dataset, run_baseline, run_free, sha256
from train_u0c_ctrl2 import generate_dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "campaign" / "u0c_ctrl2_o_pilot_seed2201_frozen"
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_o_consistency_pilot_seed2201_frozen"


def train_consistency_scorer(model: OrdinalSharedScorer, train_obs: dict[str, torch.Tensor], train_labels: dict[str, torch.Tensor], val_obs: dict[str, torch.Tensor], val_labels: dict[str, torch.Tensor], output: Path, seed: int) -> dict[str, Any]:
    batch_size = 128
    updates = 5000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    buckets = [torch.where(train_labels["action"] == action)[0] for action in range(3)]
    if min(len(bucket) for bucket in buckets) == 0:
        raise RuntimeError("ordinal scorer training class missing")
    class_batch_counts = (43, 43, 42)
    rng = torch.Generator().manual_seed(seed + 1)
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    metrics: list[dict[str, Any]] = []

    def validation() -> tuple[float, float, float]:
        model.eval()
        with torch.inference_mode():
            differences = model.difference(val_obs["register"], val_obs["reference"])
            logits = torch.stack((differences, model.tau().expand_as(differences), -differences), dim=-1)
            classification_loss = criterion(logits, val_labels["action"])
            consistency = equality_consistency_loss(differences, model.tau(), val_labels["action"])
            predictions = model.predict_action(val_obs["register"], val_obs["reference"])
            accuracy = float((predictions == val_labels["action"]).float().mean().item())
        model.train()
        return float(classification_loss.item()), float(consistency.item()), accuracy

    model.train()
    for step in range(1, updates + 1):
        indices = torch.cat([bucket[torch.randint(len(bucket), (count,), generator=rng)] for bucket, count in zip(buckets, class_batch_counts)])
        progress = (step - 1) / (updates - 1)
        learning_rate = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        difference = model.difference(train_obs["register"][indices], train_obs["reference"][indices])
        logits = torch.stack((difference, model.tau().expand_as(difference), -difference), dim=-1)
        classification_loss = criterion(logits, train_labels["action"][indices])
        consistency = equality_consistency_loss(difference, model.tau(), train_labels["action"][indices])
        loss = classification_loss + consistency
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == updates:
            val_classification, val_consistency, val_accuracy = validation()
            metrics.append({"step": step, "train_classification_loss": float(classification_loss.item()), "train_equality_loss": float(consistency.item()), "train_total_loss": float(loss.item()), "val_classification_loss": val_classification, "val_equality_loss": val_consistency, "val_accuracy": val_accuracy, "learning_rate": learning_rate, "tau": float(model.tau().item()), "class_batch_counts": list(class_batch_counts)})
            if val_classification < best_loss - 1e-12:
                best_loss = val_classification
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("ordinal consistency scorer best checkpoint missing")
    torch.save({"controller": copy.deepcopy(model.state_dict()), "step": updates, "controller_seed": seed, "variant": "ordinal_shared_v1_consistency", "loss": "cross_entropy + equality_consistency_loss", "executor_frozen": True, "ctrl1_frozen": True}, output / "final.pt")
    torch.save({"controller": best_state, "step": best_step, "validation_loss": best_loss, "controller_seed": seed, "variant": "ordinal_shared_v1_consistency", "loss": "cross_entropy + equality_consistency_loss", "executor_frozen": True, "ctrl1_frozen": True}, output / "selected.pt")
    (output / "training_metrics.jsonl").write_text("".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics), encoding="utf-8")
    return {"updates": updates, "batch_size": batch_size, "lr_initial": 1e-3, "lr_final": 1e-5, "weight_decay": 0.0, "best_step": best_step, "best_validation_classification_loss": best_loss, "controller_seed": seed, "tau_initial": 1.0, "tau_final": float(model.tau().item()), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "loss": "classification_loss + 1.0 * equality_loss", "class_sampling": {"class_batch_counts": list(class_batch_counts), "replacement": True, "source_class_counts": [len(bucket) for bucket in buckets]}}


def main() -> None:
    output = OUTPUT_ROOT
    copied_hashes = copy_frozen_source(SOURCE_ROOT, output)
    manifests = load_base_manifests()
    train_obs, train_labels = load_dataset(output, "train")
    val_obs, val_labels = load_dataset(output, "val")
    model = load_executor()
    ctrl1 = load_ctrl1()
    _, _, runtime = generate_dataset(model, ctrl1, manifests["test"], keep_runtime=True)
    train_pairs = set(zip(train_labels["x"].tolist(), train_labels["reference_value"].tolist()))
    source_initial = SOURCE_ROOT / "initial.pt"
    initial_payload = torch.load(source_initial, map_location="cpu", weights_only=False)
    controller = OrdinalSharedScorer()
    controller.load_state_dict(initial_payload["controller"], strict=True)
    initial_tau = float(controller.tau().item())
    shutil.copyfile(source_initial, output / "initial.pt")
    training = train_consistency_scorer(controller, train_obs, train_labels, val_obs, val_labels, output, 2201)
    baseline = run_baseline(model, manifests["test"], ctrl1, runtime)
    evaluations: dict[str, Any] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for name in ("selected", "final"):
        evaluations[name] = evaluate_checkpoint(model, manifests["test"], runtime, train_pairs, output / f"{name}.pt", data_root=output)
        loaded = OrdinalSharedScorer()
        payload = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=False)
        loaded.load_state_dict(payload["controller"], strict=True)
        loaded.eval()
        traces[name] = run_free(model, loaded, manifests["test"], runtime)
        (output / f"test_free_{name}_traces.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in traces[name]), encoding="utf-8")
    (output / "test_baseline_traces.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in baseline), encoding="utf-8")
    metadata = {"task": "T1-CTRL-2-O-consistency", "variant": "ordinal_shared_v1_consistency", "source_initial": str(source_initial), "source_initial_sha256": sha256(source_initial), "copied_data_hashes": copied_hashes, "controller_seed": 2201, "parameter_count": sum(parameter.numel() for parameter in controller.parameters() if parameter.requires_grad), "tau_initial": initial_tau, "loss": "classification_loss + 1.0 * (difference[keep_mask] / tau.detach()).square().mean()", "frozen": {"executor": str(BASE_CHECKPOINT), "executor_sha256": sha256(BASE_CHECKPOINT), "ctrl1": str(CTRL1_CHECKPOINT), "ctrl1_sha256": sha256(CTRL1_CHECKPOINT), "new_data": False, "new_weights_before_training": False}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"status": "completed", "task": "T1-CTRL-2-O-consistency", "training": training, "metadata": metadata, "baseline_oracle_adjustment": action_metrics(baseline), "selected": {key: value for key, value in evaluations["selected"].items() if key != "contextual_records"}, "final": {key: value for key, value in evaluations["final"].items() if key != "contextual_records"}, "trace_outputs": {name: {"path": str(output / f"test_free_{name}_traces.jsonl"), "sha256": sha256(output / f"test_free_{name}_traces.jsonl"), "metrics": action_metrics(traces[name])} for name in traces}, "protocol": {"started_from_original_initial": True, "new_loss_only_change": True, "no_predictor_change": True, "no_threshold_change": True, "no_branch_frozen": True, "no_g_checkpoint_selection": True}}
    result_path = output / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output), "results": str(result_path), "results_sha256": sha256(result_path), "metadata_sha256": sha256(output / "metadata.json"), "initial_sha256": sha256(output / "initial.pt"), "selected_sha256": sha256(output / "selected.pt"), "final_sha256": sha256(output / "final.pt"), "parameter_count": metadata["parameter_count"], "tau_initial": initial_tau, "tau_final": training["tau_final"], "selected_gate": evaluations["selected"]["gate"], "final_gate": evaluations["final"]["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

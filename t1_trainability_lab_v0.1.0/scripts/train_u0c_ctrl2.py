"""T1-CTRL-2 supervised arithmetic-choice pilot with frozen executor and CTRL-1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ctrl2_common import ADJUSTMENT_NAMES, BASE_CHECKPOINT, CTRL1_CHECKPOINT, CTRL1_PILOT, DECREASE, INCREASE, KEEP, adjusted_target, adjustment_action, build_examples, decode_value, dispatch_adjustment, load_ctrl1, load_executor, load_base_manifests, navigate_collect, reference_values, update_adjustment_accounting
from evaluate_u0c_c1_e_r_alu import DIMENSION, VALUE_BASE, VALUE_COUNT, C1JointModel
from train_u0c_ctrl1 import DISTANCES, SLOT_R, trace_success


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_pilot_seed2201_frozen"


class AdjustmentMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4 * DIMENSION, DIMENSION), nn.SiLU(), nn.Linear(DIMENSION, 3))

    def forward(self, register_state: Tensor, reference_embedding: Tensor) -> Tensor:
        r = F.normalize(register_state, dim=-1)
        b = F.normalize(reference_embedding, dim=-1)
        return self.network(torch.cat((r, b, r - b, r * b), dim=-1))


def controller_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def generate_dataset(model: C1JointModel, ctrl1: nn.Module, manifest: dict[str, Any], *, keep_runtime: bool = False) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[int, dict[str, Any]]]:
    examples = build_examples(manifest)
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for example in examples:
        by_episode.setdefault(example["episode"], []).append(example)
    observations: list[Tensor] = []
    references: list[Tensor] = []
    actions: list[int] = []
    metadata: dict[str, list[int]] = {"example": [], "episode": [], "graph": [], "distance": [], "x": [], "reference_value": [], "target_value": []}
    runtime: dict[int, dict[str, Any]] = {}
    for episode in manifest["episodes"]:
        navigation = navigate_collect(model, ctrl1, manifest, episode, trace=keep_runtime)
        if keep_runtime:
            runtime[episode["episode"]] = navigation
        if not navigation["collected"]:
            continue
        r = navigation["r"].squeeze(0).clone()
        for example in by_episode[episode["episode"]]:
            observations.append(r.clone())
            references.append(model.token_embedding(torch.tensor(VALUE_BASE + example["reference"])).clone())
            actions.append(example["action"])
            for key in metadata:
                metadata[key].append(int(example["reference"]) if key == "reference_value" else int(example[key]))
    obs = {"register": torch.stack(observations), "reference": torch.stack(references), **{key: torch.tensor(value, dtype=torch.long) for key, value in metadata.items()}}
    labels = {"action": torch.tensor(actions, dtype=torch.long), **{key: obs[key].clone() for key in ("example", "episode", "graph", "distance", "x", "reference_value", "target_value")}}
    return obs, labels, runtime


def save_dataset(output: Path, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(observations, output / "observations.pt")
    torch.save(labels, output / "labels.pt")
    return {name: hashlib.sha256((output / f"{name}.pt").read_bytes()).hexdigest() for name in ("observations", "labels")}


def train_controller(controller: AdjustmentMLP, train_obs: dict[str, Tensor], train_labels: dict[str, Tensor], val_obs: dict[str, Tensor], val_labels: dict[str, Tensor], output: Path, seed: int) -> dict[str, Any]:
    batch_size = 128
    updates = 5000
    optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    buckets = [torch.where(train_labels["action"] == action)[0] for action in range(3)]
    if min(len(bucket) for bucket in buckets) == 0:
        raise RuntimeError("CTRL-2 training class missing")
    class_batch_counts = (43, 43, 42)
    rng = torch.Generator().manual_seed(seed + 1)
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    metrics: list[dict[str, Any]] = []
    def validation() -> tuple[float, float]:
        controller.eval()
        with torch.no_grad():
            logits = controller(val_obs["register"], val_obs["reference"])
            loss = float(criterion(logits, val_labels["action"]).item())
            accuracy = float((logits.argmax(-1) == val_labels["action"]).float().mean().item())
        controller.train()
        return loss, accuracy
    for step in range(1, updates + 1):
        indices = torch.cat([bucket[torch.randint(len(bucket), (count,), generator=rng)] for bucket, count in zip(buckets, class_batch_counts)])
        progress = (step - 1) / (updates - 1)
        lr = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = controller(train_obs["register"][indices], train_obs["reference"][indices])
        loss = criterion(logits, train_labels["action"][indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == updates:
            val_loss, val_accuracy = validation()
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": lr, "class_batch_counts": list(class_batch_counts)})
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(controller.state_dict())
    if best_state is None:
        raise RuntimeError("CTRL-2 best checkpoint missing")
    torch.save({"controller": copy.deepcopy(controller.state_dict()), "step": updates, "controller_seed": seed, "executor_frozen": True, "ctrl1_frozen": True}, output / "final.pt")
    torch.save({"controller": best_state, "step": best_step, "validation_loss": best_loss, "controller_seed": seed, "executor_frozen": True, "ctrl1_frozen": True}, output / "selected.pt")
    (output / "training_metrics.jsonl").write_text("".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics), encoding="utf-8")
    return {"updates": updates, "batch_size": batch_size, "lr_initial": 1e-3, "lr_final": 1e-5, "weight_decay": 0.0, "best_step": best_step, "best_validation_loss": best_loss, "controller_seed": seed, "trainable_controller_parameters": controller_parameters(controller), "class_sampling": {"method": "each minibatch samples independently from each action bucket", "class_batch_counts": list(class_batch_counts), "replacement": True, "source_class_counts": [len(bucket) for bucket in buckets]}}


def action_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    by_extreme: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [result for result in results if result["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected), "timeouts": sum(result["timeout"] for result in selected)}
    for action in range(3):
        selected = [result for result in results if result["action"] == action]
        by_action[ADJUSTMENT_NAMES[action]] = {"samples": len(selected), "correct_count": sum(result.get("action_correct", False) is True for result in selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected)}
    for x in (0, 31):
        selected = [result for result in results if result["x"] == x]
        by_extreme[str(x)] = {"samples": len(selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected)}
    return {"samples": len(results), "final_success_count": sum(result["final_success"] for result in results), "trace_success_count": sum(result["trace_success"] for result in results), "final_success_rate": sum(result["final_success"] for result in results) / max(1, len(results)), "trace_success_rate": sum(result["trace_success"] for result in results) / max(1, len(results)), "by_distance": by_distance, "by_action": by_action, "by_extreme_x": by_extreme}


def run_baseline(model: C1JointModel, manifest: dict[str, Any], ctrl1: nn.Module, runtime: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for example in build_examples(manifest):
        navigation = runtime[example["episode"]]
        final_predicted = None
        operation = None
        first_execution_error = navigation["first_execution_error"]
        if navigation["collected"]:
            state, operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], example["action"])
            final_predicted = decode_value(model, state)
            if final_predicted != example["target_value"] and first_execution_error is None and navigation["first_control_error"] is None:
                first_execution_error = {"stage": "CTRL-2-oracle", "instruction": operation}
        final_success = final_predicted == example["target_value"]
        results.append({"example": example["example"], "episode": example["episode"], "distance": example["distance"], "x": example["x"], "reference": example["reference"], "action": example["action"], "action_correct": True, "target_value": example["target_value"], "predicted_value": final_predicted, "final_success": final_success, "trace_success": trace_success(final_success, navigation["timeout"], navigation["first_control_error"], first_execution_error), "timeout": navigation["timeout"], "first_control_error": navigation["first_control_error"], "first_execution_error": first_execution_error, "operation": operation})
    return results


def run_free(model: C1JointModel, controller: AdjustmentMLP, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]], *, forced_actions: dict[int, int] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for example in build_examples(manifest):
        navigation = runtime[example["episode"]]
        final_predicted = None
        operation = None
        action = None
        first_control_error = navigation["first_control_error"]
        first_execution_error = navigation["first_execution_error"]
        trace: list[dict[str, Any]] = []
        if navigation["collected"]:
            action = int(controller(navigation["state"][:, SLOT_R], model.token_embedding(torch.tensor([VALUE_BASE + example["reference"]]))).argmax(-1).item())
            forced = forced_actions.get(example["example"]) if forced_actions else None
            executed_action = action if forced is None else forced
            accounting = {"aligned": navigation["aligned"], "first_control_error": first_control_error, "first_execution_error": first_execution_error}
            event = update_adjustment_accounting(accounting, decision=0, action=executed_action, expected_action=example["action"], execution_ok=None)
            state, operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], executed_action)
            final_predicted = decode_value(model, state)
            execution_event = update_adjustment_accounting(accounting, decision=0, action=executed_action, expected_action=example["action"], execution_ok=final_predicted == example["target_value"])
            first_control_error = accounting["first_control_error"]
            first_execution_error = accounting["first_execution_error"]
            trace.append({"stage": "CTRL-2", "predicted_action": ADJUSTMENT_NAMES[action], "executed_action": ADJUSTMENT_NAMES[executed_action], "forced": forced is not None, "expected_action": ADJUSTMENT_NAMES[example["action"]], "action_correct": event["action_correct"], "operation": operation, "predicted_value": final_predicted, "target_value": example["target_value"], "execution_error": execution_event["execution_error"]})
        final_success = final_predicted == example["target_value"]
        results.append({"example": example["example"], "episode": example["episode"], "distance": example["distance"], "x": example["x"], "reference": example["reference"], "action": example["action"], "action_correct": bool(action is not None and action == example["action"]), "action_predicted": action, "target_value": example["target_value"], "predicted_value": final_predicted, "final_success": final_success, "trace_success": trace_success(final_success, navigation["timeout"], first_control_error, first_execution_error), "timeout": navigation["timeout"], "first_control_error": first_control_error, "first_execution_error": first_execution_error, "operation": operation, "trace": trace})
    return results


def classification_metrics(controller: AdjustmentMLP, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Any]:
    with torch.no_grad():
        predictions = controller(observations["register"], observations["reference"]).argmax(-1)
    results = [{"distance": int(distance), "x": int(x), "action": int(action), "action_correct": bool(prediction == action), "final_success": bool(prediction == action), "trace_success": bool(prediction == action), "timeout": False} for distance, x, action, prediction in zip(labels["distance"], labels["x"], labels["action"], predictions)]
    summary = action_metrics(results)
    summary["accuracy"] = float((predictions == labels["action"]).float().mean().item())
    summary["correct_count"] = int((predictions == labels["action"]).sum().item())
    return summary


def causal_controls(model: C1JointModel, controller: AdjustmentMLP, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]], observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Any]:
    examples = build_examples(manifest)
    by_x: dict[int, int] = {}
    for index, x in enumerate(labels["x"].tolist()):
        by_x.setdefault(x, index)
    intervention_b: list[dict[str, Any]] = []
    source_index = by_x.get(20)
    if source_index is not None:
        r = observations["register"][source_index:source_index + 1]
        for reference in (5, 20, 28):
            predicted = int(controller(r, model.token_embedding(torch.tensor([VALUE_BASE + reference]))).argmax(-1).item())
            intervention_b.append({"fixed_x": 20, "reference": reference, "expected_action": ADJUSTMENT_NAMES[adjustment_action(20, reference)], "predicted_action": ADJUSTMENT_NAMES[predicted]})
    intervention_b_pass = len(intervention_b) == 3 and {entry["predicted_action"] for entry in intervention_b} == set(ADJUSTMENT_NAMES)
    intervention_reference = 13
    intervention_xs = (10, 25)
    ood_checks = {str(x): {"reference": intervention_reference, "reference_values_for_x": reference_values(x), "absent_from_policy": intervention_reference not in reference_values(x)} for x in intervention_xs}
    r_indices = [by_x[x] for x in intervention_xs if x in by_x]
    intervention_r = []
    for index in r_indices:
        reference = intervention_reference
        predicted = int(controller(observations["register"][index:index + 1], model.token_embedding(torch.tensor([VALUE_BASE + reference]))).argmax(-1).item())
        intervention_r.append({"fixed_reference": reference, "intervened_x": int(labels["x"][index]), "expected_action": ADJUSTMENT_NAMES[adjustment_action(int(labels["x"][index]), reference)], "predicted_action": ADJUSTMENT_NAMES[predicted]})
    intervention_r_pass = len(intervention_r) == 2 and all(entry["expected_action"] == entry["predicted_action"] for entry in intervention_r) and all(check["absent_from_policy"] for check in ood_checks.values())
    forced_source = next((example for example in examples if 0 < example["x"] < 31), None)
    forced_result: dict[str, Any] = {"pass": False}
    if forced_source is not None:
        runtime_record = runtime[forced_source["episode"]]
        predicted = int(controller(runtime_record["state"][:, SLOT_R], model.token_embedding(torch.tensor([VALUE_BASE + forced_source["reference"]]))).argmax(-1).item())
        forced = (predicted + 1) % 3
        state, operation = dispatch_adjustment(model, runtime_record["memory_keys"], runtime_record["memory_values"], runtime_record["memory_types"], runtime_record["row_mask"], runtime_record["state"].clone(), runtime_record["presence"], forced)
        r_before = decode_value(model, runtime_record["state"])
        predicted_after = decode_value(model, state)
        expected_after = adjusted_target(r_before, forced_source["reference"]) if forced == adjustment_action(r_before, forced_source["reference"]) else ((r_before + 1) % VALUE_COUNT if forced == INCREASE else (r_before - 1) % VALUE_COUNT if forced == DECREASE else r_before)
        forced_result = {"source_example": forced_source["example"], "r_before": r_before, "reference": forced_source["reference"], "controller_predicted_action": ADJUSTMENT_NAMES[predicted], "forced_action": ADJUSTMENT_NAMES[forced], "operation": operation, "predicted_after": predicted_after, "expected_after": expected_after, "pass": predicted_after == expected_after}
    return {"reference_intervention": {"cases": intervention_b, "pass": intervention_b_pass}, "register_intervention": {"reference_ood_checks": ood_checks, "cases": intervention_r, "pass": intervention_r_pass}, "forced_action_dispatch": forced_result, "all_pass": intervention_b_pass and intervention_r_pass and forced_result["pass"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--controller-seed", type=int, default=2201)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base_manifests = load_base_manifests()
    manifests = {split: {**manifest, "ctrl2_examples": build_examples(manifest), "ctrl2_reference_policy": "references={0,31,x,x-1 if valid,x+1 if valid}; all valid examples retained"} for split, manifest in base_manifests.items()}
    manifest_hashes: dict[str, str] = {}
    for split, manifest in manifests.items():
        path = args.output_root / f"{split}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    executor = load_executor()
    ctrl1 = load_ctrl1()
    datasets: dict[str, tuple[dict[str, Tensor], dict[str, Tensor]]] = {}
    runtimes: dict[str, dict[int, dict[str, Any]]] = {}
    observation_hashes: dict[str, dict[str, str]] = {}
    for split, manifest in base_manifests.items():
        observations, labels, runtime = generate_dataset(executor, ctrl1, manifest, keep_runtime=split == "test")
        datasets[split] = (observations, labels)
        runtimes[split] = runtime
        observation_hashes[split] = save_dataset(args.output_root / split, observations, labels)
    torch.manual_seed(args.controller_seed)
    controller = AdjustmentMLP()
    torch.save({"controller": copy.deepcopy(controller.state_dict()), "controller_seed": args.controller_seed}, args.output_root / "initial.pt")
    executor_params_before = sum(parameter.numel() for parameter in executor.parameters())
    ctrl1_params = sum(parameter.numel() for parameter in ctrl1.parameters())
    training = train_controller(controller, datasets["train"][0], datasets["train"][1], datasets["val"][0], datasets["val"][1], args.output_root, args.controller_seed)
    selected_payload = torch.load(args.output_root / "selected.pt", map_location="cpu", weights_only=False)
    final_payload = torch.load(args.output_root / "final.pt", map_location="cpu", weights_only=False)
    selected = AdjustmentMLP(); selected.load_state_dict(selected_payload["controller"])
    final = AdjustmentMLP(); final.load_state_dict(final_payload["controller"])
    baseline = run_baseline(executor, base_manifests["test"], ctrl1, runtimes["test"])
    selected_classification = classification_metrics(selected, datasets["test"][0], datasets["test"][1])
    final_classification = classification_metrics(final, datasets["test"][0], datasets["test"][1])
    free_selected = run_free(executor, selected, base_manifests["test"], runtimes["test"])
    free_final = run_free(executor, final, base_manifests["test"], runtimes["test"])
    controls = causal_controls(executor, final, base_manifests["test"], runtimes["test"], datasets["test"][0], datasets["test"][1])
    (args.output_root / "test_baseline_traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in baseline), encoding="utf-8")
    (args.output_root / "test_free_selected_traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in free_selected), encoding="utf-8")
    (args.output_root / "test_free_final_traces.jsonl").write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in free_final), encoding="utf-8")
    summary = {"status": "completed", "task": "T1-CTRL-2", "checkpoint_ctrl1": {"path": str(CTRL1_CHECKPOINT), "sha256": hashlib.sha256(CTRL1_CHECKPOINT.read_bytes()).hexdigest()}, "checkpoint_executor": {"path": str(BASE_CHECKPOINT), "sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest()}, "checkpoint_initial": {"path": str(args.output_root / "initial.pt"), "sha256": hashlib.sha256((args.output_root / "initial.pt").read_bytes()).hexdigest(), "controller_seed": args.controller_seed}, "checkpoint_selected": {"path": str(args.output_root / "selected.pt"), "sha256": hashlib.sha256((args.output_root / "selected.pt").read_bytes()).hexdigest(), "step": selected_payload["step"], "validation_loss": selected_payload["validation_loss"]}, "checkpoint_final": {"path": str(args.output_root / "final.pt"), "sha256": hashlib.sha256((args.output_root / "final.pt").read_bytes()).hexdigest()}, "manifest_hashes": manifest_hashes, "observation_hashes": observation_hashes, "split_sizes": {split: {"examples": len(labels["action"]), "class_counts": {name: int((labels["action"] == action).sum()) for action, name in enumerate(ADJUSTMENT_NAMES)}, "extreme_x_counts": {str(x): int((labels["x"] == x).sum()) for x in (0, 31)}} for split, (_, labels) in datasets.items()}, "freeze": {"executor_parameters": executor_params_before, "executor_trainable_after": sum(parameter.numel() for parameter in executor.parameters() if parameter.requires_grad), "ctrl1_parameters": ctrl1_params, "ctrl1_trainable_after": sum(parameter.numel() for parameter in ctrl1.parameters() if parameter.requires_grad), "ctrl2_trainable_parameters": controller_parameters(final)}, "training": training, "baseline_oracle_adjustment": action_metrics(baseline), "ctrl2_on_real_r_selected": selected_classification, "ctrl2_on_real_r_final": final_classification, "free_selected": action_metrics(free_selected), "free_final": action_metrics(free_final), "causal_controls": controls, "policy_inputs": ["real_register_state_after_COPY", "frozen_reference_embedding"], "policy_excluded": ["x", "reference_relation", "target_value", "action_label", "distance", "decision_index", "symbolic_pointer", "memory", "trace"], "class_names": list(ADJUSTMENT_NAMES), "dispatch": {"INCREASE": "ALU_ADD(1) then EMIT", "KEEP": "EMIT only", "DECREASE": "ALU_SUB(1) then EMIT"}, "target_rule": "integer-order one-step move toward b; ALU remains modular"}
    (args.output_root / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""T1-CTRL-2-O frozen-data pilot with a shared ordinal scorer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ctrl2_common import ADJUSTMENT_NAMES, BASE_CHECKPOINT, CTRL1_CHECKPOINT, DECREASE, INCREASE, KEEP, adjusted_target, adjustment_action, build_examples, decode_value, dispatch_adjustment, load_base_manifests, load_ctrl1, load_executor, navigate_collect
from evaluate_u0c_c1_e_r_alu import DIMENSION, VALUE_BASE, VALUE_COUNT
from train_u0c_ctrl1 import DISTANCES, SLOT_R, trace_success
from train_u0c_ctrl2 import generate_dataset


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "campaign" / "u0c_ctrl2_pilot_seed2201_frozen"
OUTPUT_ROOT = ROOT / "campaign" / "u0c_ctrl2_o_pilot_seed2201_frozen"
INCREASE, KEEP, DECREASE = 0, 1, 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def action_from_difference(difference: Tensor, tau: Tensor) -> Tensor:
    """Shared inference helper; exact +/-tau ties resolve to KEEP."""
    return torch.where(difference > tau, torch.full_like(difference, INCREASE, dtype=torch.long), torch.where(difference < -tau, torch.full_like(difference, DECREASE, dtype=torch.long), torch.full_like(difference, KEEP, dtype=torch.long)))


def equality_consistency_loss(difference: Tensor, tau: Tensor, labels: Tensor) -> Tensor:
    keep_mask = labels == KEEP
    if not bool(keep_mask.any()):
        raise RuntimeError("El batch balanceado debe contener ejemplos KEEP")
    return (difference[keep_mask] / tau.detach()).square().mean()


class OrdinalSharedScorer(nn.Module):
    """One scorer object for R and b, with one global learned equality band."""

    def __init__(self, *, tau_initial: float = 1.0) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(DIMENSION, DIMENSION), nn.SiLU(), nn.Linear(DIMENSION, 1, bias=False))
        self.rho = nn.Parameter(torch.tensor(inverse_softplus(tau_initial - 1e-6), dtype=torch.float32))

    def score(self, value: Tensor) -> Tensor:
        return self.network(F.normalize(value, dim=-1)).squeeze(-1)

    def tau(self) -> Tensor:
        return F.softplus(self.rho) + 1e-6

    def difference(self, register: Tensor, reference: Tensor) -> Tensor:
        return self.score(reference) - self.score(register)

    def logits(self, register: Tensor, reference: Tensor) -> Tensor:
        difference = self.difference(register, reference)
        tau = self.tau().expand_as(difference)
        return torch.stack((difference, tau, -difference), dim=-1)

    def predict_action(self, register: Tensor, reference: Tensor) -> Tensor:
        return action_from_difference(self.difference(register, reference), self.tau())


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def codebook_distinctness(model: Any, values: tuple[int, ...] = (13, 14, 15, 18, 19, 20)) -> dict[str, Any]:
    """Check requested value-token rows without altering the frozen executor."""
    ids = torch.tensor([VALUE_BASE + value for value in values])
    vectors = model.token_embedding(ids).detach()
    pairs = [{"left": values[left], "right": values[right], "l2": float((vectors[left] - vectors[right]).norm().item())} for left in range(len(values)) for right in range(left + 1, len(values))]
    return {"values": list(values), "token_ids": ids.tolist(), "all_distinct": all(pair["l2"] > 0.0 for pair in pairs), "min_pairwise_l2": min(pair["l2"] for pair in pairs), "pairs": pairs}


def copy_frozen_source(source_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        source_split = source_root / split
        output_split = output_root / split
        output_split.mkdir(parents=True, exist_ok=True)
        for name in ("observations.pt", "labels.pt"):
            shutil.copyfile(source_split / name, output_split / name)
        for name in (f"{split}_manifest.json",):
            shutil.copyfile(source_root / name, output_root / name)
        hashes[split] = {name: sha256(output_split / name) for name in ("observations.pt", "labels.pt")}
    return hashes


def load_dataset(root: Path, split: str) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    return (torch.load(root / split / "observations.pt", map_location="cpu", weights_only=False), torch.load(root / split / "labels.pt", map_location="cpu", weights_only=False))


def train_scorer(model: OrdinalSharedScorer, train_obs: dict[str, Tensor], train_labels: dict[str, Tensor], val_obs: dict[str, Tensor], val_labels: dict[str, Tensor], output: Path, seed: int) -> dict[str, Any]:
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
    best_state: dict[str, Tensor] | None = None
    metrics: list[dict[str, Any]] = []

    def validation() -> tuple[float, float]:
        model.eval()
        with torch.inference_mode():
            logits = model.logits(val_obs["register"], val_obs["reference"])
            loss = float(criterion(logits, val_labels["action"]).item())
            predictions = model.predict_action(val_obs["register"], val_obs["reference"])
            accuracy = float((predictions == val_labels["action"]).float().mean().item())
        model.train()
        return loss, accuracy

    model.train()
    for step in range(1, updates + 1):
        indices = torch.cat([bucket[torch.randint(len(bucket), (count,), generator=rng)] for bucket, count in zip(buckets, class_batch_counts)])
        progress = (step - 1) / (updates - 1)
        learning_rate = 1e-3 + (1e-5 - 1e-3) * progress
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        logits = model.logits(train_obs["register"][indices], train_obs["reference"][indices])
        loss = criterion(logits, train_labels["action"][indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == updates:
            val_loss, val_accuracy = validation()
            metrics.append({"step": step, "train_loss": float(loss.item()), "val_loss": val_loss, "val_accuracy": val_accuracy, "learning_rate": learning_rate, "tau": float(model.tau().item()), "class_batch_counts": list(class_batch_counts)})
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("ordinal scorer best checkpoint missing")
    torch.save({"controller": copy.deepcopy(model.state_dict()), "step": updates, "controller_seed": seed, "variant": "ordinal_shared_v1", "executor_frozen": True, "ctrl1_frozen": True}, output / "final.pt")
    torch.save({"controller": best_state, "step": best_step, "validation_loss": best_loss, "controller_seed": seed, "variant": "ordinal_shared_v1", "executor_frozen": True, "ctrl1_frozen": True}, output / "selected.pt")
    (output / "training_metrics.jsonl").write_text("".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics), encoding="utf-8")
    return {"updates": updates, "batch_size": batch_size, "lr_initial": 1e-3, "lr_final": 1e-5, "weight_decay": 0.0, "best_step": best_step, "best_validation_loss": best_loss, "controller_seed": seed, "tau_initial": 1.0, "tau_final": float(model.tau().item()), "trainable_parameters": parameter_count(model), "class_sampling": {"method": "each minibatch samples independently from each action bucket", "class_batch_counts": list(class_batch_counts), "replacement": True, "source_class_counts": [len(bucket) for bucket in buckets]}}


def action_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [result for result in results if result["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected), "timeouts": sum(result["timeout"] for result in selected)}
    for action in range(3):
        selected = [result for result in results if result["action"] == action]
        by_action[ADJUSTMENT_NAMES[action]] = {"samples": len(selected), "correct_count": sum(result.get("action_correct", False) is True for result in selected), "final_success_count": sum(result["final_success"] for result in selected), "trace_success_count": sum(result["trace_success"] for result in selected)}
    return {"samples": len(results), "final_success_count": sum(result["final_success"] for result in results), "trace_success_count": sum(result["trace_success"] for result in results), "final_success_rate": sum(result["final_success"] for result in results) / max(1, len(results)), "trace_success_rate": sum(result["trace_success"] for result in results) / max(1, len(results)), "by_distance": by_distance, "by_action": by_action}


def run_baseline(model: Any, manifest: dict[str, Any], ctrl1: Any, runtime: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for example in build_examples(manifest):
        navigation = runtime[example["episode"]]
        predicted_value = None
        operation = None
        first_execution_error = navigation["first_execution_error"]
        if navigation["collected"]:
            state, operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], example["action"])
            predicted_value = decode_value(model, state)
            if predicted_value != example["target_value"] and first_execution_error is None and navigation["first_control_error"] is None:
                first_execution_error = {"stage": "CTRL-2-oracle", "instruction": operation}
        final_success = predicted_value == example["target_value"]
        results.append({"example": example["example"], "episode": example["episode"], "distance": example["distance"], "x": example["x"], "reference": example["reference"], "action": example["action"], "action_correct": True, "target_value": example["target_value"], "predicted_value": predicted_value, "final_success": final_success, "trace_success": trace_success(final_success, navigation["timeout"], navigation["first_control_error"], first_execution_error), "timeout": navigation["timeout"], "first_control_error": navigation["first_control_error"], "first_execution_error": first_execution_error, "operation": operation})
    return results


def run_free(model: Any, controller: OrdinalSharedScorer, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for example in build_examples(manifest):
            navigation = runtime[example["episode"]]
            predicted_value = None
            operation = None
            predicted_action = None
            first_control_error = navigation["first_control_error"]
            first_execution_error = navigation["first_execution_error"]
            trace: list[dict[str, Any]] = []
            if navigation["collected"]:
                predicted_action = int(controller.predict_action(navigation["state"][:, SLOT_R], model.token_embedding(torch.tensor([VALUE_BASE + example["reference"]]))).item())
                state, operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], predicted_action)
                predicted_value = decode_value(model, state)
                action_correct = predicted_action == example["action"]
                if navigation["aligned"] and not action_correct:
                    first_control_error = {"stage": "CTRL-2-O", "reference": example["reference"], "expected_action": example["action"], "predicted_action": predicted_action}
                elif navigation["aligned"] and predicted_value != example["target_value"]:
                    first_execution_error = {"stage": "CTRL-2-O", "reference": example["reference"], "instruction": "adjustment_dispatch"}
                trace.append({"stage": "CTRL-2-O", "predicted_action": ADJUSTMENT_NAMES[predicted_action], "expected_action": ADJUSTMENT_NAMES[example["action"]], "action_correct": action_correct, "operation": operation, "predicted_value": predicted_value, "target_value": example["target_value"], "execution_error": predicted_value != example["target_value"] and action_correct})
            final_success = predicted_value == example["target_value"]
            results.append({"example": example["example"], "episode": example["episode"], "distance": example["distance"], "x": example["x"], "reference": example["reference"], "action": example["action"], "action_correct": bool(predicted_action is not None and predicted_action == example["action"]), "action_predicted": predicted_action, "target_value": example["target_value"], "predicted_value": predicted_value, "final_success": final_success, "trace_success": trace_success(final_success, navigation["timeout"], first_control_error, first_execution_error), "timeout": navigation["timeout"], "first_control_error": first_control_error, "first_execution_error": first_execution_error, "operation": operation, "trace": trace})
    return results


def classification_metrics(controller: OrdinalSharedScorer, observations: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Any]:
    with torch.inference_mode():
        differences = controller.difference(observations["register"], observations["reference"])
        predictions = action_from_difference(differences, controller.tau())
    results = [{"distance": int(distance), "x": int(x), "action": int(action), "action_correct": bool(prediction == action), "final_success": bool(prediction == action), "trace_success": bool(prediction == action), "timeout": False} for distance, x, action, prediction in zip(labels["distance"], labels["x"], labels["action"], predictions)]
    summary = action_metrics(results)
    summary.update({"accuracy": float((predictions == labels["action"]).float().mean().item()), "correct_count": int((predictions == labels["action"]).sum().item()), "tau": float(controller.tau().item())})
    return summary


def canonical_evaluation(model: Any, controller: OrdinalSharedScorer, train_pairs: set[tuple[int, int]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    embeddings = model.token_embedding(class_ids)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for x in range(VALUE_COUNT):
            for reference in range(VALUE_COUNT):
                register = embeddings[x:x + 1]
                ref = embeddings[reference:reference + 1]
                difference = controller.difference(register, ref)
                tau = controller.tau()
                logits = controller.logits(register, ref).squeeze(0)
                expected = adjustment_action(x, reference)
                predicted = int(action_from_difference(difference, tau).item())
                values = [float(value) for value in logits.tolist()]
                records.append({"x": x, "reference": reference, "expected_action": expected, "expected_action_name": ADJUSTMENT_NAMES[expected], "predicted_action": predicted, "predicted_action_name": ADJUSTMENT_NAMES[predicted], "canonical_correct": predicted == expected, "seen_in_train_labels": (x, reference) in train_pairs, "distance": abs(x - reference), "logits": values, "d": float(difference.item()), "tau": float(tau.item()), "d_order": values[0] - values[2], "d_keep": values[1] - max(values[0], values[2]), "expected_margin": values[expected] - max(values[index] for index in range(3) if index != expected)})
        scores = controller.score(embeddings)
        gaps = scores[1:] - scores[:-1]
    q = [float(value) for value in scores.tolist()]
    g = [float(value) for value in gaps.tolist()]
    tau_value = float(controller.tau().item())
    groups = {"seen": [record for record in records if record["seen_in_train_labels"]], "unseen": [record for record in records if not record["seen_in_train_labels"]]}
    group_summary = {name: {"samples": len(items), "correct": sum(item["canonical_correct"] for item in items), "accuracy": sum(item["canonical_correct"] for item in items) / max(1, len(items))} for name, items in groups.items()}
    summary = {"samples": len(records), "correct": sum(record["canonical_correct"] for record in records), "exact": all(record["canonical_correct"] for record in records), "accuracy": sum(record["canonical_correct"] for record in records) / len(records), "groups": group_summary, "q_k": q, "g_k": g, "g_min": min(g), "tau": tau_value, "g_min_gt_tau": min(g) > tau_value, "distant_failures_if_g_min_gt_tau": [record for record in records if not record["canonical_correct"] and record["distance"] > 1], "distant_failure_inconsistency": bool(min(g) > tau_value and any(not record["canonical_correct"] and record["distance"] > 1 for record in records))}
    return summary, records, {"q_k": q, "g_k": g, "g_min": min(g), "tau": tau_value}


def aggregate_contextual(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_distance: dict[str, Any] = {}
    by_action: dict[str, Any] = {}
    for distance in DISTANCES:
        selected = [record for record in records if record["distance"] == distance]
        by_distance[str(distance)] = {"samples": len(selected), "decision_correct": sum(record["decision_correct"] for record in selected), "oracle_final_success": sum(record["oracle_final_success"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected), "timeouts": sum(record["timeout"] for record in selected)}
    for action in range(3):
        selected = [record for record in records if record["expected_action"] == action]
        by_action[ADJUSTMENT_NAMES[action]] = {"samples": len(selected), "decision_correct": sum(record["decision_correct"] for record in selected), "oracle_final_success": sum(record["oracle_final_success"] for record in selected), "final_success": sum(record["final_success"] for record in selected), "trace_success": sum(record["trace_success"] for record in selected)}
    total = max(1, len(records))
    return {"samples": len(records), "decision_correct": sum(record["decision_correct"] for record in records), "oracle_final_success": sum(record["oracle_final_success"] for record in records), "final_success": sum(record["final_success"] for record in records), "trace_success": sum(record["trace_success"] for record in records), "decision_accuracy": sum(record["decision_correct"] for record in records) / total, "oracle_final_success_rate": sum(record["oracle_final_success"] for record in records) / total, "final_success_rate": sum(record["final_success"] for record in records) / total, "trace_success_rate": sum(record["trace_success"] for record in records) / total, "by_distance": by_distance, "by_action": by_action}


def contextual_evaluation(model: Any, controller: OrdinalSharedScorer, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]], canonical_records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_by_pair = {(record["x"], record["reference"]): record for record in canonical_records}
    records: list[dict[str, Any]] = []
    epsilon_records: list[dict[str, Any]] = []
    pair_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT)
    embeddings = model.token_embedding(class_ids)
    with torch.inference_mode():
        for episode in manifest["episodes"]:
            navigation = runtime[episode["episode"]]
            x = navigation["symbolic_x"]
            r_real = navigation["state"][:, SLOT_R].clone()
            epsilon_records.append({"episode": episode["episode"], "graph": episode["graph"], "x": x, "score_r_real": float(controller.score(r_real).item()), "score_e_x": float(controller.score(embeddings[x:x + 1]).item()), "epsilon_R": float((controller.score(r_real) - controller.score(embeddings[x:x + 1])).abs().item())})
            for reference in range(VALUE_COUNT):
                expected = adjustment_action(x, reference)
                target = adjusted_target(x, reference)
                canonical = canonical_by_pair[(x, reference)]
                if navigation["collected"]:
                    logits = controller.logits(r_real, embeddings[reference:reference + 1]).squeeze(0)
                    predicted_action = int(controller.predict_action(r_real, embeddings[reference:reference + 1]).item())
                    oracle_state, _ = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], expected)
                    predicted_state, operation = dispatch_adjustment(model, navigation["memory_keys"], navigation["memory_values"], navigation["memory_types"], navigation["row_mask"], navigation["state"].clone(), navigation["presence"], predicted_action)
                    oracle_value = decode_value(model, oracle_state)
                    predicted_value = decode_value(model, predicted_state)
                    decision_correct = predicted_action == expected
                    oracle_success = oracle_value == target
                    final_success = predicted_value == target
                else:
                    predicted_action = None
                    logits = torch.full((3,), float("nan"))
                    operation = None
                    oracle_value = predicted_value = None
                    decision_correct = oracle_success = final_success = False
                first_control_error = navigation["first_control_error"]
                first_execution_error = navigation["first_execution_error"]
                if navigation["aligned"] and not decision_correct:
                    first_control_error = {"stage": "CTRL-2-O", "reference": reference, "expected_action": expected, "predicted_action": predicted_action}
                elif navigation["aligned"] and decision_correct and not final_success:
                    first_execution_error = {"stage": "CTRL-2-O", "reference": reference, "instruction": "adjustment_dispatch"}
                trace_ok = trace_success(final_success, navigation["timeout"], first_control_error, first_execution_error)
                category = "agreement_correct" if canonical["canonical_correct"] and decision_correct else "shared_error" if not canonical["canonical_correct"] and not decision_correct else "contextual_regression" if canonical["canonical_correct"] else "contextual_recovery"
                values = [float(value) for value in logits.tolist()]
                record = {"episode": episode["episode"], "graph": episode["graph"], "distance": episode["distance"], "x": x, "reference": reference, "expected_action": expected, "predicted_action": predicted_action, "target_value": target, "oracle_predicted_value": oracle_value, "predicted_value": predicted_value, "oracle_final_success": oracle_success, "decision_correct": decision_correct, "final_success": final_success, "trace_success": trace_ok, "timeout": navigation["timeout"], "canonical_correct": canonical["canonical_correct"], "comparison_category": category, "operation": operation, "controller_logits": values, "d": float((controller.difference(r_real, embeddings[reference:reference + 1])).item()) if navigation["collected"] else None, "tau": float(controller.tau().item())}
                records.append(record)
                pair_rows.setdefault((x, reference), []).append(record)
    table = [{"x": x, "reference": reference, "expected_action": ADJUSTMENT_NAMES[adjustment_action(x, reference)], "samples": len(items), "decision_correct": sum(item["decision_correct"] for item in items), "oracle_final_success": sum(item["oracle_final_success"] for item in items), "final_success": sum(item["final_success"] for item in items), "trace_success": sum(item["trace_success"] for item in items), "canonical_correct": sum(item["canonical_correct"] for item in items)} for (x, reference), items in sorted(pair_rows.items())]
    return aggregate_contextual(records), records, epsilon_records, table


def epsilon_summary(epsilon_records: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(item["epsilon_R"] for item in epsilon_records)
    if not values:
        return {"samples": 0}
    def percentile(fraction: float) -> float:
        position = fraction * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight
    return {"samples": len(values), "min": values[0], "p50": percentile(0.50), "p90": percentile(0.90), "p95": percentile(0.95), "p99": percentile(0.99), "p999": percentile(0.999), "max": values[-1], "mean": sum(values) / len(values)}


def build_evaluation_gate(canonical: dict[str, Any], original_pilot: dict[str, Any], contextual: dict[str, Any], pair_table: list[dict[str, Any]]) -> dict[str, Any]:
    original_global = original_pilot["correct_count"] == original_pilot["samples"]
    original_by_action = all(item["correct_count"] == item["samples"] for item in original_pilot["by_action"].values())
    original_by_distance = all(item["final_success_count"] == item["samples"] for item in original_pilot["by_distance"].values())
    contextual_by_action = all(item["samples"] == 0 or item["decision_correct"] / item["samples"] >= 0.999 and item["final_success"] / item["samples"] >= 0.999 and item["trace_success"] / item["samples"] >= 0.999 for item in contextual["by_action"].values())
    contextual_by_distance = all(item["samples"] == 0 or item["decision_correct"] / item["samples"] >= 0.999 and item["final_success"] / item["samples"] >= 0.999 and item["trace_success"] / item["samples"] >= 0.999 for item in contextual["by_distance"].values())
    return {"canonical_exact": canonical["exact"], "original_pilot_global": original_global, "original_pilot_by_action": original_by_action, "original_pilot_by_distance": original_by_distance, "original_pilot": original_global and original_by_action and original_by_distance, "contextual_decision": contextual["decision_accuracy"] >= 0.999, "contextual_final": contextual["final_success_rate"] >= 0.999, "contextual_trace": contextual["trace_success_rate"] >= 0.999, "contextual_by_action": contextual_by_action, "contextual_by_distance": contextual_by_distance, "all_contextual_pairs": all(item["samples"] == item["decision_correct"] == item["oracle_final_success"] == item["final_success"] == item["trace_success"] for item in pair_table), "executor_oracle": contextual["oracle_final_success"] == contextual["samples"], "pass": canonical["exact"] and original_global and original_by_action and original_by_distance and contextual["decision_accuracy"] >= 0.999 and contextual["final_success_rate"] >= 0.999 and contextual["trace_success_rate"] >= 0.999 and contextual_by_action and contextual_by_distance and all(item["samples"] == item["decision_correct"] == item["oracle_final_success"] == item["final_success"] == item["trace_success"] for item in pair_table) and contextual["oracle_final_success"] == contextual["samples"]}


def evaluate_checkpoint(model: Any, manifest: dict[str, Any], runtime: dict[int, dict[str, Any]], train_pairs: set[tuple[int, int]], checkpoint: Path, *, data_root: Path = PILOT_ROOT) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    controller = OrdinalSharedScorer()
    controller.load_state_dict(payload["controller"], strict=True)
    controller.eval()
    canonical_summary, canonical_records, ordinal_metrics = canonical_evaluation(model, controller, train_pairs)
    contextual_summary, contextual_records, epsilon_records, pair_table = contextual_evaluation(model, controller, manifest, runtime, canonical_records)
    test_observations, test_labels = load_dataset(data_root, "test")
    canonical_classification = classification_metrics(controller, test_observations, test_labels)
    gate = build_evaluation_gate(canonical_summary, canonical_classification, contextual_summary, pair_table)
    return {"checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)}, "tau": float(controller.tau().item()), "canonical": canonical_summary, "canonical_records": canonical_records, "ordinal_metrics": ordinal_metrics, "classification_on_original_pilot_test": canonical_classification, "contextual": contextual_summary, "contextual_pair_table": pair_table, "epsilon_R": epsilon_records, "epsilon_R_summary": epsilon_summary(epsilon_records), "contextual_records": contextual_records, "gate": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--source-pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--controller-seed", type=int, default=2201)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    copied_hashes = copy_frozen_source(args.source_pilot_root, args.output_root)
    manifests = load_base_manifests()
    train_obs, train_labels = load_dataset(args.output_root, "train")
    val_obs, val_labels = load_dataset(args.output_root, "val")
    test_obs, test_labels = load_dataset(args.output_root, "test")
    model = load_executor()
    ctrl1 = load_ctrl1()
    _, _, runtime = generate_dataset(model, ctrl1, manifests["test"], keep_runtime=True)
    train_pairs = set(zip(train_labels["x"].tolist(), train_labels["reference_value"].tolist()))
    torch.manual_seed(args.controller_seed)
    controller = OrdinalSharedScorer()
    initial_tau = float(controller.tau().item())
    torch.save({"controller": copy.deepcopy(controller.state_dict()), "controller_seed": args.controller_seed, "variant": "ordinal_shared_v1", "tau_initial": initial_tau}, args.output_root / "initial.pt")
    training = train_scorer(controller, train_obs, train_labels, val_obs, val_labels, args.output_root, args.controller_seed)
    baseline = run_baseline(model, manifests["test"], ctrl1, runtime)
    evaluations: dict[str, Any] = {}
    trace_outputs: dict[str, list[dict[str, Any]]] = {}
    for name in ("selected", "final"):
        evaluation = evaluate_checkpoint(model, manifests["test"], runtime, train_pairs, args.output_root / f"{name}.pt")
        evaluations[name] = evaluation
        loaded = OrdinalSharedScorer()
        payload = torch.load(args.output_root / f"{name}.pt", map_location="cpu", weights_only=False)
        loaded.load_state_dict(payload["controller"], strict=True)
        loaded.eval()
        trace_outputs[name] = run_free(model, loaded, manifests["test"], runtime)
        (args.output_root / f"test_free_{name}_traces.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in trace_outputs[name]), encoding="utf-8")
    (args.output_root / "test_baseline_traces.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in baseline), encoding="utf-8")
    metadata = {"task": "T1-CTRL-2-O", "variant": "ordinal_shared_v1", "source_pilot_root": str(args.source_pilot_root), "copied_observation_hashes": copied_hashes, "train_pair_count": len(train_pairs), "train_pair_set": "copied labels only", "architecture": "Linear(64,64)->SiLU->Linear(64,1,bias=False) plus rho", "parameter_count": parameter_count(controller), "controller_seed": args.controller_seed, "tau_initial": initial_tau, "frozen": {"executor": str(BASE_CHECKPOINT), "executor_sha256": sha256(BASE_CHECKPOINT), "ctrl1": str(CTRL1_CHECKPOINT), "ctrl1_sha256": sha256(CTRL1_CHECKPOINT), "weights_changed": False, "temperature_changed": False, "representation_changed": False, "new_data": False}}
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"status": "completed", "task": "T1-CTRL-2-O", "variant": "ordinal_shared_v1", "training": training, "metadata": metadata, "baseline_oracle_adjustment": action_metrics(baseline), "selected": {key: value for key, value in evaluations["selected"].items() if key != "contextual_records"}, "final": {key: value for key, value in evaluations["final"].items() if key != "contextual_records"}, "trace_outputs": {name: {"path": str(args.output_root / f"test_free_{name}_traces.jsonl"), "sha256": sha256(args.output_root / f"test_free_{name}_traces.jsonl"), "metrics": action_metrics(trace_outputs[name])} for name in trace_outputs}, "protocol": {"no_rule_based_prediction_correction": True, "action_helper": "action_from_difference(d,tau), strict inequalities, +/-tau -> KEEP", "g_no_checkpoint_selection": True}}
    result_path = args.output_root / "results.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "results": str(result_path), "results_sha256": sha256(result_path), "metadata_sha256": sha256(args.output_root / "metadata.json"), "initial_sha256": sha256(args.output_root / "initial.pt"), "selected_sha256": sha256(args.output_root / "selected.pt"), "final_sha256": sha256(args.output_root / "final.pt"), "parameter_count": parameter_count(controller), "tau_initial": initial_tau, "tau_final": training["tau_final"], "selected_gate": evaluations["selected"]["gate"], "final_gate": evaluations["final"]["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

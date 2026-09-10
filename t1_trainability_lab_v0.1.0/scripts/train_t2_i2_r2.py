"""Train T2-I2-R2 with frozen R1 writer and behavioral cardinality loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from t2_i2_r1_semantic_writer import CompetitiveSemanticWriter, architecture_report, parameter_count, tensorize
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT, SOURCE_ROOT
from train_t2_i2_r1 import dummy_values, instruction_for_label
from train_u0c_ctrl2_o import sha256

ROOT = Path(__file__).resolve().parents[1]; CAMPAIGN_ROOT = ROOT / "campaign"; ACTIONS = (0, 1, 2, 3, 5)


def output_for_seed(seed: int) -> Path: return CAMPAIGN_ROOT / f"t2_i2_r2_seed{seed}"
def checkpoint_for_seed(seed: int) -> Path: return output_for_seed(seed) / "final.pt"


def load_source() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return torch.load(SOURCE_ROOT / "train" / "observations.pt", weights_only=False), torch.load(SOURCE_ROOT / "train" / "labels.pt", weights_only=False)


def first_panel_row(labels: dict[str, torch.Tensor], constraints: tuple[int, int], value: int, action: int) -> int | None:
    lower = value if constraints == (1, 0) else 0; forbidden = value if constraints == (0, 1) else 0; mask = (labels["constraints"][:, 0] == constraints[0]) & (labels["constraints"][:, 1] == constraints[1]) & (labels["lower"] == lower) & (labels["forbidden"] == forbidden) & (labels["action"] == action); rows = torch.where(mask)[0]; return int(rows[0]) if len(rows) else None


def build_panels(labels: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    panels: dict[str, dict[str, object]] = {}
    for kind, constraints, values in (("NONE", (0, 0), (0,)), ("FLOOR", (1, 0), range(32)), ("AVOID", (0, 1), range(32))):
        for value in values:
            rows = [{"action": action, "row_idx": row} for action in ACTIONS if (row := first_panel_row(labels, constraints, value, action)) is not None]
            panels[f"{kind}:{value}"] = {"kind": kind, "value": value, "rows": rows, "J": len(rows), "actions": [item["action"] for item in rows]}
    return panels


def panel_manifest_hash(manifest: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def panel_key(constraints: tuple[int, int], lower: int, forbidden: int) -> str:
    return f"{'NONE' if constraints == (0, 0) else 'FLOOR' if constraints == (1, 0) else 'AVOID'}:{0 if constraints == (0, 0) else lower if constraints == (1, 0) else forbidden}"


def cardinality_loss_loop(supervisor: LatentConditionedSupervisor, slots: torch.Tensor, indices: torch.Tensor, labels: dict[str, torch.Tensor], observations: dict[str, torch.Tensor], panels: dict[str, dict[str, object]], p_empty_train: torch.Tensor) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for batch_index, row_index in enumerate(indices.tolist()):
        constraints = tuple(int(item) for item in labels["constraints"][row_index].tolist()); lower = int(labels["lower"][row_index]); forbidden = int(labels["forbidden"][row_index]); panel = panels[panel_key(constraints, lower, forbidden)]; row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], dtype=torch.long)
        features = observations["features"][row_ids]; p_empty = p_empty_train[row_ids]
        logits0 = supervisor(features, slots[batch_index, 0].expand(len(row_ids), -1)); logits1 = supervisor(features, slots[batch_index, 1].expand(len(row_ids), -1)); inactive0 = F.kl_div(F.log_softmax(logits0, dim=-1), p_empty, reduction="none").sum(dim=-1).mean(); inactive1 = F.kl_div(F.log_softmax(logits1, dim=-1), p_empty, reduction="none").sum(dim=-1).mean()
        if constraints == (0, 0):
            terms.append((inactive0 + inactive1) / 2)
        else:
            targets = labels["action"][row_ids]; active0 = F.cross_entropy(logits0, targets); active1 = F.cross_entropy(logits1, targets); terms.append(torch.minimum((active0 + inactive1) / 2, (active1 + inactive0) / 2))
    if not terms:
        return torch.zeros((), device=slots.device, dtype=slots.dtype)
    return torch.stack(terms).mean()


def cardinality_loss(supervisor: LatentConditionedSupervisor, slots: torch.Tensor, indices: torch.Tensor, labels: dict[str, torch.Tensor], observations: dict[str, torch.Tensor], panels: dict[str, dict[str, object]], p_empty_train: torch.Tensor) -> torch.Tensor:
    """Vectorized equivalent of cardinality_loss_loop; panels are padded to J_max."""
    return cardinality_loss_grouped(supervisor, slots, indices, labels, observations, panels, p_empty_train)


def cardinality_loss_grouped(supervisor: LatentConditionedSupervisor, slots: torch.Tensor, indices: torch.Tensor, labels: dict[str, torch.Tensor], observations: dict[str, torch.Tensor], panels: dict[str, dict[str, object]], p_empty_train: torch.Tensor) -> torch.Tensor:
    """Vectorize by identical panel shape/key, preserving loop semantics exactly."""
    batch = len(indices)
    if batch == 0:
        return torch.zeros((), device=slots.device, dtype=slots.dtype)
    keys = [panel_key(tuple(int(value) for value in labels["constraints"][row].tolist()), int(labels["lower"][row]), int(labels["forbidden"][row])) for row in indices.tolist()]
    grouped_terms: list[torch.Tensor] = []
    for key in dict.fromkeys(keys):
        positions = [index for index, item in enumerate(keys) if item == key]; panel = panels[key]; row_ids = torch.tensor([item["row_idx"] for item in panel["rows"]], device=slots.device, dtype=torch.long); features = observations["features"][row_ids]; p_empty = p_empty_train[row_ids]; targets = labels["action"][row_ids]; local_slots = slots[positions]; count = len(positions); logits0 = supervisor(features.repeat(count, 1), local_slots[:, 0].repeat_interleave(len(row_ids), dim=0)); logits1 = supervisor(features.repeat(count, 1), local_slots[:, 1].repeat_interleave(len(row_ids), dim=0)); logits0 = logits0.reshape(count, len(row_ids), -1); logits1 = logits1.reshape(count, len(row_ids), -1); ce0 = F.cross_entropy(logits0.reshape(-1, logits0.shape[-1]), targets.repeat(count), reduction="none").reshape(count, len(row_ids)).mean(dim=1); ce1 = F.cross_entropy(logits1.reshape(-1, logits1.shape[-1]), targets.repeat(count), reduction="none").reshape(count, len(row_ids)).mean(dim=1); n0 = F.kl_div(F.log_softmax(logits0, dim=-1), p_empty.expand(count, -1, -1), reduction="none").sum(dim=-1).mean(dim=1); n1 = F.kl_div(F.log_softmax(logits1, dim=-1), p_empty.expand(count, -1, -1), reduction="none").sum(dim=-1).mean(dim=1); constraint_batch = labels["constraints"][indices[torch.tensor(positions)]]; is_zero = (constraint_batch[:, 0] == 0) & (constraint_batch[:, 1] == 0); grouped_terms.append(torch.where(is_zero, (n0 + n1) / 2, torch.minimum((ce0 + n1) / 2, (ce1 + n0) / 2)))
    return torch.cat(grouped_terms).mean()


def train(writer: CompetitiveSemanticWriter, supervisor: LatentConditionedSupervisor, observations: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], panels: dict[str, dict[str, object]], p_empty_train: torch.Tensor, seed: int) -> dict[str, object]:
    goals = ((0, 0), (1, 0), (0, 1)); actions = ACTIONS; buckets = {goal: {action: torch.where((labels["constraints"][:, 0] == goal[0]) & (labels["constraints"][:, 1] == goal[1]) & (labels["action"] == action))[0] for action in actions} for goal in goals}; counts = {(0, 0): {0: 8, 1: 8, 2: 8, 5: 8}, (1, 0): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}, (0, 1): {0: 8, 1: 8, 2: 8, 3: 8, 5: 16}}; optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3, weight_decay=0.0); generator = torch.Generator().manual_seed(seed + 1); last_main = torch.tensor(0.0); last_card = torch.tensor(0.0)
    for step in range(1, 5001):
        selected: list[torch.Tensor] = []; forms: list[int] = []
        for goal, action_counts in counts.items():
            for action, count in action_counts.items():
                half = count // 2; bucket = buckets[goal][action];
                if not len(bucket): raise RuntimeError(f"missing bucket {goal}/{action}")
                selected.extend((bucket[torch.randint(len(bucket), (half,), generator=generator)], bucket[torch.randint(len(bucket), (count - half,), generator=generator)])); two = count - half; forms.extend((1,) * half + (2,) * (two // 2) + (3,) * (two - two // 2))
        indices = torch.cat(selected); texts = [instruction_for_label(tuple(labels["constraints"][index].tolist()), int(labels["lower"][index]), int(labels["forbidden"][index]), position, form) for position, (index, form) in enumerate(zip(indices.tolist(), forms))]; token_ids, lengths = tensorize(texts); optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * ((step - 1) / 4999); optimizer.zero_grad(set_to_none=True); slots = writer(token_ids, lengths); condition = slots.sum(dim=1); last_main = F.cross_entropy(supervisor(observations["features"][indices], condition), labels["action"][indices]); last_card = cardinality_loss(supervisor, slots, indices, labels, observations, panels, p_empty_train); (last_main + last_card).backward(); optimizer.step()
    return {"updates": 5000, "seed": seed, "batch_size": 128, "lambda_card": 1.0, "trainable_parameters": parameter_count(writer), "supervisor_trainable_parameters": sum(parameter.numel() for parameter in supervisor.parameters() if parameter.requires_grad), "final_main_loss": float(last_main.detach()), "final_cardinality_loss": float(last_card.detach()), "one_clause_fraction": 0.5, "two_clause_fraction": 0.5, "cardinality_loss_implementation": "vectorized_padded_panels_two_supervisor_forwards"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output = output_for_seed(args.seed) if args.output_root is None else args.output_root
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    observations, labels = load_source()
    panels = build_panels(labels)
    manifest = {"task": "T2-I2-R2", "source_split": "train", "actions": list(ACTIONS), "panels": panels}
    manifest_path = output / "panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha = sha256(manifest_path)
    writer = CompetitiveSemanticWriter()
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    with torch.no_grad():
        p_empty_train = torch.softmax(supervisor(observations["features"], torch.zeros((len(observations["features"]), 32))), dim=-1).detach()
    training = train(writer, supervisor, observations, labels, panels, p_empty_train, args.seed)
    checkpoint = checkpoint_for_seed(args.seed) if args.output_root is None else output / "final.pt"
    torch.save({"writer": writer.state_dict(), "seed": args.seed, "updates": 5000, "trainable_parameters": parameter_count(writer), "supervisor_trainable_parameters": 0, "lambda_card": 1.0, "panel_manifest_sha256": manifest_sha}, checkpoint)
    report = architecture_report(writer) | {"task": "T2-I2-R2", "lambda_card": 1.0, "panel_manifest_sha256": manifest_sha, "panel_J": {key: item["J"] for key, item in panels.items()}, "p_empty": "per train state, precomputed detached before optimizer"}
    (output / "architecture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"status": "trained", "task": "T2-I2-R2", "seed": args.seed, "training": training, "checkpoint": sha256(checkpoint), "panel_manifest_sha256": manifest_sha, "architecture_report": report}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()

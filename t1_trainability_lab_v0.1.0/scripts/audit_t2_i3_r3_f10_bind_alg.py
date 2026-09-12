"""Frozen Writer R3 F=10 binding autopsy; no COMP, training, or G5."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch import nn

from audit_t2_i3_comp0_reg_alg import COMP_CHECKPOINT, encode, load_runtime, sha256
from evaluate_u0c_ctrl7_trained import run_learned
from t2_i3_common import MANIFEST_PATH, load_writer


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
WRITER_CHECKPOINT = CAMPAIGN / "t2_i2_r3_seed6301" / "final.pt"
REG_SOURCE = CAMPAIGN / "t2_i3_comp0_g4_reg_alg_seed6401" / "results.json"
RESIDUAL_SOURCE = CAMPAIGN / "t2_i3_comp0_g4_residual_alg_seed6401" / "results.json"
ROLES = ("AT_LEAST", "VALUE_L", "AND", "AVOID", "VALUE_F")


class Adapter(nn.Module):
    def __init__(self, core: nn.Module, condition: torch.Tensor) -> None:
        super().__init__()
        self.core = core
        self.condition = condition

    def forward(self, features: torch.Tensor, _constraints: torch.Tensor) -> torch.Tensor:
        return self.core(features, self.condition.expand(features.shape[0], -1))


def scalar_stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def inspect(writer, text: str) -> dict:
    from t2_i2_r3_semantic_writer import tensorize

    token_ids, lengths = tensorize([text])
    with torch.no_grad():
        details = writer(token_ids, lengths, return_details=True)
    probabilities = details["routing_probabilities"][0]
    # Persisted detail field is per semantic slot; per-token semantic mass is
    # the sum of slot0/slot1 routing probabilities.
    masses = probabilities[:2].sum(dim=0)
    values = details["values"][0]
    # Writer forward uses slot_output(values) before weighted summation.
    writes = writer.slot_output(values)
    contributions = probabilities[:, :, None] * writes[None, :, :]
    return {"tokens": text.split(), "token_ids": token_ids[0, : lengths[0]].tolist(), "routing_probabilities": probabilities[:, : lengths[0]].tolist(), "semantic_mass": masses[: lengths[0]].tolist(), "contribution_norms": torch.linalg.vector_norm(contributions[:, : lengths[0]], dim=-1).tolist(), "output_slots": details["slots"][0].tolist(), "output_slot_norms": torch.linalg.vector_norm(details["slots"][0], dim=-1).tolist()}


def aggregate_panel(rows: list[dict]) -> dict:
    by_f: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        f = str(row["forbidden"])
        by_f.setdefault(f, {role: {"slot0": [], "slot1": [], "null": [], "semantic_mass": [], "slot0_contribution_norm": [], "slot1_contribution_norm": [], "null_contribution_norm": []} for role in ROLES})
        for index, role in enumerate(ROLES):
            target = by_f[f][role]
            target["slot0"].append(row["routing_probabilities"][0][index])
            target["slot1"].append(row["routing_probabilities"][1][index])
            target["null"].append(row["routing_probabilities"][2][index])
            target["semantic_mass"].append(row["semantic_mass"][index])
            target["slot0_contribution_norm"].append(row["contribution_norms"][0][index])
            target["slot1_contribution_norm"].append(row["contribution_norms"][1][index])
            target["null_contribution_norm"].append(row["contribution_norms"][2][index])
    return {f: {role: {key: scalar_stats(values) for key, values in metrics.items()} for role, metrics in roles.items()} for f, roles in by_f.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6401)
    args = parser.parse_args()
    if args.seed != 6401:
        raise ValueError("F10 binding source is frozen to seed6401")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    candidates = manifest["calibration"] + manifest["heldout"]
    if len(candidates) != 992 or len({item["digest"] for item in candidates}) != 992:
        raise RuntimeError("manifest candidate universe is not 992 unique pairs")
    writer = load_writer()
    rows = []
    for pair in candidates:
        text = f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}"
        rows.append({"digest": pair["digest"], "lower": pair["lower"], "forbidden": pair["forbidden"], **inspect(writer, text)})
    panel_by_f = aggregate_panel(rows)
    f10 = panel_by_f["10"]
    others = {role: {key: [] for key in f10[role]} for role in ROLES}
    for f, roles in panel_by_f.items():
        if f == "10":
            continue
        for role, metrics in roles.items():
            for key, value in metrics.items():
                others[role][key].append(value["mean"])
    comparison = {role: {key: {"f10_mean": f10[role][key]["mean"], "other_f_mean": statistics.fmean(values), "delta": f10[role][key]["mean"] - statistics.fmean(values)} for key, values in metrics.items()} for role, metrics in others.items()}
    atomic = inspect(writer, "AVOID VALUE_10")
    joint_f10 = [row for row in rows if row["forbidden"] == 10]
    runtime = load_runtime()
    residual = json.loads(RESIDUAL_SOURCE.read_text(encoding="utf-8"))
    residual_digests = [case["digest"] for case in residual["cases"]]
    by_digest = {row["digest"]: row for row in joint_f10}
    control2 = []
    for digest in residual_digests:
        row = by_digest[digest]
        pair = {"lower": row["lower"], "forbidden": row["forbidden"]}
        _manifests, _fixed, episodes, base_manifest, core, executor, ctrl1, scorer = runtime
        values = {}
        for slot_name, condition in (("E0", torch.tensor(row["output_slots"][0], dtype=torch.float32)), ("E1", torch.tensor(row["output_slots"][1], dtype=torch.float32)), ("B", torch.tensor(row["output_slots"][0], dtype=torch.float32) + torch.tensor(row["output_slots"][1], dtype=torch.float32))):
            floor = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], row["lower"], 0, (1, 0))
            avoid = run_learned(executor, ctrl1, scorer, Adapter(core, condition), base_manifest, episodes[10], 0, row["forbidden"], (0, 1))
            values[slot_name] = {"FLOOR": bool(floor["success"]), "AVOID": bool(avoid["success"])}
        control2.append({"digest": digest, "lower": row["lower"], "forbidden": row["forbidden"], "E0": values["E0"], "E1": values["E1"], "B": values["B"]})
    result = {"status": "completed", "task": "T2-I3-R3-F10-BIND-ALG", "training": False, "model_mutated": False, "comp_used": False, "z_used": False, "g5_touched": False, "source_artifacts": {"writer_checkpoint_sha256": sha256(WRITER_CHECKPOINT), "manifest_sha256": sha256(MANIFEST_PATH), "residual_artifact_sha256": sha256(RESIDUAL_SOURCE), "reg_alg_artifact_sha256": sha256(REG_SOURCE)}, "panel": {"candidate_count": 992, "pairs_per_forbidden": 31, "rows": rows, "aggregate_by_forbidden": panel_by_f, "f10_vs_other31": comparison}, "control1": {"atomic_instruction": "AVOID VALUE_10", "atomic": atomic, "joint_f10_count": len(joint_f10), "joint_f10": joint_f10}, "control2": {"cases": control2}, "causal_equation": "E_s = sum_t p(s|t) * w_t", "causal_role_mapping": list(ROLES)}
    output = CAMPAIGN / f"t2_i3_r3_f10_bind_alg_seed{args.seed}" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256(output), "status": result["status"], "panel": {"rows": 992, "f10_rows": 31}, "control1": {"atomic": "AVOID VALUE_10", "joint_rows": 31}, "control2_cases": len(control2), "source_hashes": result["source_artifacts"], "g5_touched": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

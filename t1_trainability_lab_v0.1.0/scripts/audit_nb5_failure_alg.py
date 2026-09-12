"""Frozen NB5 failure-geometry audit.

This script is analysis-only. It never constructs an optimizer, calls backward
except for the isolated Part H diagnostic, or writes any checkpoint/manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from nb5_fresh import NB5CoreEncoder, NB5GateOnlyEncoder
from train_t2_i0_baseline_b import CTRL7_CHECKPOINT, LatentConditionedSupervisor
from train_t2_i2_r2 import load_source


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CAMPAIGN = ROOT / "campaign"
MANIFEST_ROOT = CAMPAIGN / "nb5_manifests"
OUT = CAMPAIGN / "nb5_failure_alg"
SEEDS = (6801, 6802, 6803, 6804, 6805)
SCRIPT_PATH = Path(__file__).resolve()
PLACEHOLDER = "__SELF_HASH__"
MASKED_SCORE = -1.0e30


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dump_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric: {value}")
    return value


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile of empty values")
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite(ordered[lower])
    weight = position - lower
    return finite(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must have equal nonzero length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return finite(numerator / (left_norm * right_norm))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            result[indexed[position][0]] = average
        cursor = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def tensor_list(tensor: torch.Tensor) -> list[float]:
    return [finite(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def value_logits(executor: torch.nn.Module, state: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return executor.register_decoder(state, codebook)


def top2_margin(logits: torch.Tensor) -> float:
    values = logits.reshape(-1).topk(2).values
    return finite(float((values[0] - values[1]).item()))


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def encode_tokens(encoder: NB5CoreEncoder, tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [encoder.vocab.encode_name(token) for token in tokens]
    return torch.tensor([ids], dtype=torch.long), torch.tensor([len(ids)], dtype=torch.long)


def inverse_permutation(permutation: list[int]) -> dict[int, int]:
    inverse = {value: index for index, value in enumerate(permutation)}
    if len(inverse) != VALUE_COUNT or set(inverse) != set(range(VALUE_COUNT)):
        raise ValueError("manifest permutation is not a bijection over 0..31")
    return inverse


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "NB5-fresh-lexical-cipher-v1":
        raise ValueError(f"unexpected manifest schema for seed {seed}")
    if len(manifest.get("train", [])) != 96 or len(manifest.get("test", [])) != 1984:
        raise ValueError(f"unexpected manifest row counts for seed {seed}")
    if len([row for row in manifest["test"] if row["order"] == "natural"]) != 992:
        raise ValueError(f"natural test count is not 992 for seed {seed}")
    if len([row for row in manifest["test"] if row["order"] == "reverse"]) != 992:
        raise ValueError(f"reverse test count is not 992 for seed {seed}")
    return manifest, path


def load_frozen(seed: int, manifest: dict[str, Any]) -> tuple[NB5CoreEncoder, NB5GateOnlyEncoder, Path, Path]:
    stage_a_path = CAMPAIGN / f"nb5_stage_a_{seed}" / "stage_a.pt"
    stage_b_path = CAMPAIGN / f"nb5_stage_b_{seed}" / "gate.pt"
    stage_a_payload = torch.load(stage_a_path, map_location="cpu", weights_only=False)
    stage_b_payload = torch.load(stage_b_path, map_location="cpu", weights_only=False)
    stage_a = NB5CoreEncoder(manifest, seed)
    stage_a.load_state_dict(stage_a_payload["encoder"], strict=True)
    stage_a.eval()
    stage_b = NB5GateOnlyEncoder(stage_a_payload["encoder"])
    stage_b.w_c.data.copy_(stage_b_payload["w_c"])
    stage_b.b_c.data.copy_(stage_b_payload["b_c"])
    stage_b.eval()
    for parameter in list(stage_a.parameters()) + list(stage_b.parameters()):
        parameter.requires_grad_(True)
    return stage_a, stage_b, stage_a_path, stage_b_path


def atomic_geometry(
    stage_a: NB5CoreEncoder,
    executor: torch.nn.Module,
    manifest: dict[str, Any],
    codebook: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    permutation = [int(value) for value in manifest["permutation"]]
    floor_operator = manifest["operator_floor"]
    avoid_operator = manifest["operator_avoid"]
    for index, value in enumerate(permutation):
        floor_ids, floor_lengths = encode_tokens(stage_a, [floor_operator, arg_name(index)])
        avoid_ids, avoid_lengths = encode_tokens(stage_a, [avoid_operator, arg_name(index)])
        with torch.no_grad():
            floor_output = stage_a(floor_ids, floor_lengths, return_details=True)
            avoid_output = stage_a(avoid_ids, avoid_lengths, return_details=True)
            floor_logits = value_logits(executor, torch.cat((floor_output[0], torch.zeros_like(floor_output[0])), -1), codebook)
            avoid_logits = value_logits(executor, torch.cat((avoid_output[1], torch.zeros_like(avoid_output[1])), -1), codebook)
        floor_loss = F.cross_entropy(floor_logits, torch.tensor([value], dtype=torch.long))
        avoid_loss = F.cross_entropy(avoid_logits, torch.tensor([value], dtype=torch.long))
        rows.append(
            {
                "arg_index": index,
                "arg": arg_name(index),
                "value": value,
                "value_decode_margin_floor": top2_margin(floor_logits),
                "value_decode_margin_avoid": top2_margin(avoid_logits),
                "atomic_floor": {
                    "loss": finite(float(floor_loss.item())),
                    "margin": top2_margin(floor_logits),
                    "decoded_value": int(floor_logits.argmax(-1).item()),
                    "exact": bool(int(floor_logits.argmax(-1).item()) == value),
                },
                "atomic_avoid": {
                    "loss": finite(float(avoid_loss.item())),
                    "margin": top2_margin(avoid_logits),
                    "decoded_value": int(avoid_logits.argmax(-1).item()),
                    "exact": bool(int(avoid_logits.argmax(-1).item()) == value),
                },
            }
        )
    return rows


def score_tables(stage_b: NB5GateOnlyEncoder, manifest: dict[str, Any]) -> dict[str, Any]:
    tables: dict[str, list[float]] = {name: [] for name in ("F_self", "F_cross", "A_self", "A_cross")}
    effective_tables: dict[str, list[float]] = {name: [] for name in tables}
    gates: list[dict[str, float]] = []
    floor_operator = manifest["operator_floor"]
    avoid_operator = manifest["operator_avoid"]
    for index in range(VALUE_COUNT):
        row: dict[str, Any] = {"arg_index": index, "value": int(manifest["permutation"][index])}
        for operator, role, name in (
            (floor_operator, "F", "self"),
            (avoid_operator, "F", "cross"),
            (avoid_operator, "A", "self"),
            (floor_operator, "A", "cross"),
        ):
            ids, lengths = encode_tokens(stage_b, [operator, arg_name(index)])
            with torch.no_grad():
                output = stage_b(ids, lengths, return_details=True)
            score = output[6 if role == "F" else 7][0, 1]
            gate = output[9][0, 1]
            raw_score = finite(float(score.item()))
            effective_score = raw_score if float(gate.item()) > 0.5 else MASKED_SCORE
            tables[f"{role}_{name}"].append(raw_score)
            effective_tables[f"{role}_{name}"].append(effective_score)
            row[f"{role}_{name}"] = raw_score
            row[f"effective_{role}_{name}"] = effective_score
            row[f"gate_{operator}"] = finite(float(gate.item()))
        gates.append(row)
    return {"tables": tables, "effective_tables": effective_tables, "rows": gates, "effective_score_mask": "raw score when gate c_i > 0.5, otherwise -1e30 sentinel for -infinity"}


def observe_case(stage_b: NB5GateOnlyEncoder, executor: torch.nn.Module, case: dict[str, Any], codebook: torch.Tensor) -> dict[str, Any]:
    ids, lengths = encode_tokens(stage_b, list(case["tokens"]))
    with torch.no_grad():
        output = stage_b(ids, lengths, return_details=True)
        sf, sa, vt, gate = output[6][0], output[7][0], output[8][0], output[9][0]
        valid = torch.where(gate > 0.5)[0]
        if len(valid) == 0:
            raise RuntimeError("Stage B gate selected no token")
        floor_pointer = int(valid[torch.argmax(2.5 * sf[valid])].item())
        avoid_pointer = int(valid[torch.argmax(2.5 * sa[valid])].item())
        floor_value = (vt[floor_pointer] * gate[floor_pointer]).unsqueeze(0)
        avoid_value = (vt[avoid_pointer] * gate[avoid_pointer]).unsqueeze(0)
        floor_logits = value_logits(executor, torch.cat((floor_value, torch.zeros_like(floor_value)), -1), codebook)
        avoid_logits = value_logits(executor, torch.cat((avoid_value, torch.zeros_like(avoid_value)), -1), codebook)
    floor_value = int(floor_logits.argmax(-1).item())
    avoid_value = int(avoid_logits.argmax(-1).item())
    return {
        "lower": int(case["lower"]),
        "forbidden": int(case["forbidden"]),
        "floor_pointer_token": case["tokens"][floor_pointer],
        "avoid_pointer_token": case["tokens"][avoid_pointer],
        "floor_decoded_value": floor_value,
        "avoid_decoded_value": avoid_value,
        "floor_failure": floor_value != int(case["lower"]),
        "avoid_failure": avoid_value != int(case["forbidden"]),
        "floor_decode_margin": top2_margin(floor_logits),
        "avoid_decode_margin": top2_margin(avoid_logits),
    }


def predicted_case(case: dict[str, Any], tables: dict[str, list[float]], inverse: dict[int, int]) -> dict[str, Any]:
    lower = int(case["lower"])
    forbidden = int(case["forbidden"])
    lower_index = inverse[lower]
    forbidden_index = inverse[forbidden]
    delta_f = tables["F_self"][lower_index] - tables["F_cross"][forbidden_index]
    delta_a = tables["A_self"][forbidden_index] - tables["A_cross"][lower_index]
    return {
        "lower": lower,
        "forbidden": forbidden,
        "delta_F": finite(delta_f),
        "delta_A": finite(delta_a),
        "floor_failure": delta_f <= 0.0,
        "avoid_failure": delta_a <= 0.0,
    }


def matrix_from_cases(cases: list[dict[str, Any]], field: str) -> list[list[bool | None]]:
    matrix: list[list[bool | None]] = [[None for _ in range(VALUE_COUNT)] for _ in range(VALUE_COUNT)]
    for case in cases:
        matrix[int(case["lower"])][int(case["forbidden"])] = bool(case[field])
    return matrix


def role_overlap(self_scores: list[float], cross_scores: list[float], failures: list[dict[str, Any]], role: str) -> dict[str, Any]:
    pairs = [(i, j, self_scores[i] - cross_scores[j]) for i in range(VALUE_COUNT) for j in range(VALUE_COUNT) if i != j]
    gaps = [gap for _, _, gap in pairs]
    target_field = "floor_failure" if role == "FLOOR" else "avoid_failure"
    target_values = [bool(item[target_field]) for item in failures]
    return {
        "self": {
            "min": finite(min(self_scores)),
            "p10": percentile(self_scores, 10),
            "p25": percentile(self_scores, 25),
            "median": percentile(self_scores, 50),
            "p75": percentile(self_scores, 75),
            "p90": percentile(self_scores, 90),
            "max": finite(max(self_scores)),
        },
        "cross": {
            "min": finite(min(cross_scores)),
            "p10": percentile(cross_scores, 10),
            "p25": percentile(cross_scores, 25),
            "median": percentile(cross_scores, 50),
            "p75": percentile(cross_scores, 75),
            "p90": percentile(cross_scores, 90),
            "max": finite(max(cross_scores)),
        },
        "global_gap_exact_G": finite(min(gaps)),
        "global_gap_approx_min_self_minus_max_cross": finite(min(self_scores) - max(cross_scores)),
        "expected_sign_prediction": {
            "G_positive_implies_exhaustive_role_separation": min(gaps) > 0.0,
            "approximate_gap_positive": min(self_scores) - max(cross_scores) > 0.0,
            "predicted_failure_count": sum(target_values),
        },
        "self_below_max_cross_count": sum(value < max(cross_scores) for value in self_scores),
        "cross_above_min_self_count": sum(value > min(self_scores) for value in cross_scores),
        "negative_pair_count": sum(gap < 0.0 for gap in gaps),
        "nonpositive_pair_count": sum(gap <= 0.0 for gap in gaps),
    }


def failure_profiles(
    failures: list[dict[str, Any]], role: str, permutation: list[int]
) -> dict[str, dict[str, list[int]]]:
    inverse = inverse_permutation(permutation)
    arg_target = [0] * VALUE_COUNT
    arg_competitor = [0] * VALUE_COUNT
    value_target = [0] * VALUE_COUNT
    value_competitor = [0] * VALUE_COUNT
    for case in failures:
        lower = int(case["lower"])
        forbidden = int(case["forbidden"])
        failed = bool(case["floor_failure"] if role == "FLOOR" else case["avoid_failure"])
        if failed:
            if role == "FLOOR":
                target_value = lower
                competitor_value = forbidden
            else:
                target_value = forbidden
                competitor_value = lower
            value_target[target_value] += 1
            value_competitor[competitor_value] += 1
            arg_target[inverse[target_value]] += 1
            arg_competitor[inverse[competitor_value]] += 1
    return {
        "arg_aligned": {
            "target_failure_count": arg_target,
            "competitor_failure_count": arg_competitor,
            "total_failure_count": [a + b for a, b in zip(arg_target, arg_competitor)],
        },
        "value_aligned": {
            "target_failure_count": value_target,
            "competitor_failure_count": value_competitor,
            "total_failure_count": [a + b for a, b in zip(value_target, value_competitor)],
        },
    }


def top5(values: list[int], permutation: list[int]) -> list[dict[str, int]]:
    return [
        {"arg_index": index, "value": int(permutation[index]), "count": int(values[index])}
        for index in sorted(range(VALUE_COUNT), key=lambda item: (-values[item], item))[:5]
    ]


def top5_values(values: list[int], permutation: list[int]) -> list[dict[str, int]]:
    inverse = inverse_permutation(permutation)
    return [
        {"value": value, "arg_index": inverse[value], "count": int(values[value])}
        for value in sorted(range(VALUE_COUNT), key=lambda item: (-values[item], item))[:5]
    ]


def aligned_profile(values: list[int], permutation: list[int], by: str) -> list[int]:
    if by == "ARG":
        return list(values)
    result = [0] * VALUE_COUNT
    for index, value in enumerate(permutation):
        result[value] = values[index]
    return result


def profiles_for_role(failures: list[dict[str, Any]], role: str, permutation: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = failure_profiles(failures, role, permutation)
    output["per_seed"] = output["arg_aligned"]
    for alignment in ("arg_aligned", "value_aligned"):
        top = top5 if alignment == "arg_aligned" else top5_values
        output[alignment]["top5_target"] = top(output[alignment]["target_failure_count"], permutation)
        output[alignment]["top5_competitor"] = top(output[alignment]["competitor_failure_count"], permutation)
        output[alignment]["top5_total"] = top(output[alignment]["total_failure_count"], permutation)
    output["weak_self"] = output["arg_aligned"]["target_failure_count"]
    output["cross_attractor"] = output["arg_aligned"]["competitor_failure_count"]
    return output


def spearman_matrices(seed_profiles: list[dict[str, Any]], alignment: str) -> dict[str, list[list[float | None]]]:
    names = ("target_failure_count", "competitor_failure_count", "total_failure_count")
    matrix: dict[str, list[list[float | None]]] = {}
    for name in names:
        vectors = [profile[alignment][name] for profile in seed_profiles]
        matrix[name] = [[spearman(left, right) for right in vectors] for left in vectors]
    return matrix


def aggregate_profile(seed_profiles: list[dict[str, Any]], alignment: str) -> dict[str, Any]:
    names = ("target_failure_count", "competitor_failure_count", "total_failure_count")
    result: dict[str, Any] = {}
    for name in names:
        result[name] = [sum(profile[alignment][name][index] for profile in seed_profiles) for index in range(VALUE_COUNT)]
    if alignment == "arg_aligned":
        top = top5
        permutation = list(range(VALUE_COUNT))
    else:
        top = top5_values
        permutation = list(range(VALUE_COUNT))
    result["top5_target"] = top(result["target_failure_count"], permutation)
    result["top5_competitor"] = top(result["competitor_failure_count"], permutation)
    result["top5_total"] = top(result["total_failure_count"], permutation)
    return result


def gradient_probe(
    stage_a: NB5CoreEncoder,
    executor: torch.nn.Module,
    supervisor: LatentConditionedSupervisor,
    observations: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    manifest: dict[str, Any],
    codebook: torch.Tensor,
    role: str,
    arg_index: int,
) -> dict[str, Any]:
    stage_a.zero_grad(set_to_none=True)
    operator = manifest["operator_avoid"] if role == "AVOID" else manifest["operator_floor"]
    ids, lengths = encode_tokens(stage_a, [operator, arg_name(arg_index)])
    output = stage_a(ids, lengths, return_details=True)
    sf, sa = output[6], output[7]
    sf.retain_grad()
    sa.retain_grad()
    mode = output[2]
    target_value = int(manifest["permutation"][arg_index])
    constraints = (0, 1) if role == "AVOID" else (1, 0)
    mask = (labels["constraints"][:, 0] == constraints[0]) & (labels["constraints"][:, 1] == constraints[1])
    value_field = labels["forbidden"] if role == "AVOID" else labels["lower"]
    source_rows = torch.where(mask & (value_field == target_value))[0]
    if not len(source_rows):
        raise RuntimeError(f"missing atomic source row for {role} {arg_index}")
    source = int(source_rows[0].item())
    behavior_logits = supervisor(observations["features"][source : source + 1], mode)
    behavior_loss = F.cross_entropy(behavior_logits, labels["action"][source : source + 1])
    state = output[1] if role == "AVOID" else output[0]
    reference_logits = value_logits(executor, torch.cat((state, torch.zeros_like(state)), -1), codebook)
    reference_loss = F.cross_entropy(reference_logits, torch.tensor([target_value], dtype=torch.long))
    total_loss = behavior_loss + reference_loss
    total_loss.backward()
    present_query = sa if role == "AVOID" else sf
    absent_query = sf if role == "AVOID" else sa
    present_q = stage_a.q_a if role == "AVOID" else stage_a.q_f
    absent_q = stage_a.q_f if role == "AVOID" else stage_a.q_a
    present_score_grad = present_query.grad[0, 1] if present_query.grad is not None else None
    absent_score_grad = absent_query.grad[0, 1] if absent_query.grad is not None else None
    present_q_grad = present_q.grad
    absent_q_grad = absent_q.grad
    return {
        "role": role,
        "arg_index": arg_index,
        "arg": arg_name(arg_index),
        "value": target_value,
        "source_row_index": source,
        "operator": operator,
        "loss_provenance": "single atomic row with the Stage A objective L_behavior + L_ref; frozen supervisor/executor; no optimizer/update",
        "L_behavior": finite(float(behavior_loss.detach().item())),
        "L_ref": finite(float(reference_loss.detach().item())),
        "L_behavior_plus_L_ref": finite(float(total_loss.detach().item())),
        "q_gradients": {
            "F": {"present": role == "FLOOR", "norm": finite(float(stage_a.q_f.grad.norm().item())) if stage_a.q_f.grad is not None else 0.0},
            "A": {"present": role == "AVOID", "norm": finite(float(stage_a.q_a.grad.norm().item())) if stage_a.q_a.grad is not None else 0.0},
            "selected_present_norm": finite(float(present_q_grad.norm().item())) if present_q_grad is not None else 0.0,
            "selected_absent_norm": finite(float(absent_q_grad.norm().item())) if absent_q_grad is not None else 0.0,
        },
        "score_gradients_at_ARG": {
            "present_self": finite(float(present_score_grad.item())) if present_score_grad is not None else 0.0,
            "absent_cross": finite(float(absent_score_grad.item())) if absent_score_grad is not None else 0.0,
            "absent_cross_exactly_zero": absent_score_grad is None or float(absent_score_grad.item()) == 0.0,
        },
        "reference_decoded_value": int(reference_logits.argmax(-1).item()),
        "reference_margin": top2_margin(reference_logits),
    }


def quadrant_summary(cases: list[dict[str, Any]], inverse: dict[int, int], tables: dict[str, list[float]]) -> dict[str, Any]:
    deltas = [predicted_case(case, tables, inverse) for case in cases]
    quadrants = {"both_pass": 0, "floor_fail_only": 0, "avoid_fail_only": 0, "both_fail": 0}
    distance_rows: list[float] = []
    delta_f: list[float] = []
    delta_a: list[float] = []
    for item in deltas:
        floor_fail = item["floor_failure"]
        avoid_fail = item["avoid_failure"]
        if not floor_fail and not avoid_fail:
            quadrants["both_pass"] += 1
        elif floor_fail and not avoid_fail:
            quadrants["floor_fail_only"] += 1
        elif not floor_fail and avoid_fail:
            quadrants["avoid_fail_only"] += 1
        else:
            quadrants["both_fail"] += 1
        delta_f.append(item["delta_F"])
        delta_a.append(item["delta_A"])
        distance_rows.append(math.sqrt(max(item["delta_F"], 0.0) ** 2 + max(item["delta_A"], 0.0) ** 2))
    return {
        "counts": quadrants,
        "pearson_delta_F_delta_A": pearson(delta_f, delta_a),
        "spearman_delta_F_delta_A": spearman(delta_f, delta_a),
        "distance_to_both_fail_region": {
            "metric": "sqrt(max(delta_F,0)^2 + max(delta_A,0)^2), Euclidean distance to {delta_F<=0, delta_A<=0}",
            "min": finite(min(distance_rows)),
            "median": percentile(distance_rows, 50),
            "mean": finite(sum(distance_rows) / len(distance_rows)),
        },
        "geometric_explanation": "Both failure requires the same pair to lie in the southwest quadrant of (delta_F, delta_A); observed points remain outside that closed quadrant. This is an empirical five-seed regularity, not an architecture theorem.",
        "min_max_delta": finite(min(max(left, right) for left, right in zip(delta_f, delta_a))),
    }


def reverse_sanity(stage_b: NB5GateOnlyEncoder, executor: torch.nn.Module, manifest: dict[str, Any], codebook: torch.Tensor, natural_cases: list[dict[str, Any]]) -> dict[str, Any]:
    reverse_rows = [row for row in manifest["test"] if row["order"] == "reverse"]
    reverse_observed = [observe_case(stage_b, executor, row, codebook) for row in reverse_rows]
    natural_f = {(int(row["lower"]), int(row["forbidden"])) for row in natural_cases if row["floor_failure"]}
    reverse_f = {(int(row["lower"]), int(row["forbidden"])) for row in reverse_observed if row["floor_failure"]}
    natural_a = {(int(row["lower"]), int(row["forbidden"])) for row in natural_cases if row["avoid_failure"]}
    reverse_a = {(int(row["lower"]), int(row["forbidden"])) for row in reverse_observed if row["avoid_failure"]}
    return {
        "natural_cases": len(natural_cases),
        "reverse_cases": len(reverse_observed),
        "floor_failure_sets_equal": natural_f == reverse_f,
        "avoid_failure_sets_equal": natural_a == reverse_a,
        "pass": natural_f == reverse_f and natural_a == reverse_a,
    }


def source_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    supervisor = LatentConditionedSupervisor(CTRL7_CHECKPOINT)
    supervisor.eval()
    for parameter in supervisor.parameters():
        parameter.requires_grad_(False)
    observations, labels = load_source()
    codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)).detach()

    source_files = [
        SCRIPT_PATH,
        SCRIPTS / "nb5_fresh.py",
        SCRIPTS / "ctrl2_common.py",
        SCRIPTS / "train_nb5_stage_a_6801.py",
        SCRIPTS / "train_nb5_stage_b_6801.py",
        SCRIPTS / "audit_nb5_secondary.py",
        MANIFEST_ROOT / "nb5_v2_semantic_check.json",
        CAMPAIGN / "nb5_secondary" / "results.json",
        CTRL7_CHECKPOINT,
    ]
    for seed in SEEDS:
        source_files.extend((MANIFEST_ROOT / f"manifest_{seed}_v2.json", CAMPAIGN / f"nb5_stage_a_{seed}" / "stage_a.pt", CAMPAIGN / f"nb5_stage_b_{seed}" / "gate.pt"))

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        manifest, manifest_path = load_manifest(seed)
        stage_a, stage_b, stage_a_path, stage_b_path = load_frozen(seed, manifest)
        inverse = inverse_permutation([int(value) for value in manifest["permutation"]])
        atomic = atomic_geometry(stage_a, executor, manifest, codebook)
        score_data = score_tables(stage_b, manifest)
        for atomic_row, score_row in zip(atomic, score_data["rows"]):
            atomic_row["gate_c"] = score_row[f"gate_{manifest['operator_floor']}"]
        natural_rows = [row for row in manifest["test"] if row["order"] == "natural"]
        observed = [observe_case(stage_b, executor, row, codebook) for row in natural_rows]
        predicted = [predicted_case(row, score_data["effective_tables"], inverse) for row in natural_rows]
        observed_matrix = {
            "floor": matrix_from_cases(observed, "floor_failure"),
            "avoid": matrix_from_cases(observed, "avoid_failure"),
        }
        predicted_matrix = {
            "floor": matrix_from_cases(predicted, "floor_failure"),
            "avoid": matrix_from_cases(predicted, "avoid_failure"),
        }
        if observed_matrix != predicted_matrix:
            raise RuntimeError(f"Part B mismatch for seed {seed}: predicted_failure_matrix != observed_failure_matrix; no artifact written")
        overlap = {
            "FLOOR": role_overlap(score_data["tables"]["F_self"], score_data["tables"]["F_cross"], observed, "FLOOR"),
            "AVOID": role_overlap(score_data["tables"]["A_self"], score_data["tables"]["A_cross"], observed, "AVOID"),
        }
        profiles = {
            role: profiles_for_role(observed, role, [int(value) for value in manifest["permutation"]])
            for role in ("FLOOR", "AVOID")
        }
        g = {
            "FLOOR": {
                "self_vs_target_failure_spearman": spearman(score_data["tables"]["F_self"], profiles["FLOOR"]["weak_self"]),
                "cross_vs_competitor_failure_spearman": spearman(score_data["tables"]["F_cross"], profiles["FLOOR"]["cross_attractor"]),
                "expected_signs": {"self_vs_target": "negative", "cross_vs_competitor": "positive"},
            },
            "AVOID": {
                "self_vs_target_failure_spearman": spearman(score_data["tables"]["A_self"], profiles["AVOID"]["weak_self"]),
                "cross_vs_competitor_failure_spearman": spearman(score_data["tables"]["A_cross"], profiles["AVOID"]["cross_attractor"]),
                "expected_signs": {"self_vs_target": "negative", "cross_vs_competitor": "positive"},
            },
        }
        quadrant = quadrant_summary(natural_rows, inverse, score_data["effective_tables"])
        probes = [
            gradient_probe(stage_a, executor, supervisor, observations, labels, manifest, codebook, "AVOID", 0),
            gradient_probe(stage_a, executor, supervisor, observations, labels, manifest, codebook, "FLOOR", 0),
        ]
        per_seed.append(
            {
                "seed": seed,
                "manifest": source_record(manifest_path),
                "stage_a_checkpoint": source_record(stage_a_path),
                "stage_b_checkpoint": source_record(stage_b_path),
                "part_A": {"rows": atomic, "row_count": len(atomic), "atomic_floor_exact": sum(row["atomic_floor"]["exact"] for row in atomic), "atomic_avoid_exact": sum(row["atomic_avoid"]["exact"] for row in atomic)},
                "part_B": {"cases": 992, "observed_failure_matrix": observed_matrix, "predicted_failure_matrix": predicted_matrix, "prediction_score_tables": score_data["effective_tables"], "bitwise_equal": True, "floor_992_of_992": True, "avoid_992_of_992": True, "prediction_basis": "four gate-masked score tables; raw score retained separately for Parts C/D/G"},
                "part_C": overlap,
                "part_D": overlap,
                "part_E": profiles,
                "part_F": {role: {"weak_self": profiles[role]["weak_self"], "cross_attractor": profiles[role]["cross_attractor"], "weak_self_rate": [finite(value / 31.0) for value in profiles[role]["weak_self"]], "cross_attractor_rate": [finite(value / 31.0) for value in profiles[role]["cross_attractor"]], "by_value": {"weak_self": profiles[role]["value_aligned"]["target_failure_count"], "cross_attractor": profiles[role]["value_aligned"]["competitor_failure_count"]}} for role in ("FLOOR", "AVOID")},
                "part_G": g,
                "part_H": {"probe_arg_index": 0, "probes": probes, "atomic_32_of_32_floor": sum(row["atomic_floor"]["exact"] for row in atomic) == 32, "atomic_32_of_32_avoid": sum(row["atomic_avoid"]["exact"] for row in atomic) == 32},
                "part_I": quadrant,
                "natural_failure_counts": {"floor": sum(row["floor_failure"] for row in observed), "avoid": sum(row["avoid_failure"] for row in observed), "both": sum(row["floor_failure"] and row["avoid_failure"] for row in observed)},
                "score_tables": score_data,
            }
        )

    cross_seed: dict[str, Any] = {}
    for role in ("FLOOR", "AVOID"):
        role_profiles = [item["part_E"][role] for item in per_seed]
        cross_seed[role] = {"arg_aligned": spearman_matrices(role_profiles, "arg_aligned"), "value_aligned": spearman_matrices(role_profiles, "value_aligned"), "aggregate_arg_aligned": aggregate_profile(role_profiles, "arg_aligned"), "aggregate_value_aligned": aggregate_profile(role_profiles, "value_aligned")}

    aggregate_g: dict[str, Any] = {}
    for role, self_name, cross_name in (("FLOOR", "F_self", "F_cross"), ("AVOID", "A_self", "A_cross")):
        self_values: list[float] = []
        cross_values: list[float] = []
        target_values: list[float] = []
        competitor_values: list[float] = []
        for item in per_seed:
            self_values.extend(item["score_tables"]["tables"][self_name])
            cross_values.extend(item["score_tables"]["tables"][cross_name])
            target_values.extend(item["part_E"][role]["weak_self"])
            competitor_values.extend(item["part_E"][role]["cross_attractor"])
        aggregate_g[role] = {"self_vs_target_failure_spearman": spearman(self_values, target_values), "cross_vs_competitor_failure_spearman": spearman(cross_values, competitor_values), "expected_signs": {"self_vs_target": "negative", "cross_vs_competitor": "positive"}}

    worst_candidates = [
        {"seed": seed_item["seed"], "role": role, "G": seed_item["part_C"][role]["global_gap_exact_G"]}
        for seed_item in per_seed
        for role in ("FLOOR", "AVOID")
    ]
    worst = min(worst_candidates, key=lambda row: row["G"])
    reverse = []
    for seed in SEEDS:
        manifest, _ = load_manifest(seed)
        _, stage_b, _, _ = load_frozen(seed, manifest)
        natural_rows = [row for row in manifest["test"] if row["order"] == "natural"]
        observed_natural = [observe_case(stage_b, executor, row, codebook) for row in natural_rows]
        reverse.append({"seed": seed, **reverse_sanity(stage_b, executor, manifest, codebook, observed_natural)})

    artifact: dict[str, Any] = {
        "status": "completed",
        "task": "T2-NOBYPASS-1-NB5-FAILURE-ALG",
        "training": False,
        "weights_updated": False,
        "new_joint_sequences_executed": False,
        "existing_joint_test_rows_evaluated": {"natural_per_seed": 992, "reverse_per_seed_final_sanity": 992},
        "analysis_scope": "natural test rows for all Parts A-I; reverse only final failure-set sanity",
        "executive_summary": {
            "natural_cases_per_seed": 992,
            "all_part_B_gates": True,
            "natural_failure_counts": {str(item["seed"]): item["natural_failure_counts"] for item in per_seed},
            "all_fail_both_zero": all(item["natural_failure_counts"]["both"] == 0 for item in per_seed),
            "worst_geometry": worst,
            "all_atomic_floor_32_of_32": all(item["part_H"]["atomic_32_of_32_floor"] for item in per_seed),
            "all_atomic_avoid_32_of_32": all(item["part_H"]["atomic_32_of_32_avoid"] for item in per_seed),
        },
        "source_hashes": {str(path): source_record(path) for path in source_files},
        "parts": {
            "A": {"per_seed": [{"seed": item["seed"], **item["part_A"]} for item in per_seed]},
            "B": {"per_seed": [{"seed": item["seed"], **item["part_B"]} for item in per_seed]},
            "C": {"per_seed": [{"seed": item["seed"], **item["part_C"]} for item in per_seed]},
            "D": {"per_seed": [{"seed": item["seed"], **item["part_D"]} for item in per_seed]},
            "E": {"per_seed": [{"seed": item["seed"], **item["part_E"]} for item in per_seed], "cross_seed": cross_seed},
            "F": {"per_seed": [{"seed": item["seed"], **item["part_F"]} for item in per_seed]},
            "G": {"per_seed": [{"seed": item["seed"], **item["part_G"]} for item in per_seed], "aggregate": aggregate_g, "worst_geometry_atomic_evidence": {"worst": worst, "all_seed_atomic_floor": [item["part_H"]["atomic_32_of_32_floor"] for item in per_seed], "all_seed_atomic_avoid": [item["part_H"]["atomic_32_of_32_avoid"] for item in per_seed]}},
            "H": {"per_seed": [{"seed": item["seed"], **item["part_H"]} for item in per_seed], "provenance": "Stage A encoder state loaded from stage_a.pt; actual Stage A single-row L_behavior + L_ref formulas reproduced; frozen executor/supervisor; autograd only; no optimizer or parameter update."},
            "I": {"per_seed": [{"seed": item["seed"], **item["part_I"]} for item in per_seed]},
        },
        "final_reverse_sanity": reverse,
        "artifacts": {"results_json": str(OUT / "results.json"), "analysis_script": str(SCRIPT_PATH)},
        "next_recommended": "Treat seeds 6801-6805 as development/diagnostic evidence; validate any causal objective change on a fresh 6901-6905 battery.",
        "score_table_resolution": {"raw_geometry_tables": "Parts C/D/G use raw s_F/s_A values from atomic [operator, ARG_i] probes.", "part_B_effective_tables": "Part B/I use four deterministic gate-masked tables: raw score when c_i > 0.5, -1e30 sentinel otherwise, matching existing Stage B selector c>0.5.", "why": "Raw four-table inequalities alone do not reproduce seed 6804 gate exclusions; first observed discrepancy is natural (lower=0, forbidden=29), where A_self-A_cross=0.732486 but target ARG_07 has c=0.492545 and is excluded.", "no_case_specific_values": True},
        "risks": ["Legacy Stage A/B results.json metadata has stale seed fields and may reference pre-v2 manifest hashes; this artifact uses raw hashes of actual v2 manifests and frozen checkpoints on disk.", "G>0 is sufficient for exhaustive role separation, while G<=0 only permits overlap; sign is not itself a failure-count theorem.", "Part B exactness depends on existing Stage B gate selector semantics; raw unmasked scores are retained and reported separately."],
        "skill_resolution": {"graphify": "No .codegraph/ index present; targeted filesystem inspection used.", "work_unit_commits": "Single focused analysis work unit; no commit requested or created.", "shared": "Read before work."},
        "artifact_self_hash": PLACEHOLDER,
        "artifact_hash_basis": "SHA-256 of final canonical UTF-8 JSON bytes with artifact_self_hash replaced by __SELF_HASH__.",
    }
    artifact_hash = sha256_bytes(dump_bytes(artifact))
    artifact["artifact_self_hash"] = artifact_hash
    final_bytes = dump_bytes(artifact)
    verification = dict(artifact)
    verification["artifact_self_hash"] = PLACEHOLDER
    if sha256_bytes(dump_bytes(verification)) != artifact_hash:
        raise RuntimeError("artifact self-hash verification failed")
    (OUT / "results.json").write_bytes(final_bytes)
    print(json.dumps({"status": artifact["status"], "artifact": str(OUT / "results.json"), "artifact_sha256": artifact_hash, "part_B": "992/992 per seed", "reverse_sanity": all(item["pass"] for item in reverse)}, indent=2))


if __name__ == "__main__":
    main()

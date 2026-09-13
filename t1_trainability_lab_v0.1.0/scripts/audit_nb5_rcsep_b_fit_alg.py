"""Read-only RCSEP-B fit/separability/oracle audit.

This script loads frozen RCSEP-A cores, frozen RCSEP-B gates, v2 manifests,
and the frozen decoder.  Part C optimizes only a standalone NumPy/SciPy
17-vector; it never constructs a torch optimizer or updates model parameters.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linprog, minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ctrl2_common import load_executor  # noqa: E402
from nb5_fresh import NB5GateOnlyEncoder  # noqa: E402


CAMPAIGN = ROOT / "campaign"
MANIFEST_ROOT = CAMPAIGN / "nb5_manifests"
OUTPUT_DIR = CAMPAIGN / "nb5_rcsep_b_fit_alg"
OUTPUT_PATH = OUTPUT_DIR / "results.json"
SEEDS = (6801, 6802, 6803, 6804, 6805)
ARG_TOKENS = [f"ARG_{index:02d}" for index in range(32)]
STRUCTURE_TOKENS = ["OP_X", "OP_Y", "LINK"]
DOMAIN_TOKENS = ARG_TOKENS + STRUCTURE_TOKENS
LEGACY_TOKENS = DOMAIN_TOKENS + ["NOOP"]
TAU = 3.0545
ALPHA = 2.0
POINTER_FIELDS = ("pointer_F", "pointer_A", "decode_F", "decode_A", "RAW", "CANON", "RAW==CANON")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finite(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def arg_index(token: str) -> int | None:
    return int(token[4:]) if token.startswith("ARG_") else None


def validate_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "NB5-fresh-lexical-cipher-v1":
        raise ValueError(f"unexpected manifest schema for seed {seed}")
    if int(manifest.get("domain_seed")) != seed:
        raise ValueError(f"manifest domain seed mismatch for seed {seed}")
    if list(manifest["token_ids"]) != ARG_TOKENS + ["LINK", "OP_X", "OP_Y"]:
        raise ValueError(f"unexpected token order for seed {seed}")
    permutation = [int(value) for value in manifest["permutation"]]
    if sorted(permutation) != list(range(32)):
        raise ValueError(f"manifest permutation is not a bijection for seed {seed}")
    test = manifest.get("test", [])
    if len(test) != 1984:
        raise ValueError(f"test row count is not 1984 for seed {seed}")
    if sum(row["order"] == "natural" for row in test) != 992:
        raise ValueError(f"natural test count is not 992 for seed {seed}")
    if sum(row["order"] == "reverse" for row in test) != 992:
        raise ValueError(f"reverse test count is not 992 for seed {seed}")
    if any(len(row.get("tokens", [])) != 5 for row in test):
        raise ValueError(f"unexpected test sequence length for seed {seed}")
    return manifest, path


def load_frozen_seed(seed: int, manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    stage_a_path = CAMPAIGN / f"nb5_rcsep_a_{seed}" / "stage_a.pt"
    stage_a_results_path = CAMPAIGN / f"nb5_rcsep_a_{seed}" / "results.json"
    gate_path = CAMPAIGN / f"nb5_rcsep_b_gate_{seed}" / "gate.pt"
    gate_results_path = CAMPAIGN / f"nb5_rcsep_b_gate_{seed}" / "results.json"
    for path in (stage_a_path, stage_a_results_path, gate_path, gate_results_path):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen source for seed {seed}: {path}")

    stage_a_results = json.loads(stage_a_results_path.read_text(encoding="utf-8"))
    gate_results = json.loads(gate_results_path.read_text(encoding="utf-8"))
    stage_a_sha = sha256_file(stage_a_path)
    gate_sha = sha256_file(gate_path)
    manifest_sha = sha256_file(manifest_path)
    if stage_a_results.get("checkpoint", {}).get("sha256") != stage_a_sha:
        raise ValueError(f"RCSEP-A result/checkpoint hash mismatch for seed {seed}")
    gate_payload = torch.load(gate_path, map_location="cpu", weights_only=False)
    if gate_payload.get("stage_a_checkpoint_sha256") != stage_a_sha:
        raise ValueError(f"gate/stage_a hash mismatch for seed {seed}")
    if gate_payload.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"gate/manifest hash mismatch for seed {seed}")
    if gate_results.get("gate_checkpoint_sha256") != gate_sha:
        raise ValueError(f"RCSEP-B result/checkpoint hash mismatch for seed {seed}")

    stage_a_payload = torch.load(stage_a_path, map_location="cpu", weights_only=False)
    encoder = NB5GateOnlyEncoder(stage_a_payload["encoder"])
    with torch.no_grad():
        encoder.w_c.copy_(gate_payload["w_c"])
        encoder.b_c.copy_(gate_payload["b_c"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "stage_a_path": stage_a_path,
        "gate_path": gate_path,
        "stage_a_results_path": stage_a_results_path,
        "gate_results_path": gate_results_path,
        "stage_a_results": stage_a_results,
        "gate_results": gate_results,
        "gate_payload": gate_payload,
        "encoder": encoder,
        "source_hashes": {
            "stage_a_core": source_record(stage_a_path),
            "rcsep_b_gate": source_record(gate_path),
            "manifest_v2": source_record(manifest_path),
            "rcsep_a_results": source_record(stage_a_results_path),
            "rcsep_b_results": source_record(gate_results_path),
        },
    }


def decoder_codebook(executor: torch.nn.Module) -> torch.Tensor:
    from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT

    return executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long))


def part_a(seed_data: dict[str, Any], executor: torch.nn.Module, codebook: torch.Tensor) -> dict[str, Any]:
    encoder = seed_data["encoder"]
    ids = torch.tensor([encoder.vocab.encode_name(token) for token in DOMAIN_TOKENS], dtype=torch.long)
    with torch.no_grad():
        embeddings = encoder.embedding(ids)
        value_state = encoder.w_v(embeddings)
        zeros = torch.zeros_like(value_state)
        logits = executor.register_decoder(torch.cat((value_state, zeros), dim=-1), codebook)
        top2 = torch.topk(logits, 2, dim=-1).values
        margins = top2[:, 0] - top2[:, 1]
        teachers = torch.sigmoid(ALPHA * (margins - TAU))
        hard = margins > TAU

    rows = []
    for index, token in enumerate(DOMAIN_TOKENS):
        category = "ARG" if token in ARG_TOKENS else "STRUCTURE"
        rows.append(
            {
                "token": token,
                "category": category,
                "arg_index": arg_index(token),
                "internal_index": int(ids[index].item()),
                "raw_embedding": [finite(value) for value in embeddings[index].tolist()],
                "margin": finite(margins[index].item()),
                "teacher_y": finite(teachers[index].item()),
                "teacher_yhat": int(hard[index].item()),
            }
        )
    failures = [
        {"token": row["token"], "reason": "ARG teacher_yhat != 1", "teacher_yhat": row["teacher_yhat"], "margin": row["margin"]}
        for row in rows
        if row["category"] == "ARG" and row["teacher_yhat"] != 1
    ]
    failures.extend(
        {"token": row["token"], "reason": "STRUCTURE teacher_yhat != 0", "teacher_yhat": row["teacher_yhat"], "margin": row["margin"]}
        for row in rows
        if row["category"] == "STRUCTURE" and row["teacher_yhat"] != 0
    )
    return {
        "status": "pass" if not failures else "fail",
        "token_count": len(rows),
        "arg_hard_positive": sum(row["teacher_yhat"] == 1 for row in rows if row["category"] == "ARG"),
        "structure_hard_zero": sum(row["teacher_yhat"] == 0 for row in rows if row["category"] == "STRUCTURE"),
        "threshold": TAU,
        "teacher": "sigmoid(2 * (margin - 3.0545)); hard label margin > 3.0545",
        "rows": rows,
        "failures": failures,
    }


def svm_part(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray([row["raw_embedding"] for row in rows], dtype=np.float64)
    labels = np.asarray([1.0 if row["category"] == "ARG" else -1.0 for row in rows], dtype=np.float64)
    x_aug = np.concatenate((x, np.ones((len(rows), 1))), axis=1)
    signed_design = labels[:, None] * x_aug

    feasibility = linprog(
        np.zeros(x_aug.shape[1]),
        A_ub=-signed_design,
        b_ub=-np.ones(len(rows)),
        bounds=[(None, None)] * x_aug.shape[1],
        method="highs-ds",
    )
    start = feasibility.x if feasibility.success and feasibility.x is not None else np.linalg.lstsq(x_aug, labels, rcond=None)[0]

    def objective(value: np.ndarray) -> float:
        return float(0.5 * np.dot(value[:-1], value[:-1]))

    def gradient(value: np.ndarray) -> np.ndarray:
        return np.concatenate((value[:-1], np.zeros(1)))

    constrained = minimize(
        objective,
        start,
        jac=gradient,
        constraints={"type": "ineq", "fun": lambda value: signed_design @ value - 1.0, "jac": lambda value: signed_design},
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 2000, "disp": False},
    )
    value = np.asarray(constrained.x, dtype=np.float64)
    scores = x_aug @ value
    signed_scores = labels * scores
    norm = float(np.linalg.norm(value[:-1]))
    tol = 1.0e-6
    arg_rows = [index for index, row in enumerate(rows) if row["category"] == "ARG"]
    structure_rows = [index for index, row in enumerate(rows) if row["category"] == "STRUCTURE"]
    support = [rows[index]["token"] for index, margin in enumerate(signed_scores) if abs(margin - 1.0) <= tol]
    violations = [rows[index]["token"] for index, margin in enumerate(signed_scores) if margin < 1.0 - tol]
    feasible_solution = bool(np.all(signed_scores >= 1.0 - 1.0e-5))
    separable = bool(feasibility.success and feasible_solution)
    return {
        "status": "separable" if separable else "not_separable_or_solver_failed",
        "labels": "ARG=+1, OP_X/OP_Y/LINK=-1 from Part A hard labels only",
        "solver": {
            "feasibility": {
                "method": "scipy.optimize.linprog(method='highs-ds')",
                "success": bool(feasibility.success),
                "status": int(feasibility.status),
                "message": str(feasibility.message),
            },
            "hard_margin": {
                "method": "scipy.optimize.minimize(method='SLSQP')",
                "success": bool(constrained.success),
                "status": int(constrained.status),
                "message": str(constrained.message),
                "iterations": int(getattr(constrained, "nit", -1)),
                "function_evaluations": int(getattr(constrained, "nfev", -1)),
            },
        },
        "parameters": {"w": [finite(item) for item in value[:-1]], "b": finite(value[-1])},
        "min_score_ARG": finite(np.min(scores[arg_rows])),
        "max_score_structure": finite(np.max(scores[structure_rows])),
        "score_gap_ARG_minus_structure": finite(np.min(scores[arg_rows]) - np.max(scores[structure_rows])),
        "min_signed_margin": finite(np.min(signed_scores)),
        "geometric_hard_margin": finite(np.min(signed_scores) / norm) if norm > 0.0 else None,
        "max_hard_margin_normalized_at_margin_1": finite(1.0 / norm) if norm > 0.0 else None,
        "support": {"count": len(support), "tokens": support, "tolerance": tol},
        "violations": {"count": len(violations), "tokens": violations, "tolerance": tol},
        "separable": separable,
        "special_ARG_07_seed_6804": next(
            {
                "token": row["token"],
                "score": finite(scores[index]),
                "signed_margin": finite(signed_scores[index]),
                "classification": bool(scores[index] > 0.0),
            }
            for index, row in enumerate(rows)
            if row["token"] == "ARG_07"
        ),
    }


def bce_and_gradient(value: np.ndarray, features: np.ndarray, teacher: np.ndarray) -> tuple[float, np.ndarray]:
    logits = features @ value
    loss = np.mean(np.logaddexp(0.0, logits) - teacher * logits)
    gradient = features.T @ (expit(logits) - teacher) / len(teacher)
    return finite(loss), np.asarray([finite(item) for item in gradient], dtype=np.float64)


def current_gate_bce(
    seed_data: dict[str, Any],
    a_rows: list[dict[str, Any]],
    executor: torch.nn.Module,
    codebook: torch.Tensor,
) -> dict[str, Any]:
    encoder = seed_data["encoder"]
    ids = torch.tensor([encoder.vocab.encode_name(token) for token in LEGACY_TOKENS], dtype=torch.long)
    with torch.no_grad():
        embeddings = encoder.embedding(ids)
        value_state = encoder.w_v(embeddings)
        logits = executor.register_decoder(torch.cat((value_state, torch.zeros_like(value_state)), dim=-1), codebook)
        top2 = torch.topk(logits, 2, dim=-1).values
        teacher = torch.sigmoid(ALPHA * (top2[:, 0] - top2[:, 1] - TAU)).cpu().numpy().astype(np.float64)
    gate_w = seed_data["gate_payload"]["w_c"].detach().cpu().numpy().astype(np.float64)
    gate_b = finite(seed_data["gate_payload"]["b_c"].item())
    all_features = np.concatenate((embeddings.cpu().numpy().astype(np.float64), np.ones((len(LEGACY_TOKENS), 1))), axis=1)
    all_value = np.concatenate((gate_w, np.asarray([gate_b])))
    all_logits = all_features @ all_value
    all_bce = float(np.mean(np.logaddexp(0.0, all_logits) - teacher * all_logits))
    domain_features = all_features[: len(DOMAIN_TOKENS)]
    domain_teacher = teacher[: len(DOMAIN_TOKENS)]
    domain_logits = domain_features @ all_value
    domain_bce = float(np.mean(np.logaddexp(0.0, domain_logits) - domain_teacher * domain_logits))
    reported = finite(seed_data["gate_results"]["B2"]["bce_final"])
    return {
        "existing_reported_bce": reported,
        "existing_reported_bce_provenance": "RCSEP-B B2.bce_final; train_gate scores LEGACY_TOKENS = 35 domain tokens plus NOOP",
        "existing_reported_bce_includes_legacy_NOOP": True,
        "recomputed_36_token_bce_including_NOOP": finite(all_bce),
        "reported_matches_recomputed_36": bool(abs(reported - all_bce) < 1.0e-6),
        "recomputed_35_domain_bce_excluding_NOOP": finite(domain_bce),
        "gate_parameters_current_Adam": {"w": [finite(item) for item in gate_w], "b": gate_b},
        "domain_rows_used": len(a_rows),
    }


def soft_fit_part(seed_data: dict[str, Any], a_rows: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    features = np.concatenate(
        (np.asarray([row["raw_embedding"] for row in a_rows], dtype=np.float64), np.ones((len(a_rows), 1))), axis=1
    )
    teacher = np.asarray([row["teacher_y"] for row in a_rows], dtype=np.float64)
    initial = np.asarray(current["gate_parameters_current_Adam"]["w"] + [current["gate_parameters_current_Adam"]["b"]], dtype=np.float64)

    def fun(value: np.ndarray) -> float:
        return bce_and_gradient(value, features, teacher)[0]

    def jac(value: np.ndarray) -> np.ndarray:
        return bce_and_gradient(value, features, teacher)[1]

    fitted = minimize(
        fun,
        initial,
        jac=jac,
        method="L-BFGS-B",
        options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 10000, "maxls": 100},
    )
    value = np.asarray(fitted.x, dtype=np.float64)
    optimum, gradient = bce_and_gradient(value, features, teacher)
    scores = features @ value
    confidence = expit(scores)
    arg_confidence = confidence[:32]
    structure_confidence = confidence[32:]
    hard = confidence > 0.5
    return {
        "status": "completed",
        "objective": "(1/35) * sum BCEWithLogits(e_t @ w + b, y_t), Part A continuous teacher y",
        "optimizer": {
            "method": "scipy.optimize.minimize(method='L-BFGS-B')",
            "success": bool(fitted.success),
            "status": int(fitted.status),
            "message": str(fitted.message),
            "iterations": int(getattr(fitted, "nit", -1)),
            "function_evaluations": int(getattr(fitted, "nfev", -1)),
            "gradient_evaluations": int(getattr(fitted, "njev", -1)),
            "analytic_gradient": "features.T @ (sigmoid(features @ [w,b]) - y) / 35",
        },
        "current_Adam_gate": current,
        "optimum_BCE_35_domain": optimum,
        "gradient_norm_at_optimum": finite(np.linalg.norm(gradient)),
        "min_c_ARG": finite(np.min(arg_confidence)),
        "max_c_structure": finite(np.max(structure_confidence)),
        "hard_classification": {"ARG_positive_structure_zero": bool(np.all(hard[:32]) and not np.any(hard[32:])), "correct": int(np.sum(hard[:32])) + int(np.sum(~hard[32:])), "total": 35},
        "parameters": {"w": [finite(item) for item in value[:-1]], "b": finite(value[-1])},
        "per_token_scores_and_confidence": [
            {"token": row["token"], "logit": finite(scores[index]), "c": finite(confidence[index]), "hard_selected": bool(hard[index])}
            for index, row in enumerate(a_rows)
        ],
    }


def oracle_eval_order(
    seed_data: dict[str, Any],
    executor: torch.nn.Module,
    codebook: torch.Tensor,
    margin_by_token: dict[str, float],
    order: str,
) -> list[dict[str, Any]]:
    encoder = seed_data["encoder"]
    manifest = seed_data["manifest"]
    inverse = {int(value): index for index, value in enumerate(manifest["permutation"])}
    rows = []
    for case in manifest["test"]:
        if case["order"] != order:
            continue
        token_ids = torch.tensor([[encoder.vocab.encode_name(token) for token in case["tokens"]]], dtype=torch.long)
        lengths = torch.tensor([len(case["tokens"])], dtype=torch.long)
        with torch.no_grad():
            output = encoder(token_ids, lengths, return_details=True)
            sf, sa, vt, gate = output[6][0], output[7][0], output[8][0], output[9][0]
            oracle_h = torch.tensor([margin_by_token[token] > TAU for token in case["tokens"]], dtype=torch.bool)
            eligible = torch.where(oracle_h)[0]
            lower = int(case["lower"])
            forbidden = int(case["forbidden"])
            expected_floor_token = f"ARG_{inverse[lower]:02d}"
            expected_avoid_token = f"ARG_{inverse[forbidden]:02d}"
            floor_pointer = int(eligible[torch.argmax(sf[eligible])].item()) if len(eligible) else None
            avoid_pointer = int(eligible[torch.argmax(sa[eligible])].item()) if len(eligible) else None
            if floor_pointer is not None and avoid_pointer is not None:
                floor_state = (vt[floor_pointer] * gate[floor_pointer]).unsqueeze(0)
                avoid_state = (vt[avoid_pointer] * gate[avoid_pointer]).unsqueeze(0)
                floor_logits = executor.register_decoder(torch.cat((floor_state, torch.zeros_like(floor_state)), -1), codebook)[0]
                avoid_logits = executor.register_decoder(torch.cat((avoid_state, torch.zeros_like(avoid_state)), -1), codebook)[0]
                decode_f = int(floor_logits.argmax().item())
                decode_a = int(avoid_logits.argmax().item())
                canonical_f = int(executor.register_decoder(codebook[decode_f].unsqueeze(0), codebook)[0].argmax().item())
                canonical_a = int(executor.register_decoder(codebook[decode_a].unsqueeze(0), codebook)[0].argmax().item())
            else:
                decode_f = decode_a = canonical_f = canonical_a = None
        pointer_f_token = case["tokens"][floor_pointer] if floor_pointer is not None else None
        pointer_a_token = case["tokens"][avoid_pointer] if avoid_pointer is not None else None
        row = {
            "digest": case.get("digest"),
            "order": order,
            "lower": lower,
            "forbidden": forbidden,
            "tokens": case["tokens"],
            "oracle_eligible_tokens": [case["tokens"][int(index)] for index in eligible.tolist()],
            "oracle_eligible_count": int(len(eligible)),
            "pointer_F_token": pointer_f_token,
            "pointer_A_token": pointer_a_token,
            "pointer_F": bool(pointer_f_token == expected_floor_token),
            "pointer_A": bool(pointer_a_token == expected_avoid_token),
            "decode_F_value": decode_f,
            "decode_A_value": decode_a,
            "decode_F": bool(decode_f == lower),
            "decode_A": bool(decode_a == forbidden),
            "canonical_F_value": canonical_f,
            "canonical_A_value": canonical_a,
            "RAW": bool(decode_f == lower and decode_a == forbidden),
            "CANON": bool(canonical_f == lower and canonical_a == forbidden),
            "RAW==CANON": bool(decode_f is not None and decode_f == canonical_f and decode_a == canonical_a),
        }
        rows.append(row)
    return rows


def order_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cases": len(rows), **{field: sum(bool(row[field]) for row in rows) for field in POINTER_FIELDS}}


def order_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"digest": row["digest"], "lower": row["lower"], "forbidden": row["forbidden"], "tokens": row["tokens"], "failed_fields": [field for field in POINTER_FIELDS if not row[field]], "row": row}
        for row in rows
        if not all(row[field] for field in POINTER_FIELDS)
    ]


def reverse_equality(natural: list[dict[str, Any]], reverse: list[dict[str, Any]]) -> dict[str, Any]:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[Any, ...]]:
        return {(int(row["lower"]), int(row["forbidden"])): tuple(row[field] for field in POINTER_FIELDS) for row in rows}

    left = keyed(natural)
    right = keyed(reverse)
    keys = sorted(set(left) | set(right))
    mismatches = [
        {"lower": key[0], "forbidden": key[1], "natural": left.get(key), "reverse": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    ]
    return {"equal": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches}


def part_d(seed_data: dict[str, Any], a_rows: list[dict[str, Any]], executor: torch.nn.Module, codebook: torch.Tensor) -> dict[str, Any]:
    margins = {row["token"]: row["margin"] for row in a_rows}
    natural = oracle_eval_order(seed_data, executor, codebook, margins, "natural")
    reverse = oracle_eval_order(seed_data, executor, codebook, margins, "reverse")
    summaries = {"natural": order_summary(natural), "reverse": order_summary(reverse)}
    failures = {"natural": order_failures(natural), "reverse": order_failures(reverse)}
    equality = reverse_equality(natural, reverse)
    passed = bool(
        all(summary["cases"] == 992 and all(summary[field] == 992 for field in POINTER_FIELDS) for summary in summaries.values())
        and equality["equal"]
    )
    return {
        "status": "pass" if passed else "fail",
        "eligibility": "oracle h_t = 1[margin_t > 3.0545] from frozen decoder over [w_v(e_t), zeros32]",
        "eligibility_does_not_use_stage_b_gate": True,
        "queries": "exact manifest token sequences; frozen encoder sf/sa queries unchanged",
        "payload_semantics": "vt[selected] * existing frozen RCSEP-B c[selected] after oracle eligibility selection",
        "pointer_fields": {
            "pointer_F": "sf argmax over oracle-eligible sequence tokens equals expected lower ARG token",
            "pointer_A": "sa argmax over oracle-eligible sequence tokens equals expected forbidden ARG token",
            "decode_F": "decoder argmax equals lower VALUE",
            "decode_A": "decoder argmax equals forbidden VALUE",
            "RAW": "decode_F and decode_A",
            "CANON": "canonical decoder round-trip equals expected lower/forbidden VALUE",
            "RAW==CANON": "raw decoded VALUE ids equal canonical round-trip ids",
        },
        "natural": summaries["natural"],
        "reverse": summaries["reverse"],
        "exact_reverse_equality": equality,
        "bad_results": failures,
    }


def artifact_hash(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return sha256_bytes(payload)


def main() -> None:
    if OUTPUT_DIR.exists() or OUTPUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite new audit output: {OUTPUT_DIR}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        codebook = decoder_codebook(executor).detach()

    per_seed = []
    loaded_seeds = []
    halt_evidence = []
    for seed in SEEDS:
        manifest, manifest_path = validate_manifest(seed)
        seed_data = load_frozen_seed(seed, manifest, manifest_path)
        a = part_a(seed_data, executor, codebook)
        entry: dict[str, Any] = {"seed": seed, "source_hashes": seed_data["source_hashes"], "part_A": a}
        loaded_seeds.append((seed_data, a, entry))
        if a["status"] != "pass":
            halt_evidence.extend({"seed": seed, **failure} for failure in a["failures"])
        per_seed.append(entry)

    # Part A is a global gate: no B-D work is allowed for any seed on failure.
    if not halt_evidence:
        for seed_data, a, entry in loaded_seeds:
            current = current_gate_bce(seed_data, a["rows"], executor, codebook)
            b = svm_part(a["rows"])
            c = soft_fit_part(seed_data, a["rows"], current)
            d = part_d(seed_data, a["rows"], executor, codebook)
            entry.update({"part_B": b, "part_C": c, "part_D": d})

    a_pass = not halt_evidence
    artifact: dict[str, Any] = {
        "status": "completed" if a_pass else "halted_part_A",
        "task": "T2-NOBYPASS-2-RCSEP-B-FIT-ALG",
        "executive_summary": {
            "part_A": "5/5 seeds pass" if a_pass else f"HALT before Parts B-D; {len(halt_evidence)} precise hard-label failures",
            "part_B": "ran only after all Part A hard-label requirements passed" if a_pass else "not run",
            "part_C": "ran only after all Part A hard-label requirements passed" if a_pass else "not run",
            "part_D": "ran only after all Part A hard-label requirements passed" if a_pass else "not run",
            "no_adjustment": True,
        },
        "authorization": {
            "id": "T2-NOBYPASS-2-RCSEP-B-FIT-ALG",
            "read_only_diagnostics": True,
            "frozen_inputs": "RCSEP-A stage_a.pt, RCSEP-B gate.pt, v2 manifests, frozen executor",
            "joint_training_or_sequences_added": False,
        },
        "no_training": True,
        "no_checkpoint_changes": True,
        "no_manifest_changes": True,
        "no_existing_artifact_changes": True,
        "optimizer_scope": "Part B/C optimize standalone SciPy vectors only; no model/checkpoint optimizer or parameter update",
        "source_hashes": {"script": source_record(Path(__file__).resolve())},
        "per_seed": per_seed,
        "halt_evidence": halt_evidence,
        "artifacts": {"script": str(Path(__file__).resolve()), "consolidated_results": str(OUTPUT_PATH)},
        "next_recommended": "Treat 6801-6805 as development/diagnostic evidence; validate any follow-up on fresh 6901-6905 domains.",
        "risks": [
            "Part B/C are geometry diagnostics on 35 raw token embeddings; they do not train or alter any checkpoint.",
            "Existing RCSEP-B reported B2 BCE includes legacy NOOP; this artifact separately reports recomputed 35-domain BCE.",
            "Oracle Part D changes only eligibility mask; frozen gate c remains payload weighting per existing HARDPTR semantics.",
        ],
        "skill_resolution": {
            "graphify": "Read before work; existing .codegraph targeted requested source and filesystem inspection completed.",
            "work_unit_commits": "Single focused read-only audit work unit; no commit requested.",
            "shared": "Read before work.",
        },
    }
    artifact["artifact_self_hash"] = artifact_hash(artifact)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "artifact": str(OUTPUT_PATH), "artifact_self_hash": artifact["artifact_self_hash"], "seeds": [entry["seed"] for entry in per_seed]}, sort_keys=True))


if __name__ == "__main__":
    main()

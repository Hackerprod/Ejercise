"""Read-only RCSEP-C no-gate HARDPTR audit over frozen RCSEP-A cores."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from ctrl2_common import BASE_CHECKPOINT, load_executor  # noqa: E402
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT  # noqa: E402
from nb5_fresh import NB5CoreEncoder  # noqa: E402
from train_u0c_c1_joint import C0_CHECKPOINT, U0A_CHECKPOINT  # noqa: E402


CAMPAIGN = ROOT / "campaign"
MANIFEST_ROOT = CAMPAIGN / "nb5_manifests"
OUTPUT_DIR = CAMPAIGN / "nb5_rcsep_c_nogate_hardptr_alg"
OUTPUT_PATH = OUTPUT_DIR / "results.json"
SEEDS = (6801, 6802, 6803, 6804, 6805)
ORDERS = ("natural", "reverse")
ARG_TOKENS = tuple(f"ARG_{index:02d}" for index in range(VALUE_COUNT))
STRUCTURE_TOKENS = ("LINK", "OP_X", "OP_Y")
POINTER_FIELDS = ("pointer_F", "pointer_A", "decode_F", "decode_A", "RAW", "CANON", "RAW==CANON")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def arg_name(index: int) -> str:
    return f"ARG_{index:02d}"


def token_type(token: str) -> str:
    return "ARG" if token.startswith("ARG_") else token


def first_argmax(scores: torch.Tensor, positions: range | list[int]) -> int:
    candidates = list(positions)
    if not candidates:
        raise ValueError("argmax requires at least one real position")
    return min(candidates, key=lambda position: (-finite(scores[position].item()), position))


def inverse_permutation(permutation: list[int]) -> dict[int, int]:
    inverse = {value: index for index, value in enumerate(permutation)}
    if len(inverse) != VALUE_COUNT or set(inverse) != set(range(VALUE_COUNT)):
        raise ValueError("manifest permutation is not a bijection over 0..31")
    return inverse


def validate_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "NB5-fresh-lexical-cipher-v1":
        raise ValueError(f"unexpected manifest schema for seed {seed}")
    if int(manifest.get("domain_seed")) != seed:
        raise ValueError(f"manifest domain seed mismatch for seed {seed}")
    expected_tokens = list(ARG_TOKENS) + list(STRUCTURE_TOKENS)
    if list(manifest.get("token_ids", {})) != expected_tokens:
        raise ValueError(f"unexpected manifest token order for seed {seed}")
    inverse = inverse_permutation([int(value) for value in manifest["permutation"]])
    test = manifest.get("test", [])
    if len(test) != 1984:
        raise ValueError(f"test row count is not 1984 for seed {seed}")
    if sum(row.get("order") == "natural" for row in test) != 992:
        raise ValueError(f"natural test count is not 992 for seed {seed}")
    if sum(row.get("order") == "reverse" for row in test) != 992:
        raise ValueError(f"reverse test count is not 992 for seed {seed}")
    for row in test:
        if len(row.get("tokens", [])) != 5 or len(row.get("token_ids", [])) != 5:
            raise ValueError(f"test row is not five real tokens for seed {seed}")
        lower_arg = arg_name(inverse[int(row["lower"])])
        forbidden_arg = arg_name(inverse[int(row["forbidden"])])
        expected_tokens = (
            ["OP_X", lower_arg, "LINK", "OP_Y", forbidden_arg]
            if row["order"] == "natural"
            else ["OP_Y", forbidden_arg, "LINK", "OP_X", lower_arg]
        )
        if row["tokens"] != expected_tokens:
            raise ValueError(f"unexpected test token structure for seed {seed}")
        expected_ids = [int(manifest["token_ids"][token]) for token in expected_tokens]
        if [int(value) for value in row["token_ids"]] != expected_ids:
            raise ValueError(f"test token ids do not match token names for seed {seed}")
    for order in ORDERS:
        pairs = [(int(row["lower"]), int(row["forbidden"])) for row in test if row["order"] == order]
        if len(set(pairs)) != 992 or any(lower == forbidden for lower, forbidden in pairs):
            raise ValueError(f"test pair catalog is invalid for {seed}/{order}")
    return manifest, path


def load_frozen_seed(seed: int, manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    stage_a_path = CAMPAIGN / f"nb5_rcsep_a_{seed}" / "stage_a.pt"
    stage_a_results_path = CAMPAIGN / f"nb5_rcsep_a_{seed}" / "results.json"
    for path in (stage_a_path, stage_a_results_path):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen RCSEP-A source for seed {seed}: {path}")
    stage_a_results = json.loads(stage_a_results_path.read_text(encoding="utf-8"))
    stage_a_sha = sha256_file(stage_a_path)
    if stage_a_results.get("checkpoint", {}).get("sha256") != stage_a_sha:
        raise ValueError(f"RCSEP-A result/checkpoint hash mismatch for seed {seed}")
    payload = torch.load(stage_a_path, map_location="cpu", weights_only=False)
    encoder = NB5CoreEncoder(manifest, seed)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "stage_a_path": stage_a_path,
        "stage_a_results_path": stage_a_results_path,
        "stage_a_results": stage_a_results,
        "encoder": encoder,
        "source_hashes": {
            "stage_a_checkpoint": source_record(stage_a_path),
            "stage_a_results": source_record(stage_a_results_path),
            "manifest_v2": source_record(manifest_path),
        },
    }


def decoder_codebook(executor: torch.nn.Module) -> torch.Tensor:
    class_ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)
    return executor.token_embedding(class_ids).detach()


def decode_state(executor: torch.nn.Module, codebook: torch.Tensor, state: torch.Tensor, target_value: int) -> dict[str, Any]:
    zeros = torch.zeros((1, state.shape[-1]), dtype=state.dtype, device=state.device)
    logits = executor.register_decoder(torch.cat((state.unsqueeze(0), zeros), dim=-1), codebook)[0]
    decoded = int(logits.argmax().item())
    canonical_logits = executor.register_decoder(codebook[decoded].unsqueeze(0), codebook)[0]
    canonical = int(canonical_logits.argmax().item())
    top = logits.topk(2).values
    return {
        "decoded_value": decoded,
        "canonical_value": canonical,
        "target_score": finite(logits[target_value].item()),
        "winner_score": finite(logits[decoded].item()),
        "winner_margin": finite((top[0] - top[1]).item()),
    }


def margin_evidence(tokens: list[str], scores: torch.Tensor, target_position: int, other_position: int) -> dict[str, Any]:
    real_positions = range(len(tokens))
    structural_positions = [position for position in real_positions if tokens[position] in STRUCTURE_TOKENS]
    if not structural_positions:
        raise ValueError("case has no structural positions")
    arg_winner = other_position
    structure_winner = first_argmax(scores, structural_positions)
    target_score = finite(scores[target_position].item())
    arg_winner_score = finite(scores[arg_winner].item())
    structure_winner_score = finite(scores[structure_winner].item())
    return {
        "target_arg": tokens[target_position],
        "target_position": target_position,
        "target_score": target_score,
        "arg_competitor": {
            "token": tokens[arg_winner],
            "winner_type": "ARG",
            "winner_position": arg_winner,
            "winner_score": arg_winner_score,
            "margin": finite(target_score - arg_winner_score),
            "positive": bool(target_score - arg_winner_score > 0.0),
        },
        "structure_competitor": {
            "token": tokens[structure_winner],
            "winner_type": token_type(tokens[structure_winner]),
            "winner_position": structure_winner,
            "winner_score": structure_winner_score,
            "margin": finite(target_score - structure_winner_score),
            "positive": bool(target_score - structure_winner_score > 0.0),
        },
    }


def pointer_evidence(tokens: list[str], scores: torch.Tensor, target_position: int) -> dict[str, Any]:
    winner_position = first_argmax(scores, range(len(tokens)))
    target_score = finite(scores[target_position].item())
    winner_score = finite(scores[winner_position].item())
    return {
        "target_token": tokens[target_position],
        "target_position": target_position,
        "target_score": target_score,
        "winner_token": tokens[winner_position],
        "winner_type": token_type(tokens[winner_position]),
        "winner_position": winner_position,
        "winner_score": winner_score,
        "margin": finite(target_score - winner_score),
        "pass": winner_position == target_position,
        "selection_rule": "first/min-position argmax over every real position 0 <= t < length",
    }


def failure_record(
    seed: int,
    order: str,
    lower: int,
    forbidden: int,
    role: str,
    target_arg: str,
    failure_kind: str,
    evidence: dict[str, Any],
    decoded: dict[str, Any],
) -> dict[str, Any]:
    if failure_kind in ("ARG-vs-ARG margin", "ARG-vs-structure margin"):
        winner = evidence["arg_competitor"] if failure_kind == "ARG-vs-ARG margin" else evidence["structure_competitor"]
        return {
            "seed": seed,
            "order": order,
            "L": lower,
            "F": forbidden,
            "role": role,
            "target_ARG": target_arg,
            "failure_kind": failure_kind,
            "winning_token": winner["token"],
            "winner_type": winner["winner_type"],
            "target_score": evidence["target_score"],
            "winner_score": winner["winner_score"],
            "margin": winner["margin"],
            "winner_position": winner["winner_position"],
        }
    pointer = evidence["pointer"]
    return {
        "seed": seed,
        "order": order,
        "L": lower,
        "F": forbidden,
        "role": role,
        "target_ARG": target_arg,
        "failure_kind": failure_kind,
        "winning_token": pointer["winner_token"],
        "winner_type": pointer["winner_type"],
        "target_score": pointer["target_score"],
        "winner_score": pointer["winner_score"],
        "margin": pointer["margin"],
        "winner_position": pointer["winner_position"],
        "decoded_value": decoded["decoded_value"],
        "canonical_value": decoded["canonical_value"],
        "decoder_winner_score": decoded["winner_score"],
        "decoder_margin": decoded["winner_margin"],
    }


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "min": finite(tensor.min().item()),
        "p01": finite(torch.quantile(tensor, 0.01).item()),
        "p05": finite(torch.quantile(tensor, 0.05).item()),
        "p25": finite(torch.quantile(tensor, 0.25).item()),
        "p50": finite(torch.quantile(tensor, 0.50).item()),
        "p75": finite(torch.quantile(tensor, 0.75).item()),
        "p95": finite(torch.quantile(tensor, 0.95).item()),
        "p99": finite(torch.quantile(tensor, 0.99).item()),
        "max": finite(tensor.max().item()),
    }


def evaluate_case(
    seed: int,
    order: str,
    case: dict[str, Any],
    encoder: NB5CoreEncoder,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens = list(case["tokens"])
    length = len(tokens)
    ids = torch.tensor([[encoder.vocab.encode_name(token) for token in tokens]], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.no_grad():
        output = encoder(ids, lengths, return_details=True)
        sf = output[6][0]
        sa = output[7][0]
        value_vectors = output[8][0]

        lower = int(case["lower"])
        forbidden = int(case["forbidden"])
        target_args = {"FLOOR": arg_name(int(case["lower_arg_index"])), "AVOID": arg_name(int(case["forbidden_arg_index"]))}
        floor_target = tokens.index(target_args["FLOOR"])
        avoid_target = tokens.index(target_args["AVOID"])
        other_floor = tokens.index(target_args["AVOID"])
        other_avoid = tokens.index(target_args["FLOOR"])
        pointer_f = first_argmax(sf, range(length))
        pointer_a = first_argmax(sa, range(length))
        floor_pointer = pointer_evidence(tokens, sf, floor_target)
        avoid_pointer = pointer_evidence(tokens, sa, avoid_target)
        floor_margin = margin_evidence(tokens, sf, floor_target, other_floor)
        avoid_margin = margin_evidence(tokens, sa, avoid_target, other_avoid)
        floor_decode = decode_state(executor, codebook, value_vectors[pointer_f], lower)
        avoid_decode = decode_state(executor, codebook, value_vectors[pointer_a], forbidden)

    floor_decode["target_value"] = lower
    avoid_decode["target_value"] = forbidden
    floor_decode["decode_pass"] = floor_decode["decoded_value"] == lower
    avoid_decode["decode_pass"] = avoid_decode["decoded_value"] == forbidden
    floor_decode["canonical_pass"] = floor_decode["canonical_value"] == lower
    avoid_decode["canonical_pass"] = avoid_decode["canonical_value"] == forbidden
    floor_decode["raw_canon_equal"] = floor_decode["decoded_value"] == floor_decode["canonical_value"]
    avoid_decode["raw_canon_equal"] = avoid_decode["decoded_value"] == avoid_decode["canonical_value"]

    row = {
        "seed": seed,
        "order": order,
        "case_index": int(case["case_index"]),
        "digest": case.get("digest"),
        "L": lower,
        "F": forbidden,
        "tokens": tokens,
        "token_ids": [int(value) for value in case["token_ids"]],
        "length": length,
        "real_positions": list(range(length)),
        "target_ARGS": target_args,
        "raw_scores": {"s_F": [finite(value) for value in sf[:length].tolist()], "s_A": [finite(value) for value in sa[:length].tolist()]},
        "pointer_F": pointer_f == floor_target,
        "pointer_A": pointer_a == avoid_target,
        "pointer_index_F": pointer_f,
        "pointer_index_A": pointer_a,
        "pointer_evidence": {"FLOOR": floor_pointer, "AVOID": avoid_pointer},
        "margins": {"FLOOR": floor_margin, "AVOID": avoid_margin},
        "selected_state_formula": "W_v(e_j) exactly; no gate weighting",
        "selected_states": {"FLOOR": [finite(value) for value in value_vectors[pointer_f].tolist()], "AVOID": [finite(value) for value in value_vectors[pointer_a].tolist()]},
        "decode_evidence": {"FLOOR": floor_decode, "AVOID": avoid_decode},
        "decode_F": floor_decode["decode_pass"],
        "decode_A": avoid_decode["decode_pass"],
        "RAW": bool(floor_decode["decode_pass"] and avoid_decode["decode_pass"]),
        "CANON": bool(floor_decode["canonical_pass"] and avoid_decode["canonical_pass"]),
        "RAW==CANON": bool(floor_decode["raw_canon_equal"] and avoid_decode["raw_canon_equal"]),
    }
    failures: list[dict[str, Any]] = []
    for role, margin, pointer, decoded, target_arg in (
        ("FLOOR", floor_margin, floor_pointer, floor_decode, target_args["FLOOR"]),
        ("AVOID", avoid_margin, avoid_pointer, avoid_decode, target_args["AVOID"]),
    ):
        for metric, label in (("arg_competitor", "ARG-vs-ARG margin"), ("structure_competitor", "ARG-vs-structure margin")):
            if not margin[metric]["positive"]:
                failures.append(failure_record(seed, order, lower, forbidden, role, target_arg, label, {**margin, "pointer": pointer}, decoded))
        if not pointer["pass"]:
            failures.append(failure_record(seed, order, lower, forbidden, role, target_arg, "pointer failure", {**margin, "pointer": pointer}, decoded))
        if not decoded["decode_pass"]:
            failures.append(failure_record(seed, order, lower, forbidden, role, target_arg, "decode failure", {**margin, "pointer": pointer}, decoded))
    if not row["CANON"]:
        for role, decoded, target_arg in (("FLOOR", floor_decode, target_args["FLOOR"]), ("AVOID", avoid_decode, target_args["AVOID"])):
            if not decoded["canonical_pass"]:
                failures.append(failure_record(seed, order, lower, forbidden, role, target_arg, "canonical failure", {**(floor_margin if role == "FLOOR" else avoid_margin), "pointer": floor_pointer if role == "FLOOR" else avoid_pointer}, decoded))
    return row, failures


def order_summary(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 992:
        raise ValueError(f"expected 992 rows, got {len(rows)}")
    margin_values = {
        role: {
            metric: [float(row["margins"][role][metric]["margin"]) for row in rows]
            for metric in ("arg_competitor", "structure_competitor")
        }
        for role in ("FLOOR", "AVOID")
    }
    return {
        "cases": len(rows),
        **{field: sum(bool(row[field]) for row in rows) for field in POINTER_FIELDS},
        "exigency": {
            role: {
                "M_ARG_positive": sum(row["margins"][role]["arg_competitor"]["positive"] for row in rows),
                "M_STRUCT_positive": sum(row["margins"][role]["structure_competitor"]["positive"] for row in rows),
                "denominator": 992,
            }
            for role in ("FLOOR", "AVOID")
        },
        "margin_quantiles": {
            role: {metric: quantiles(values) for metric, values in metrics.items()}
            for role, metrics in margin_values.items()
        },
        "failure_counts": {
            kind: sum(item["failure_kind"] == kind for item in failures)
            for kind in ("ARG-vs-ARG margin", "ARG-vs-structure margin", "pointer failure", "decode failure", "canonical failure")
        },
        "failing_case_count": len({(item["L"], item["F"]) for item in failures}),
    }


def reverse_outcome_equality(natural: list[dict[str, Any]], reverse: list[dict[str, Any]]) -> dict[str, Any]:
    fields = POINTER_FIELDS

    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[bool, ...]]:
        return {(int(row["L"]), int(row["F"])): tuple(bool(row[field]) for field in fields) for row in rows}

    left = keyed(natural)
    right = keyed(reverse)
    keys = sorted(set(left) | set(right))
    mismatches = [
        {"L": key[0], "F": key[1], "natural": left.get(key), "reverse": right.get(key)}
        for key in keys
        if left.get(key) != right.get(key)
    ]
    return {"key": "(lower,forbidden)", "outcome_fields": list(fields), "equal": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches}


def artifact_hash(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return sha256_bytes(payload)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite new audit output: {OUTPUT_PATH}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        codebook = decoder_codebook(executor)

    per_seed: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for seed in SEEDS:
        manifest, manifest_path = validate_manifest(seed)
        inverse = inverse_permutation([int(value) for value in manifest["permutation"]])
        seed_data = load_frozen_seed(seed, manifest, manifest_path)
        grouped: dict[str, list[dict[str, Any]]] = {order: [] for order in ORDERS}
        seed_failures: dict[str, list[dict[str, Any]]] = {order: [] for order in ORDERS}
        for order in ORDERS:
            cases = [row for row in manifest["test"] if row["order"] == order]
            for case_index, original_case in enumerate(cases):
                lower = int(original_case["lower"])
                forbidden = int(original_case["forbidden"])
                case = {**original_case, "case_index": case_index, "lower_arg_index": inverse[lower], "forbidden_arg_index": inverse[forbidden]}
                row, failures = evaluate_case(seed, order, case, seed_data["encoder"], executor, codebook)
                grouped[order].append(row)
                seed_failures[order].extend(failures)
        equality = reverse_outcome_equality(grouped["natural"], grouped["reverse"])
        seed_entry = {
            "seed": seed,
            "source_hashes": seed_data["source_hashes"],
            "orders": {
                order: {"summary": order_summary(grouped[order], seed_failures[order]), "failures": seed_failures[order], "cases": grouped[order]}
                for order in ORDERS
            },
            "exact_reverse_outcome_equality": equality,
        }
        per_seed.append(seed_entry)
        for order in ORDERS:
            all_failures.extend(seed_failures[order])

    flat_failures = [item for group in all_failures for item in group]
    all_summaries = [seed["orders"][order]["summary"] for seed in per_seed for order in ORDERS]
    all_requirements = all(
        summary["cases"] == 992
        and all(summary[field] == 992 for field in POINTER_FIELDS)
        and all(summary["exigency"][role][metric] == 992 for role in ("FLOOR", "AVOID") for metric in ("M_ARG_positive", "M_STRUCT_positive"))
        for summary in all_summaries
    )
    reverse_equal = all(seed["exact_reverse_outcome_equality"]["equal"] for seed in per_seed)
    passed = bool(all_requirements and reverse_equal)

    source_hashes = {
        "audit_script": source_record(SCRIPT_DIR / Path(__file__).name),
        "nb5_fresh.py": source_record(SCRIPT_DIR / "nb5_fresh.py"),
        "ctrl2_common.py": source_record(SCRIPT_DIR / "ctrl2_common.py"),
        "train_u0c_c1_joint.py": source_record(SCRIPT_DIR / "train_u0c_c1_joint.py"),
        "unified.py": source_record(ROOT / "t1_trainability" / "unified.py"),
        "executor_base_checkpoint": source_record(BASE_CHECKPOINT),
        "executor_u0a_checkpoint": source_record(U0A_CHECKPOINT),
        "executor_c0_checkpoint": source_record(C0_CHECKPOINT),
        "rcsep_a_train_script": source_record(SCRIPT_DIR / "train_nb5_rcsep_a.py"),
    }
    artifact: dict[str, Any] = {
        "status": "diagnostic_complete",
        "classification": "pass" if passed else "fail",
        "task": "T2-NOBYPASS-2-RCSEP-C-NOGATE-HARDPTR-ALG",
        "authorization": {
            "id": "T2-NOBYPASS-2-RCSEP-C-NOGATE-HARDPTR-ALG",
            "read_only_diagnostics": True,
            "frozen_inputs": "RCSEP-A stage_a.pt, frozen executor, v2 manifests",
            "joint_training_or_sequences_added": False,
        },
        "executive_summary": {
            "result": "PASS only when every count, margin exigency, and reverse outcome equality passes" if passed else "FAIL: one or more required counts, margins, or reverse outcome equalities failed",
            "all_count_and_margin_requirements": all_requirements,
            "reverse_outcome_equality_all_seeds": reverse_equal,
            "failure_evidence_records": len(flat_failures),
            "no_pass_inferred_from_partial_counts": True,
        },
        "protocol": {
            "no_gate_loaded": True,
            "no_gate_used": True,
            "gate_path": None,
            "no_numeric_confidence": True,
            "tau_used": False,
            "no_training": True,
            "no_optimizer": True,
            "no_checkpoint_changes": True,
            "no_manifest_changes": True,
            "no_existing_artifact_changes": True,
            "no_eligibility_filter": True,
            "no_value_eligibility": True,
            "real_length_only": True,
            "padding_excluded_solely_by_real_length": True,
            "raw_score_source": "encoder return_details raw sf/sa at every real token position",
            "pointer_F": "first/min-position argmax raw s_F over every real t < length",
            "pointer_A": "first/min-position argmax raw s_A over every real t < length",
            "selected_state": "W_v(e_j) exactly; no gate weighting",
            "decoder": "frozen register_decoder with frozen VALUE codebook; canonical round-trip uses decoded codebook vector",
        },
        "scope": {"seeds": list(SEEDS), "orders": list(ORDERS), "cases_per_seed_order": 992, "total_cases": len(SEEDS) * len(ORDERS) * 992, "roles": ["FLOOR", "AVOID"]},
        "source_hashes": source_hashes,
        "per_seed": per_seed,
        "failure_evidence": flat_failures,
        "artifacts": {"script": str(Path(__file__).resolve()), "consolidated_results": str(OUTPUT_PATH)},
        "next_recommended": "Treat 6801-6805 as development/diagnostic evidence; validate any follow-up on fresh 6901-6905 domains.",
        "risks": [
            "This is frozen diagnostic evidence, not fresh validation of a new domain.",
            "Positive margin exigency is stricter than pointer success because ties or structural wins remain separately visible.",
            "Reverse equality compares exact outcome booleans by (lower,forbidden), not positional token locations.",
        ],
        "skill_resolution": {
            "graphify": "Read before work; existing .codegraph targeted source exploration completed.",
            "work_unit_commits": "Single focused read-only audit work unit; no commit requested.",
            "shared": "Read before work.",
        },
    }
    artifact["artifact_self_hash"] = artifact_hash(artifact)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "classification": artifact["classification"], "artifact": str(OUTPUT_PATH), "artifact_self_hash": artifact["artifact_self_hash"], "failure_evidence": len(flat_failures)}, sort_keys=True))


if __name__ == "__main__":
    main()

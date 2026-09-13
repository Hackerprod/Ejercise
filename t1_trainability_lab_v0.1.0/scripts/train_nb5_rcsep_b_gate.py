"""Train and audit the authorized RCSEP-B numeric-confidence gate.

Only the 35 named domain-token gate rows plus the legacy NOOP row are scored.
The RCSEP-A encoder parameters and all decoder parameters remain frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from nb5_fresh import NB5GateOnlyEncoder


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign"
MANIFEST_ROOT = CAMPAIGN / "nb5_manifests"
SCRIPT_PATH = Path(__file__).resolve()
SEEDS = (6801, 6802, 6803, 6804, 6805)
UPDATES = 5000
ALPHA = 2.0
TAU = 3.0545
INITIAL_LR = 1.0e-3
FINAL_LR = 1.0e-5
DOMAIN_TOKENS = [f"ARG_{index:02d}" for index in range(32)] + ["OP_X", "OP_Y", "LINK"]
LEGACY_TOKENS = DOMAIN_TOKENS + ["NOOP"]
POINTER_FIELDS = ("pointer_F", "pointer_A", "decode_F", "decode_A", "RAW", "CANON", "RAW==CANON")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def finite(value: float) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"non-finite metric: {result}")
    return result


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    payload = bytearray()
    payload.extend(str(value.dtype).encode("ascii"))
    payload.extend(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    payload.extend(value.view(torch.uint8).numpy().tobytes())
    return sha256_bytes(bytes(payload))


def state_digest(state: dict[str, torch.Tensor]) -> str:
    records = {name: tensor_digest(state[name]) for name in sorted(state)}
    return sha256_bytes(json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must have equal nonzero length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = sum((a - left_mean) ** 2 for a in left) ** 0.5
    right_norm = sum((b - right_mean) ** 2 for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return finite(numerator / (left_norm * right_norm))


def load_manifest(seed: int) -> tuple[dict[str, Any], Path]:
    path = MANIFEST_ROOT / f"manifest_{seed}_v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "NB5-fresh-lexical-cipher-v1":
        raise ValueError(f"unexpected manifest schema for seed {seed}")
    if int(manifest.get("domain_seed")) != seed:
        raise ValueError(f"manifest domain seed mismatch for seed {seed}")
    if len(manifest.get("test", [])) != 1984:
        raise ValueError(f"unexpected test row count for seed {seed}")
    if sum(row["order"] == "natural" for row in manifest["test"]) != 992:
        raise ValueError(f"natural test count is not 992 for seed {seed}")
    if sum(row["order"] == "reverse" for row in manifest["test"]) != 992:
        raise ValueError(f"reverse test count is not 992 for seed {seed}")
    if list(manifest["token_ids"]) != [f"ARG_{index:02d}" for index in range(32)] + ["LINK", "OP_X", "OP_Y"]:
        raise ValueError(f"unexpected token order for seed {seed}")
    return manifest, path


def load_gate(seed: int, manifest: dict[str, Any]) -> tuple[NB5GateOnlyEncoder, Path, dict[str, torch.Tensor], dict[str, str]]:
    checkpoint = CAMPAIGN / f"nb5_rcsep_a_{seed}" / "stage_a.pt"
    result_path = CAMPAIGN / f"nb5_rcsep_a_{seed}" / "results.json"
    if not checkpoint.exists() or not result_path.exists():
        raise FileNotFoundError(f"missing RCSEP-A source artifact for seed {seed}")
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint)
    if source_result.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError(f"RCSEP-A result/checkpoint hash mismatch for seed {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = NB5GateOnlyEncoder(payload["encoder"])
    core_before = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters() if name not in ("w_c", "b_c")}
    core_hashes_before = {name: tensor_digest(value) for name, value in core_before.items()}
    for name, parameter in encoder.named_parameters():
        if name not in ("w_c", "b_c"):
            parameter.requires_grad_(False)
    if any(parameter.requires_grad for name, parameter in encoder.named_parameters() if name not in ("w_c", "b_c")):
        raise RuntimeError("core parameter was not frozen")
    if not encoder.w_c.requires_grad or not encoder.b_c.requires_grad:
        raise RuntimeError("gate parameters are not trainable")
    if encoder.w_c.shape != (16,) or encoder.b_c.shape != torch.Size([]):
        raise RuntimeError("unexpected fresh gate parameter shapes")
    if not torch.equal(encoder.w_c.detach(), torch.zeros(16)) or not torch.equal(encoder.b_c.detach(), torch.zeros(())):
        raise RuntimeError("gate parameters were not freshly zero initialized")
    return encoder, checkpoint, core_before, core_hashes_before


def build_token_table(
    encoder: NB5GateOnlyEncoder,
    executor: torch.nn.Module,
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor, torch.Tensor]:
    ids = torch.tensor([encoder.vocab.encode_name(token) for token in LEGACY_TOKENS], dtype=torch.long)
    embeddings = encoder.embedding(ids)
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long))
        value_state = encoder.w_v(embeddings)
        logits = executor.register_decoder(torch.cat((value_state, torch.zeros_like(value_state)), -1), codebook)
        margins = logits.topk(2, dim=-1).values[:, 0] - logits.topk(2, dim=-1).values[:, 1]
        teacher = torch.sigmoid(ALPHA * (margins - TAU)).detach()
    row_by_token: dict[str, dict[str, Any]] = {}
    for position, token in enumerate(LEGACY_TOKENS):
        row_by_token[token] = {
            "token": token,
            "category": "ARG" if token.startswith("ARG_") else ("NOOP" if token == "NOOP" else "STRUCTURE"),
            "arg_index": int(token[4:]) if token.startswith("ARG_") else None,
            "internal_index": int(ids[position].item()),
            "margin": finite(margins[position].item()),
            "teacher_y": finite(teacher[position].item()),
        }
    return [row_by_token[token] for token in DOMAIN_TOKENS], row_by_token["NOOP"], embeddings, codebook


def train_gate(
    encoder: NB5GateOnlyEncoder,
    embeddings: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[float, float]:
    optimizer = torch.optim.AdamW([encoder.w_c, encoder.b_c], lr=INITIAL_LR, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    last_loss = torch.tensor(0.0)
    for step in range(1, UPDATES + 1):
        prediction = embeddings @ encoder.w_c + encoder.b_c
        loss = criterion(prediction, teacher)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite gate loss at step {step}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = INITIAL_LR + (FINAL_LR - INITIAL_LR) * (step - 1) / (UPDATES - 1)
        last_loss = loss.detach()
    with torch.no_grad():
        final_logits = embeddings @ encoder.w_c + encoder.b_c
        final_loss = criterion(final_logits, teacher)
    return finite(last_loss.item()), finite(final_loss.item())


def gate_rows(
    encoder: NB5GateOnlyEncoder,
    domain_rows: list[dict[str, Any]],
    noop_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = domain_rows + [noop_row]
    ids = torch.tensor([int(row["internal_index"]) for row in rows], dtype=torch.long)
    with torch.no_grad():
        confidence = torch.sigmoid(encoder.embedding(ids) @ encoder.w_c + encoder.b_c)
    for row, value in zip(rows, confidence.tolist()):
        row["learned_c"] = finite(value)
        row["hard_selected"] = bool(value > 0.5)
    domain = rows[: len(domain_rows)]
    arg_rows = [row for row in domain if row["category"] == "ARG"]
    structure_rows = [row for row in domain if row["category"] == "STRUCTURE"]
    min_arg = min(arg_rows, key=lambda row: (row["learned_c"], row["token"]))
    max_structure = max(structure_rows, key=lambda row: (row["learned_c"], row["token"]))
    b1 = {
        "domain_token_count": len(domain),
        "arg_token_count": len(arg_rows),
        "structure_token_count": len(structure_rows),
        "min_c_ARG": min_arg["learned_c"],
        "min_c_ARG_token": min_arg["token"],
        "min_c_ARG_gt_0.5": bool(min_arg["learned_c"] > 0.5),
        "max_c_structure": max_structure["learned_c"],
        "max_c_structure_token": max_structure["token"],
        "max_c_OP_X_OP_Y_LINK_lt_0.5": bool(max_structure["learned_c"] < 0.5),
        "pass": bool(min_arg["learned_c"] > 0.5 and max_structure["learned_c"] < 0.5),
        "noop_reported_separately": True,
    }
    return rows, b1


def b2_report(rows: list[dict[str, Any]], final_bce: float) -> dict[str, Any]:
    domain = rows[:35]
    arg_rows = [row for row in domain if row["category"] == "ARG"]
    structure_rows = [row for row in domain if row["category"] == "STRUCTURE"]
    b2 = {
        "domain_token_count": len(domain),
        "bce_final": final_bce,
        "teacher_vs_learned_correlation": pearson([row["teacher_y"] for row in domain], [row["learned_c"] for row in domain]),
        "min_teacher_ARG": min(arg_rows, key=lambda row: (row["teacher_y"], row["token"]))["teacher_y"],
        "min_teacher_ARG_token": min(arg_rows, key=lambda row: (row["teacher_y"], row["token"]))["token"],
        "max_teacher_structure": max(structure_rows, key=lambda row: (row["teacher_y"], row["token"]))["teacher_y"],
        "max_teacher_structure_token": max(structure_rows, key=lambda row: (row["teacher_y"], row["token"]))["token"],
        "pass": bool(len(domain) == 35 and all(torch.isfinite(torch.tensor(row["learned_c"])) for row in domain)),
        "correlation_method": "Pearson correlation over 35 named domain tokens",
    }
    return b2


def evaluate_order(
    encoder: NB5GateOnlyEncoder,
    executor: torch.nn.Module,
    manifest: dict[str, Any],
    codebook: torch.Tensor,
    order: str,
) -> list[dict[str, Any]]:
    inverse = {int(value): index for index, value in enumerate(manifest["permutation"])}
    rows: list[dict[str, Any]] = []
    for case in manifest["test"]:
        if case["order"] != order:
            continue
        token_ids = torch.tensor([[encoder.vocab.encode_name(token) for token in case["tokens"]]], dtype=torch.long)
        lengths = torch.tensor([len(case["tokens"])], dtype=torch.long)
        with torch.no_grad():
            output = encoder(token_ids, lengths, return_details=True)
            sf, sa, vt, gate = output[6][0], output[7][0], output[8][0], output[9][0]
            eligible = torch.where(gate > 0.5)[0]
            lower = int(case["lower"])
            forbidden = int(case["forbidden"])
            expected_floor_token = f"ARG_{inverse[lower]:02d}"
            expected_avoid_token = f"ARG_{inverse[forbidden]:02d}"
            if len(eligible):
                floor_pointer = int(eligible[torch.argmax(sf[eligible])].item())
                avoid_pointer = int(eligible[torch.argmax(sa[eligible])].item())
                floor_state = (vt[floor_pointer] * gate[floor_pointer]).unsqueeze(0)
                avoid_state = (vt[avoid_pointer] * gate[avoid_pointer]).unsqueeze(0)
                floor_logits = executor.register_decoder(torch.cat((floor_state, torch.zeros_like(floor_state)), -1), codebook)[0]
                avoid_logits = executor.register_decoder(torch.cat((avoid_state, torch.zeros_like(avoid_state)), -1), codebook)[0]
                decode_f = int(floor_logits.argmax().item())
                decode_a = int(avoid_logits.argmax().item())
                canonical_f = int(executor.register_decoder(codebook[decode_f].unsqueeze(0), codebook)[0].argmax().item())
                canonical_a = int(executor.register_decoder(codebook[decode_a].unsqueeze(0), codebook)[0].argmax().item())
                pointer_f_token = case["tokens"][floor_pointer]
                pointer_a_token = case["tokens"][avoid_pointer]
                row = {
                    "order": order,
                    "lower": lower,
                    "forbidden": forbidden,
                    "eligible_count": int(len(eligible)),
                    "eligible_tokens": [case["tokens"][int(index)] for index in eligible.tolist()],
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
                    "RAW==CANON": bool(decode_f == canonical_f and decode_a == canonical_a),
                }
            else:
                row = {
                    "order": order,
                    "lower": lower,
                    "forbidden": forbidden,
                    "eligible_count": 0,
                    "eligible_tokens": [],
                    "pointer_F_token": None,
                    "pointer_A_token": None,
                    "pointer_F": False,
                    "pointer_A": False,
                    "decode_F_value": None,
                    "decode_A_value": None,
                    "decode_F": False,
                    "decode_A": False,
                    "canonical_F_value": None,
                    "canonical_A_value": None,
                    "RAW": False,
                    "CANON": False,
                    "RAW==CANON": False,
                }
            rows.append(row)
    return rows


def order_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"cases": len(rows), **{field: sum(bool(row[field]) for row in rows) for field in POINTER_FIELDS}}


def reverse_equals_natural(natural: list[dict[str, Any]], reverse: list[dict[str, Any]]) -> bool:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[Any, ...]]:
        return {
            (int(row["lower"]), int(row["forbidden"])): tuple(row[field] for field in POINTER_FIELDS)
            for row in rows
        }

    return keyed(natural) == keyed(reverse)


def run_seed(seed: int, executor: torch.nn.Module) -> dict[str, Any]:
    manifest, manifest_path = load_manifest(seed)
    encoder, stage_a_path, core_before, core_hashes_before = load_gate(seed, manifest)
    domain_rows, noop_row, embeddings, codebook = build_token_table(encoder, executor)
    teacher = torch.tensor([row["teacher_y"] for row in domain_rows + [noop_row]], dtype=torch.float32)
    last_bce, final_bce = train_gate(encoder, embeddings, teacher)
    core_after = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters() if name not in ("w_c", "b_c")}
    core_hashes_after = {name: tensor_digest(value) for name, value in core_after.items()}
    core_tensor_checks = {name: {"before": core_hashes_before[name], "after": core_hashes_after[name], "unchanged": core_before[name].equal(core_after[name])} for name in sorted(core_before)}
    core_state_before = state_digest(core_before)
    core_state_after = state_digest(core_after)
    rows, b1 = gate_rows(encoder, domain_rows, noop_row)
    b2 = b2_report(rows, final_bce)
    natural_rows = evaluate_order(encoder, executor, manifest, codebook, "natural")
    reverse_rows = evaluate_order(encoder, executor, manifest, codebook, "reverse")
    natural_summary = order_summary(natural_rows)
    reverse_summary = order_summary(reverse_rows)
    all_metrics_992 = lambda summary: bool(summary["cases"] == 992 and all(summary[field] == 992 for field in POINTER_FIELDS))
    b3 = {
        "natural": natural_summary,
        "reverse": reverse_summary,
        "reverse_equals_natural": reverse_equals_natural(natural_rows, reverse_rows),
        "natural_all_metrics_992": all_metrics_992(natural_summary),
        "reverse_all_metrics_992": all_metrics_992(reverse_summary),
        "pass": bool(
            reverse_equals_natural(natural_rows, reverse_rows)
            and all_metrics_992(natural_summary)
            and all_metrics_992(reverse_summary)
        ),
        "criterion": "natural and reverse pointer/decode/RAW/CANON metrics must each be 992/992 and match exactly",
        "definitions": {
            "HARDPTR": "h=1[c>0.5] over valid sequence tokens",
            "pointer_F": "raw sf argmax over eligible HARDPTR tokens equals expected lower ARG token",
            "pointer_A": "raw sa argmax over eligible HARDPTR tokens equals expected forbidden ARG token",
            "decoder_state": "vt[pointer] * c[pointer], matching existing NB5 Stage B formal semantics",
            "decode_F": "decoder argmax equals lower VALUE",
            "decode_A": "decoder argmax equals forbidden VALUE",
            "RAW": "decode_F and decode_A",
            "CANON": "canonical decoder round-trip of each raw decoded VALUE equals expected lower/forbidden VALUE",
            "RAW==CANON": "raw decoded lower/forbidden VALUE ids equal their canonical round-trip ids",
            "reverse_requirement": "natural and reverse outcome maps must be identical",
        },
        "rows": {"natural": natural_rows, "reverse": reverse_rows},
    }
    output = CAMPAIGN / f"nb5_rcsep_b_gate_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    gate_checkpoint = output / "gate.pt"
    torch.save(
        {
            "w_c": encoder.w_c.detach(),
            "b_c": encoder.b_c.detach(),
            "stage_a_checkpoint_sha256": sha256_file(stage_a_path),
            "manifest_sha256": sha256_file(manifest_path),
            "alpha": ALPHA,
            "tau": TAU,
            "updates": UPDATES,
            "trainable_parameters": 17,
            "core_parameter_state_sha256_before": core_state_before,
            "core_parameter_state_sha256_after": core_state_after,
        },
        gate_checkpoint,
    )
    core_unchanged = bool(all(item["unchanged"] for item in core_tensor_checks.values()))
    b0_pass = bool(core_state_before == core_state_after and core_unchanged)
    all_gates_pass = bool(b0_pass and b1["pass"] and b2["pass"] and b3["pass"])
    result = {
        "status": "completed",
        "task": "T2-NOBYPASS-2-RCSEP-B-GATE",
        "seed": seed,
        "objective": {
            "z_t": "e_t @ w_c + b_c",
            "c_t": "sigmoid(z_t)",
            "teacher": "sigmoid(2 * (margin - 3.0545)) detached",
            "margin": "top1-top2 frozen decoder margin over [w_v(e_t), zeros32]",
            "loss": "BCEWithLogitsLoss(z_t, y_t)",
            "alpha": ALPHA,
            "tau": TAU,
            "tokens": "35 named domain tokens plus legacy NOOP training row",
            "joint_train_examples": 0,
            "ground_truth_arguments_in_forward": False,
        "teacher_in_inference": False,
        "new_loss": False,
        "value_token_conditioning": False,
        "gate_initialization": {"w_c": {"shape": [16], "initial": "zeros"}, "b_c": {"shape": [], "initial": "zeros"}},
            "optimizer": "AdamW only [w_c, b_c]",
            "learning_rate": {"initial": INITIAL_LR, "final": FINAL_LR, "schedule": "linear over 5000 updates"},
            "weight_decay": 0.0,
            "updates": UPDATES,
            "legacy_training_rows": {"ARG": 32, "OP_X": 1, "OP_Y": 1, "LINK": 1, "NOOP": 1, "total": 36},
        },
        "source_hashes": {
            "stage_a_checkpoint": source_record(stage_a_path),
            "manifest": source_record(manifest_path),
            "rcsep_a_results": source_record(CAMPAIGN / f"nb5_rcsep_a_{seed}" / "results.json"),
            "gate_training_script": source_record(SCRIPT_PATH),
        },
        "B0": {
            "core_parameter_state_sha256_before": core_state_before,
            "core_parameter_state_sha256_after": core_state_after,
            "core_state_hash_identical": bool(core_state_before == core_state_after),
            "every_core_tensor_unchanged": core_unchanged,
            "tensor_checks": core_tensor_checks,
            "pass": b0_pass,
        },
        "B1": b1,
        "B2": b2,
        "B3": b3,
        "domain_token_table": rows[:35],
        "legacy_noop": next(row for row in rows if row["token"] == "NOOP"),
        "training": {"last_update_bce_pre_step": last_bce, "final_bce_post_training": final_bce},
        "artifacts": {"stage_a_source": str(stage_a_path), "gate": str(gate_checkpoint), "results": str(output / "results.json")},
        "gate_checkpoint_sha256": sha256_file(gate_checkpoint),
        "special_seed_6804_ARG_07_equivalent": next(row for row in rows if row["token"] == "ARG_07") if seed == 6804 else None,
        "all_required_gates_pass": all_gates_pass,
        "risks": [
            "RCSEP-A seeds remain development/diagnostic evidence; fresh validation requires a new battery.",
            "B1 is a requested threshold result and may fail; no seed-specific adjustment is applied.",
        ],
    }
    result["status"] = "completed_pass" if all_gates_pass else "completed_with_gate_failures"
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    executor = load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    results = [run_seed(seed, executor) for seed in SEEDS]
    audit_dir = CAMPAIGN / "nb5_rcsep_b_gate_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "completed_pass" if all(result["all_required_gates_pass"] for result in results) else "completed_with_gate_failures",
        "task": "T2-NOBYPASS-2-RCSEP-B-GATE",
        "objective": results[0]["objective"],
        "training_scope": "five frozen RCSEP-A cores; no Stage A rerun; no legacy Stage B output modification",
        "per_seed": [
            {
                "seed": result["seed"],
                "status": result["status"],
                "all_required_gates_pass": result["all_required_gates_pass"],
                "gate_checkpoint_sha256": result["gate_checkpoint_sha256"],
                "B0": result["B0"],
                "B1": result["B1"],
                "B2": result["B2"],
                "B3": {"natural": result["B3"]["natural"], "reverse": result["B3"]["reverse"], "reverse_equals_natural": result["B3"]["reverse_equals_natural"], "pass": result["B3"]["pass"]},
                "artifacts": result["artifacts"],
            }
            for result in results
        ],
        "artifacts": {"per_seed_results": [result["artifacts"]["results"] for result in results], "audit": str(audit_dir / "results.json")},
        "next_recommended": "Treat 6801-6805 as development/diagnostic evidence; validate any follow-up on fresh 6901-6905 domains.",
        "risks": ["A failed B1 threshold is reported as a result, never repaired or relabeled PASS."],
        "skill_resolution": {"graphify": "Targeted filesystem inspection after existing CodeGraph query did not expose requested symbols.", "work_unit_commits": "Single focused implementation/audit work unit; no commit requested.", "shared": "Read before work."},
    }
    placeholder = "__SELF_HASH__"
    artifact["artifact_self_hash"] = placeholder
    artifact_hash = sha256_bytes((json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    artifact["artifact_self_hash"] = artifact_hash
    audit_path = audit_dir / "results.json"
    audit_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "audit": str(audit_path), "artifact_sha256": artifact_hash, "seeds": {str(result["seed"]): result["status"] for result in results}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

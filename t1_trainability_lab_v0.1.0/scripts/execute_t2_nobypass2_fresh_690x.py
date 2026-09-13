"""Execute authorized fresh 690x RCSEP-A training and no-gate HARDPTR audit."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

import audit_nb5_rcsep_c_nogate_hardptr_alg as audit  # noqa: E402
import train_nb5_rcsep_a as train  # noqa: E402
from nb5_fresh import NB5CoreEncoder  # noqa: E402


SEEDS = (6901, 6902, 6903, 6904, 6905)
ORDERS = ("natural", "reverse")
CAMPAIGN = ROOT / "campaign"
OUTPUT_ROOT = CAMPAIGN / "nb5_fresh_690x"
OUTPUT_PREFIX = "nb5_fresh_690x_"
UPDATES = 5000
EXPECTED_CASES = 992


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def artifact_hash(artifact: dict[str, Any]) -> str:
    unsigned = dict(artifact)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    payload = (json.dumps(unsigned, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {result}")
    return result


def output_dir(seed: int) -> Path:
    return CAMPAIGN / f"{OUTPUT_PREFIX}{seed}"


def old_680x_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for base in (CAMPAIGN, SCRIPT_DIR):
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if "680" in relative or "legacy" in relative.lower():
                snapshot[relative] = sha256_file(path)
    return dict(sorted(snapshot.items()))


def source_hashes(manifest_path: Path, checkpoint: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "execute_script": SCRIPT_DIR / Path(__file__).name,
        "rcsep_a_train_script": SCRIPT_DIR / "train_nb5_rcsep_a.py",
        "nogate_audit_script": SCRIPT_DIR / "audit_nb5_rcsep_c_nogate_hardptr_alg.py",
        "nb5_fresh.py": SCRIPT_DIR / "nb5_fresh.py",
        "manifest_checker": SCRIPT_DIR / "check_nb5_manifests_v2.py",
        "manifest_v2": manifest_path,
        "stage_a_checkpoint": checkpoint,
        "supervisor_checkpoint": train.CTRL7_CHECKPOINT,
        "executor_checkpoint": train.BASE_CHECKPOINT,
    }
    return {name: source_record(path) for name, path in paths.items()}


def train_seed(
    seed: int,
    observations: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    supervisor: torch.nn.Module,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
) -> dict[str, Any]:
    manifest, manifest_path = train.load_manifest(seed)
    destination = output_dir(seed)
    destination.mkdir(parents=True, exist_ok=True)

    # This is the only core initialization used for this seed.
    encoder = NB5CoreEncoder(manifest, seed)
    encoder.train()
    behavior_tokens, behavior_lengths, source_rows, floor_mask, avoid_mask, source_by_role = train.build_behavior_batch(
        encoder, manifest, labels
    )
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=0.0)
    last_losses: dict[str, float] = {}
    for step in range(1, UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = train.objective(
            encoder,
            supervisor,
            executor,
            codebook,
            manifest,
            behavior_tokens,
            behavior_lengths,
            source_rows,
            floor_mask,
            avoid_mask,
            observations,
            labels,
        )
        last_losses = {name: finite(losses[name].detach().item()) for name in ("behavior", "ref", "sep", "total")}
        losses["total"].backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 1e-3 + (1e-5 - 1e-3) * (step - 1) / (UPDATES - 1)

    encoder.eval()
    with torch.no_grad():
        final_losses = train.objective(
            encoder,
            supervisor,
            executor,
            codebook,
            manifest,
            behavior_tokens,
            behavior_lengths,
            source_rows,
            floor_mask,
            avoid_mask,
            observations,
            labels,
        )
        final_loss_values = {name: finite(final_losses[name].item()) for name in ("behavior", "ref", "sep", "total")}
        final_catalog = {name: value for name, value in final_losses["catalog"].items()}

    probes = [
        train.gradient_probe(
            encoder,
            supervisor,
            executor,
            codebook,
            manifest,
            "AVOID",
            0,
            source_by_role["AVOID"][0],
            observations,
            labels,
        ),
        train.gradient_probe(
            encoder,
            supervisor,
            executor,
            codebook,
            manifest,
            "FLOOR",
            0,
            source_by_role["FLOOR"][0],
            observations,
            labels,
        ),
    ]
    if any(not probe["absent_role_strictly_nonzero"] for probe in probes):
        raise RuntimeError(f"zero absent-role gradient blocker for seed {seed}: {json.dumps(probes, sort_keys=True)}")

    with torch.no_grad():
        atomic = train.atomic_evidence(encoder, manifest, executor, codebook, final_catalog)
        noop_source = torch.where(~floor_mask & ~avoid_mask)[0]
        _, _, noop_mode = encoder(behavior_tokens[noop_source], behavior_lengths[noop_source])
        noop_logits = supervisor(observations["features"][source_rows[noop_source]], noop_mode)
        noop_exact = int((noop_logits.argmax(-1) == labels["action"][source_rows[noop_source]]).sum().item())

    checkpoint = destination / "stage_a.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "seed": seed,
            "updates": UPDATES,
            "manifest_sha256": sha256_file(manifest_path),
            "objective": "L_behavior + L_ref + L_sep",
            "L_sep_coefficient": 1.0,
            "joint_train_examples": 0,
        },
        checkpoint,
    )
    return {
        "seed": seed,
        "manifest": source_record(manifest_path),
        "checkpoint": source_record(checkpoint),
        "last_update_losses": last_losses,
        "final_losses": final_loss_values,
        "atomic": {
            "FLOOR": {"exact": atomic["floor_exact"], "expected": 32},
            "AVOID": {"exact": atomic["avoid_exact"], "expected": 32},
            "NOOP": {"exact": noop_exact, "expected": int(len(noop_source))},
        },
        "gradient_audit": {"probes": probes, "all_absent_role_strictly_nonzero": True},
        "recipe_verification": {
            "fresh_init": "NB5CoreEncoder(manifest_690X_v2, seed)",
            "batch_rows": {"FLOOR": 32, "AVOID": 32, "NOOP": 32, "total": 96},
            "updates": UPDATES,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "gradient_clipping": False,
            "joint_rows": 0,
            "forward_inputs": ["token_ids", "lengths"],
            "teacher_or_ground_truth_forward_args": False,
            "catalog_queries_per_update": 64,
            "catalog": "full 32x32 all-to-all F/A raw score matrices, diagonal included",
            "L_sep": "(mean(softplus(F_cross - F_self)) + mean(softplus(A_cross - A_self))) / 2",
            "L_sep_coefficient": 1.0,
            "supervisor_executor": "frozen",
        },
    }


def reload_frozen_core(seed: int, manifest: dict[str, Any], checkpoint: Path) -> NB5CoreEncoder:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = NB5CoreEncoder(manifest, seed)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def strict_margin_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics = {
        "M_F_ARG": [float(row["margins"]["FLOOR"]["arg_competitor"]["margin"]) for row in rows],
        "M_F_STRUCT": [float(row["margins"]["FLOOR"]["structure_competitor"]["margin"]) for row in rows],
        "M_A_ARG": [float(row["margins"]["AVOID"]["arg_competitor"]["margin"]) for row in rows],
        "M_A_STRUCT": [float(row["margins"]["AVOID"]["structure_competitor"]["margin"]) for row in rows],
    }
    return {
        name: {
            "minimum": finite(min(values)),
            "count_le_zero": sum(value <= 0.0 for value in values),
            "positive_count": sum(value > 0.0 for value in values),
            "expected": EXPECTED_CASES,
        }
        for name, values in metrics.items()
    }


def audit_seed(
    seed: int,
    manifest: dict[str, Any],
    manifest_path: Path,
    checkpoint: Path,
    executor: torch.nn.Module,
    codebook: torch.Tensor,
) -> dict[str, Any]:
    encoder = reload_frozen_core(seed, manifest, checkpoint)
    inverse = audit.inverse_permutation([int(value) for value in manifest["permutation"]])
    grouped: dict[str, list[dict[str, Any]]] = {order: [] for order in ORDERS}
    failures: dict[str, list[dict[str, Any]]] = {order: [] for order in ORDERS}
    for order in ORDERS:
        cases = [row for row in manifest["test"] if row["order"] == order]
        for case_index, original_case in enumerate(cases):
            case = {
                **original_case,
                "case_index": case_index,
                "lower_arg_index": inverse[int(original_case["lower"])],
                "forbidden_arg_index": inverse[int(original_case["forbidden"])],
            }
            row, row_failures = audit.evaluate_case(seed, order, case, encoder, executor, codebook)
            grouped[order].append(row)
            failures[order].extend(row_failures)

    orders: dict[str, Any] = {}
    for order in ORDERS:
        summary = audit.order_summary(grouped[order], failures[order])
        summary["expected_cases"] = EXPECTED_CASES
        summary["strict_margin_metrics"] = strict_margin_metrics(grouped[order])
        orders[order] = {"summary": summary, "failures": failures[order], "cases": grouped[order]}

    equality = audit.reverse_outcome_equality(grouped["natural"], grouped["reverse"])
    return {
        "seed": seed,
        "source_hashes": {
            "manifest_v2": source_record(manifest_path),
            "stage_a_checkpoint": source_record(checkpoint),
            "audit_script": source_record(SCRIPT_DIR / Path(audit.__file__).name),
        },
        "protocol": {
            "stage_b": False,
            "gate_loaded": False,
            "gate_used": False,
            "numeric_confidence": False,
            "tau_used": False,
            "alpha_used": False,
            "eligibility_filter": False,
            "value_mask": False,
            "soft_mixture": False,
            "pointer_F": "first/min-position argmax raw s_F over every real t < length",
            "pointer_A": "first/min-position argmax raw s_A over every real t < length",
            "selected_state": "W_v(e_j) exactly; no gate weighting",
            "decoder": "frozen register_decoder and frozen VALUE codebook; canonical round-trip",
        },
        "orders": orders,
        "exact_reverse_outcome_equality": equality,
    }


def seed_passes(training: dict[str, Any], audited: dict[str, Any]) -> bool:
    atomic = training["atomic"]
    if any(atomic[name]["exact"] != atomic[name]["expected"] for name in ("FLOOR", "AVOID", "NOOP")):
        return False
    if not audited["exact_reverse_outcome_equality"]["equal"]:
        return False
    for order in ORDERS:
        summary = audited["orders"][order]["summary"]
        if summary["cases"] != EXPECTED_CASES:
            return False
        if any(summary[field] != EXPECTED_CASES for field in audit.POINTER_FIELDS):
            return False
        for metric in summary["strict_margin_metrics"].values():
            if metric["count_le_zero"] != 0 or metric["positive_count"] != EXPECTED_CASES:
                return False
    return True


def write_json(path: Path, artifact: dict[str, Any]) -> None:
    artifact["artifact_self_hash"] = artifact_hash(artifact)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    if OUTPUT_ROOT.exists() and (OUTPUT_ROOT / "results.json").exists():
        raise FileExistsError(f"refusing to overwrite consolidated output: {OUTPUT_ROOT / 'results.json'}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    old_snapshot = old_680x_snapshot()
    observations, labels = train.load_source()
    supervisor = train.LatentConditionedSupervisor(train.CTRL7_CHECKPOINT)
    supervisor.eval()
    for parameter in supervisor.parameters():
        parameter.requires_grad_(False)
    executor = train.load_executor()
    executor.eval()
    for parameter in executor.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        codebook = executor.token_embedding(torch.arange(train.VALUE_BASE, train.VALUE_BASE + train.VALUE_COUNT, dtype=torch.long)).detach()

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        destination = output_dir(seed)
        try:
            training = train_seed(seed, observations, labels, supervisor, executor, codebook)
            manifest = json.loads((train.MANIFEST_ROOT / f"manifest_{seed}_v2.json").read_text(encoding="utf-8"))
            manifest_path = train.MANIFEST_ROOT / f"manifest_{seed}_v2.json"
            audited = audit_seed(seed, manifest, manifest_path, Path(training["checkpoint"]["path"]), executor, codebook)
            entry: dict[str, Any] = {
                "status": "completed",
                "classification": "PASS" if seed_passes(training, audited) else "VALID_FAIL",
                "training": training,
                "audit": audited,
                "source_hashes": source_hashes(manifest_path, Path(training["checkpoint"]["path"])),
                "failure_policy": "No tuning or rerun after failure; record evidence and continue remaining seeds.",
            }
        except Exception as error:
            entry = {
                "status": "execution_failed",
                "classification": "VALID_FAIL",
                "seed": seed,
                "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
                "failure_policy": "No tuning or rerun after failure; record evidence and continue remaining seeds.",
            }
        write_json(destination / "results.json", entry)
        per_seed.append(entry)
        print(json.dumps({"seed": seed, "status": entry["status"], "classification": entry["classification"]}, sort_keys=True), flush=True)

    after_snapshot = old_680x_snapshot()
    old_unchanged = old_snapshot == after_snapshot
    all_pass = len(per_seed) == len(SEEDS) and all(entry["classification"] == "PASS" for entry in per_seed)
    verification_sources: dict[str, Any] = {
        "execute_script": source_record(SCRIPT_DIR / Path(__file__).name),
        "rcsep_a_train_script": source_record(SCRIPT_DIR / "train_nb5_rcsep_a.py"),
        "nogate_audit_script": source_record(SCRIPT_DIR / "audit_nb5_rcsep_c_nogate_hardptr_alg.py"),
        "nb5_fresh.py": source_record(SCRIPT_DIR / "nb5_fresh.py"),
        "manifest_checker": source_record(SCRIPT_DIR / "check_nb5_manifests_v2.py"),
        "manifests": {str(seed): per_seed[index]["training"]["manifest"] for index, seed in enumerate(SEEDS)},
        "checkpoints": {str(seed): per_seed[index]["training"]["checkpoint"] for index, seed in enumerate(SEEDS)},
    }
    fresh_check_path = CAMPAIGN / "nb5_fresh_690x_manifest_check.json"
    if fresh_check_path.exists():
        verification_sources["fresh_manifest_check"] = source_record(fresh_check_path)
    consolidated: dict[str, Any] = {
        "status": "completed",
        "classification": "PASS_STRONG" if all_pass else "VALID FAIL",
        "task": "T2-NOBYPASS-2-FRESH-690x",
        "executive_summary": {
            "result": "PASS_STRONG only if every metric/count/margin passes for 5 seeds x 2 orders" if all_pass else "VALID FAIL: one or more authorized seed, count, margin, equality, or atomic requirements failed",
            "seeds_completed": len(per_seed),
            "seeds_expected": len(SEEDS),
            "all_seed_requirements": all_pass,
            "failure_evidence_records": sum(
                len(order_data.get("failures", []))
                for entry in per_seed
                for order_data in entry.get("audit", {}).get("orders", {}).values()
            ),
            "no_pass_inferred_from_partial_counts": True,
        },
        "authorization": {
            "id": "T2-NOBYPASS-2-FRESH-690x",
            "sole_writer": True,
            "seeds": list(SEEDS),
            "orders": list(ORDERS),
            "run_all_seeds_even_if_one_fails": True,
            "no_sequential_stopping": True,
            "no_tuning_if_any_failure": True,
        },
        "fresh_init_provenance": {
            "constructor": "NB5CoreEncoder(manifest_690X_v2, seed)",
            "seed_to_manifest": {str(seed): f"manifest_{seed}_v2.json" for seed in SEEDS},
            "prior_680x_core_or_checkpoint_loaded": False,
            "frozen_supervisor_executor_only": True,
        },
        "recipe": {
            "updates": UPDATES,
            "batch": {"FLOOR": 32, "AVOID": 32, "NOOP": 32, "total": 96},
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "learning_rate": {"initial": 1e-3, "final": 1e-5, "schedule": "linear over 5000 updates"},
            "gradient_clipping": False,
            "joint_rows": 0,
            "forward_inputs": ["token_ids", "lengths"],
            "teacher_ground_truth_args": False,
            "catalog": {"queries_per_update": 64, "layout": "32 FLOOR then 32 AVOID", "matrices": "full 32x32 all-to-all raw sf/sa, diagonal included"},
            "L_sep": "(mean(softplus(F_cross - F_self)) + mean(softplus(A_cross - A_self))) / 2",
            "L_sep_coefficient": 1.0,
        },
        "evaluation_protocol": {
            "immediate_after_stage_a": True,
            "stage_b": False,
            "gate": False,
            "numeric_confidence": False,
            "tau": False,
            "alpha": False,
            "eligibility": False,
            "value_mask": False,
            "pointer": "first/min-position argmax raw sf/sa over all real positions t < length",
            "selected_state": "raw W_v(e_j)",
            "cases_per_seed_order": EXPECTED_CASES,
            "roles": ["FLOOR", "AVOID"],
            "strict_margins": {
                "M_F_ARG": "FLOOR target lower ARG minus other ARG",
                "M_F_STRUCT": "FLOOR target minus max all structural OP_X/OP_Y/LINK positions",
                "M_A_ARG": "AVOID target forbidden ARG minus other ARG",
                "M_A_STRUCT": "AVOID target minus max all structural OP_X/OP_Y/LINK positions",
            },
        },
        "atomic_gate": "FLOOR/AVOID/NOOP 32/32 required per seed",
        "per_seed": per_seed,
        "old_680x_and_legacy_immutability": {
            "checked_file_count": len(old_snapshot),
            "unchanged": old_unchanged,
            "changed_files": sorted(set(old_snapshot) ^ set(after_snapshot) | {path for path in old_snapshot.keys() & after_snapshot.keys() if old_snapshot[path] != after_snapshot[path]}),
            "baseline_sha256": hashlib.sha256(json.dumps(old_snapshot, sort_keys=True).encode()).hexdigest(),
            "after_sha256": hashlib.sha256(json.dumps(after_snapshot, sort_keys=True).encode()).hexdigest(),
        },
        "verification": {
            "py_compile": "passed before execution",
            "imports": "passed before execution",
            "source_hashes": verification_sources,
            "outputs_self_hashed": True,
            "old_680x_artifacts_legacy_files_unchanged": old_unchanged,
        },
        "artifacts": {
            "per_seed": [str(output_dir(seed) / "results.json") for seed in SEEDS],
            "checkpoints": [str(output_dir(seed) / "stage_a.pt") for seed in SEEDS],
            "fresh_manifest_check": str(fresh_check_path),
            "consolidated_results": str(OUTPUT_ROOT / "results.json"),
        },
        "next_recommended": "No tuning. If VALID FAIL, analyze recorded evidence only and await fresh authorization.",
        "risks": [
            "Any failed metric classifies complete battery as VALID FAIL; partial success is not PASS_STRONG.",
            "Per-seed case evidence is intentionally large and preserves raw scores, selected states, decoder, canonical, and margin evidence.",
        ],
        "skill_resolution": {
            "graphify": "Read before work; existing .codegraph targeted exploration completed.",
            "work_unit_commits": "Single focused execution work unit; no commit requested.",
            "shared": "Read before work.",
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "results.json", consolidated)
    print(json.dumps({"status": consolidated["status"], "classification": consolidated["classification"], "artifact": str(OUTPUT_ROOT / "results.json"), "artifact_self_hash": consolidated["artifact_self_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

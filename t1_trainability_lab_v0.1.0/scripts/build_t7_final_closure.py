"""Build the immutable T7 final-closure inventory without model execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "t7_final_closure" / "t7_final_closure.json"
SEEDS_FRESH = (7801, 7802, 7803, 7804, 7805)
SEEDS_DEVELOPMENT = (7701, 7702, 7703, 7704, 7705)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def records(paths: list[Path]) -> list[dict[str, Any]]:
    return [record(path) for path in sorted(set(paths), key=lambda item: item.as_posix())]


def load(relative: str) -> dict[str, Any]:
    with require(relative).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(relative)
    return value


def write_self_hashed(payload: dict[str, Any]) -> tuple[str, str]:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(encoded)
    return digest, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    fresh_root = ROOT / "campaign" / "t7_noop_none_lexical_paired_fresh"
    development_root = ROOT / "campaign" / "t7_noop_none_lexical_paired_development"
    conformance_root = ROOT / "campaign" / "t7_noop_none_lexical_paired_development_conformance"
    fresh_preparation = ROOT / "campaign" / "t7_noop_none_lexical_fresh_preparation"
    development_preparation = ROOT / "campaign" / "t7_noop_none_lexical_development_preparation"
    stage_a_fresh = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_fresh"
    stage_a_development = ROOT / "campaign" / "t7_noop_none_lexical_stage_a_development"
    stage_b_fresh = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_fresh"
    stage_b_development = ROOT / "campaign" / "t7_noop_none_lexical_stage_b_development"
    stage_c_fresh = ROOT / "campaign" / "t7_noop_none_lexical_stage_c_fresh"
    stage_c_development = ROOT / "campaign" / "t7_noop_none_lexical_stage_c_development"

    fresh_eval = load("campaign/t7_noop_none_lexical_paired_fresh/results.json")
    reconstructed = load("campaign/t7_final_closure/audit_t7_paired_fresh_reconstructed.json")
    stage_b = load("campaign/t7_noop_none_lexical_stage_b_fresh/results.json")
    stage_c = load("campaign/t7_noop_none_lexical_stage_c_fresh/results.json")

    code_paths = [
        "scripts/prepare_t7_noop_none_lexical_fresh_preparation.py",
        "scripts/execute_t7_noop_none_stage_a_fresh.py",
        "scripts/execute_t7_noop_none_stage_b_fresh.py",
        "scripts/repair_t7_noop_none_stage_b_atomic_sanity.py",
        "scripts/execute_t7_noop_none_stage_c_fresh.py",
        "scripts/evaluate_t7_noop_none_lexical_paired_fresh.py",
        "scripts/prepare_t7_noop_none_lexical_development.py",
        "scripts/execute_t7_noop_none_stage_a_development.py",
        "scripts/execute_t7_noop_none_stage_b_development.py",
        "scripts/execute_t7_noop_none_stage_c_development.py",
        "scripts/evaluate_t7_noop_none_lexical_paired_development.py",
        "scripts/evaluate_t7_noop_none_lexical_paired_development_conformance.py",
        "scripts/check_t7_noop_none_lexical_development.py",
        "scripts/assemble_t7_noop_none_lexical_development_preparation.py",
        "scripts/resell_t7_noop_none_lexical_manifests.py",
        "scripts/audit_t7_paired_fresh_independent.py",
        "scripts/audit_t7_paired_fresh_reconstructed.py",
        "scripts/ctrl2_common.py",
    ]
    code_registry = records([require(path) for path in code_paths])

    fresh_manifests = records(list((fresh_preparation / "manifests").glob("manifest_*.json")))
    development_manifests = records(list((development_preparation / "manifests").glob("manifest_*.json")))
    checkpoints_a = records(list(stage_a_fresh.rglob("final.pt")) + list(stage_a_development.rglob("final.pt")))
    checkpoints_c = records(list(stage_c_fresh.rglob("final.pt")) + list(stage_c_development.rglob("final.pt")))
    calibrations_b = records(list(stage_b_fresh.rglob("calibration_*.json")) + list(stage_b_development.rglob("calibration_*.json")))
    stage_results = records(
        [
            path
            for root in (fresh_preparation, development_preparation, stage_a_fresh, stage_a_development, stage_b_fresh, stage_b_development, stage_c_fresh, stage_c_development, fresh_root, development_root, conformance_root)
            for path in root.rglob("*.json")
            if path.name == "results.json" or path.name.startswith("result_") or path.name in {"independent_checker.json", "conformance_specification.json", "writer_byte_exact_probe.json"}
        ]
    )
    evidence_jsonl = records(list(fresh_root.rglob("cases.jsonl")) + list(development_root.rglob("cases.jsonl")) + list(conformance_root.rglob("cases.jsonl")))
    audit_records = records([require("scripts/audit_t7_paired_fresh_independent.py"), require("scripts/audit_t7_paired_fresh_independent_result.json"), require("scripts/audit_t7_paired_fresh_reconstructed.py"), require("campaign/t7_final_closure/audit_t7_paired_fresh_reconstructed.json")])
    runtime_checkpoint = require("campaign/u0c_c1_lossnorm_anneal_seed101_12000/final.pt")

    fresh_metrics = []
    for seed_result in fresh_eval["seeds"]:
        seed = int(seed_result["seed"])
        fresh_metrics.append({"seed": seed, "base": seed_result["evaluation"]["base"], "augmented": seed_result["evaluation"]["augmented"], "pairs": seed_result["evaluation"]["pairs"], "cases": seed_result["evaluation"]["base"]["counts_by_size"]})

    payload = {
        "schema": "t7-final-closure-v1",
        "status": "closed",
        "classification": "T7-NOOP-NONE-LEXICAL-FRESH: CLOSED/PASS_STRONG",
        "scope": {"fresh_seeds": list(SEEDS_FRESH), "development_seeds": list(SEEDS_DEVELOPMENT), "development_role": "development/diagnostic permanent", "fresh_role": "evidence fresh", "no_t8": True, "no_new_training_or_forwards": True},
        "architecture_real_recipe": {"binder": "k_t = LN(W_k[e_{t-1},e_t] + a_k)", "score": "s_{r,t} = Q_r^T k_t / 4", "selection": "selection among real candidates plus NULL", "value_path": "shared W_v", "readout": "decoder/codebook approved via decode_states"},
        "stages": {"A": {"name": "atomic base binder", "recipe": "4-role atomic binder training with RCSEP-N+BG", "training_multi_clause": 0}, "B": {"name": "NULL midpoint calibration", "recipe": "b_r = N_r_max + (P_r_min - N_r_max)/2", "catalog": "base catalog without NOOP", "intervals": 20}, "C": {"name": "explicit NOOP learning", "recipe": "L_C = (1/136)*sum_r sum_{z in U} softplus(s_r(z;e_N)-b_r)", "trainable": "only new 16-component e_N", "core_frozen": True, "thresholds_frozen": True, "supervision": "NOOP receives EXPLICIT irrelevance supervision including its local contexts -- it is NOT an unknown operator rejected spontaneously."}},
        "provenance_scopes": {"development": "7701-7705 = development/diagnostic permanent", "fresh": "7801-7805 = evidence fresh", "valid_development_reference": "campaign/t7_noop_none_lexical_paired_development_conformance/", "invalid_historical_variant": "campaign/t7_noop_none_lexical_paired_development/ remains preserved as historical/invalid +2-fixed attempt, not a replication", "valid_b_sanity": {"path": "scripts/repair_t7_noop_none_stage_b_atomic_sanity.py", "commit": "fea9ea43f255893451230db4714360d409e0363f", "decoder": "decode_states"}},
        "final_claim": "En cinco dominios léxicos fresh, la receta de entrenamiento atómico del binder, calibración NULL por catálogo y aprendizaje explícito de un embedding NOOP de 16 componentes preservó exactamente presencia, ausencia y binding en los 15 subconjuntos no vacíos de cuatro roles, al añadir una cláusula numérica irrelevante y recorrer las órdenes predeclaradas, sin entrenamiento multi-cláusula ni modificación del core o de los umbrales durante el aprendizaje de NOOP.",
        "fresh_domain_clarification": "Dominios fresh significa replicación de la receta con nuevos dominios e inicializaciones, NO que un único modelo congelado entendiera ciphers nuevos sin entrenamiento.",
        "limits": {"demonstrated": ["maximum one appearance per role", "one NOOP", "four roles", "32-value numeric vocabulary", "controlled local grammar"], "not_demonstrated": ["free text", "repeated roles", "generalization to arbitrary N", "new executor primitives", "general reasoning"]},
        "metrics": {"fresh_per_seed": fresh_metrics, "global": {"calibration_intervals": 20, "c1_noop_certificates": 20, "atomic_base_controls": 640, "base_multi_clause": 297600, "augmented_instructions": 1251200, "base_total": 298240, "paired_comparisons": 1251200}, "evaluator_observed": fresh_eval["observed_totals"], "reconstructed_observed": reconstructed["global_observed"], "witness_availability": "Every cases.jsonl row retains seed, case_id, pair_id, subset, order, D, assignment, tokens, role position, and raw score fields; minima witnesses are recorded by the reconstructed audit.", "pair_independence_note": "las comparaciones de pareja no son evaluaciones adicionales, y las 1.549.440 filas no constituyen 1.549.440 réplicas estadísticamente independientes -- la batería contiene 5 entrenamientos/dominios fresh y una enumeración estructurada de casos dentro de cada uno.", "local_evidence_note": "Los JSONL permanecen locales -- sus hashes identifican el contenido pero no equivalen a haberlo publicado ni permiten a un tercero recalcularlo sin obtener los archivos o regenerarlos."},
        "incidents_separated": {"implementation_incidents": ["buggy argmax-based B sanity attempt", "historical +2-fixed conformance attempt", "Stage-A self-hash transcription error"], "chat_transcription_errors": ["Any prior chat hash transcription errors are not scientific results and are excluded from counts."], "scientific_results": "Only valid A/B/C fresh artifacts, corrected conformance reference, PASS_STRONG paired evaluation, and PASS_RECONSTRUCTED audit contribute to closure counts.", "invalid_attempts_counted_as_replications": 0},
        "audits": {"old_scope_preserved": record(require("scripts/audit_t7_paired_fresh_independent_result.json")), "new_reconstructed": reconstructed, "new_audit_script": record(require("scripts/audit_t7_paired_fresh_reconstructed.py")), "real_discrepancy_against_accepted_pass_strong": False},
        "artifacts": {"fresh_manifests": fresh_manifests, "development_manifests": development_manifests, "checkpoints_A": checkpoints_a, "checkpoints_C": checkpoints_c, "calibrations_B": calibrations_b, "stage_results": stage_results, "evidence_jsonl": evidence_jsonl, "audit_files": audit_records, "code_executed": code_registry, "runtime": {"checkpoint": record(runtime_checkpoint), "file_sha256": fresh_eval["runtime_integrity"]["before"]["file_sha256"], "tensor_state_hash": fresh_eval["runtime_integrity"]["before"]["tensor_state_hash"], "codebook_hash": fresh_eval["runtime_integrity"]["before"]["codebook_hash"]}},
        "stage_status": {"A": "PASS", "B": "CLOSED/PASS", "C": "CLOSED/PASS", "paired_fresh": "PASS_STRONG", "reconstructed_audit": reconstructed["classification"]},
        "no_next_stage": "No next stage; T7 final closure complete.",
    }
    digest, file_sha = write_self_hashed(payload)
    print(json.dumps({"status": payload["status"], "classification": payload["classification"], "artifact": OUTPUT.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": file_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

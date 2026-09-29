"""Read-only TAIL-REPEAT test-time recurrence scaling over frozen K-curve checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch


HERE = Path(__file__).resolve().parent
CURVE_ROOT = HERE / "results" / "k_curve_primary_20260927_md286"
BLOCK_A_REPORT_PATH = CURVE_ROOT / "block_A_report.json"
BLOCK_B_REPORT_PATH = CURVE_ROOT / "block_B_report.json"
C_EVALUATION_REPORT_PATH = HERE / "results" / "coverage_c_distribution_evaluation_20260928_md291" / "coverage_c_distribution_evaluation_report.json"
B_REPORT_PATH = HERE / "results" / "coverage_b_distribution_diagnostic_20260927_md290_retry1" / "coverage_b_distribution_report.json"
DEFAULT_OUTPUT_DIR = HERE / "results" / "test_time_recurrence_scaling_a_20260928_md295"
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
TRAINING_ROUNDS = (1, 4, 6)
INFERENCE_ROUNDS = (1, 2, 4, 6, 8, 12, 16)
VALL_TARGETS = 231424
V0_TARGETS = 30720
GUARDRAIL = 1e-4
T_CRIT_90_DF4 = 2.131846786326383

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_coverage_b2_document_macro_check as coverage_b2  # noqa: E402
import run_k_curve_primary_block as curve_runner  # noqa: E402
import run_backend_quality_qualification as quality  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    unsigned = dict(value)
    signature = unsigned.pop(field, None)
    if not signature or signature != quality._canonical_hash(unsigned):
        raise ValueError(f"{label} self-hash mismatch")
    return str(signature)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    quality._write_json(path, value)


def _load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[int, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    qualification = quality._load_qualification_manifest()
    block_a = json.loads(BLOCK_A_REPORT_PATH.read_text(encoding="utf-8"))
    block_b = json.loads(BLOCK_B_REPORT_PATH.read_text(encoding="utf-8"))
    block_a_sig = _check_self_hash(block_a, "self_sha256", "K-CURVE-A block A report")
    block_b_sig = _check_self_hash(block_b, "self_sha256", "K-CURVE-A block B report")
    if block_a.get("status") != "COMPLETE" or block_b.get("status") != "COMPLETE":
        raise PermissionError("TTR scaling requires both completed K-CURVE-A blocks")
    if block_a.get("qualification_manifest_sha256") != qualification["manifest_sha256"] or block_b.get("qualification_manifest_sha256") != qualification["manifest_sha256"]:
        raise ValueError("K-CURVE-A qualification-manifest identity mismatch")

    c_report = json.loads(C_EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
    c_sig = _check_self_hash(c_report, "report_self_sha256", "Coverage-C final evaluation report")
    if c_report.get("status") != "COVERAGE_C_COMPLETE" or c_report.get("primary_metric") != "VALL_ALL_FULL_CHUNKS token-weighted NLL":
        raise PermissionError("TTR scaling requires closed Coverage-C and its selected VALL primary metric")
    b_report = json.loads(B_REPORT_PATH.read_text(encoding="utf-8"))
    b_sig = _check_self_hash(b_report, "report_self_sha256", "Coverage-B report")
    if b_report.get("status") != "DISTRIBUTION_DIAGNOSTIC_COMPLETE":
        raise ValueError("TTR validation input requires completed Coverage-B distribution strata")
    validation_manifest, chunks_by_document, payload_sha = coverage_b2._load_validation_chunks(b_report)
    if validation_manifest["strata"]["VALL_ALL_FULL_CHUNKS"]["target_tokens"] != VALL_TARGETS:
        raise ValueError("VALL validation target count drift")
    fast_model_class = coverage_b2.coverage_b.ce.OmegaCoreLMFast
    fast_model_module = sys.modules[fast_model_class.__module__]
    fast_model_path = Path(fast_model_module.__file__).resolve()

    checkpoints: list[dict[str, Any]] = []
    for seed in SEEDS:
        block = block_a if seed in (20260913, 20260914) else block_b
        for k_train in TRAINING_ROUNDS:
            route_key = f"native_K{k_train}_seed_{seed}"
            route = block["routes"].get(route_key)
            if route is None or route.get("route_pass") is not True or route.get("status") != "COMPLETE":
                raise ValueError(f"K-CURVE-A checkpoint route missing/failed: {route_key}")
            run_report_path = Path(route["run_report_path"])
            if _sha256_file(run_report_path) != route["run_report_sha256"]:
                raise ValueError(f"K-CURVE-A run report hash mismatch for {route_key}")
            run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
            if (
                run_report.get("status") != "COMPLETE"
                or run_report.get("backend") != "native"
                or int(run_report.get("K", -1)) != k_train
                or int(run_report.get("seed", -1)) != seed
                or int(run_report.get("updates", -1)) != 2000
                or run_report.get("test_split_loaded") is not False
                or int(run_report.get("endpoint_nll_validation", {}).get("tokens", -1)) != V0_TARGETS
            ):
                raise ValueError(f"K-CURVE-A run report identity mismatch for {route_key}")
            checkpoint_path = Path(run_report["run_dir"]) / "checkpoint_02000.pt"
            sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".identity.json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            checkpoint_sha = _sha256_file(checkpoint_path)
            if checkpoint_sha != sidecar.get("sha256"):
                raise ValueError(f"K-CURVE-A endpoint checkpoint/sidecar SHA mismatch for {route_key}")
            identity = run_report.get("identity", {})
            if quality._canonical_hash(identity) != sidecar.get("identity_sha256"):
                raise ValueError(f"K-CURVE-A endpoint checkpoint identity hash mismatch for {route_key}")
            checkpoints.append(
                {
                    "checkpoint_rounds": k_train,
                    "seed": seed,
                    "condition": f"K{k_train}",
                    "block": block["block"],
                    "run_report_path": str(run_report_path),
                    "run_report_sha256": route["run_report_sha256"],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "stored_training_endpoint_v0_nll": float(run_report["endpoint_nll_validation"]["nll"]),
                    "stored_training_endpoint_v0_tokens": int(run_report["endpoint_nll_validation"]["tokens"]),
                }
            )
    if len(checkpoints) != 15:
        raise AssertionError("TTR checkpoint grid must contain exactly 15 frozen checkpoints")

    source_identity = {
        "qualification_manifest_sha256": qualification["manifest_sha256"],
        "block_a_report_path": str(BLOCK_A_REPORT_PATH),
        "block_a_report_file_sha256": _sha256_file(BLOCK_A_REPORT_PATH),
        "block_a_report_self_sha256": block_a_sig,
        "block_b_report_path": str(BLOCK_B_REPORT_PATH),
        "block_b_report_file_sha256": _sha256_file(BLOCK_B_REPORT_PATH),
        "block_b_report_self_sha256": block_b_sig,
        "coverage_c_evaluation_report_path": str(C_EVALUATION_REPORT_PATH),
        "coverage_c_evaluation_report_file_sha256": _sha256_file(C_EVALUATION_REPORT_PATH),
        "coverage_c_evaluation_report_self_sha256": c_sig,
        "coverage_b_report_self_sha256": b_sig,
        "validation_manifest_self_sha256": validation_manifest["manifest_sha256"],
        "validation_payload_sha256": payload_sha,
        "r1_evaluator_path": b_report["r1_evaluator_identity"]["path"],
        "r1_evaluator_sha256": b_report["r1_evaluator_identity"]["sha256"],
        "fast_model_path": str(fast_model_path),
        "fast_model_sha256": _sha256_file(fast_model_path),
    }
    return block_a, block_b, c_report, chunks_by_document, checkpoints, source_identity


def _load_checkpoint_model(
    checkpoint: dict[str, Any],
    ce: Any,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint["checkpoint_path"])
    if _sha256_file(path) != checkpoint["checkpoint_sha256"]:
        raise ValueError(f"checkpoint changed after source inventory: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    run_report_path = Path(checkpoint["run_report_path"])
    if _sha256_file(run_report_path) != checkpoint["run_report_sha256"]:
        raise ValueError(f"run report changed after checkpoint inventory: {run_report_path}")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    sidecar = json.loads(path.with_suffix(path.suffix + ".identity.json").read_text(encoding="utf-8"))
    if (
        int(payload.get("completed_updates", -1)) != 2000
        or int(payload.get("next_update", -1)) != 2000
        or int(identity.get("K", -1)) != int(checkpoint["checkpoint_rounds"])
        or int(identity.get("seed", -1)) != int(checkpoint["seed"])
        or identity.get("backend") != "native"
        or identity != run_report.get("identity")
        or run_report.get("status") != "COMPLETE"
        or int(run_report.get("updates", -1)) != 2000
        or sidecar.get("identity_sha256") != quality._canonical_hash(identity)
        or payload.get("model") is None
    ):
        raise ValueError(f"checkpoint payload update/K/seed/backend mismatch: {path}")
    model = ce.fresh_model(int(checkpoint["seed"]), int(checkpoint["checkpoint_rounds"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    k_train = int(checkpoint["checkpoint_rounds"])
    if model.variant != "shared" or model.rounds != k_train or len(model.blocks) != 1:
        raise ValueError(f"expected tied/shared fast model with one CoreMLP for {path}")
    if tuple(model.depth_embedding.weight.shape) != (k_train, model.dimension) or tuple(model.gate_logits.shape) != (k_train, model.dimension):
        raise ValueError(f"round-specific parameter table shape mismatch for {path}")
    return model, {
        "checkpoint_rounds": k_train,
        "seed": int(checkpoint["seed"]),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "model_state_sha256": quality._value_hash(model.state_dict()),
        "shared_core_count": len(model.blocks),
        "depth_embedding_shape": list(model.depth_embedding.weight.shape),
        "gate_logits_shape": list(model.gate_logits.shape),
    }


def _make_inference_adapter(base_model: torch.nn.Module, k_infer: int) -> tuple[torch.nn.Module, dict[str, Any]]:
    k_train = int(base_model.rounds)
    if k_infer not in INFERENCE_ROUNDS or k_infer < 1:
        raise ValueError(f"inference rounds outside frozen grid: {k_infer}")
    adapter = copy.deepcopy(base_model)
    round_map = [index if index < k_train else k_train - 1 for index in range(k_infer)]
    tail_count = max(0, k_infer - k_train)
    tables_extended = False
    if tail_count:
        depth_source = adapter.depth_embedding.weight.detach().clone()
        gate_source = adapter.gate_logits.detach().clone()
        depth_tail = depth_source[-1:].expand(tail_count, -1).clone()
        gate_tail = gate_source[-1:].expand(tail_count, -1).clone()
        expanded_depth = torch.cat((depth_source, depth_tail), dim=0)
        expanded_gate = torch.cat((gate_source, gate_tail), dim=0)
        adapter.depth_embedding = torch.nn.Embedding.from_pretrained(expanded_depth, freeze=False)
        adapter.gate_logits = torch.nn.Parameter(expanded_gate)
        tables_extended = True
    adapter.rounds = k_infer
    adapter.eval()
    if len(adapter.blocks) != 1 or adapter.variant != "shared":
        raise ValueError("inference adapter must reuse the one tied CoreMLP")
    if adapter.depth_embedding.num_embeddings < k_infer or adapter.gate_logits.shape[0] < k_infer:
        raise AssertionError("inference round table does not cover the requested recurrence")
    for round_index, source_index in enumerate(round_map):
        if not torch.equal(adapter.depth_embedding.weight[round_index], base_model.depth_embedding.weight[source_index]):
            raise AssertionError(f"depth embedding mapping mismatch at inference round {round_index}")
        if not torch.equal(adapter.gate_logits[round_index], base_model.gate_logits[source_index]):
            raise AssertionError(f"gate-logit mapping mismatch at inference round {round_index}")
    return adapter, {
        "checkpoint_rounds": k_train,
        "inference_rounds": k_infer,
        "continuation_policy": "TAIL_REPEAT",
        "round_index_mapping": round_map,
        "tail_repeated_round_count": tail_count,
        "tail_repeat_source_index": k_train - 1 if tail_count else None,
        "in_memory_round_tables_extended": tables_extended,
        "shared_core_count": len(adapter.blocks),
        "checkpoint_modified": False,
    }


def _evaluate_configuration(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    chunks_by_document: dict[int, list[dict[str, Any]]],
    k_infer: int,
) -> dict[str, Any]:
    adapter, policy_identity = _make_inference_adapter(model, k_infer)
    adapter_hash_before = quality._value_hash(adapter.state_dict())
    v0_chunks = [chunks_by_document[doc][0] for doc in range(60)]
    v0 = coverage_b2.coverage_b.r1.evaluate_validation(adapter, [{"tokens": row["tokens"]} for row in v0_chunks])
    per_document: list[dict[str, Any]] = []
    for doc_order in range(60):
        rows = chunks_by_document[doc_order]
        metric = coverage_b2.coverage_b.r1.evaluate_validation(adapter, [{"tokens": row["tokens"]} for row in rows])
        if not metric["finite"] or int(metric["tokens"]) != 512 * len(rows):
            raise ValueError(f"invalid Vall score at Ktrain={checkpoint['checkpoint_rounds']}, seed={checkpoint['seed']}, Kinfer={k_infer}, document={doc_order}")
        per_document.append(
            {
                "document_order_index": doc_order,
                "full_text_sha256": rows[0]["full_text_sha256"],
                "chunk_count": len(rows),
                "target_tokens": int(metric["tokens"]),
                "nll": float(metric["nll"]),
            }
        )
    adapter_hash_after = quality._value_hash(adapter.state_dict())
    if adapter_hash_before != adapter_hash_after:
        raise RuntimeError("adapter parameters changed during inference-only scoring")
    del adapter
    vall_tokens = sum(row["target_tokens"] for row in per_document)
    vall_weighted_nll = sum(row["nll"] * row["target_tokens"] for row in per_document) / vall_tokens
    vall_doc_macro = sum(row["nll"] for row in per_document) / len(per_document)
    checkpoint_sha_after = _sha256_file(Path(checkpoint["checkpoint_path"]))
    if checkpoint_sha_after != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint file changed during TTR inference")
    if not v0["finite"] or int(v0["tokens"]) != V0_TARGETS or vall_tokens != VALL_TARGETS:
        raise ValueError("TTR validation target count/finite guard mismatch")
    return {
        **policy_identity,
        "V0_CHUNK0": {"nll": float(v0["nll"]), "tokens": int(v0["tokens"]), "finite": bool(v0["finite"]), "decision_metric": False},
        "VALL_TOKEN_WEIGHTED": {"nll": vall_weighted_nll, "tokens": vall_tokens, "finite": True},
        "VALL_DOCUMENT_MACRO": {"nll": vall_doc_macro, "documents": len(per_document), "per_document": per_document},
        "adapter_state_dict_sha256_before": adapter_hash_before,
        "adapter_state_dict_sha256_after": adapter_hash_after,
        "checkpoint_sha256_before": checkpoint["checkpoint_sha256"],
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_modified": False,
    }


def _delta_summary(seed_rows: list[dict[str, Any]], field: str, label: str) -> dict[str, Any]:
    deltas = [float(row[field]) for row in seed_rows]
    mean = sum(deltas) / len(deltas)
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1))
    half = T_CRIT_90_DF4 * sample_sd / math.sqrt(len(deltas))
    return {
        "metric": label,
        "n": len(deltas),
        "per_seed_delta": {str(row["seed"]): row[field] for row in seed_rows},
        "mean_delta": mean,
        "sample_sd": sample_sd,
        "t_critical_90_two_sided_df4": T_CRIT_90_DF4,
        "ci90": [mean - half, mean + half],
        "decision_threshold": None,
    }


def run_scaling(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"TTR scaling output is immutable; resume explicitly: {output_dir}")
    block_a, block_b, c_report, chunks_by_document, checkpoints, source_identity = _load_sources()
    if resume and (output_dir / "ttr_scaling_progress.json").is_file():
        report = json.loads((output_dir / "ttr_scaling_progress.json").read_text(encoding="utf-8"))
        if report.get("source_identity") != source_identity:
            raise ValueError("TTR progress/source identity mismatch on resume")
        report.pop("report_self_sha256", None)
    else:
        report = {
            "schema": "omega-test-time-recurrence-scaling-a-v1",
            "unit": "OMEGA-TEST-TIME-RECURRENCE-SCALING-A",
            "status": "INFERENCE_EVALUATION_IN_PROGRESS",
            "source_identity": source_identity,
            "policy": {
                "primary_continuation_policy": "TAIL_REPEAT",
                "checkpoint_rounds": list(TRAINING_ROUNDS),
                "inference_rounds": list(INFERENCE_ROUNDS),
                "round_mapping": "r<K_train uses trained row r; r>=K_train repeats the final trained depth_embedding and gate_logits row K_train-1",
                "shared_core": "one FastWorkspaceUpdateBlock in the tied variant is reused on every inferred recurrent round",
                "validation_primary": "VALL token-weighted on the 60 frozen documents / 452 full chunks / 231424 targets",
                "validation_secondary": "VALL document-macro, equal mean of each document's token-mean NLL",
                "V0_CHUNK0": "reported for historical continuity only; excluded from round-scaling decision",
                "precision": "FP32",
                "device": "CPU",
                "torch_intraop": 4,
                "torch_interop": 1,
                "checkpoint_modified": False,
                "optimizer_updates": 0,
                "teacher_forward_used": False,
                "hidden_cache_used": False,
                "native_dll_used_for_evaluation": False,
                "test_split_loaded": False,
                "secondary_CYCLE_policy": "NOT_RUN; await TAIL_REPEAT result review before authorizing",
            },
            "checkpoint_matrix": checkpoints,
            "checkpoint_results": {},
            "scientific_updates": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        output_dir.mkdir(parents=True, exist_ok=False)

    expected_keys = {f"K{k}_seed_{seed}" for k in TRAINING_ROUNDS for seed in SEEDS}
    if set(report.get("checkpoint_results", {})) - expected_keys:
        raise ValueError("TTR progress contains an unexpected training checkpoint")
    r2, _p0, ce, _bridge, _modules = quality._load_real_dependencies()
    ce.validate_policy()
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415
    ce.fresh_model = curve_runner._fresh_model_factory(ce, technical_model)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("TTR inference requires CPU FP32 torch 4/1")

    checkpoint_by_key = {f"K{row['checkpoint_rounds']}_seed_{row['seed']}": row for row in checkpoints}
    if "checkpoint_preflight" not in report:
        preflight_rows = []
        for checkpoint in checkpoints:
            model, identity = _load_checkpoint_model(checkpoint, ce)
            preflight_rows.append(
                {
                    "checkpoint_rounds": checkpoint["checkpoint_rounds"],
                    "seed": checkpoint["seed"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "model_state_sha256": identity["model_state_sha256"],
                    "shared_core_count": identity["shared_core_count"],
                    "depth_embedding_shape": identity["depth_embedding_shape"],
                    "gate_logits_shape": identity["gate_logits_shape"],
                    "pass": True,
                }
            )
            del model
            gc.collect()
        report["checkpoint_preflight"] = preflight_rows
        _write_json(output_dir / "ttr_scaling_progress.json", report)
    elif len(report["checkpoint_preflight"]) != 15 or any(row.get("pass") is not True for row in report["checkpoint_preflight"]):
        raise ValueError("TTR resume checkpoint preflight matrix is incomplete or failed")

    for k_train in TRAINING_ROUNDS:
        for seed in SEEDS:
            key = f"K{k_train}_seed_{seed}"
            result = report["checkpoint_results"].setdefault(
                key,
                {
                    "checkpoint_rounds": k_train,
                    "seed": seed,
                    "condition": f"K{k_train}",
                    "continuation_policy": "TAIL_REPEAT",
                    "inference_results": {},
                },
            )
            checkpoint = checkpoint_by_key[key]
            if "model_state_sha256" not in result:
                model, identity = _load_checkpoint_model(checkpoint, ce)
                result.update(identity)
                result["checkpoint_matrix_identity"] = {field: checkpoint[field] for field in ("block", "run_report_path", "run_report_sha256", "checkpoint_path")}
            else:
                model, identity = _load_checkpoint_model(checkpoint, ce, quality.BE376_SHA256)
                if identity["model_state_sha256"] != result["model_state_sha256"]:
                    raise ValueError(f"loaded model-state identity drift on resume: {key}")
            for k_infer in INFERENCE_ROUNDS:
                infer_key = str(k_infer)
                if infer_key in result["inference_results"]:
                    continue
                metric = _evaluate_configuration(model, checkpoint, chunks_by_document, k_infer)
                metric["checkpoint_rounds"] = k_train
                metric["inference_rounds"] = k_infer
                metric["continuation_policy"] = "TAIL_REPEAT"
                metric["checkpoint_modified"] = False
                result["inference_results"][infer_key] = metric
                report["checkpoint_results"][key] = result
                _write_json(output_dir / "ttr_scaling_progress.json", report)
            if len(result["inference_results"]) != len(INFERENCE_ROUNDS):
                raise AssertionError(f"TTR inference grid incomplete for {key}")
            if _sha256_file(Path(checkpoint["checkpoint_path"])) != checkpoint["checkpoint_sha256"]:
                raise RuntimeError(f"frozen checkpoint changed during TTR evaluation for {key}")
            nominal_v0 = float(result["inference_results"][str(k_train)]["V0_CHUNK0"]["nll"])
            expected_v0 = float(checkpoint["stored_training_endpoint_v0_nll"])
            v0_diff = nominal_v0 - expected_v0
            if abs(v0_diff) > GUARDRAIL:
                raise ValueError(f"nominal K V0 portability check failed for {key}: delta={v0_diff}")
            result["nominal_k_v0_portability"] = {
                "stored_training_endpoint_v0_nll": expected_v0,
                "inference_at_training_rounds_v0_nll": nominal_v0,
                "absolute_difference": abs(v0_diff),
                "guardrail": GUARDRAIL,
                "pass": True,
            }
            result["checkpoint_modified"] = False
            result["model_state_sha256_after_all_inference"] = quality._value_hash(model.state_dict())
            if result["model_state_sha256_after_all_inference"] != result["model_state_sha256"]:
                raise RuntimeError(f"base checkpoint model state changed during TTR evaluation for {key}")
            del model
            gc.collect()
            _write_json(output_dir / "ttr_scaling_progress.json", report)

    if set(report["checkpoint_results"]) != expected_keys:
        raise AssertionError("TTR scaling did not evaluate all 15 frozen checkpoints")
    summaries: dict[str, Any] = {}
    v0_descriptive: dict[str, Any] = {}
    for k_train in TRAINING_ROUNDS:
        training_condition = f"K{k_train}"
        token_weighted_by_kinfer: dict[str, Any] = {}
        document_macro_by_kinfer: dict[str, Any] = {}
        v0_by_kinfer: dict[str, Any] = {}
        for k_infer in INFERENCE_ROUNDS:
            token_rows = []
            macro_rows = []
            v0_rows = []
            for seed in SEEDS:
                checkpoint_result = report["checkpoint_results"][f"K{k_train}_seed_{seed}"]
                baseline = checkpoint_result["inference_results"][str(k_train)]
                current = checkpoint_result["inference_results"][str(k_infer)]
                token_rows.append(
                    {
                        "seed": seed,
                        "delta": float(current["VALL_TOKEN_WEIGHTED"]["nll"]) - float(baseline["VALL_TOKEN_WEIGHTED"]["nll"]),
                        "NLL_K_infer": float(current["VALL_TOKEN_WEIGHTED"]["nll"]),
                        "NLL_K_train": float(baseline["VALL_TOKEN_WEIGHTED"]["nll"]),
                    }
                )
                macro_rows.append(
                    {
                        "seed": seed,
                        "delta": float(current["VALL_DOCUMENT_MACRO"]["nll"]) - float(baseline["VALL_DOCUMENT_MACRO"]["nll"]),
                        "NLL_K_infer": float(current["VALL_DOCUMENT_MACRO"]["nll"]),
                        "NLL_K_train": float(baseline["VALL_DOCUMENT_MACRO"]["nll"]),
                    }
                )
                v0_rows.append(
                    {
                        "seed": seed,
                        "nll": float(current["V0_CHUNK0"]["nll"]),
                        "note": "descriptive historical continuity only; not used for the round-scaling conclusion",
                    }
                )
            def summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
                values = [float(row["delta"]) for row in rows]
                mean = sum(values) / len(values)
                sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
                half = T_CRIT_90_DF4 * sample_sd / math.sqrt(len(values))
                return {
                    "metric": metric,
                    "n": len(values),
                    "per_seed": rows,
                    "mean_delta": mean,
                    "sample_sd": sample_sd,
                    "t_critical_90_two_sided_df4": T_CRIT_90_DF4,
                    "ci90": [mean - half, mean + half],
                    "decision_threshold": None,
                }
            token_weighted_by_kinfer[str(k_infer)] = summarize(token_rows, "VALL_TOKEN_WEIGHTED")
            document_macro_by_kinfer[str(k_infer)] = summarize(macro_rows, "VALL_DOCUMENT_MACRO")
            v0_by_kinfer[str(k_infer)] = {
                "per_seed_nll": v0_rows,
                "mean_nll": sum(row["nll"] for row in v0_rows) / len(v0_rows),
                "decision_metric": False,
            }
        summaries[training_condition] = {
            "checkpoint_rounds": k_train,
            "inference_rounds_grid": list(INFERENCE_ROUNDS),
            "primary_VALL_token_weighted_delta_vs_K_train": token_weighted_by_kinfer,
            "secondary_VALL_document_macro_delta_vs_K_train": document_macro_by_kinfer,
        }
        v0_descriptive[training_condition] = v0_by_kinfer

    report["primary_metric"] = "VALL token-weighted NLL"
    report["secondary_metric"] = "VALL document-macro NLL"
    report["delta_definition"] = "NLL(K_infer) - NLL(K_train), paired within each frozen checkpoint/seed"
    report["summary_by_checkpoint_rounds"] = summaries
    report["V0_CHUNK0_descriptive_only"] = v0_descriptive
    report["status"] = "TEST_TIME_RECURRENCE_SCALING_A_COMPLETE"
    report["checkpoint_modified"] = False
    report["scientific_updates"] = 0
    report["test_split_loaded"] = False
    report["teacher_forward_used"] = False
    report["hidden_cache_used"] = False
    report["native_dll_used_for_evaluation"] = False
    report["secondary_CYCLE_policy"] = "NOT_RUN; TAIL_REPEAT results require review first"
    report["report_self_sha256"] = quality._canonical_hash(report)
    _write_json(output_dir / "test_time_recurrence_scaling_a_report.json", report)
    _write_json(output_dir / "ttr_scaling_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-test-time-recurrence-scaling-a-go", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.confirm_test_time_recurrence_scaling_a_go:
        parser.error("OMEGA-TEST-TIME-RECURRENCE-SCALING-A requires the explicit post-C GO")
    output_dir = args.output_dir.resolve()
    try:
        report = run_scaling(output_dir, resume=args.resume)
    except Exception as error:
        if output_dir.exists():
            _write_json(
                output_dir / "ttr_scaling_failure.json",
                {
                    "status": "FAILED_RUNTIME",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "scientific_updates": 0,
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint_count": len(report["checkpoint_results"]),
                "round_grid": report["policy"]["inference_rounds"],
                "primary_metric": report["primary_metric"],
                "summaries": report["summary_by_checkpoint_rounds"],
                "report": str(output_dir / "test_time_recurrence_scaling_a_report.json"),
                "scientific_updates": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

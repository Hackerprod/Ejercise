"""Evaluate update-2000 weights with only gate_logits restored from update 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_backend_quality_qualification as quality  # noqa: E402

REPO_ROOT = HERE.parents[2]
LAB_ROOT = REPO_ROOT / "t1_trainability_lab_v0.1.0"
CAMPAIGN = LAB_ROOT / "campaign"
R1_SCOPE = CAMPAIGN / "omega_core_lm_0_r1_scientific_scoping_a"
CE_DIR = CAMPAIGN / "omega_ce_only_baseline"
EXPANDED_DIR = CAMPAIGN / "omega_expanded_frozen_validation"
INPUTS_DIR = HERE / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
QUALIFICATION_MANIFEST = INPUTS_DIR / "qualification_manifest.json"
NORMAL_PREFLIGHT = HERE / "results" / "r1_state_causal_diagnostics_20260926T181530" / "normal_portability_preflight.json"
ANALYSIS_REPORT = HERE / "results" / "formal_quality_analysis.json"
BLOCK_REPORTS = (
    HERE / "results" / "block_A" / "block_A_report.json",
    HERE / "results" / "block_B" / "block_B_report.json",
)
UPDATE_ZERO = "checkpoint_00000.pt"
UPDATE_FINAL = "checkpoint_02000.pt"
EXPECTED_FINAL_UPDATE = 2000
EXPECTED_DOCUMENT_COUNT = 60
EXPECTED_TOKENS = 30_720
SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
K_VALUES = (1, 4)
BACKENDS = ("pytorch", "native")
T_CRIT_90_DF4 = 2.131846786326383


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone(memory_format=torch.preserve_format)


def _state_hash(state: dict[str, Any]) -> str:
    hashed = {
        name: _tensor_hash(value) if torch.is_tensor(value) else value
        for name, value in sorted(state.items())
    }
    return _canonical_hash(hashed)


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "stride": list(value.stride()) if value.layout == torch.strided else None,
        "storage_offset": int(value.storage_offset()) if value.layout == torch.strided else None,
    }


def _load_metadata() -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any, dict[str, Any]]:
    manifest = json.loads(QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    signature = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError("sealed qualification manifest self-hash mismatch")

    preflight = json.loads(NORMAL_PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("checkpoint_count") != 20:
        raise PermissionError("NORMAL same-machine preflight is not PASS for all 20 checkpoints")
    if preflight.get("validation_manifest_sha256") != manifest["validation_manifest"]["verification"]["manifest_sha256"]:
        raise ValueError("NORMAL preflight validation-manifest identity drift")
    if preflight.get("normal_scorer", {}).get("sha256") != manifest["source_identities"]["r1_scope_runner"]["sha256"]:
        raise ValueError("stored NORMAL scorer identity differs from sealed R1 evaluator")
    if any(float(row["absolute_difference_nats_per_token"]) > 1.0e-4 or row.get("pass") is not True for row in preflight["checks"]):
        raise PermissionError("one or more NORMAL checkpoints exceed fixed same-host 1e-4 guardrail")

    if str(R1_SCOPE) not in sys.path:
        sys.path.insert(0, str(R1_SCOPE))
    if str(CE_DIR) not in sys.path:
        sys.path.insert(0, str(CE_DIR))
    if str(EXPANDED_DIR) not in sys.path:
        sys.path.insert(0, str(EXPANDED_DIR))
    import run_scientific_scoping_a as r1  # noqa: PLC0415
    import run_omega_ce_only_baseline as ce  # noqa: PLC0415
    import run_omega_expanded_frozen_validation as expanded  # noqa: PLC0415

    identities = {
        "r1_evaluator": {"path": str(Path(r1.__file__).resolve()), "sha256": _sha256_file(Path(r1.__file__).resolve())},
        "model_factory": {"path": str(Path(ce.__file__).resolve()), "sha256": _sha256_file(Path(ce.__file__).resolve())},
        "validation_loader": {"path": str(Path(expanded.__file__).resolve()), "sha256": _sha256_file(Path(expanded.__file__).resolve())},
        "torch_version": str(torch.__version__),
        "cpu": preflight.get("cpu"),
    }
    for name, source_key in (("r1_evaluator", "r1_scope_runner"), ("model_factory", "ce_model_factory"), ("validation_loader", "expanded_validation_runner")):
        if identities[name]["path"] != str(Path(manifest["source_identities"][source_key]["path"]).resolve()) or identities[name]["sha256"] != manifest["source_identities"][source_key]["sha256"]:
            raise ValueError(f"GATE-INIT source identity drift: {name}")
    policy = ce.validate_policy()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("GATE-INIT requires PyTorch intra-op/inter-op 4/1")
    identities["runtime_policy"] = policy
    return manifest, preflight, r1, ce, expanded, identities


def _load_run_rows(manifest: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    runs: dict[tuple[int, int, str], dict[str, Any]] = {}
    expected_statuses = {"A": ("20260913", "20260914"), "B": ("20260915", "20260916", "20260917")}
    for block in ("A", "B"):
        path = HERE / "results" / f"block_{block}" / f"block_{block}_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE" or report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"Block {block} report status/manifest drift")
        for item in report["runs"]:
            key = (int(item["seed"]), int(item["K"]), str(item["backend"]))
            if key in runs:
                raise ValueError(f"duplicate checkpoint route {key}")
            runs[key] = item
    expected = {(seed, k, backend) for seed in SEEDS for k in K_VALUES for backend in BACKENDS}
    if set(runs) != expected or len(runs) != 20:
        raise ValueError("GATE-INIT requires exactly five seeds×K1/K4×both backends")
    return runs


def _load_checkpoint_model(
    route: dict[str, Any], checkpoint_name: str, ce: Any, manifest: dict[str, Any]
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    run_report = route["run_report"]
    run_dir = Path(run_report["run_dir"])
    checkpoint_path = run_dir / checkpoint_name
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".identity.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    digest = _sha256_file(checkpoint_path)
    if digest != sidecar.get("sha256"):
        raise ValueError(f"checkpoint/sidecar hash mismatch: {checkpoint_path}")
    # Use mmap + weights-only deserialization; no AdamW object is created and
    # optimizer state is discarded without restore or use.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("identity") != run_report.get("identity"):
        raise ValueError(f"checkpoint/run-report identity mismatch: {checkpoint_path}")
    identity = payload["identity"]
    if sidecar.get("identity_sha256") != _canonical_hash(identity):
        raise ValueError(f"checkpoint sidecar identity hash mismatch: {checkpoint_path}")
    expected_update = 0 if checkpoint_name == UPDATE_ZERO else EXPECTED_FINAL_UPDATE
    if int(payload.get("completed_updates", -1)) != expected_update or int(payload.get("next_update", -1)) != expected_update:
        raise ValueError(f"checkpoint update identity mismatch: {checkpoint_path}")
    payload.pop("optimizer", None)
    state_dict = payload["model"]
    if any(not bool(torch.isfinite(value).all()) for value in state_dict.values() if torch.is_tensor(value) and value.is_floating_point()):
        raise FloatingPointError(f"non-finite model tensor in checkpoint: {checkpoint_path}")
    model = ce.fresh_model(int(identity["seed"]), int(identity["K"]))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    copied_state = {name: value.detach().cpu().clone(memory_format=torch.preserve_format) for name, value in state_dict.items()}
    result = {
        "path": str(checkpoint_path),
        "sha256": digest,
        "update": expected_update,
        "backend": identity["backend"],
        "K": int(identity["K"]),
        "seed": int(identity["seed"]),
        "model_state_sha256": _state_hash(copied_state),
        "gate_logits_metadata": _tensor_metadata(copied_state["gate_logits"]),
        "optimizer_restored": False,
        "optimizer_state_used": False,
    }
    return model, result, copied_state


def _normal_rows_by_route(preflight: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {(int(row["seed"]), int(row["K"]), str(row["backend"])): row for row in preflight["checks"]}


def _eval_gate_init_model(r1: Any, ce: Any, docs: list[dict[str, Any]], final_state: dict[str, torch.Tensor], initial_gate: torch.Tensor, seed: int, rounds: int) -> dict[str, Any]:
    model = ce.fresh_model(seed, rounds)
    candidate_state = {name: value.detach().clone(memory_format=torch.preserve_format) for name, value in final_state.items()}
    candidate_state["gate_logits"] = initial_gate.detach().clone(memory_format=torch.preserve_format)
    untouched_comparisons = []
    for name, value in final_state.items():
        if name == "gate_logits":
            continue
        replacement = candidate_state[name]
        equal = torch.equal(value, replacement)
        untouched_comparisons.append({"name": name, "torch_equal": equal, "reference_meta": _tensor_metadata(value), "replacement_meta": _tensor_metadata(replacement), "metadata_equal": _tensor_metadata(value) == _tensor_metadata(replacement)})
        if not equal or _tensor_metadata(value) != _tensor_metadata(replacement):
            raise AssertionError(f"GATE-INIT changed non-gate tensor: {name}")
    if tuple(candidate_state["gate_logits"].shape) != tuple(final_state["gate_logits"].shape) or candidate_state["gate_logits"].dtype != final_state["gate_logits"].dtype:
        raise ValueError("update0 gate tensor shape/dtype differs from update2000")
    model.load_state_dict(candidate_state, strict=True)
    model.eval()
    before_hash = _state_hash(model.state_dict())
    with torch.no_grad():
        metric = r1.evaluate_validation(model, docs)
    after_state = model.state_dict()
    after_hash = _state_hash(after_state)
    if before_hash != after_hash:
        raise RuntimeError("weights changed during GATE-INIT evaluation")
    if not bool(metric.get("finite")) or int(metric.get("tokens", -1)) != EXPECTED_TOKENS:
        raise FloatingPointError("GATE-INIT evaluator returned invalid NLL/tokens")
    return {
        "nll_gate_init": float(metric["nll"]),
        "tokens": int(metric["tokens"]),
        "finite": bool(metric["finite"]),
        "weights_unchanged_during_scoring": True,
        "gate_replaced_tensor_metadata": _tensor_metadata(candidate_state["gate_logits"]),
        "gate_replaced_tensor_sha256": _tensor_hash(candidate_state["gate_logits"]),
        "only_modified_key": "gate_logits",
        "all_other_model_tensors_torch_equal": all(row["torch_equal"] and row["metadata_equal"] for row in untouched_comparisons),
        "untouched_tensor_comparisons": untouched_comparisons,
        "scored_model_state_sha256": after_hash,
    }


def _summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError("GATE-INIT summary requires all five seeds")
    mean = sum(values) / 5.0
    sd = float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item())
    half = 2.131846786326383 * sd / math.sqrt(5.0)
    return {"seed_values": values, "n": 5, "mean": mean, "sample_sd": sd, "ci90": [mean-half, mean+half], "t_critical_90_two_sided_df4": 2.131846786326383}


def run_gate_init(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"GATE-INIT output is immutable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, preflight, r1, ce, expanded, source_ids = _load_metadata()
    manifest_signature = manifest["manifest_sha256"]
    documents, validation_manifest, _tokenizer = expanded._load_real_validation()
    if len(documents) != EXPECTED_DOCUMENT_COUNT or validation_manifest["manifest_sha256"] != manifest["validation_manifest"]["verification"]["manifest_sha256"]:
        raise ValueError("frozen validation60 identity drift")
    if validation_manifest["dataset"]["split"] != "validation":
        raise ValueError("GATE-INIT must only score validation split")

    runs: dict[tuple[int, int, str], dict[str, Any]] = {}
    for block in ("A", "B"):
        block_path = HERE / "results" / f"block_{block}" / f"block_{block}_report.json"
        report = json.loads(block_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE" or report.get("qualification_manifest_sha256") != manifest_signature or report.get("test_split_loaded") is not False:
            raise ValueError(f"Block {block} checkpoint source report identity/status mismatch")
        for route in report["runs"]:
            key = (int(route["seed"]), int(route["K"]), str(route["backend"]))
            if key in runs:
                raise ValueError(f"duplicate GATE-INIT route {key}")
            runs[key] = route
    expected_routes = {(seed, k, backend) for seed in SEEDS for k in K_VALUES for backend in BACKENDS}
    if set(runs) != expected_routes or len(runs) != 20:
        raise ValueError("GATE-INIT requires exactly 20 routes")
    normals = _normal_rows_by_route(preflight)
    if set(normals) != expected_routes:
        raise ValueError("stored NORMAL baseline does not contain the same 20 routes")

    # Portability check already ran on this same machine. Reuse its stored NLL;
    # only verify each checkpoint and baseline binding here. No NORMAL rescore.
    normal_by_route: dict[tuple[int, int, str], float] = {}
    checkpoint0_gate: dict[tuple[int, int, str], dict[str, Any]] = {}
    final_model_states: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
    checkpoint_identities: dict[tuple[int, int, str], dict[str, Any]] = {}
    normal_binding_rows = []
    for key in sorted(expected_routes):
        route = runs[key]
        stored_normal = normals[key]
        expected_nll = float(route["run_report"]["endpoint_nll_validation"]["nll"])
        if (
            stored_normal.get("checkpoint_sha256") is None
            or stored_normal.get("pass") is not True
            or float(stored_normal["stored_normal_nll"]) != expected_nll
            or float(stored_normal["current_host_normal_nll"]) != expected_nll
            or float(stored_normal["absolute_difference_nats_per_token"]) > 1.0e-4
        ):
            raise ValueError(f"stored NORMAL baseline binding failed for route {key}")
        final_model, final_identity, final_state = _load_checkpoint_model(route, UPDATE_FINAL, ce, manifest)
        initial_model, initial_identity, initial_state = _load_checkpoint_model(route, UPDATE_ZERO, ce, manifest)
        if stored_normal["checkpoint_sha256"] != final_identity["sha256"]:
            raise ValueError(f"stored NORMAL NLL is not bound to update-2000 checkpoint for route {key}")
        if final_identity["backend"] != key[2] or initial_identity["backend"] != key[2] or final_identity["K"] != key[1] or final_identity["seed"] != key[0] or initial_identity["update"] != 0 or final_identity["update"] != EXPECTED_FINAL_UPDATE:
            raise ValueError(f"GATE-INIT paired checkpoint route identity mismatch: {key}")
        gate0 = initial_state.get("gate_logits")
        gate2000 = final_state.get("gate_logits")
        if gate0 is None or gate2000 is None or _tensor_metadata(gate0) != _tensor_metadata(gate2000):
            raise ValueError(f"GATE-INIT raw gate tensor metadata mismatch: {key}")
        checkpoint0_gate[key] = {
            "checkpoint_identity": initial_identity,
            "gate_logits": _clone_tensor(gate0),
            "gate_logits_sha256": _tensor_hash(gate0),
            "gate_logits_metadata": _tensor_metadata(gate0),
        }
        checkpoint_identities[key] = {"update0": initial_identity, "update2000": final_identity}
        final_model_states[key] = final_state
        normal_by_route[key] = float(stored_normal["current_host_normal_nll"])
        normal_binding_rows.append({
            "seed": key[0], "K": key[1], "backend": key[2],
            "normal_nll_reused": expected_nll,
            "normal_checkpoint_sha256": stored_normal["checkpoint_sha256"],
            "update0_checkpoint_sha256": initial_identity["sha256"],
            "update2000_checkpoint_sha256": final_identity["sha256"],
            "normal_scorer_identity_matches": True,
            "checkpoint_pair_identity_matches": True,
        })
        del final_model, initial_model

    checkpoint_rows: list[dict[str, Any]] = []
    for key in sorted(expected_routes):
        seed, rounds, backend = key
        gate0_info = checkpoint0_gate[key]
        gate0 = gate0_info["gate_logits"]
        final_state = final_model_states[key]
        gate2000 = final_state.get("gate_logits")
        if gate2000 is None:
            raise KeyError(f"update-2000 checkpoint missing gate_logits for {key}")
        gate_shape = tuple(int(value) for value in gate2000.shape)
        if len(gate_shape) != 2 or gate_shape[0] != rounds:
            raise ValueError(f"gate_logits is not [K,D] for route {key}: {gate_shape}")
        initial_sigmoid = torch.sigmoid(gate0)
        final_sigmoid = torch.sigmoid(gate2000)
        replacement_result = _eval_gate_init_model(r1, ce, documents, final_state, gate0, seed, rounds)
        nll_normal = normal_by_route[key]
        nll_gate_init = float(replacement_result["nll_gate_init"])
        checkpoint_rows.append({
            "seed": seed,
            "K": rounds,
            "backend": backend,
            "nll_normal_reused": nll_normal,
            "nll_gate_init": nll_gate_init,
            "delta_gate_init": nll_gate_init - nll_normal,
            "tokens": replacement_result["tokens"],
            "finite": replacement_result["finite"],
            "update0_gate_logits_sha256": gate0_info["gate_logits_sha256"],
            "update0_gate_logits_metadata": gate0_info["gate_logits_metadata"],
            "update2000_gate_logits_sha256": _tensor_hash(gate2000),
            "update2000_gate_logits_metadata": _tensor_metadata(gate2000),
            "modified_keys": ["gate_logits"],
            "all_other_weights_exact": replacement_result["all_other_model_tensors_torch_equal"],
            "all_other_weight_comparisons": replacement_result["untouched_tensor_comparisons"],
            "gate_replacement_copy_sha256": replacement_result["gate_replaced_tensor_sha256"],
            "scored_model_state_sha256": replacement_result["scored_model_state_sha256"],
            "optimizer_restored_or_used": False,
        })

    checkpoint_summaries: dict[str, Any] = {}
    for rounds in K_VALUES:
        for backend in BACKENDS:
            group = [row for row in checkpoint_rows if row["K"] == rounds and row["backend"] == backend]
            values = [float(next(row["delta_gate_init"] for row in group if row["seed"] == seed)) for seed in SEEDS]
            checkpoint_summaries[f"K{rounds}_{backend}"] = _summary(values)

    report = {
        "schema": "omega-r1-gate-init-diagnostic-v1",
        "unit": "OMEGA-R1-STATE-CAUSAL-DIAGNOSTICS/GATE-INIT",
        "status": "COMPLETE_DIAGNOSTIC_NO_PASS_FAIL_GATE",
        "intervention": "At update2000 model, replace only gate_logits with actual tensor from the matching update0 checkpoint; no approximate constants",
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "formal_quality_verdict": json.loads(ANALYSIS_REPORT.read_text(encoding="utf-8"))["status"],
        "normal_scorer_reused": {
            "path": str(Path(r1.__file__).resolve()),
            "sha256": _sha256_file(Path(r1.__file__).resolve()),
            "stored_normal_preflight_path": str(NORMAL_PREFLIGHT),
            "stored_normal_preflight_sha256": _sha256_file(NORMAL_PREFLIGHT),
            "normal_nll_recomputed": False,
        },
        "validation_identity": {
            "manifest_sha256": validation_manifest["manifest_sha256"],
            "document_count": len(documents),
            "tokens": EXPECTED_TOKENS,
            "split": "validation",
        },
        "cpu_identity": preflight["cpu"],
        "torch_identity": preflight["torch"],
        "checkpoint_count": len(checkpoint_rows),
        "checkpoint_pair_identities": normal_binding_rows,
        "per_checkpoint_values": checkpoint_rows,
        "paired_seed_summaries_by_K_backend": checkpoint_summaries,
        "optimizer_restored_or_used": False,
        "teacher_or_hidden_cache_used": False,
        "test_split_loaded": False,
        "training_updates": 0,
        "weights_modified_on_disk": False,
        "only_parameter_key_replaced_in_scoring_copy": "gate_logits",
        "no_intervention_threshold_or_pass_fail_gate": True,
        "interpretation_predeclared": {
            "delta_positive": "gate updates improve NLL relative to update0 gate tensor",
            "delta_near_zero": "little evidence of useful gate adaptation",
            "delta_negative": "update0 gate tensor improves NLL relative to learned gate tensor",
        },
        "strict_trajectory_proximity": "FAIL_RECORDED",
        "original_causal_smoke": "FAIL_RECORDED",
        "harness_identity": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        "output_dir": str(output_dir),
    }
    report["report_self_sha256"] = _canonical_hash(report)
    quality._write_json(output_dir / "gate_init_diagnostic_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-sol-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_sol_go:
        parser.error("GATE-INIT requires explicit Sol GO")
    stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"gate_init_diagnostic_{stamp}"
    try:
        report = run_gate_init(output_dir)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(output_dir / "gate_init_diagnostic_failure.json", {
                "status": "FAILED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "output_dir": str(output_dir),
            })
        raise
    print(json.dumps({
        "status": report["status"],
        "report": str(output_dir / "gate_init_diagnostic_report.json"),
        "checkpoint_count": report["checkpoint_count"],
        "training_updates": report["training_updates"],
        "optimizer_restored_or_used": report["optimizer_restored_or_used"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

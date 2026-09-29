"""Read-only R1 transport-state reset/shuffle diagnostics over frozen checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LAB_ROOT = REPO_ROOT / "t1_trainability_lab_v0.1.0"
CAMPAIGN = LAB_ROOT / "campaign"
R1_SCOPE = CAMPAIGN / "omega_core_lm_0_r1_scientific_scoping_a"
CE_DIR = CAMPAIGN / "omega_ce_only_baseline"
EXPANDED_DIR = CAMPAIGN / "omega_expanded_frozen_validation"
INPUTS_DIR = HERE / "inputs_r1_masked_token_mean_v1_block_a_preflight_sealed_v2"
QUALIFICATION_MANIFEST = INPUTS_DIR / "qualification_manifest.json"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_backend_quality_qualification as quality  # noqa: E402

UPDATE_ZERO = "checkpoint_00000.pt"
UPDATE_FINAL = "checkpoint_02000.pt"
EXPECTED_FINAL_UPDATE = 2000
EXPECTED_DOCUMENT_COUNT = 60
EXPECTED_TOKENS = 30_720
PHYSICAL_BATCH = 8
WINDOW_TOKENS = 256
T_CRIT_90_DF4 = 2.131846786326383
NORMAL_PORTABILITY_LIMIT = 1.0e-4
ANALYSIS_REPORT = HERE / "results" / "formal_quality_analysis.json"
_OUTPUT_DIR_CREATED_BY_THIS_PROCESS = False


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


def _state_hash(state: dict[str, Any]) -> str:
    hashed: dict[str, Any] = {}
    for name, value in sorted(state.items()):
        if torch.is_tensor(value):
            hashed[name] = _tensor_hash(value)
        else:
            hashed[name] = value
    return _canonical_hash(hashed)


def _load_sealed_manifest() -> dict[str, Any]:
    manifest = json.loads(QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))
    signature = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not signature or signature != _canonical_hash(unsigned):
        raise ValueError("sealed quality manifest self-hash mismatch")
    if manifest.get("status") != "PREPARED_NO_SCIENTIFIC_TRAINING":
        raise ValueError("quality manifest has an unexpected state")
    if manifest.get("loss_contract", {}).get("id") != "R1_MASKED_TOKEN_MEAN_V1":
        raise ValueError("state diagnostics require canonical R1_MASKED_TOKEN_MEAN_V1")
    return manifest


def _source_identity_check(manifest: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    expected = manifest["source_identities"][name]
    resolved = path.resolve()
    digest = _sha256_file(resolved)
    if resolved != Path(expected["path"]).resolve() or digest != expected["sha256"]:
        raise ValueError(f"evaluator/model source identity drift: {resolved}")
    return {"path": str(resolved), "sha256": digest}


def _load_dependencies(manifest: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
    for path in (R1_SCOPE, CE_DIR, EXPANDED_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_scientific_scoping_a as r1  # noqa: PLC0415
    import run_omega_ce_only_baseline as ce  # noqa: PLC0415
    import run_omega_expanded_frozen_validation as expanded  # noqa: PLC0415

    r1_identity = _source_identity_check(manifest, "r1_scope_runner", R1_SCOPE / "run_scientific_scoping_a.py")
    ce_identity = _source_identity_check(manifest, "ce_model_factory", CE_DIR / "run_omega_ce_only_baseline.py")
    expanded_identity = _source_identity_check(manifest, "expanded_validation_runner", EXPANDED_DIR / "run_omega_expanded_frozen_validation.py")
    policy = ce.validate_policy()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("diagnostic scorer requires PyTorch intra-op/inter-op 4/1")
    return r1, ce, expanded, {
        "r1_scope_runner": r1_identity,
        "model_factory": ce_identity,
        "expanded_validation_runner": expanded_identity,
        "runtime_policy": policy,
        "cpu_identity": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "torch_identity": {
            "version": str(torch.__version__),
            "intraop_threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def _load_reported_runs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    for block, expected_seeds in (("A", [20260913, 20260914]), ("B", [20260915, 20260916, 20260917])):
        report_path = HERE / "results" / f"block_{block}" / f"block_{block}_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"completed Block {block} report is missing: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE" or report.get("qualification_manifest_sha256") is None:
            raise ValueError(f"Block {block} report is incomplete or unbound")
        if report.get("seeds") != expected_seeds or report.get("run_count") != len(expected_seeds) * 4:
            raise ValueError(f"Block {block} route count/seeds mismatch")
        if report.get("qualification_manifest_sha256") != json.loads(QUALIFICATION_MANIFEST.read_text(encoding="utf-8"))["manifest_sha256"]:
            raise ValueError(f"Block {block} report uses another qualification manifest")
        if report.get("test_split_loaded") is not False:
            raise ValueError(f"Block {block} report indicates forbidden test-split use")
        runs.extend(report["runs"])
    keys = {(int(row["seed"]), int(row["K"]), str(row["backend"])) for row in runs}
    expected = {(seed, k, backend) for seed in (20260913, 20260914, 20260915, 20260916, 20260917) for k in (1, 4) for backend in ("pytorch", "native")}
    if keys != expected or len(runs) != 20:
        raise ValueError("must have exactly five seeds × K1/K4 × both backend checkpoints")
    by_key = {(int(row["seed"]), int(row["K"]), str(row["backend"])): row for row in runs}
    return by_key, runs


def _load_checkpoint_model(
    row: dict[str, Any],
    checkpoint_name: str,
    ce: Any,
    manifest: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    run_report = row["run_report"]
    run_dir = Path(run_report["run_dir"])
    checkpoint_path = run_dir / checkpoint_name
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".identity.json")
    if not checkpoint_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"checkpoint/sidecar missing: {checkpoint_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checkpoint_file_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_file_sha256 != sidecar.get("sha256"):
        raise ValueError(f"checkpoint file/sidecar hash mismatch: {checkpoint_path}")

    # Checkpoint container includes optimizer tensors; mmap/weights-only decode
    # is used, then optimizer payload is discarded without restoring or reading it.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != "omega-backend-quality-checkpoint-v1":
        raise ValueError(f"unexpected checkpoint schema: {checkpoint_path}")
    if payload.get("identity") != run_report.get("identity"):
        raise ValueError(f"checkpoint/run-report identity mismatch: {checkpoint_path}")
    identity = payload["identity"]
    optimizer_payload_present = "optimizer" in payload
    payload.pop("optimizer", None)
    if sidecar.get("identity_sha256") != _canonical_hash(identity):
        raise ValueError(f"checkpoint sidecar identity hash mismatch: {checkpoint_path}")
    if identity.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError(f"checkpoint qualification-manifest identity mismatch: {checkpoint_path}")
    expected_update = 0 if checkpoint_name == UPDATE_ZERO else EXPECTED_FINAL_UPDATE
    if int(payload.get("completed_updates", -1)) != expected_update or int(payload.get("next_update", -1)) != expected_update:
        raise ValueError(f"checkpoint update boundary mismatch: {checkpoint_path}")
    state_dict = payload["model"]
    if any(not bool(torch.isfinite(value).all()) for value in state_dict.values() if torch.is_tensor(value) and value.is_floating_point()):
        raise FloatingPointError(f"non-finite model tensor in checkpoint: {checkpoint_path}")
    model = ce.fresh_model(int(row["seed"]), int(row["K"]))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model_state_sha256 = _state_hash(model.state_dict())
    result = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_file_sha256,
        "sidecar_sha256": sidecar.get("sha256"),
        "sidecar_identity_sha256": sidecar.get("identity_sha256"),
        "expected_sidecar_identity_sha256": _canonical_hash(identity),
        "sidecar_identity_valid": sidecar.get("identity_sha256") == _canonical_hash(identity),
        "update": expected_update,
        "backend": identity["backend"],
        "K": int(identity["K"]),
        "seed": int(identity["seed"]),
        "model_state_sha256": model_state_sha256,
        "optimizer_restored": False,
        "optimizer_state_used": False,
        "optimizer_payload_ignored": optimizer_payload_present,
    }
    if not result["sidecar_identity_valid"]:
        raise ValueError(f"checkpoint sidecar identity hash mismatch: {checkpoint_path}")
    gate_logits = state_dict.get("gate_logits")
    if gate_logits is None:
        raise KeyError(f"checkpoint missing gate_logits: {checkpoint_path}")
    return model, result, {name: _clone_cpu(value) for name, value in state_dict.items()}


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone(memory_format=torch.preserve_format)


def _normal_preflight(
    by_key: dict[tuple[int, int, str], dict[str, Any]],
    r1: Any,
    ce: Any,
    validation_docs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    final_models: dict[tuple[int, int, str], dict[str, Any]] = {}
    all_pass = True
    for seed in (20260913, 20260914, 20260915, 20260916, 20260917):
        for rounds in (1, 4):
            for backend in ("pytorch", "native"):
                key = (seed, rounds, backend)
                run = by_key[key]
                expected_nll = float(run["run_report"]["endpoint_nll_validation"]["nll"])
                expected_tokens = int(run["run_report"]["endpoint_nll_validation"]["tokens"])
                model, checkpoint_identity, state = _load_checkpoint_model(run, UPDATE_FINAL, ce, manifest)
                before_hash = _state_hash(model.state_dict())
                metric = r1.evaluate_validation(model, validation_docs)
                after_hash = _state_hash(model.state_dict())
                nll = float(metric["nll"])
                difference = abs(nll - expected_nll)
                passed = bool(metric.get("finite")) and int(metric.get("tokens", -1)) == EXPECTED_TOKENS and expected_tokens == EXPECTED_TOKENS and difference <= NORMAL_PORTABILITY_LIMIT and before_hash == after_hash
                all_pass = all_pass and passed and checkpoint_identity["sidecar_identity_valid"]
                rows.append({
                    "seed": seed,
                    "K": rounds,
                    "backend": backend,
                    "checkpoint": checkpoint_identity,
                    "stored_normal_nll": expected_nll,
                    "current_machine_normal_nll": nll,
                    "abs_difference": difference,
                    "guardrail_nats_per_token": NORMAL_PORTABILITY_LIMIT,
                    "tokens": int(metric.get("tokens", -1)),
                    "finite": bool(metric.get("finite")),
                    "weights_unchanged": before_hash == after_hash,
                    "pass": passed,
                })
                final_models[key] = {name: _clone_cpu(value) for name, value in model.state_dict().items()}
                del model
    return rows, final_models, all_pass


def _evaluate_transport_condition(
    r1: Any,
    model: torch.nn.Module,
    documents: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    if mode not in ("ANCHOR_RESET", "STATE_SHUFFLE"):
        raise ValueError(mode)
    model.eval()
    doc_nll: list[list[float | None]] = [[None, None] for _ in documents]
    group_map: list[dict[str, Any]] = []
    with torch.no_grad():
        for group_start in range(0, len(documents), PHYSICAL_BATCH):
            doc_indices = list(range(group_start, min(group_start + PHYSICAL_BATCH, len(documents))))
            group_states: list[torch.Tensor] = []
            for doc_index in doc_indices:
                source = torch.tensor(documents[doc_index]["tokens"], dtype=torch.long).unsqueeze(0)
                state0 = model.initial_state(1, device=torch.device("cpu"))
                input0 = source[:, :WINDOW_TOKENS]
                target0 = source[:, 1 : WINDOW_TOKENS + 1]
                state_after0, logits0 = r1.forward_window_adapter(model, input0, state0)
                ce0 = F.cross_entropy(logits0.reshape(-1, logits0.shape[-1]), target0.reshape(-1), reduction="none")
                doc_nll[doc_index][0] = float(ce0.sum().item())
                group_states.append(state_after0.detach().clone())

            group_size = len(doc_indices)
            permutation = [(index + 1) % group_size for index in range(group_size)]
            source_for_destination = {permutation[source_index]: source_index for source_index in range(group_size)}
            for local_destination, doc_index in enumerate(doc_indices):
                source = torch.tensor(documents[doc_index]["tokens"], dtype=torch.long).unsqueeze(0)
                input1 = source[:, WINDOW_TOKENS : 2 * WINDOW_TOKENS]
                target1 = source[:, WINDOW_TOKENS + 1 : 2 * WINDOW_TOKENS + 1]
                if mode == "ANCHOR_RESET":
                    transported_state = model.initial_state(1, device=torch.device("cpu"))
                    donor_local_index = None
                else:
                    donor_local_index = source_for_destination[local_destination]
                    transported_state = group_states[donor_local_index].detach().clone()
                state_after1, logits1 = r1.forward_window_adapter(model, input1, transported_state)
                ce1 = F.cross_entropy(logits1.reshape(-1, logits1.shape[-1]), target1.reshape(-1), reduction="none")
                doc_nll[doc_index][1] = float(ce1.sum().item())
            group_map.append({
                "validation_document_indices": doc_indices,
                "batch_size": group_size,
                "state_source_to_window1_destination": {str(source_index): int(permutation[source_index]) for source_index in range(group_size)},
                "window1_destination_to_state_source": {str(destination): int(source_for_destination[destination]) for destination in range(group_size)},
                "fixed_points": [index for index, destination in enumerate(permutation) if index == destination],
            })
    if any(value is None for pair in doc_nll for value in pair):
        raise AssertionError("one or more validation targets did not receive an NLL")
    total_nll = 0.0
    total_tokens = 0
    for nll0, nll1 in doc_nll:
        total_nll += float(nll0)
        total_tokens += WINDOW_TOKENS
        total_nll += float(nll1)
        total_tokens += WINDOW_TOKENS
    return {
        "nll": total_nll / total_tokens,
        "tokens": total_tokens,
        "finite": math.isfinite(total_nll / total_tokens),
        "mode": mode,
        "batch_size": PHYSICAL_BATCH,
        "tail_batch_size": len(documents) % PHYSICAL_BATCH,
        "fixed_point_free": all(not row["fixed_points"] for row in group_map),
        "group_permutation": group_map,
        "causal_intervention": "window0 state transport altered only before window1; inner recurrent-round anchor unchanged",
    }


def _t_summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise ValueError("state diagnostics require all five paired seeds")
    mean = sum(values) / 5.0
    sample_sd = float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item())
    half_width = T_CRIT_90_DF4 * sample_sd / math.sqrt(5.0)
    return {"seed_values": values, "n": 5, "mean": mean, "sample_sd": sample_sd, "t_critical_90_two_sided_df4": T_CRIT_90_DF4, "ci90": [mean - half_width, mean + half_width]}


def run_diagnostics(output_dir: Path) -> dict[str, Any]:
    global _OUTPUT_DIR_CREATED_BY_THIS_PROCESS
    if output_dir.exists():
        raise FileExistsError(f"state diagnostics output is immutable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _OUTPUT_DIR_CREATED_BY_THIS_PROCESS = True
    manifest = _load_sealed_manifest()
    original_tost = json.loads(ANALYSIS_REPORT.read_text(encoding="utf-8"))
    if original_tost.get("qualification_manifest_sha256") != manifest["manifest_sha256"] or original_tost.get("status") not in ("QUALIFIED", "NOT_EQUIVALENT", "INCONCLUSIVE"):
        raise ValueError("five-seed formal report/manifest identity mismatch")
    q_report_paths = [HERE / "results" / "block_A" / "block_A_report.json", HERE / "results" / "block_B" / "block_B_report.json"]
    r1, ce, expanded, identities = _load_dependencies(manifest)
    documents, val_manifest, _tokenizer = expanded._load_real_validation()
    if len(documents) != EXPECTED_DOCUMENT_COUNT or val_manifest["manifest_sha256"] != manifest["validation_manifest"]["verification"]["manifest_sha256"]:
        raise ValueError("current frozen validation loader differs from the sealed 60-document manifest")
    if val_manifest.get("dataset", {}).get("split") != "validation":
        raise ValueError("diagnostics may load validation only, never test split")
    if ANALYSIS_REPORT.read_text(encoding="utf-8").count('"seeds"') < 1:
        raise ValueError("formal five-seed analysis report malformed")

    routes: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for path in q_report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE" or report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"quality block report identity/status mismatch: {path}")
        for item in report["runs"]:
            key = (int(item["seed"]), int(item["K"]), str(item["backend"]))
            if key in by_key:
                raise ValueError(f"duplicate state diagnostics checkpoint route: {key}")
            by_key[key] = item
            routes.append({"seed": key[0], "K": key[1], "backend": key[2], "run_report_path": item["run_report_path"], "run_report_sha256": item["run_report_sha256"]})
    expected_keys = {(seed, k, backend) for seed in (20260913, 20260914, 20260915, 20260916, 20260917) for k in (1, 4) for backend in ("pytorch", "native")}
    if set(by_key) != expected_keys or len(routes) != 20:
        raise ValueError("state diagnostics require exactly 20 unique final checkpoints")

    checkpoint_identities: dict[tuple[int, int, str], dict[str, Any]] = {}
    final_model_states: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
    normal_rows: list[dict[str, Any]] = []
    normal_pass = True
    for key in sorted(expected_keys):
        row = by_key[key]
        run_report = row["run_report"]
        model, checkpoint_identity, model_state = _load_checkpoint_model(row, UPDATE_FINAL, ce, manifest)
        if int(run_report["endpoint_nll_validation"]["update"]) != EXPECTED_FINAL_UPDATE:
            raise ValueError(f"stored normal endpoint is not update 2000: {run_report['run_id']}")
        weight_hash_before = _state_hash(model.state_dict())
        metric = r1.evaluate_validation(model, documents)
        weight_hash_after = _state_hash(model.state_dict())
        stored_nll = float(run_report["endpoint_nll_validation"]["nll"])
        evaluated_nll = float(metric["nll"])
        difference = abs(evaluated_nll - stored_nll)
        passed = bool(metric["finite"]) and int(metric["tokens"]) == EXPECTED_TOKENS and difference <= NORMAL_PORTABILITY_LIMIT and weight_hash_before == weight_hash_after
        normal_pass = normal_pass and passed
        checkpoint_identities[key] = checkpoint_identity
        final_model_states[key] = model_state
        normal_rows.append({
            "seed": key[0], "K": key[1], "backend": key[2],
            "checkpoint_path": checkpoint_identity["checkpoint_path"],
            "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
            "stored_normal_nll": stored_nll,
            "current_host_normal_nll": evaluated_nll,
            "absolute_difference_nats_per_token": difference,
            "guardrail_nats_per_token": NORMAL_PORTABILITY_LIMIT,
            "tokens": int(metric["tokens"]),
            "finite": bool(metric["finite"]),
            "weights_unchanged": weight_hash_before == weight_hash_after,
            "pass": passed,
        })
        del model

    preflight = {
        "status": "PASS" if normal_pass and len(normal_rows) == 20 else "HOLD_PORTABILITY",
        "same_machine_as_training": True,
        "cpu": identities["cpu_identity"],
        "torch": identities["torch_identity"],
        "normal_scorer": identities["r1_scope_runner"],
        "validation_source": identities["expanded_validation_runner"],
        "validation_manifest_sha256": val_manifest["manifest_sha256"],
        "validation_document_count": len(documents),
        "normal_guardrail": "abs(current host NLL - stored update2000 NLL) <= 1e-4 nats/token for every checkpoint",
        "checkpoint_count": len(normal_rows),
        "checks": normal_rows,
        "ablations_started": False,
    }
    quality._write_json(output_dir / "normal_portability_preflight.json", preflight)
    if not normal_pass or len(normal_rows) != 20:
        report = {
            "schema": "omega-r1-state-causal-diagnostics-v1",
            "status": "HOLD_PORTABILITY",
            "normal_preflight": preflight,
            "ablations_started": False,
            "reason": "at least one of the 20 normal NLLs exceeded the fixed same-host portability guardrail",
            "output_dir": str(output_dir),
        }
        report["report_self_sha256"] = _canonical_hash(report)
        quality._write_json(output_dir / "state_causal_diagnostics_report.json", report)
        return report

    gate_rows: list[dict[str, Any]] = []
    for seed, k, backend in sorted(expected_keys):
        row = by_key[(seed, k, backend)]
        run_report = row["run_report"]
        update0_model, update0_identity, update0_state = _load_checkpoint_model(row, UPDATE_ZERO, ce, manifest)
        final_gate = final_model_states[(seed, k, backend)].get("gate_logits")
        initial_gate = update0_state.get("gate_logits")
        if initial_gate is None or final_gate is None or tuple(initial_gate.shape) != tuple(final_gate.shape):
            raise ValueError(f"gate_logits absent or shape drift for {run_report['run_id']}")
        if not bool(torch.isfinite(initial_gate).all()) or not bool(torch.isfinite(final_gate).all()):
            raise FloatingPointError(f"non-finite gate_logits for {run_report['run_id']}")
        initial_sigmoid = torch.sigmoid(initial_gate)
        final_sigmoid = torch.sigmoid(final_gate)
        rounds, indices = initial_gate.reshape(int(k), -1).shape
        gate_rows.append({
            "seed": seed, "K": k, "backend": backend,
            "checkpoint_0_sha256": update0_identity["checkpoint_sha256"],
            "checkpoint_2000_sha256": checkpoint_identities[(seed, k, backend)]["checkpoint_sha256"],
            "gate_shape": list(initial_gate.shape),
            "gate_values_by_round_index": [
                {
                    "round": r,
                    "index": i,
                    "logit_initial": float(initial_gate.reshape(rounds, indices)[r, i].item()),
                    "logit_update2000": float(final_gate.reshape(rounds, indices)[r, i].item()),
                    "sigmoid_initial": float(initial_sigmoid.reshape(rounds, indices)[r, i].item()),
                    "sigmoid_update2000": float(final_sigmoid.reshape(rounds, indices)[r, i].item()),
                    "delta_logit": float((final_gate.reshape(rounds, indices)[r, i] - initial_gate.reshape(rounds, indices)[r, i]).item()),
                    "delta_sigmoid": float((final_sigmoid.reshape(rounds, indices)[r, i] - initial_sigmoid.reshape(rounds, indices)[r, i]).item()),
                }
                for r in range(rounds) for i in range(indices)
            ],
            "scope_note": "mechanistic descriptive evidence only; not causal proof by itself",
        })
        del update0_model

    ablation_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        seed, k, backend = key
        base_model_state = final_model_states[key]
        normal_row = next(row for row in normal_rows if (row["seed"], row["K"], row["backend"]) == key)
        condition_results: dict[str, Any] = {}
        for mode in ("ANCHOR_RESET", "STATE_SHUFFLE"):
            model = ce.fresh_model(seed, k)
            model.load_state_dict({name: value.detach().clone(memory_format=torch.preserve_format) for name, value in base_model_state.items()}, strict=True)
            model.eval()
            weight_hash_before = _state_hash(model.state_dict())
            result = _evaluate_transport_condition(r1, model, documents, mode)
            weight_hash_after = _state_hash(model.state_dict())
            if weight_hash_before != weight_hash_after:
                raise RuntimeError(f"weights changed during {mode} evaluation for seed={seed},K={k},{backend}")
            condition_results[mode] = result
            del model
        nll_normal = normal_row["current_host_normal_nll"]
        nll_reset = float(condition_results["ANCHOR_RESET"]["nll"])
        nll_shuffle = float(condition_results["STATE_SHUFFLE"]["nll"])
        ablation_rows.append({
            "seed": seed, "K": k, "backend": backend,
            "checkpoint_update": EXPECTED_FINAL_UPDATE,
            "NLL_normal": nll_normal,
            "NLL_reset": nll_reset,
            "delta_reset": nll_reset - nll_normal,
            "NLL_shuffle": nll_shuffle,
            "delta_shuffle": nll_shuffle - nll_normal,
            "tokens": EXPECTED_TOKENS,
            "state_conditions": condition_results,
            "weights_unchanged_in_all_conditions": True,
        })

    summaries: dict[str, Any] = {}
    for k in (1, 4):
        for backend in ("pytorch", "native"):
            subset = [row for row in ablation_rows if row["K"] == k and row["backend"] == backend]
            for metric_name in ("delta_reset", "delta_shuffle"):
                values = [float(next(row[metric_name] for row in subset if row["seed"] == seed)) for seed in (20260913, 20260914, 20260915, 20260916, 20260917)]
                summaries[f"K{k}_{backend}_{metric_name}"] = _t_summary(values)

    backend_contrasts: dict[str, Any] = {}
    for k in (1, 4):
        for metric_name in ("delta_reset", "delta_shuffle"):
            values = []
            by_seed = {}
            for seed in (20260913, 20260914, 20260915, 20260916, 20260917):
                native_value = next(row[metric_name] for row in ablation_rows if row["seed"] == seed and row["K"] == k and row["backend"] == "native")
                pytorch_value = next(row[metric_name] for row in ablation_rows if row["seed"] == seed and row["K"] == k and row["backend"] == "pytorch")
                difference = float(native_value) - float(pytorch_value)
                values.append(difference)
                by_seed[str(seed)] = difference
            backend_contrasts[f"K{k}_{metric_name}_native_minus_pytorch"] = {
                "seed_values": by_seed,
                "mean": sum(values) / len(values),
                "sample_sd": float(torch.tensor(values, dtype=torch.float64).std(unbiased=True).item()),
                "ci90_descriptive_only": _t_summary(values)["ci90"],
                "equivalence_margin_defined": False,
                "interpretation": "secondary descriptive contrast; no new gate/margin",
            }

    report = {
        "schema": "omega-r1-state-causal-diagnostics-v1",
        "status": "COMPLETE_DIAGNOSTIC_NO_PASS_FAIL_GATE",
        "unit": "OMEGA-R1-STATE-CAUSAL-DIAGNOSTICS",
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "formal_quality_verdict": json.loads(ANALYSIS_REPORT.read_text(encoding="utf-8"))["status"],
        "evaluation_cpu": identities["cpu_identity"],
        "torch_build": identities["torch_identity"],
        "normal_portability_preflight": preflight,
        "normal_guardrail_pass": True,
        "reset_semantics": "W0 executed normally; W1 receives a fresh zero initial state instead of detached W0 output; internal round anchors unchanged",
        "shuffle_semantics": "W0 states are permuted between validation examples only; source index i goes to W1 destination (i+1) mod group_size; slots are never permuted",
        "state_shuffle_group_size": PHYSICAL_BATCH,
        "state_shuffle_batches": "ordered validation groups of 8; final group of 4 uses the same one-step cyclic derangement",
        "training_updates": 0,
        "weights_modified": False,
        "optimizer_restored_or_loaded": False,
        "teacher_or_hidden_cache_used": False,
        "test_split_loaded": False,
        "checkpoint_count": 20,
        "checkpoint_gate_motion": gate_rows,
        "per_checkpoint_conditions": ablation_rows,
        "paired_seed_summaries": summaries,
        "backend_contrasts_secondary": backend_contrasts,
        "historical_delta_ablation_threshold_used_as_gate": False,
        "strict_trajectory_proximity": "FAIL_RECORDED",
        "output_dir": str(output_dir),
        "harness_identity": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
    }
    report["report_self_sha256"] = _canonical_hash(report)
    quality._write_json(output_dir / "state_causal_diagnostics_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-sol-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_sol_go:
        parser.error("state diagnostics require the explicit Sol GO")
    stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"r1_state_causal_diagnostics_{stamp}"
    try:
        report = run_diagnostics(output_dir)
    except Exception as error:
        if _OUTPUT_DIR_CREATED_BY_THIS_PROCESS and output_dir.exists():
            quality._write_json(output_dir / "state_causal_diagnostics_failure.json", {
                "status": "FAILED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "output_dir": str(output_dir),
            })
        raise
    print(json.dumps({
        "status": report["status"],
        "report": str(output_dir / "state_causal_diagnostics_report.json"),
        "normal_portability_pass": report.get("normal_guardrail_pass"),
        "checkpoint_count": report.get("checkpoint_count"),
        "ablation_rows": len(report.get("per_checkpoint_conditions", [])),
        "optimizer_restored": report.get("optimizer_restored_or_loaded"),
        "training_updates": report.get("training_updates"),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "COMPLETE_DIAGNOSTIC_NO_PASS_FAIL_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

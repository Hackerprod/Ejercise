"""Run the explicitly authorized 8-update causal loss-contract smoke only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
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


ROUTE_ORDER = (("pytorch", 1), ("native", 1), ("pytorch", 4), ("native", 4))
SEED = 20260913
UPDATES_PER_ROUTE = 2
WINDOW_TOKENS = 256
EXPECTED_VALID_TOKENS = 2048


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(quality._canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _tensor_snapshot(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous().clone()


def _loss_source_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    function = quality.r1_masked_token_mean_loss
    path = Path(function.__code__.co_filename).resolve()
    digest = _sha256_file(path)
    expected = manifest["loss_contract"]["implementation"]
    if (
        path != Path(expected["path"]).resolve()
        or function.__name__ != expected["callable"]
        or digest != expected["sha256"]
        or quality.LOSS_CONTRACT != manifest["loss_contract"]["id"]
    ):
        raise ValueError("actually loaded loss callable/path/hash differs from sealed R1 loss contract")
    return {"callable": function.__name__, "module": function.__module__, "path": str(path), "sha256": digest}


def _validate_runtime_policy(ce: Any, p0: Any) -> dict[str, Any]:
    policy = ce.validate_policy()
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if int(torch.get_num_threads()) != 4 or int(torch.get_num_interop_threads()) != 1:
        raise RuntimeError("PyTorch thread contract is not 4/1")
    if int(policy["intraop_threads"]) != 4 or int(policy["interop_threads"]) != 1:
        raise RuntimeError("common model policy disagrees with PyTorch thread contract")
    if int(p0.PHYSICAL_BATCH) != 8:
        raise RuntimeError(f"runtime physical batch is {p0.PHYSICAL_BATCH}, expected 8")
    return policy


def _load_gate_module(quality: Any) -> Any:
    if str(quality.P2R0) not in sys.path:
        sys.path.insert(0, str(quality.P2R0))
    import run_omega_native_runtime_r1_bridge as local_gates  # noqa: PLC0415

    return local_gates


def _capture_route_update(
    *,
    backend: str,
    rounds: int,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    state: torch.Tensor,
    r2: Any,
    p0: Any,
    ce: Any,
    bridge: Any,
    local_gates: Any,
    state_part_weight: Any,
    source: torch.Tensor,
    pair_positions: list[int],
    pair_keys: list[list[str]],
    loss_source: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    window = update
    if window not in (0, 1):
        raise ValueError(f"smoke update must be window 0 or 1, got {window}")
    if int(valid_mask.sum().item()) != EXPECTED_VALID_TOKENS:
        raise ValueError("smoke mask must have 2,048 valid target tokens")

    model_state_before_sha256 = quality._value_hash(model.state_dict())
    state_input = state.detach() if window == 1 else state
    state_input_sha256 = quality._tensor_hash(state_input)
    optimizer.zero_grad(set_to_none=True)
    if backend == "native":
        next_state, student_logits, readout_states = r2._native_forward(
            model, inputs, state_input, state_part_weight, bridge
        )
    else:
        next_state, student_logits, readout_states = r2._reference_forward(
            model, inputs, state_input, p0
        )

    # Deliberately call the manifest-pinned canonical function directly after
    # the backend branch. No p0.hidden.base loss helper is reachable here.
    loss_terms = quality.r1_masked_token_mean_loss(
        student_logits, teacher_logits, targets, valid_mask
    )
    if loss_terms["valid_tokens"] != EXPECTED_VALID_TOKENS:
        raise ValueError("canonical loss returned unexpected valid-token count")
    loss_terms["total"].backward()
    gradients = {
        name: _tensor_snapshot(parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    missing_gradients = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    if missing_gradients:
        raise RuntimeError(f"missing pre-clip gradients: {missing_gradients}")
    pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
    clip_coefficient = min(1.0, float(ce.CLIP_NORM) / (pre_clip_norm + 1.0e-6))
    optimizer.step()
    optimizer_step = max(
        int(state_row["step"].item()) if torch.is_tensor(state_row.get("step")) else int(state_row.get("step", 0))
        for state_row in optimizer.state.values()
    )
    if optimizer_step != update + 1:
        raise AssertionError(f"expected optimizer step {update + 1}, got {optimizer_step}")

    next_state_detached = next_state.detach()
    model_state_after_sha256 = quality._value_hash(model.state_dict())
    snapshot = {
        "losses": {name: _tensor_snapshot(loss_terms[name]) for name in ("ce", "kl", "total")},
        "state_input": _tensor_snapshot(state_input),
        "readout_states": _tensor_snapshot(readout_states),
        "next_state": _tensor_snapshot(next_state_detached),
        "grads": gradients,
        "clip_norm": pre_clip_norm,
        "clip_coefficient": clip_coefficient,
        "parameters": {name: _tensor_snapshot(value) for name, value in model.named_parameters()},
        "optimizer": local_gates._optimizer_state(model, optimizer),
    }
    event = {
        "phase": "causal_smoke_update_completed",
        "status": "APPLIED",
        "backend": backend,
        "K": rounds,
        "seed": SEED,
        "update": update,
        "window": window,
        "pair": 0,
        "optimizer_step": optimizer_step,
        "optimizer_step_total": 2,
        "state_source": (
            "common_zero_initial_state"
            if window == 0
            else "route_local_window0_next_state_detached; computed before update0 AdamW and consumed with update1 weights"
        ),
        "state_input_sha256": state_input_sha256,
        "state_output_sha256": quality._tensor_hash(next_state_detached),
        "model_state_before_update_sha256": model_state_before_sha256,
        "model_state_after_update_sha256": model_state_after_sha256,
        "document_positions": pair_positions,
        "document_keys": pair_keys,
        "input_range": [window * WINDOW_TOKENS, (window + 1) * WINDOW_TOKENS],
        "target_range": [window * WINDOW_TOKENS + 1, (window + 1) * WINDOW_TOKENS + 1],
        "teacher_context_range": [0, 256] if window == 0 else [0, 512],
        "source_token_sha256": quality._tensor_hash(source),
        "teacher_logits_sha256": quality._tensor_hash(teacher_logits),
        "valid_tokens": int(loss_terms["valid_tokens"]),
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_callable": loss_source["callable"],
        "loss_source_path": loss_source["path"],
        "loss_source_sha256": loss_source["sha256"],
        "ce": float(loss_terms["ce"].detach().item()),
        "kl_tau_squared_applied": float(loss_terms["kl"].detach().item()),
        "total": float(loss_terms["total"].detach().item()),
        "pre_clip_gradient_norm": pre_clip_norm,
        "clip_norm_limit": float(ce.CLIP_NORM),
        "clip_coefficient_applied": clip_coefficient,
        "clip_intervened": clip_coefficient < 1.0,
        "model_state_sha256": model_state_after_sha256,
        "optimizer_state_sha256": quality._value_hash(optimizer.state_dict()),
        "physical_batch": 8,
        "effective_batch": 8,
        "gradient_accumulations": 1,
        "native_workers": 4 if backend == "native" else None,
        "torch_intraop_threads": 4,
        "torch_interop_threads": 1,
        "warmup_optimizer_updates": 0,
        "validation_or_test_loaded": False,
    }
    return event, next_state_detached, snapshot


def _add_failure_details(
    gate: dict[str, Any],
    reference: torch.Tensor,
    candidate: torch.Tensor,
    name: str,
    output_dir: Path,
) -> dict[str, Any]:
    left = reference.detach().cpu().contiguous()
    right = candidate.detach().cpu().contiguous()
    gate["tolerance"] = "abs(candidate-reference) <= 1.0e-5 + 1.0e-4*abs(reference); candidate finite"
    if tuple(left.shape) != tuple(right.shape):
        gate["failure_count"] = 1
        gate["failure_samples"] = [{"reason": "shape_mismatch", "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}]
        return gate
    difference = (right - left).abs()
    tolerance = 1.0e-5 + 1.0e-4 * left.abs()
    failed = (~torch.isfinite(right)) | (difference > tolerance)
    if failed.any():
        gate["failure_count"] = int(failed.sum().item())
        indices = failed.nonzero(as_tuple=False)
        gate["failure_samples"] = [
            {
                "index": [int(value) for value in index.tolist()],
                "reference": float(left[tuple(index.tolist())].item()),
                "candidate": float(right[tuple(index.tolist())].item()),
                "absolute_error": float(difference[tuple(index.tolist())].item()),
                "tolerance": float(tolerance[tuple(index.tolist())].item()),
            }
            for index in indices[:100]
        ]
        artifact_dir = output_dir / "gate_failure_tensors"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        artifact_path = artifact_dir / f"{safe_name}.pt"
        torch.save({"reference": left, "candidate": right, "failed_mask": failed, "absolute_error": difference, "tolerance": tolerance}, artifact_path)
        gate["full_failure_tensor_artifact"] = str(artifact_path)
        gate["full_failure_tensor_artifact_sha256"] = _sha256_file(artifact_path)
    return gate


def _compare_snapshots(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    update: int,
    rounds: int,
    local_gates: Any,
    output_dir: Path,
) -> dict[str, Any]:
    comparisons: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (f"loss.ce.update{update}", reference["losses"]["ce"], candidate["losses"]["ce"]),
        (f"loss.kl.update{update}", reference["losses"]["kl"], candidate["losses"]["kl"]),
        (f"loss.total.update{update}", reference["losses"]["total"], candidate["losses"]["total"]),
        (f"state_input.update{update}", reference["state_input"], candidate["state_input"]),
        (f"readout_states.update{update}", reference["readout_states"], candidate["readout_states"]),
        (f"next_state.update{update}", reference["next_state"], candidate["next_state"]),
        (f"clip.pre_norm.update{update}", torch.tensor(reference["clip_norm"], dtype=torch.float32), torch.tensor(candidate["clip_norm"], dtype=torch.float32)),
        (f"clip.coefficient.update{update}", torch.tensor(reference["clip_coefficient"], dtype=torch.float32), torch.tensor(candidate["clip_coefficient"], dtype=torch.float32)),
    ]
    if set(reference["grads"]) != set(candidate["grads"]):
        raise RuntimeError("backend gradient parameter-name sets differ")
    comparisons.extend(
        (f"grad.{name}.update{update}", reference["grads"][name], candidate["grads"][name])
        for name in reference["grads"]
    )
    if set(reference["parameters"]) != set(candidate["parameters"]):
        raise RuntimeError("backend parameter-name sets differ")
    comparisons.extend(
        (f"parameter.{name}.update{update}", reference["parameters"][name], candidate["parameters"][name])
        for name in reference["parameters"]
    )
    if set(reference["optimizer"]) != set(candidate["optimizer"]):
        raise RuntimeError("backend optimizer parameter-name sets differ")
    for name in reference["optimizer"]:
        if set(reference["optimizer"][name]) != set(candidate["optimizer"][name]):
            raise RuntimeError(f"optimizer slot sets differ for {name}")
        for slot in reference["optimizer"][name]:
            left, right = reference["optimizer"][name][slot], candidate["optimizer"][name][slot]
            if torch.is_tensor(left) and torch.is_tensor(right):
                comparisons.append((f"optimizer.{name}.{slot}.update{update}", left, right))

    rows: list[dict[str, Any]] = []
    for name, left, right in comparisons:
        gate = local_gates._tensor_gate(left, right, name=name)
        rows.append(_add_failure_details(gate, left, right, name, output_dir))
    return {
        "K": rounds,
        "update": update,
        "pass": all(bool(row["pass"]) for row in rows),
        "gate_source": str(Path(local_gates.__file__).resolve()),
        "gate_source_sha256": _sha256_file(Path(local_gates.__file__).resolve()),
        "atol": float(local_gates.ATOL),
        "rtol": float(local_gates.RTOL),
        "fallbacks_or_tolerance_changes": False,
        "tensors_compared": len(rows),
        "tensors": rows,
    }


def run_causal_smoke(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"smoke output is immutable; path already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = quality._load_qualification_manifest()
    loss_source = _loss_source_identity(manifest)
    r2, p0, ce, bridge, _modules = quality._load_real_dependencies()
    local_gates = _load_gate_module(quality)
    policy = _validate_runtime_policy(ce, p0)
    if quality._sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("selected be376 DLL changed before causal smoke")

    train_manifest = json.loads(quality.TRAIN_MANIFEST.read_text(encoding="utf-8"))
    documents, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    manifest_docs = train_manifest["documents"]
    actual_keys = [
        (row["full_text_sha256"], row["retained_513_token_sha256"])
        for row in documents
    ]
    expected_keys = [
        (row["full_text_sha256"], row["retained_513_token_sha256"])
        for row in manifest_docs
    ]
    if actual_keys != expected_keys or len(documents) != 602:
        raise ValueError("loaded train docs do not match sealed 602-document manifest")
    pair = train_manifest["pairs"][0]
    positions = [int(value) for value in pair["document_indices"]]
    pair_keys = pair["document_keys"]
    if len(positions) != 8 or len(pair_keys) != 8:
        raise ValueError("smoke pair must contain exactly eight frozen training documents")
    source = torch.tensor([documents[index]["tokens"] for index in positions], dtype=torch.long)
    windows: dict[int, dict[str, torch.Tensor]] = {}
    for window in (0, 1):
        offset = window * WINDOW_TOKENS
        inputs = source[:, offset : offset + WINDOW_TOKENS]
        targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
        if tuple(inputs.shape) != (8, 256) or tuple(targets.shape) != (8, 256):
            raise ValueError(f"smoke window {window} has unexpected input/target shape")
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, window)
        if tuple(teacher_logits.shape[:2]) != (8, 256) or int(teacher_logits.shape[-1]) != int(teacher_weight.shape[0]):
            raise ValueError(f"teacher logits have unexpected shape in window {window}: {tuple(teacher_logits.shape)}")
        windows[window] = {"inputs": inputs, "targets": targets, "valid_mask": valid_mask, "teacher_logits": teacher_logits}

    dll_identity = {
        "path": str(quality.BE376_DLL),
        "sha256": _sha256_file(quality.BE376_DLL),
        "expected_sha256": quality.BE376_SHA256,
        "native_source_commit": manifest["qualified_backend_config"]["native_source_commit"],
        "native_cpp_blob_sha256": manifest["qualified_backend_config"]["native_cpp_blob_sha256"],
    }
    cache_identity = {
        "manifest_path": manifest["hidden_cache"]["manifest"]["path"],
        "manifest_sha256": manifest["hidden_cache"]["manifest"]["sha256"],
        "cache_path": manifest["hidden_cache"]["cache_file"]["path"],
        "cache_sha256": manifest["hidden_cache"]["cache_file"]["sha256"],
        "teacher_revision": manifest["hidden_cache"]["teacher_revision"],
        "teacher_weight_sha256": quality._tensor_hash(teacher_weight),
        "teacher_bias_sha256": quality._tensor_hash(teacher_bias) if teacher_bias is not None else None,
    }
    source_identity = {
        "manifest_sha256": manifest["manifest_sha256"],
        "training_manifest_sha256": manifest["training_manifest"]["manifest_sha256"],
        "source_training_manifest_sha256": manifest["training_manifest"]["source_sha256"],
        "dataset_revision": manifest["hidden_cache"]["dataset_revision"],
        "dataset_split": "train",
        "train_document_count": len(documents),
        "pair_index": 0,
        "document_positions": positions,
        "document_keys": pair_keys,
        "source_pair_sha256": quality._tensor_hash(source),
        "teacher_logits_sha256_by_window": {str(window): quality._tensor_hash(row["teacher_logits"]) for window, row in windows.items()},
    }
    native_bridge_path = Path(bridge.__file__).resolve()
    native_bridge_sha256 = _sha256_file(native_bridge_path)
    expected_bridge = manifest["source_identities"]["r2_production_bridge"]
    if str(native_bridge_path) != str(Path(expected_bridge["path"]).resolve()) or native_bridge_sha256 != expected_bridge["sha256"]:
        raise ValueError("actually loaded native bridge path/hash differs from sealed manifest")

    common_initializations: dict[int, tuple[Path, str, dict[str, Any]]] = {}
    for rounds in (1, 4):
        init_path, init_sha = quality._prepare_common_initialization(
            ce, SEED, rounds, output_root / "common_initializations"
        )
        bundle = torch.load(init_path, map_location="cpu", weights_only=False)
        common_initializations[rounds] = (init_path, init_sha, bundle)

    route_outputs: dict[tuple[str, int], dict[str, Any]] = {}
    route_snapshots: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for backend, rounds in ROUTE_ORDER:
        init_path, init_sha, init_bundle = common_initializations[rounds]
        if _sha256_file(init_path) != init_sha or init_bundle["seed"] != SEED or init_bundle["K"] != rounds:
            raise ValueError("common smoke initialization bundle identity mismatch")
        model = ce.fresh_model(SEED, rounds)
        model.load_state_dict(init_bundle["model_state"], strict=True)
        if quality._value_hash(model.state_dict()) != init_bundle["model_state_sha256"]:
            raise ValueError("loaded smoke model differs from common initialization snapshot")
        torch.set_rng_state(init_bundle["torch_rng_state"])
        random.setstate(init_bundle["python_rng_state"])
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=ce.BASE_LR,
            betas=ce.ADAMW_BETAS,
            eps=ce.ADAMW_EPS,
            weight_decay=ce.WEIGHT_DECAY,
        )
        if optimizer.state or quality._value_hash(optimizer.state_dict()) != init_bundle["fresh_adamw_state_sha256"]:
            raise AssertionError("smoke route must start with a fresh empty AdamW state")
        state = init_bundle["initial_recurrent_state"].clone()
        if quality._tensor_hash(state) != init_bundle["initial_recurrent_state_sha256"]:
            raise ValueError("smoke recurrent initial state differs from common bundle")
        state_part_weight = None
        if backend == "native":
            bridge.configure_library(quality.BE376_DLL)
            if not bridge.runtime_abi_available():
                raise RuntimeError("selected be376 DLL lacks production runtime ABI")
            bridge.configure_runtime(4)
            state_part_weight = r2.prepare_native_model(model)

        route_id = f"{backend}_K{rounds}_seed_{SEED}"
        ledger_path = output_root / f"{route_id}.jsonl"
        snapshots: list[dict[str, Any]] = []
        route_events: list[dict[str, Any]] = []
        for update in (0, 1):
            window = windows[update]
            previous_state_sha256 = quality._tensor_hash(state)
            event, next_state, snapshot = _capture_route_update(
                backend=backend,
                rounds=rounds,
                update=update,
                model=model,
                optimizer=optimizer,
                inputs=window["inputs"],
                targets=window["targets"],
                teacher_logits=window["teacher_logits"],
                valid_mask=window["valid_mask"],
                state=state,
                r2=r2,
                p0=p0,
                ce=ce,
                bridge=bridge,
                local_gates=local_gates,
                state_part_weight=state_part_weight,
                source=source,
                pair_positions=positions,
                pair_keys=pair_keys,
                loss_source=loss_source,
                manifest=manifest,
            )
            event["route_id"] = route_id
            event["prior_state_sha256_at_forward"] = previous_state_sha256
            if update == 1:
                event["window0_detached_state_reused_exactly"] = previous_state_sha256 == route_events[0]["state_output_sha256"]
                event["window0_weights_updated_before_window1"] = event["model_state_before_update_sha256"] == route_events[0]["model_state_after_update_sha256"]
                if not event["window0_detached_state_reused_exactly"] or not event["window0_weights_updated_before_window1"]:
                    raise AssertionError("causal continuity did not reuse window0 state with window1 updated weights")
            _append_jsonl(ledger_path, event)
            route_events.append(event)
            snapshots.append(snapshot)
            state = next_state
        if len(route_events) != UPDATES_PER_ROUTE or [row["optimizer_step"] for row in route_events] != [1, 2]:
            raise AssertionError(f"route did not execute exactly two sequential optimizer steps: {route_id}")
        route_outputs[(backend, rounds)] = {
            "route_id": route_id,
            "backend": backend,
            "K": rounds,
            "seed": SEED,
            "common_initialization_path": str(init_path),
            "common_initialization_sha256": init_sha,
            "common_initial_model_state_sha256": init_bundle["model_state_sha256"],
            "initial_recurrent_state_sha256": init_bundle["initial_recurrent_state_sha256"],
            "physical_batch": 8,
            "effective_batch": 8,
            "gradient_accumulations": 1,
            "native_workers": 4 if backend == "native" else None,
            "torch_intraop_threads": 4,
            "torch_interop_threads": 1,
            "warmup_optimizer_updates": 0,
            "updates": route_events,
            "route_ledger_path": str(ledger_path),
            "route_ledger_sha256": _sha256_file(ledger_path),
        }
        route_snapshots[(backend, rounds)] = snapshots

    gates: list[dict[str, Any]] = []
    for rounds in (1, 4):
        pytorch_snapshots = route_snapshots[("pytorch", rounds)]
        native_snapshots = route_snapshots[("native", rounds)]
        for update in (0, 1):
            gates.append(_compare_snapshots(
                pytorch_snapshots[update], native_snapshots[update],
                update=update, rounds=rounds, local_gates=local_gates, output_dir=output_root,
            ))

    dll_identity["loaded_path"] = str(Path(bridge.loaded_library_path()).resolve())
    dll_identity["loaded_sha256"] = _sha256_file(Path(bridge.loaded_library_path()).resolve())
    if dll_identity["loaded_sha256"] != quality.BE376_SHA256:
        raise RuntimeError("loaded DLL hash changed during causal smoke")
    report = {
        "schema": "omega-loss-contract-causal-smoke-v1",
        "unit": "OMEGA-LOSS-CONTRACT-RECONCILIATION",
        "phase": "STEP_2_CAUSAL_SMOKE",
        "status": "PASS" if all(row["pass"] for row in gates) else "FAIL",
        "smoke_only": True,
        "scientific_quality_training": False,
        "technical_optimizer_updates": sum(len(route["updates"]) for route in route_outputs.values()),
        "expected_technical_optimizer_updates": 8,
        "route_order": [f"{backend}_K{rounds}" for backend, rounds in ROUTE_ORDER],
        "updates_per_route": UPDATES_PER_ROUTE,
        "warmup_optimizer_updates": 0,
        "validation_or_test_loaded": False,
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source": loss_source,
        "historical_helper_called": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "qualified_execution_config": manifest["qualified_backend_config"],
        "runtime_policy": policy,
        "dll_identity": dll_identity,
        "native_bridge_identity": {"path": str(native_bridge_path), "sha256": native_bridge_sha256},
        "cache_identity": cache_identity,
        "source_identity": source_identity,
        "routes": [route_outputs[(backend, rounds)] for backend, rounds in ROUTE_ORDER],
        "local_gate_summary": {
            "gate_source": str(Path(local_gates.__file__).resolve()),
            "gate_source_sha256": _sha256_file(Path(local_gates.__file__).resolve()),
            "atol": float(local_gates.ATOL),
            "rtol": float(local_gates.RTOL),
            "fallbacks_or_tolerance_changes": False,
            "comparisons": gates,
        },
        "harness": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        "output_dir": str(output_root),
    }
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_root / "causal_smoke_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-step2-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_step2_go:
        parser.error("causal smoke requires the explicit Sol Step-2 GO confirmation")
    stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_root = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / "loss_contract_causal_smoke" / stamp
    try:
        report = run_causal_smoke(output_root)
    except Exception as error:
        if output_root.exists():
            quality._write_json(output_root / "causal_smoke_failure.json", {
                "status": "FAILED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "output_dir": str(output_root),
            })
        raise
    print(json.dumps({
        "status": report["status"],
        "report": str(output_root / "causal_smoke_report.json"),
        "technical_optimizer_updates": report["technical_optimizer_updates"],
        "manifest_sha256": report["manifest_sha256"],
        "loss_contract": report["loss_contract"],
        "loss_source_sha256": report["loss_source"]["sha256"],
        "dll_sha256": report["dll_identity"]["loaded_sha256"],
        "gate_passes": sum(bool(row["pass"]) for row in report["local_gate_summary"]["comparisons"]),
        "gate_count": len(report["local_gate_summary"]["comparisons"]),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

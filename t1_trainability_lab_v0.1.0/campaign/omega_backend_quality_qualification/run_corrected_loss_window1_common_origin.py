"""Bounded K4/window1 common-origin diagnosis; never runs stable or quality training."""

from __future__ import annotations

import argparse
import copy
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
import run_loss_contract_causal_smoke as smoke  # noqa: E402


SEED = 20260913
K = 4
WINDOW_TOKENS = 256
ROUTES = ("pytorch", "native")
ORIGINAL_SMOKE = HERE / "results" / "loss_contract_causal_smoke_20260924T115525" / "causal_smoke_report.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone(memory_format=torch.preserve_format)


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return _clone_tensor(value)
    if isinstance(value, dict):
        return type(value)((key, _clone_tree(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "stride": list(value.stride()) if value.layout == torch.strided else None,
        "storage_offset": int(value.storage_offset()) if value.layout == torch.strided else None,
        "contiguous": bool(value.is_contiguous()) if value.layout == torch.strided else None,
        "requires_grad": bool(value.requires_grad),
    }


def _tensor_pair_report(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    metadata_reference = _tensor_metadata(reference)
    metadata_candidate = _tensor_metadata(candidate)
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    same_meta = metadata_reference == metadata_candidate
    equal = bool(torch.equal(reference, candidate)) if same_shape else False
    diff_count: int | None = None
    max_abs_diff: float | None = None
    max_relative_diff: float | None = None
    if same_shape:
        if equal:
            diff_count, max_abs_diff, max_relative_diff = 0, 0.0, 0.0
        elif reference.is_floating_point() and candidate.is_floating_point():
            delta = (candidate.to(dtype=torch.float64) - reference.to(dtype=torch.float64)).abs()
            if delta.numel():
                diff_count = int(torch.count_nonzero(candidate != reference).item())
                max_abs_diff = float(delta.max().item())
                max_relative_diff = float((delta / reference.to(dtype=torch.float64).abs().clamp_min(1.0e-30)).max().item())
            else:
                diff_count, max_abs_diff, max_relative_diff = 0, 0.0, 0.0
        else:
            unequal = candidate != reference
            diff_count = int(torch.count_nonzero(unequal).item())
    return {
        "torch_equal": equal,
        "metadata_equal": same_meta,
        "equal_with_metadata": bool(equal and same_meta),
        "reference_metadata": metadata_reference,
        "candidate_metadata": metadata_candidate,
        "different_element_count": diff_count,
        "max_abs_diff": max_abs_diff,
        "max_relative_diff": max_relative_diff,
    }


def _compare_tensor_mappings(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor], label: str) -> dict[str, Any]:
    reference_keys = list(reference)
    candidate_keys = list(candidate)
    rows: list[dict[str, Any]] = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            rows.append({"name": f"{label}.{name}", "key_missing": True, "reference_present": name in reference, "candidate_present": name in candidate, "equal_with_metadata": False})
            continue
        rows.append({"name": f"{label}.{name}", **_tensor_pair_report(reference[name], candidate[name])})
    return {
        "label": label,
        "reference_key_order_sha256": quality._canonical_hash(reference_keys),
        "candidate_key_order_sha256": quality._canonical_hash(candidate_keys),
        "key_order_equal": reference_keys == candidate_keys,
        "pass_exact": reference_keys == candidate_keys and all(row.get("equal_with_metadata", False) for row in rows),
        "tensor_count": len(rows),
        "tensors": rows,
    }


def _compare_optimizer_named(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    names = sorted(set(reference) | set(candidate))
    for name in names:
        if name not in reference or name not in candidate:
            rows.append({"name": name, "key_missing": True, "pass_exact": False})
            continue
        ref_slots = reference[name]
        cand_slots = candidate[name]
        if set(ref_slots) != set(cand_slots):
            rows.append({"name": name, "slot_set_equal": False, "reference_slots": sorted(ref_slots), "candidate_slots": sorted(cand_slots), "pass_exact": False})
            continue
        slot_rows = []
        for slot in sorted(ref_slots):
            left, right = ref_slots[slot], cand_slots[slot]
            if torch.is_tensor(left) and torch.is_tensor(right):
                slot_rows.append({"slot": slot, **_tensor_pair_report(left, right)})
            else:
                slot_rows.append({"slot": slot, "reference": left, "candidate": right, "equal": left == right})
        rows.append({"name": name, "slot_set_equal": True, "pass_exact": all(row.get("equal_with_metadata", row.get("equal", False)) for row in slot_rows), "slots": slot_rows})
    return {"pass_exact": set(reference) == set(candidate) and all(row.get("pass_exact", False) for row in rows), "parameter_count": len(rows), "parameters": rows}


def _input_storage_identity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    return {
        **_tensor_pair_report(reference, candidate),
        "reference_data_ptr": int(reference.data_ptr()) if reference.numel() else None,
        "candidate_data_ptr": int(candidate.data_ptr()) if candidate.numel() else None,
        "storage_independent": not reference.numel() or int(reference.data_ptr()) != int(candidate.data_ptr()),
    }


def _event_reproduces_original(rebuilt: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    exact_fields = (
        "model_state_after_update_sha256",
        "state_output_sha256",
        "optimizer_state_sha256",
        "source_token_sha256",
        "teacher_logits_sha256",
        "loss_contract",
        "loss_source_sha256",
        "valid_tokens",
        "optimizer_step",
        "ce",
        "kl_tau_squared_applied",
        "total",
        "pre_clip_gradient_norm",
        "clip_coefficient_applied",
    )
    fields = {name: {"rebuilt": rebuilt.get(name), "original": original.get(name), "equal": rebuilt.get(name) == original.get(name)} for name in exact_fields}
    return {"pass_exact": all(row["equal"] for row in fields.values()), "fields": fields}


def _storage_pointers(mapping: dict[str, torch.Tensor]) -> dict[str, int | None]:
    return {name: (int(value.data_ptr()) if value.numel() else None) for name, value in mapping.items()}


def _independent_tensor_copies(left: torch.Tensor, right: torch.Tensor) -> bool:
    return not left.numel() or int(left.data_ptr()) != int(right.data_ptr())


def run_diagnostic(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output immutable; already exists: {output_dir}")
    if not ORIGINAL_SMOKE.is_file():
        raise FileNotFoundError(f"original smoke report missing: {ORIGINAL_SMOKE}")
    original = json.loads(ORIGINAL_SMOKE.read_text(encoding="utf-8"))
    if original.get("status") != "FAIL" or original.get("phase") != "STEP_2_CAUSAL_SMOKE":
        raise ValueError("preserved original smoke report identity/status mismatch")

    manifest = quality._load_qualification_manifest()
    loss_source = smoke._loss_source_identity(manifest)
    if loss_source["sha256"] != "4f456775993c60dc58c63a160ecd292c37c5b53518e933cca64d209698927c88":
        raise ValueError("canonical corrected loss source identity changed")
    if _sha256_file(Path(smoke.__file__).resolve()) != original["harness"]["sha256"]:
        raise ValueError("causal-smoke runner source changed since original failure")
    r2, p0, ce, bridge, _modules = quality._load_real_dependencies()
    local_gates = smoke._load_gate_module(quality)
    policy = smoke._validate_runtime_policy(ce, p0)
    if _sha256_file(quality.BE376_DLL) != quality.BE376_SHA256:
        raise ValueError("selected be376 DLL hash changed")

    train_manifest = json.loads(quality.TRAIN_MANIFEST.read_text(encoding="utf-8"))
    documents, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    if len(documents) != 602:
        raise ValueError("loaded train document count differs from frozen manifest")
    pair = train_manifest["pairs"][0]
    positions = [int(index) for index in pair["document_indices"]]
    pair_keys = pair["document_keys"]
    source = torch.tensor([documents[index]["tokens"] for index in positions], dtype=torch.long)
    w0 = {
        "inputs": source[:, :WINDOW_TOKENS],
        "targets": source[:, 1 : WINDOW_TOKENS + 1],
        "valid_mask": torch.ones((8, WINDOW_TOKENS), dtype=torch.bool),
        "teacher_logits": r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, 0),
    }
    original_routes = {
        row["backend"]: row
        for row in original["routes"]
        if int(row["K"]) == K
    }
    if set(original_routes) != set(ROUTES):
        raise ValueError("original K4 Pytorch/native smoke routes are incomplete")
    original_update0 = {backend: row["updates"][0] for backend, row in original_routes.items()}
    original_update1 = {backend: row["updates"][1] for backend, row in original_routes.items()}
    if any(row.get("window") != 0 or row.get("update") != 0 for row in original_update0.values()):
        raise ValueError("original K4 prefix updates do not identify window0/update0")

    init_path = Path(original_routes["pytorch"]["common_initialization_path"])
    init_hash = original_routes["pytorch"]["common_initialization_sha256"]
    if _sha256_file(init_path) != init_hash:
        raise ValueError("preserved K4 common initialization bundle hash mismatch")
    init_bundle = torch.load(init_path, map_location="cpu", weights_only=False)
    if int(init_bundle["seed"]) != SEED or int(init_bundle["K"]) != K:
        raise ValueError("preserved K4 common initialization seed/K mismatch")
    if init_bundle["model_state_sha256"] != original_routes["pytorch"]["common_initial_model_state_sha256"]:
        raise ValueError("K4 common initialization model-state hash differs from original smoke")

    output_dir.mkdir(parents=True, exist_ok=False)
    prefix_snapshots: dict[str, dict[str, Any]] = {}
    prefix_events: dict[str, dict[str, Any]] = {}
    prefix_reproduction: dict[str, dict[str, Any]] = {}
    for backend in ROUTES:
        model = ce.fresh_model(SEED, K)
        model.load_state_dict(init_bundle["model_state"], strict=True)
        if quality._value_hash(model.state_dict()) != init_bundle["model_state_sha256"]:
            raise ValueError(f"{backend} reconstruction does not load exact common K4 init")
        torch.set_rng_state(init_bundle["torch_rng_state"])
        random.setstate(init_bundle["python_rng_state"])
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS,
            eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY,
        )
        if optimizer.state or quality._value_hash(optimizer.state_dict()) != init_bundle["fresh_adamw_state_sha256"]:
            raise AssertionError("reconstructed K4 prefix must start with fresh AdamW")
        state = _clone_tensor(init_bundle["initial_recurrent_state"])
        state_part_weight = None
        if backend == "native":
            bridge.configure_library(quality.BE376_DLL)
            if not bridge.runtime_abi_available():
                raise RuntimeError("selected be376 DLL lacks runtime ABI")
            bridge.configure_runtime(4)
            state_part_weight = r2.prepare_native_model(model)

        event, next_state, _snapshot = smoke._capture_route_update(
            backend=backend, rounds=K, update=0, model=model, optimizer=optimizer,
            inputs=w0["inputs"], targets=w0["targets"], teacher_logits=w0["teacher_logits"],
            valid_mask=w0["valid_mask"], state=state, r2=r2, p0=p0, ce=ce,
            bridge=bridge, local_gates=local_gates, state_part_weight=state_part_weight,
            source=source, pair_positions=positions, pair_keys=pair_keys,
            loss_source=loss_source, manifest=manifest,
        )
        original_event = original_update0[backend]
        event_match = _event_reproduces_original(event, original_event)
        event["route_id"] = f"{backend}_K4_seed_{SEED}_window0_prefix_reconstruction"
        event["window0_prefix_reconstruction"] = True
        prefix_events[backend] = event
        prefix_reproduction[backend] = event_match
        named_optimizer = local_gates._optimizer_state(model, optimizer)
        model_state = {name: _clone_tensor(value) for name, value in model.state_dict().items()}
        named_buffers = {name for name, _ in model.named_buffers()}
        prefix_snapshots[backend] = {
            "backend": backend,
            "K": K,
            "seed": SEED,
            "model_state": model_state,
            "parameter_names": [name for name, _ in model.named_parameters()],
            "buffer_names": sorted(named_buffers),
            "recurrent_state_window0_detached": _clone_tensor(next_state),
            "optimizer_state_dict": _clone_tree(optimizer.state_dict()),
            "optimizer_state_by_parameter": _clone_tree(named_optimizer),
            "optimizer_param_groups": _clone_tree(optimizer.state_dict()["param_groups"]),
            "torch_rng_state_after_window0": torch.get_rng_state().clone(),
            "python_rng_state_after_window0": random.getstate(),
            "window1_inputs": _clone_tensor(source[:, WINDOW_TOKENS : 2 * WINDOW_TOKENS]),
            "window1_targets": _clone_tensor(source[:, WINDOW_TOKENS + 1 : 2 * WINDOW_TOKENS + 1]),
            "window1_mask": torch.ones((8, WINDOW_TOKENS), dtype=torch.bool),
            "window0_event": event,
            "model_state_sha256": quality._value_hash(model.state_dict()),
            "recurrent_state_sha256": quality._tensor_hash(next_state),
            "optimizer_state_sha256": quality._value_hash(optimizer.state_dict()),
        }
        if not event_match["pass_exact"]:
            # Reconstruction was not byte-for-byte the original W0 prefix; do
            # not treat its derived W1 state as an original pre-forward snapshot.
            break

    prefix_reconstructed_exactly = len(prefix_snapshots) == 2 and all(row["pass_exact"] for row in prefix_reproduction.values())
    part1: dict[str, Any] = {
        "stored_original_pre_window1_full_tensors": False,
        "stored_original_pre_window1_route_data": {
            "model_state_sha256": {backend: original_update0[backend]["model_state_after_update_sha256"] for backend in ROUTES},
            "recurrent_state_sha256": {backend: original_update0[backend]["state_output_sha256"] for backend in ROUTES},
            "optimizer_state_sha256": {backend: original_update0[backend]["optimizer_state_sha256"] for backend in ROUTES},
            "window1_inputs_sha256": {backend: original_update0[backend]["source_token_sha256"] for backend in ROUTES},
            "window1_teacher_logits_sha256": {backend: original["source_identity"]["teacher_logits_sha256_by_window"]["1"] for backend in ROUTES},
        },
        "reconstructed_prefix_updates": len(prefix_snapshots),
        "reconstructed_prefix_exactly_reproduces_original_event_hashes": prefix_reconstructed_exactly,
        "reconstruction_event_comparisons": prefix_reproduction,
    }
    prefix_snapshot_path = output_dir / "reconstructed_K4_window1_prefix_snapshots.pt"
    if prefix_snapshots:
        torch.save({
            "schema": "corrected-loss-window1-prefix-snapshots-v1",
            "origin_backend_runs": prefix_snapshots,
        }, prefix_snapshot_path)
        part1["prefix_snapshot_artifact"] = str(prefix_snapshot_path)
        part1["prefix_snapshot_artifact_sha256"] = _sha256_file(prefix_snapshot_path)
    if prefix_reconstructed_exactly:
        py = prefix_snapshots["pytorch"]
        native = prefix_snapshots["native"]
        model_params = _compare_tensor_mappings(py["model_state"], native["model_state"], "model_state_parameters_and_buffers")
        recurrent_state = _tensor_pair_report(py["recurrent_state_window0_detached"], native["recurrent_state_window0_detached"])
        optimizer_moments = _compare_optimizer_named(py["optimizer_state_by_parameter"], native["optimizer_state_by_parameter"])
        param_groups_equal = py["optimizer_param_groups"] == native["optimizer_param_groups"]

        window1 = {
            "inputs": source[:, WINDOW_TOKENS : 2 * WINDOW_TOKENS],
            "targets": source[:, WINDOW_TOKENS + 1 : 2 * WINDOW_TOKENS + 1],
            "mask": torch.ones((8, WINDOW_TOKENS), dtype=torch.bool),
            "teacher_logits": r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, 1),
        }
        teacher_pytorch = _clone_tensor(window1["teacher_logits"])
        teacher_native = _clone_tensor(window1["teacher_logits"])
        part1["pre_window1_exact_comparison"] = {
            "model_parameters_and_buffers": model_params,
            "recurrent_state_input": recurrent_state,
            "tokens": _tensor_pair_report(py["window1_inputs"], native["window1_inputs"]),
            "targets": _tensor_pair_report(py["window1_targets"], native["window1_targets"]),
            "mask": _tensor_pair_report(py["window1_mask"], native["window1_mask"]),
            "teacher_logits": _tensor_pair_report(teacher_pytorch, teacher_native),
            "teacher_logits_sha256_matches_original_by_route": {
                backend: quality._tensor_hash(window1["teacher_logits"]) == original_update1[backend]["teacher_logits_sha256"]
                for backend in ROUTES
            },
            "window1_source_hash_matches_original_by_route": {
                backend: quality._tensor_hash(source) == original_update1[backend]["source_token_sha256"]
                for backend in ROUTES
            },
            "optimizer_moments_and_step": optimizer_moments,
            "optimizer_param_groups_equal": param_groups_equal,
            "model_state_sha256": {backend: prefix_snapshots[backend]["model_state_sha256"] for backend in ROUTES},
            "recurrent_state_sha256": {backend: prefix_snapshots[backend]["recurrent_state_sha256"] for backend in ROUTES},
            "optimizer_state_sha256": {backend: prefix_snapshots[backend]["optimizer_state_sha256"] for backend in ROUTES},
        }
        original_window1_inputs_match = (
            all(part1["pre_window1_exact_comparison"]["teacher_logits_sha256_matches_original_by_route"].values())
            and all(part1["pre_window1_exact_comparison"]["window1_source_hash_matches_original_by_route"].values())
        )
        part1["window1_teacher_and_source_match_original_smoke"] = original_window1_inputs_match
        if not original_window1_inputs_match:
            report = {
                "schema": "corrected-loss-window1-common-origin-v1",
                "unit": "OMEGA-LOSS-CONTRACT-RECONCILIATION",
                "phase": "CORRECTED-LOSS-WINDOW1-COMMON-ORIGIN",
                "status": "PREFIX_RECONSTRUCTION_MISMATCH",
                "classification": "WINDOW1_TEACHER_OR_SOURCE_DIFFERS_FROM_ORIGINAL; COMMON_ORIGIN_NOT_RUN",
                "original_smoke_report_path": str(ORIGINAL_SMOKE),
                "original_smoke_report_sha256": _sha256_file(ORIGINAL_SMOKE),
                "loss_contract": quality.LOSS_CONTRACT,
                "loss_source": loss_source,
                "manifest_sha256": manifest["manifest_sha256"],
                "part1": part1,
                "prefix_events": prefix_events,
                "prefix_reproduction": prefix_reproduction,
                "technical_update_budget": {"prefix_reconstruction": 2, "common_origin_window1": 0, "total": 2, "maximum_authorized": 4},
                "output_dir": str(output_dir),
            }
            report["report_self_sha256"] = quality._canonical_hash(report)
            quality._write_json(output_dir / "window1_common_origin_report.json", report)
            return report
        # Part 2: two independent copies of the single PyTorch W0 origin.
        origin = prefix_snapshots["pytorch"]
        window1_by_backend: dict[str, dict[str, torch.Tensor]] = {}
        common_models: dict[str, torch.nn.Module] = {}
        common_optimizers: dict[str, torch.optim.Optimizer] = {}
        common_states: dict[str, torch.Tensor] = {}
        clone_reports: dict[str, Any] = {}
        for backend in ROUTES:
            model = ce.fresh_model(SEED, K)
            model.load_state_dict(_clone_tree(origin["model_state"]), strict=True)
            if quality._value_hash(model.state_dict()) != origin["model_state_sha256"]:
                raise ValueError(f"{backend} W1 model copy differs from PyTorch W0 origin")
            torch.set_rng_state(origin["torch_rng_state_after_window0"].clone())
            random.setstate(origin["python_rng_state_after_window0"])
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS,
                eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY,
            )
            optimizer.load_state_dict(_clone_tree(origin["optimizer_state_dict"]))
            state = _clone_tensor(origin["recurrent_state_window0_detached"])
            data_copy = {key: _clone_tensor(value) for key, value in window1.items()}
            named_state = local_gates._optimizer_state(model, optimizer)
            model_state_now = {name: _clone_tensor(value) for name, value in model.state_dict().items()}
            model_origin_compare = _compare_tensor_mappings(origin["model_state"], model_state_now, f"{backend}_W1_copy_model")
            state_origin_compare = _tensor_pair_report(origin["recurrent_state_window0_detached"], state)
            optimizer_origin_compare = _compare_optimizer_named(origin["optimizer_state_by_parameter"], named_state)
            clone_reports[backend] = {
                "model_matches_origin_exactly": model_origin_compare["pass_exact"],
                "model_state": model_origin_compare,
                "state_matches_origin_exactly": state_origin_compare["equal_with_metadata"],
                "state": state_origin_compare,
                "optimizer_matches_origin_exactly": optimizer_origin_compare["pass_exact"],
                "optimizer": optimizer_origin_compare,
                "window1_data": {key: _tensor_pair_report(window1[key], data_copy[key]) for key in window1},
                "model_state_data_ptrs": _storage_pointers(model_state_now),
                "state_data_ptr": int(state.data_ptr()),
                "optimizer_state_data_ptrs": {
                    parameter_name: {
                        slot: int(value.data_ptr()) if torch.is_tensor(value) and value.numel() else None
                        for slot, value in slots.items()
                    }
                    for parameter_name, slots in named_state.items()
                },
            }
            common_models[backend] = model
            common_optimizers[backend] = optimizer
            common_states[backend] = state
            window1_by_backend[backend] = data_copy

        independent_storage: dict[str, Any] = {
            "model_state": {
                name: int(common_models["pytorch"].state_dict()[name].data_ptr()) != int(common_models["native"].state_dict()[name].data_ptr())
                for name in origin["model_state"]
            },
            "recurrent_state": int(common_states["pytorch"].data_ptr()) != int(common_states["native"].data_ptr()),
            "window1_data": {
                key: _independent_tensor_copies(window1_by_backend["pytorch"][key], window1_by_backend["native"][key])
                for key in window1_by_backend["pytorch"]
            },
            "optimizer_state": {},
        }
        py_named = local_gates._optimizer_state(common_models["pytorch"], common_optimizers["pytorch"])
        native_named = local_gates._optimizer_state(common_models["native"], common_optimizers["native"])
        for parameter_name in py_named:
            independent_storage["optimizer_state"][parameter_name] = {
                slot: (not torch.is_tensor(py_named[parameter_name][slot]) or not py_named[parameter_name][slot].numel() or int(py_named[parameter_name][slot].data_ptr()) != int(native_named[parameter_name][slot].data_ptr()))
                for slot in py_named[parameter_name]
            }
        independent_storage["pass"] = (
            all(independent_storage["model_state"].values())
            and independent_storage["recurrent_state"]
            and all(independent_storage["window1_data"].values())
            and all(all(slots.values()) for slots in independent_storage["optimizer_state"].values())
        )

        common_origin_copy_exact = all(
            clone_reports[backend][key]
            for backend in ROUTES
            for key in ("model_matches_origin_exactly", "state_matches_origin_exactly", "optimizer_matches_origin_exactly")
        ) and independent_storage["pass"]
        part2_gates: dict[str, Any] = {}
        w1_events: dict[str, Any] = {}
        w1_snapshots: dict[str, dict[str, Any]] = {}
        if common_origin_copy_exact:
            for backend in ROUTES:
                model = common_models[backend]
                optimizer = common_optimizers[backend]
                state_part_weight = None
                if backend == "native":
                    bridge.configure_library(quality.BE376_DLL)
                    if not bridge.runtime_abi_available():
                        raise RuntimeError("selected be376 DLL lacks runtime ABI during common-origin W1")
                    bridge.configure_runtime(4)
                    state_part_weight = r2.prepare_native_model(model)
                d = window1_by_backend[backend]
                event, next_state, snapshot = smoke._capture_route_update(
                    backend=backend, rounds=K, update=1, model=model, optimizer=optimizer,
                    inputs=d["inputs"], targets=d["targets"], teacher_logits=d["teacher_logits"],
                    valid_mask=d["mask"], state=common_states[backend], r2=r2, p0=p0,
                    ce=ce, bridge=bridge, local_gates=local_gates,
                    state_part_weight=state_part_weight, source=source,
                    pair_positions=positions, pair_keys=pair_keys,
                    loss_source=loss_source, manifest=manifest,
                )
                event["route_id"] = f"{backend}_K4_seed_{SEED}_window1_common_pytorch_origin_DISCARDABLE"
                event["state_source"] = "common PyTorch K4 window0 next_state detached; never recomputed"
                event["state_source_backend"] = "pytorch"
                event["origin_optimizer_step_before_window1"] = 1
                event["discardable_update"] = True
                event["trajectory_continued"] = False
                w1_events[backend] = event
                w1_snapshots[backend] = snapshot
            part2_gate = smoke._compare_snapshots(
                w1_snapshots["pytorch"], w1_snapshots["native"],
                update=1, rounds=K, local_gates=local_gates, output_dir=output_dir,
            )
            part2_gates["K4_common_origin_window1"] = part2_gate
        else:
            part2_gates["K4_common_origin_window1"] = {"pass": False, "not_run": True, "reason": "independent snapshots did not match origin exactly"}

        diagnostic_status = "LOCAL_PASS" if common_origin_copy_exact and part2_gates["K4_common_origin_window1"]["pass"] else "LOCAL_FAIL"
        if diagnostic_status == "LOCAL_PASS":
            if not model_params["pass_exact"] or not recurrent_state["equal_with_metadata"]:
                classification = "TRAJECTORY_PROXIMITY_DISCREPANCY_IN_THIS_CASE"
            else:
                classification = "COMMON_ORIGIN_PASS_BUT_ORIGINAL_FAILURE_NOT_EXPLAINED_BY_DIVERGENT_PARAMETER_OR_STATE_HASHES"
        else:
            classification = "LOCAL_DISCREPANCY_REMAINS"
        part2 = {
            "origin_backend": "pytorch",
            "origin_stage": "K4 seed 20260913 after W0 forward/backward/clip/AdamW; W0 next_state detached before consumption",
            "origin_optimizer_step": 1,
            "copies_are_independent": independent_storage,
            "copy_equals_origin_exactly": clone_reports,
            "common_origin_copy_exact": common_origin_copy_exact,
            "window1_events": w1_events,
            "gates": part2_gates,
            "discardable_updates": True,
            "classification": classification,
        }
        report = {
            "schema": "corrected-loss-window1-common-origin-v1",
            "unit": "OMEGA-LOSS-CONTRACT-RECONCILIATION",
            "phase": "CORRECTED-LOSS-WINDOW1-COMMON-ORIGIN",
            "status": diagnostic_status,
            "classification": classification,
            "original_smoke_report_path": str(ORIGINAL_SMOKE),
            "original_smoke_status_unchanged": original["status"],
            "original_smoke_report_sha256": _sha256_file(ORIGINAL_SMOKE),
            "loss_contract": quality.LOSS_CONTRACT,
            "loss_source": loss_source,
            "manifest_sha256": manifest["manifest_sha256"],
            "dll_sha256": _sha256_file(quality.BE376_DLL),
            "native_bridge_sha256": _sha256_file(Path(bridge.__file__).resolve()),
            "teacher_cache_manifest_sha256": manifest["hidden_cache"]["manifest"]["sha256"],
            "teacher_cache_sha256": manifest["hidden_cache"]["cache_file"]["sha256"],
            "runtime_policy": policy,
            "K": K,
            "seed": SEED,
            "valid_tokens": 2048,
            "part1": part1,
            "part2": part2,
            "prefix_events": prefix_events,
            "prefix_reproduction": prefix_reproduction,
            "part2_gate_sources": {
                "path": str(Path(local_gates.__file__).resolve()),
                "sha256": _sha256_file(Path(local_gates.__file__).resolve()),
                "atol": float(local_gates.ATOL),
                "rtol": float(local_gates.RTOL),
                "fallback_or_tolerance_changed": False,
            },
            "technical_update_budget": {"prefix_reconstruction": 2, "common_origin_window1": len(w1_events), "total": 2 + len(w1_events), "maximum_authorized": 4},
            "output_dir": str(output_dir),
        }
    else:
        report = {
            "schema": "corrected-loss-window1-common-origin-v1",
            "unit": "OMEGA-LOSS-CONTRACT-RECONCILIATION",
            "phase": "CORRECTED-LOSS-WINDOW1-COMMON-ORIGIN",
            "status": "PREFIX_RECONSTRUCTION_MISMATCH",
            "classification": "ORIGINAL_PRE_WINDOW1_STATE_NOT_RECONSTRUCTED_EXACTLY; COMMON_ORIGIN_NOT_RUN",
            "original_smoke_report_path": str(ORIGINAL_SMOKE),
            "original_smoke_report_sha256": _sha256_file(ORIGINAL_SMOKE),
            "loss_contract": quality.LOSS_CONTRACT,
            "loss_source": loss_source,
            "manifest_sha256": manifest["manifest_sha256"],
            "prefix_events": prefix_events,
            "prefix_reproduction": prefix_reproduction,
            "part1": part1,
            "technical_optimizer_updates": 2,
            "maximum_authorized_updates": 4,
            "common_origin_w1_updates": 0,
            "output_dir": str(output_dir),
        }
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_dir / "window1_common_origin_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-sol-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_sol_go:
        parser.error("diagnostic requires the explicit Sol GO")
    stamp = time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"corrected_loss_window1_common_origin_{stamp}"
    try:
        report = run_diagnostic(output_dir)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(output_dir / "diagnostic_failure.json", {
                "status": "FAILED_RUNTIME",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "output_dir": str(output_dir),
            })
        raise
    print(json.dumps({
        "status": report["status"],
        "classification": report["classification"],
        "report": str(output_dir / "window1_common_origin_report.json"),
        "updates": report.get("technical_update_budget", {"total": report.get("technical_optimizer_updates", 0)}),
        "part1_exact": report.get("part1", {}).get("reconstructed_prefix_exactly_reproduces_original_event_hashes"),
        "part2_gate_pass": report.get("part2", {}).get("gates", {}).get("K4_common_origin_window1", {}).get("pass"),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "LOCAL_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

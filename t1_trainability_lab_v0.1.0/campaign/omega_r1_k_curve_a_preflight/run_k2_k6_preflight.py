"""Bounded K2/K6 common-origin and causal preflight for OMEGA-R1-K-CURVE-A.

This runner uses only the seed-20260913 first frozen training pair. All optimizer
updates are disposable technical checks; it does not create scientific runs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
from typing import Any

for _name in ("KMP_BLOCKTIME", "KMP_LIBRARY", "KMP_SETTINGS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"):
    os.environ.pop(_name, None)

import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
P2R0 = HERE.parent / "omega_native_runtime_p2r0"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_backend_quality_qualification as quality  # noqa: E402
import run_loss_contract_causal_smoke as smoke  # noqa: E402


SEED = 20260913
DEPTHS = (2, 6)
BACKENDS = ("pytorch", "native")
WINDOW_TOKENS = 256
EXPECTED_BATCH = 8
EXPECTED_VALID_TOKENS = 2048
EXPECTED_LOSS_SHA256 = "4f456775993c60dc58c63a160ecd292c37c5b53518e933cca64d209698927c88"
EXPECTED_DLL_SHA256 = "be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return type(value)((key, _clone_tree(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _storage_pointers(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(value.data_ptr())] if value.numel() else []
    if isinstance(value, dict):
        return [pointer for item in value.values() for pointer in _storage_pointers(item)]
    if isinstance(value, (list, tuple)):
        return [pointer for item in value for pointer in _storage_pointers(item)]
    return []


def _fresh_model(ce: Any, technical_model: Any, seed: int, rounds: int) -> torch.nn.Module:
    """Use the frozen reference-then-F initialization convention for any K."""
    ce.r1.configure_cpu_runtime()
    ce.set_seed(seed)
    reference = technical_model.OmegaCoreLM0R1Technical(
        vocab_size=ce.TOKENIZER_VOCAB,
        dimension=ce.DIMENSION,
        slots=ce.SLOTS,
        rounds=rounds,
        variant="shared",
    ).to(dtype=torch.float32)
    try:
        return ce.OmegaCoreLMFast.from_reference(reference).to(dtype=torch.float32)
    finally:
        del reference


def _make_common_initialization(
    *, ce: Any, technical_model: Any, seed: int, rounds: int, output_dir: Path
) -> tuple[dict[str, Any], Path, str]:
    model = _fresh_model(ce, technical_model, seed, rounds)
    model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ce.BASE_LR,
        betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS,
        weight_decay=ce.WEIGHT_DECAY,
    )
    if optimizer.state:
        raise AssertionError("common initialization must have fresh empty AdamW state")
    initial_state = model.initial_state(EXPECTED_BATCH, device=torch.device("cpu")).detach().cpu().clone()
    bundle = {
        "schema": "omega-r1-k-curve-discardable-common-init-v1",
        "seed": seed,
        "K": rounds,
        "architecture": "R1 shared; reference construction then OmegaCoreLMFast.from_reference",
        "model_state": model_state,
        "model_state_sha256": quality._value_hash(model_state),
        "initial_recurrent_state": initial_state,
        "initial_recurrent_state_sha256": quality._tensor_hash(initial_state),
        "fresh_adamw_state_dict": copy.deepcopy(optimizer.state_dict()),
        "fresh_adamw_state_sha256": quality._value_hash(optimizer.state_dict()),
        "fresh_adamw_state_empty": not bool(optimizer.state_dict()["state"]),
        "torch_rng_state": torch.get_rng_state().clone(),
        "python_rng_state": random.getstate(),
        "scientific_checkpoint": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"common_init_K{rounds}_seed_{seed}.pt"
    if path.exists():
        raise FileExistsError(f"immutable preflight initialization exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return bundle, path, _sha256_file(path)


def _new_optimizer(ce: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=ce.BASE_LR,
        betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS,
        weight_decay=ce.WEIGHT_DECAY,
    )


def _prepare_backend(backend: str, *, quality: Any, r2: Any, bridge: Any, model: torch.nn.Module) -> Any:
    if backend != "native":
        return None
    bridge.configure_library(quality.BE376_DLL)
    if not bridge.runtime_abi_available():
        raise RuntimeError("be37623d runtime ABI is unavailable")
    bridge.configure_runtime(4)
    return r2.prepare_native_model(model)


def _capture(
    *,
    backend: str,
    rounds: int,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: torch.Tensor,
    window: dict[str, torch.Tensor],
    source: torch.Tensor,
    positions: list[int],
    pair_keys: list[list[str]],
    r2: Any,
    p0: Any,
    ce: Any,
    bridge: Any,
    local_gates: Any,
    state_part_weight: Any,
    loss_source: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    return smoke._capture_route_update(
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


def _run_common_origin(
    *,
    rounds: int,
    bundle: dict[str, Any],
    output_dir: Path,
    ce: Any,
    technical_model: Any,
    r2: Any,
    p0: Any,
    bridge: Any,
    local_gates: Any,
    windows: dict[int, dict[str, torch.Tensor]],
    source: torch.Tensor,
    positions: list[int],
    pair_keys: list[list[str]],
    loss_source: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    w0: dict[str, dict[str, Any]] = {}
    w0_snapshots: dict[str, dict[str, Any]] = {}
    origin_rng: dict[str, Any] = {}

    for backend in BACKENDS:
        model = _fresh_model(ce, technical_model, SEED, rounds)
        model.load_state_dict(_clone_tree(bundle["model_state"]), strict=True)
        if quality._value_hash(model.state_dict()) != bundle["model_state_sha256"]:
            raise ValueError(f"K{rounds}/{backend} W0 model did not load the shared initialization")
        torch.set_rng_state(bundle["torch_rng_state"].clone())
        random.setstate(bundle["python_rng_state"])
        model.train()
        optimizer = _new_optimizer(ce, model)
        if optimizer.state or quality._value_hash(optimizer.state_dict()) != bundle["fresh_adamw_state_sha256"]:
            raise AssertionError("W0 arm did not start with a fresh AdamW state")
        state = bundle["initial_recurrent_state"].detach().clone()
        state_part_weight = _prepare_backend(backend, quality=quality, r2=r2, bridge=bridge, model=model)
        event, next_state, snapshot = _capture(
            backend=backend,
            rounds=rounds,
            update=0,
            model=model,
            optimizer=optimizer,
            state=state,
            window=windows[0],
            source=source,
            positions=positions,
            pair_keys=pair_keys,
            r2=r2,
            p0=p0,
            ce=ce,
            bridge=bridge,
            local_gates=local_gates,
            state_part_weight=state_part_weight,
            loss_source=loss_source,
            manifest=manifest,
        )
        event["route_id"] = f"{backend}_K{rounds}_seed_{SEED}_W0_COMMON_INIT_DISCARDABLE"
        event["preflight_phase"] = "common_origin_w0"
        event["discardable_update"] = True
        w0[backend] = {
            "model": model,
            "optimizer": optimizer,
            "next_state": next_state.detach().clone(),
            "event": event,
            "optimizer_state_dict": _clone_tree(optimizer.state_dict()),
        }
        w0_snapshots[backend] = snapshot
        if backend == "pytorch":
            origin_rng = {"torch": torch.get_rng_state().clone(), "python": random.getstate()}

    w0_gate = smoke._compare_snapshots(
        w0_snapshots["pytorch"],
        w0_snapshots["native"],
        update=0,
        rounds=rounds,
        local_gates=local_gates,
        output_dir=output_dir,
    )

    origin_model = w0["pytorch"]["model"]
    origin_optimizer_state = _clone_tree(w0["pytorch"]["optimizer_state_dict"])
    origin_state = w0["pytorch"]["next_state"].detach().clone()
    origin_model_state = _clone_tree(origin_model.state_dict())
    origin_optimizer_named = local_gates._optimizer_state(origin_model, w0["pytorch"]["optimizer"])
    w1: dict[str, dict[str, Any]] = {}
    w1_snapshots: dict[str, dict[str, Any]] = {}
    copy_checks: dict[str, Any] = {}
    copy_objects: dict[str, dict[str, Any]] = {}

    for backend in BACKENDS:
        model = _fresh_model(ce, technical_model, SEED, rounds)
        model.load_state_dict(_clone_tree(origin_model_state), strict=True)
        optimizer = _new_optimizer(ce, model)
        optimizer.load_state_dict(_clone_tree(origin_optimizer_state))
        state = origin_state.detach().clone()
        data = {name: value.detach().clone() for name, value in windows[1].items()}
        copied_model_state = model.state_dict()
        copied_optimizer_named = local_gates._optimizer_state(model, optimizer)
        model_origin_equal = quality._value_hash(copied_model_state) == quality._value_hash(origin_model_state)
        optimizer_origin_equal = quality._value_hash(copied_optimizer_named) == quality._value_hash(origin_optimizer_named)
        state_origin_equal = torch.equal(state, origin_state)
        model_origin_independent = all(
            not value.numel() or int(value.data_ptr()) != int(origin_model_state[name].data_ptr())
            for name, value in copied_model_state.items()
        )
        optimizer_origin_independent = bool(_storage_pointers(copied_optimizer_named)) and not (
            set(_storage_pointers(copied_optimizer_named)) & set(_storage_pointers(origin_optimizer_named))
        )
        state_origin_independent = int(state.data_ptr()) != int(origin_state.data_ptr())
        data_origin_independent = all(
            not value.numel() or int(value.data_ptr()) != int(windows[1][name].data_ptr())
            for name, value in data.items()
        )
        copy_checks[backend] = {
            "model_matches_pytorch_w0_origin_exactly": model_origin_equal,
            "optimizer_matches_pytorch_w0_origin_exactly": optimizer_origin_equal,
            "recurrent_state_matches_pytorch_w0_origin_exactly": state_origin_equal,
            "model_storage_independent_from_origin": model_origin_independent,
            "optimizer_storage_independent_from_origin": optimizer_origin_independent,
            "recurrent_state_storage_independent_from_origin": state_origin_independent,
            "window1_data_storage_independent_from_origin": data_origin_independent,
        }
        copy_objects[backend] = {
            "model": model,
            "optimizer": optimizer,
            "state": state,
            "data": data,
            "optimizer_named": copied_optimizer_named,
        }
        if not all(
            copy_checks[backend][key]
            for key in (
                "model_matches_pytorch_w0_origin_exactly",
                "optimizer_matches_pytorch_w0_origin_exactly",
                "recurrent_state_matches_pytorch_w0_origin_exactly",
                "model_storage_independent_from_origin",
                "optimizer_storage_independent_from_origin",
                "recurrent_state_storage_independent_from_origin",
                "window1_data_storage_independent_from_origin",
            )
        ):
            raise AssertionError(f"K{rounds}/{backend} W1 copy did not preserve an independent common origin")

    py_copy = copy_objects["pytorch"]
    native_copy = copy_objects["native"]
    copies_independent = {
        "model_state": not (set(_storage_pointers(py_copy["model"].state_dict())) & set(_storage_pointers(native_copy["model"].state_dict()))),
        "optimizer_state": not bool(
            set(_storage_pointers(py_copy["optimizer_named"]))
            & set(_storage_pointers(native_copy["optimizer_named"]))
        ),
        "recurrent_state": int(py_copy["state"].data_ptr()) != int(native_copy["state"].data_ptr()),
        "window1_data": all(
            not py_copy["data"][name].numel()
            or int(py_copy["data"][name].data_ptr()) != int(native_copy["data"][name].data_ptr())
            for name in py_copy["data"]
        ),
    }
    copies_independent["pass"] = all(copies_independent.values())
    if not copies_independent["pass"]:
        raise AssertionError(f"K{rounds} common-origin W1 copies share storage")

    for backend in BACKENDS:
        current = copy_objects[backend]
        current["model"].train()
        torch.set_rng_state(origin_rng["torch"].clone())
        random.setstate(origin_rng["python"])
        state_part_weight = _prepare_backend(
            backend, quality=quality, r2=r2, bridge=bridge, model=current["model"]
        )
        event, next_state, snapshot = _capture(
            backend=backend,
            rounds=rounds,
            update=1,
            model=current["model"],
            optimizer=current["optimizer"],
            state=current["state"],
            window=current["data"],
            source=source,
            positions=positions,
            pair_keys=pair_keys,
            r2=r2,
            p0=p0,
            ce=ce,
            bridge=bridge,
            local_gates=local_gates,
            state_part_weight=state_part_weight,
            loss_source=loss_source,
            manifest=manifest,
        )
        event["route_id"] = f"{backend}_K{rounds}_seed_{SEED}_W1_COMMON_PYTORCH_W0_ORIGIN_DISCARDABLE"
        event["preflight_phase"] = "common_origin_w1"
        event["state_source"] = "detached PyTorch W0 next_state, before any W1 update"
        event["state_source_backend"] = "pytorch"
        event["origin_optimizer_step_before_window1"] = 1
        event["discardable_update"] = True
        w1[backend] = {"event": event, "next_state": next_state.detach().clone()}
        w1_snapshots[backend] = snapshot

    w1_gate = smoke._compare_snapshots(
        w1_snapshots["pytorch"],
        w1_snapshots["native"],
        update=1,
        rounds=rounds,
        local_gates=local_gates,
        output_dir=output_dir,
    )

    w0_model_equal = quality._value_hash(w0["pytorch"]["model"].state_dict()) == quality._value_hash(
        w0["native"]["model"].state_dict()
    )
    w0_state_equal = torch.equal(w0["pytorch"]["next_state"], w0["native"]["next_state"])
    if w0_gate["pass"] and w1_gate["pass"]:
        classification = (
            "TRAJECTORY_PROXIMITY_DISCREPANCY_IN_THIS_CASE"
            if not w0_model_equal or not w0_state_equal
            else "COMMON_ORIGIN_LOCAL_PASS"
        )
    else:
        classification = "COMMON_ORIGIN_LOCAL_GATE_FAIL"

    return {
        "K": rounds,
        "seed": SEED,
        "origin": "one common K-specific initialization; W1 origin is PyTorch W0 post-update model/AdamW/state",
        "common_origin_copy_exact": True,
        "copies_are_storage_independent": copies_independent,
        "window0": {
            "updates": {backend: w0[backend]["event"] for backend in BACKENDS},
            "native_vs_pytorch_gate": w0_gate,
            "post_update_model_state_equal_bitwise": w0_model_equal,
            "post_update_recurrent_state_equal_bitwise": w0_state_equal,
        },
        "window1": {
            "updates": {backend: w1[backend]["event"] for backend in BACKENDS},
            "copy_checks_vs_pytorch_w0_origin": copy_checks,
            "native_vs_pytorch_gate": w1_gate,
        },
        "classification": classification,
        "common_origin_pass": bool(w0_gate["pass"] and w1_gate["pass"] and copies_independent["pass"]),
        "technical_optimizer_updates": 4,
        "scientific_updates": 0,
    }


def _run_independent_causal_smoke(
    *,
    rounds: int,
    bundle: dict[str, Any],
    output_dir: Path,
    ce: Any,
    technical_model: Any,
    r2: Any,
    p0: Any,
    bridge: Any,
    local_gates: Any,
    windows: dict[int, dict[str, torch.Tensor]],
    source: torch.Tensor,
    positions: list[int],
    pair_keys: list[list[str]],
    loss_source: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    routes: dict[str, dict[str, Any]] = {}
    route_snapshots: dict[str, list[dict[str, Any]]] = {}
    for backend in BACKENDS:
        model = _fresh_model(ce, technical_model, SEED, rounds)
        model.load_state_dict(_clone_tree(bundle["model_state"]), strict=True)
        if quality._value_hash(model.state_dict()) != bundle["model_state_sha256"]:
            raise ValueError(f"K{rounds}/{backend} causal smoke did not load the shared initialization")
        torch.set_rng_state(bundle["torch_rng_state"].clone())
        random.setstate(bundle["python_rng_state"])
        model.train()
        optimizer = _new_optimizer(ce, model)
        state = bundle["initial_recurrent_state"].detach().clone()
        state_part_weight = _prepare_backend(backend, quality=quality, r2=r2, bridge=bridge, model=model)
        events: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        for update in (0, 1):
            incoming_state_sha = quality._tensor_hash(state)
            weights_before_sha = quality._value_hash(model.state_dict())
            event, next_state, snapshot = _capture(
                backend=backend,
                rounds=rounds,
                update=update,
                model=model,
                optimizer=optimizer,
                state=state,
                window=windows[update],
                source=source,
                positions=positions,
                pair_keys=pair_keys,
                r2=r2,
                p0=p0,
                ce=ce,
                bridge=bridge,
                local_gates=local_gates,
                state_part_weight=state_part_weight,
                loss_source=loss_source,
                manifest=manifest,
            )
            event["route_id"] = f"{backend}_K{rounds}_seed_{SEED}_independent_causal_smoke_DISCARDABLE"
            event["preflight_phase"] = "independent_causal_smoke"
            event["prior_state_sha256_at_forward"] = incoming_state_sha
            event["weights_before_sha256"] = weights_before_sha
            event["discardable_update"] = True
            if update == 1:
                event["window0_detached_state_reused_exactly"] = incoming_state_sha == events[0]["state_output_sha256"]
                event["window0_weights_updated_before_window1"] = weights_before_sha == events[0]["model_state_after_update_sha256"]
                if not event["window0_detached_state_reused_exactly"] or not event["window0_weights_updated_before_window1"]:
                    raise AssertionError(f"K{rounds}/{backend} causal W0->W1 continuity was broken")
            events.append(event)
            snapshots.append(snapshot)
            state = next_state.detach().clone()
        if [event["optimizer_step"] for event in events] != [1, 2]:
            raise AssertionError(f"K{rounds}/{backend} did not perform exactly two sequential optimizer updates")
        routes[backend] = {"events": events, "optimizer_updates": len(events)}
        route_snapshots[backend] = snapshots

    gates = {
        f"W{update}": smoke._compare_snapshots(
            route_snapshots["pytorch"][update],
            route_snapshots["native"][update],
            update=update,
            rounds=rounds,
            local_gates=local_gates,
            output_dir=output_dir,
        )
        for update in (0, 1)
    }
    return {
        "K": rounds,
        "seed": SEED,
        "route_updates": routes,
        "pytorch_vs_native_gates": gates,
        "both_routes_have_two_updates": all(routes[backend]["optimizer_updates"] == 2 for backend in BACKENDS),
        "causal_continuity_verified": all(
            event.get("window0_detached_state_reused_exactly") is True
            and event.get("window0_weights_updated_before_window1") is True
            for backend in BACKENDS
            for event in routes[backend]["events"][1:]
        ),
        "technical_optimizer_updates": 4,
        "scientific_updates": 0,
    }


def run_preflight(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"preflight output is immutable; already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = quality._load_qualification_manifest()
    loss_source = smoke._loss_source_identity(manifest)
    if loss_source["sha256"] != EXPECTED_LOSS_SHA256:
        raise ValueError("canonical R1 masked-token-mean loss SHA256 mismatch")

    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    r1, _expanded = modules
    local_gates = smoke._load_gate_module(quality)
    policy = smoke._validate_runtime_policy(ce, p0)
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("preflight requires torch intraop/inter-op 4/1")
    if _sha256_file(quality.BE376_DLL) != EXPECTED_DLL_SHA256 or quality.BE376_SHA256 != EXPECTED_DLL_SHA256:
        raise ValueError("certified be37623d DLL identity mismatch")

    technique_path = SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415

    source_harnesses = {
        "canonical_loss": loss_source["path"],
        "causal_smoke_harness": str(Path(smoke.__file__).resolve()),
        "common_origin_harness": str(CAMPAIGN / "run_corrected_loss_window1_common_origin.py"),
        "native_bridge": str(Path(bridge.__file__).resolve()),
        "native_r2_runner": str(Path(r2.__file__).resolve()),
        "preflight_runner": str(Path(__file__).resolve()),
    }
    source_hashes_before = {name: _sha256_file(Path(path)) for name, path in source_harnesses.items()}
    train_manifest = json.loads(quality.TRAIN_MANIFEST.read_text(encoding="utf-8"))
    docs, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    expected_keys = [
        (row["full_text_sha256"], row["retained_513_token_sha256"])
        for row in train_manifest["documents"]
    ]
    actual_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in docs]
    if len(docs) != 602 or actual_keys != expected_keys:
        raise ValueError("loaded 602 training documents differ from the frozen source manifest")

    pair = train_manifest["pairs"][0]
    positions = [int(value) for value in pair["document_indices"]]
    pair_keys = pair["document_keys"]
    if len(positions) != EXPECTED_BATCH or len(pair_keys) != EXPECTED_BATCH:
        raise ValueError("preflight must use the first frozen physical batch of 8 training documents")
    source = torch.tensor([docs[position]["tokens"] for position in positions], dtype=torch.long)
    windows: dict[int, dict[str, torch.Tensor]] = {}
    for window in (0, 1):
        offset = window * WINDOW_TOKENS
        inputs = source[:, offset : offset + WINDOW_TOKENS]
        targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
        teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, window)
        if tuple(inputs.shape) != (EXPECTED_BATCH, WINDOW_TOKENS) or tuple(targets.shape) != (EXPECTED_BATCH, WINDOW_TOKENS):
            raise ValueError(f"window {window} input/target shape mismatch")
        if tuple(teacher_logits.shape[:2]) != (EXPECTED_BATCH, WINDOW_TOKENS):
            raise ValueError(f"window {window} teacher-logit shape mismatch")
        windows[window] = {
            "inputs": inputs,
            "targets": targets,
            "teacher_logits": teacher_logits,
            "valid_mask": torch.ones_like(targets, dtype=torch.bool),
        }

    manifest_identity = manifest["manifest_sha256"]
    report: dict[str, Any] = {
        "schema": "omega-r1-k-curve-k2-k6-preflight-v1",
        "unit": "OMEGA-R1-K-CURVE-A",
        "status": "IN_PROGRESS",
        "phase": "K2_K6_COMMON_ORIGIN_AND_CAUSAL_SMOKE",
        "seed": SEED,
        "depths": list(DEPTHS),
        "backends": list(BACKENDS),
        "qualification_manifest_sha256": manifest_identity,
        "training_manifest_sha256": manifest["training_manifest"]["manifest_sha256"],
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source": loss_source,
        "native_dll": {
            "path": str(quality.BE376_DLL),
            "sha256": _sha256_file(quality.BE376_DLL),
            "expected_sha256": EXPECTED_DLL_SHA256,
        },
        "native_bridge": {"path": str(Path(bridge.__file__).resolve()), "sha256": _sha256_file(Path(bridge.__file__).resolve())},
        "runtime_policy": policy,
        "torch": {
            "version": str(torch.__version__),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "cpu": {"processor": platform.processor(), "machine": platform.machine(), "platform": platform.platform()},
        "data": {
            "dataset_revision": train_manifest["dataset"]["revision"],
            "split": "train only; first predeclared cyclic pair",
            "training_document_count": len(docs),
            "pair_index": 0,
            "document_positions": positions,
            "document_keys": pair_keys,
            "source_pair_sha256": quality._tensor_hash(source),
            "teacher_cache_manifest_sha256": manifest["hidden_cache"]["manifest"]["sha256"],
            "teacher_cache_sha256": manifest["hidden_cache"]["cache_file"]["sha256"],
            "teacher_logits_sha256_by_window": {str(window): quality._tensor_hash(item["teacher_logits"]) for window, item in windows.items()},
            "valid_tokens_per_update": EXPECTED_VALID_TOKENS,
            "validation_or_test_loaded": False,
        },
        "training_contract": {
            "scientific_training": False,
            "technical_optimizer_updates_are_discardable": True,
            "batch_physical_effective": [EXPECTED_BATCH, EXPECTED_BATCH],
            "bptt_window_tokens": WINDOW_TOKENS,
            "window1_state": "route-local detached W0 output for independent smoke; PyTorch W0 common origin for common-origin test",
            "optimizer": "fresh AdamW per independent route; canonical configured hyperparameters; clip 1.0",
            "teacher_forward": False,
            "cached_teacher_logits_used": True,
            "hidden_cache_used_for_logits": True,
            "test_split_loaded": False,
        },
        "source_harnesses": source_harnesses,
        "source_hashes_before": source_hashes_before,
        "common_initializations": {},
        "per_depth": {},
        "technical_optimizer_updates": 0,
        "sentinel_started": False,
        "output_dir": str(output_dir),
    }
    quality._write_json(output_dir / "preflight_progress.json", report)

    init_dir = output_dir / "common_initializations"
    for rounds in DEPTHS:
        depth_dir = output_dir / f"K{rounds}"
        depth_dir.mkdir(parents=True, exist_ok=False)
        bundle, init_path, init_sha = _make_common_initialization(
            ce=ce,
            technical_model=technical_model,
            seed=SEED,
            rounds=rounds,
            output_dir=init_dir,
        )
        report["common_initializations"][str(rounds)] = {
            "path": str(init_path),
            "sha256": init_sha,
            "model_state_sha256": bundle["model_state_sha256"],
            "initial_recurrent_state_sha256": bundle["initial_recurrent_state_sha256"],
            "fresh_adamw_state_sha256": bundle["fresh_adamw_state_sha256"],
            "fresh_adamw_state_empty": bundle["fresh_adamw_state_empty"],
            "historical_checkpoint_loaded": False,
        }
        common = _run_common_origin(
            rounds=rounds,
            bundle=bundle,
            output_dir=depth_dir / "common_origin",
            ce=ce,
            technical_model=technical_model,
            r2=r2,
            p0=p0,
            bridge=bridge,
            local_gates=local_gates,
            windows=windows,
            source=source,
            positions=positions,
            pair_keys=pair_keys,
            loss_source=loss_source,
            manifest=manifest,
        )
        smoke_result = _run_independent_causal_smoke(
            rounds=rounds,
            bundle=bundle,
            output_dir=depth_dir / "independent_smoke",
            ce=ce,
            technical_model=technical_model,
            r2=r2,
            p0=p0,
            bridge=bridge,
            local_gates=local_gates,
            windows=windows,
            source=source,
            positions=positions,
            pair_keys=pair_keys,
            loss_source=loss_source,
            manifest=manifest,
        )
        independent_w0_pass = bool(smoke_result["pytorch_vs_native_gates"]["W0"]["pass"])
        independent_w1_pass = bool(smoke_result["pytorch_vs_native_gates"]["W1"]["pass"])
        common_pass = bool(common["common_origin_pass"])
        if not independent_w0_pass:
            trajectory_classification = "INDEPENDENT_W0_LOCAL_GATE_FAIL"
        elif not independent_w1_pass and common_pass:
            trajectory_classification = "FAIL_RECORDED_TRAJECTORY_PROXIMITY"
        elif not independent_w1_pass:
            trajectory_classification = "INDEPENDENT_W1_FAIL_WITH_COMMON_ORIGIN_NOT_PASSING"
        else:
            trajectory_classification = "INDEPENDENT_SMOKE_NUMERIC_GATES_PASS"
        report["per_depth"][str(rounds)] = {
            "common_origin": common,
            "independent_causal_smoke": smoke_result,
            "independent_smoke_classification": trajectory_classification,
            "common_origin_pass": common_pass,
            "independent_w0_numeric_gate_pass": independent_w0_pass,
            "independent_w1_numeric_gate_pass": independent_w1_pass,
            "independent_w1_failure_is_nonblocking_trajectory_proximity": (
                trajectory_classification == "FAIL_RECORDED_TRAJECTORY_PROXIMITY"
            ),
        }
        report["technical_optimizer_updates"] += common["technical_optimizer_updates"] + smoke_result["technical_optimizer_updates"]
        quality._write_json(output_dir / "preflight_progress.json", report)

    all_common_pass = all(report["per_depth"][str(k)]["common_origin_pass"] for k in DEPTHS)
    all_w0_pass = all(report["per_depth"][str(k)]["independent_w0_numeric_gate_pass"] for k in DEPTHS)
    recorded_trajectory = any(
        report["per_depth"][str(k)]["independent_w1_failure_is_nonblocking_trajectory_proximity"]
        for k in DEPTHS
    )
    all_independent_gates_pass = all(
        report["per_depth"][str(k)]["independent_w1_numeric_gate_pass"] for k in DEPTHS
    )
    if all_common_pass and all_w0_pass:
        report["status"] = "PASS_WITH_TRAJECTORY_PROXIMITY_RECORDED" if recorded_trajectory else (
            "PASS" if all_independent_gates_pass else "PASS_COMMON_ORIGIN_INDEPENDENT_W1_RECORDED"
        )
    else:
        report["status"] = "HOLD_COMMON_ORIGIN_OR_W0_NUMERICAL_GATE_FAIL"
    report["common_origin_pass_all_depths"] = all_common_pass
    report["independent_w0_gate_pass_all_depths"] = all_w0_pass
    report["independent_w1_gate_pass_all_depths"] = all_independent_gates_pass
    report["trajectory_proximity_fail_recorded"] = recorded_trajectory
    report["sentinel_eligible_after_judge_review"] = bool(all_common_pass and all_w0_pass)
    report["source_hashes_after"] = {name: _sha256_file(Path(path)) for name, path in source_harnesses.items()}
    if report["source_hashes_after"] != source_hashes_before:
        raise RuntimeError("one or more source files changed during K2/K6 preflight")
    report["sentinel_started"] = False
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_dir / "k2_k6_preflight_report.json", report)
    quality._write_json(output_dir / "preflight_progress.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-k-curve-preflight-go", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_k_curve_preflight_go:
        parser.error("K2/K6 preflight requires the explicit OMEGA-R1-K-CURVE-A GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"k2_k6_preflight_{stamp}"
    try:
        report = run_preflight(output_dir)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "preflight_failure.json",
                {
                    "status": "FAILED_RUNTIME",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "sentinel_started": False,
                    "output_dir": str(output_dir),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "common_origin_pass_all_depths": report["common_origin_pass_all_depths"],
                "independent_w0_gate_pass_all_depths": report["independent_w0_gate_pass_all_depths"],
                "independent_w1_gate_pass_all_depths": report["independent_w1_gate_pass_all_depths"],
                "trajectory_proximity_fail_recorded": report["trajectory_proximity_fail_recorded"],
                "technical_optimizer_updates": report["technical_optimizer_updates"],
                "sentinel_started": report["sentinel_started"],
                "report": str(output_dir / "k2_k6_preflight_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["common_origin_pass_all_depths"] and report["independent_w0_gate_pass_all_depths"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

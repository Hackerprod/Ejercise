"""OMEGA-SHARED-GRAD-CANCELLATION-AUDIT, W0/update0 only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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

import numpy as np
import psutil
import torch


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "omega_backend_quality_qualification"
P2R0 = HERE.parent / "omega_native_runtime_p2r0"
LAB_ROOT = HERE.parents[1]
SCRIPTS = LAB_ROOT / "scripts"
for _path in (CAMPAIGN, P2R0, SCRIPTS, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_backend_quality_qualification as quality  # noqa: E402
import run_loss_contract_causal_smoke as smoke  # noqa: E402
import run_k2_k6_preflight as k_preflight  # noqa: E402


SEEDS = (20260913, 20260914, 20260915, 20260916, 20260917)
DEPTHS = (1, 2, 4, 6)
EXPECTED_LOSS_SHA256 = "4f456775993c60dc58c63a160ecd292c37c5b53518e933cca64d209698927c88"
EXPECTED_DLL_SHA256 = "be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc"
WINDOW_TOKENS = 256
EXPECTED_BATCH = 8
TARGET_COUNT = 2048
TARGET_UPDATES = len(SEEDS) * len(DEPTHS)
TARGET_WEIGHT_SHAPE = (128, 512)
TARGET_ELEMENT_COUNT = TARGET_WEIGHT_SHAPE[0] * TARGET_WEIGHT_SHAPE[1]
BIN_LABELS = ("<10", "10-10^2", "10^2-10^3", "10^3-10^4", "10^4-10^5", "10^5-10^6", ">=10^6", "infinity")
BIN_EDGES = (10.0, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_model(ce: Any, technical_model: Any, seed: int, rounds: int) -> torch.nn.Module:
    if rounds in getattr(ce, "KS", ()):
        return ce.fresh_model(seed, rounds)
    return k_preflight._fresh_model(ce, technical_model, seed, rounds)


def _make_bundle(
    *,
    ce: Any,
    technical_model: Any,
    seed: int,
    rounds: int,
    init_dir: Path,
) -> tuple[dict[str, Any], Path, str]:
    init_dir.mkdir(parents=True, exist_ok=True)
    existing = init_dir / f"common_init_K{rounds}_seed_{seed}.pt"
    if existing.is_file():
        digest = _sha256_file(existing)
        bundle = torch.load(existing, map_location="cpu", weights_only=False)
        if int(bundle.get("seed", -1)) != seed or int(bundle.get("K", -1)) != rounds:
            raise ValueError(f"existing audit common init identity mismatch: {existing}")
        return bundle, existing, digest
    if rounds in getattr(ce, "KS", ()):
        path, digest = quality._prepare_common_initialization(ce, seed, rounds, init_dir)
        return torch.load(path, map_location="cpu", weights_only=False), path, digest
    bundle, path, digest = k_preflight._make_common_initialization(
        ce=ce, technical_model=technical_model, seed=seed, rounds=rounds, output_dir=init_dir
    )
    return bundle, path, digest


def _run_fp32_origin(
    *,
    backend: str,
    seed: int,
    rounds: int,
    bundle: dict[str, Any],
    window: dict[str, torch.Tensor],
    source: torch.Tensor,
    positions: list[int],
    pair_keys: list[list[str]],
    manifest: dict[str, Any],
    loss_source: dict[str, Any],
    ce: Any,
    r2: Any,
    p0: Any,
    bridge: Any,
    local_gates: Any,
    technical_model: Any,
) -> dict[str, Any]:
    model = _fresh_model(ce, technical_model, seed, rounds)
    model.load_state_dict(k_preflight._clone_tree(bundle["model_state"]), strict=True)
    if quality._value_hash(model.state_dict()) != bundle["model_state_sha256"]:
        raise ValueError(f"{backend} K{rounds} seed{seed} model differs from common W0 origin")
    torch.set_rng_state(bundle["torch_rng_state"].clone())
    random.setstate(bundle["python_rng_state"])
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ce.BASE_LR,
        betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS,
        weight_decay=ce.WEIGHT_DECAY,
    )
    if optimizer.state or quality._value_hash(optimizer.state_dict()) != bundle["fresh_adamw_state_sha256"]:
        raise AssertionError("audit W0 optimizer must be fresh")
    state = bundle["initial_recurrent_state"].detach().clone()
    state_part_weight = None
    if backend == "native":
        bridge.configure_library(quality.BE376_DLL)
        if not bridge.runtime_abi_available():
            raise RuntimeError("certified native runtime ABI unavailable")
        bridge.configure_runtime(4)
        state_part_weight = r2.prepare_native_model(model)

    postclip: dict[str, torch.Tensor] = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def clipping_hook(parameters: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        norm = original_clip(parameters, *args, **kwargs)
        postclip["blocks.0.fc2.weight"] = model.blocks[0].fc2.weight.grad.detach().clone()
        return norm

    torch.nn.utils.clip_grad_norm_ = clipping_hook
    try:
        event, next_state, snapshot = smoke._capture_route_update(
            backend=backend,
            rounds=rounds,
            update=0,
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
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip

    result = {
        "backend": backend,
        "event": event,
        "preclip_gradient": snapshot["grads"]["blocks.0.fc2.weight"].detach().cpu().clone(),
        "postclip_gradient": postclip["blocks.0.fc2.weight"].detach().cpu().clone(),
        "clip_norm": float(snapshot["clip_norm"]),
        "clip_coefficient": float(snapshot["clip_coefficient"]),
        "exp_avg": snapshot["optimizer"]["blocks.0.fc2.weight"]["exp_avg"].detach().cpu().clone(),
        "exp_avg_sq": snapshot["optimizer"]["blocks.0.fc2.weight"]["exp_avg_sq"].detach().cpu().clone(),
        "weight_after_adamw": snapshot["parameters"]["blocks.0.fc2.weight"].detach().cpu().clone(),
        "losses": {name: float(snapshot["losses"][name].detach().item()) for name in ("ce", "kl", "total")},
        "model_state_sha256": quality._value_hash(model.state_dict()),
        "optimizer_state_sha256": quality._value_hash(optimizer.state_dict()),
    }
    del model, optimizer, snapshot
    return result


def _fp64_round_gradients(
    *, ce: Any, technical_model: Any, seed: int, rounds: int, bundle: dict[str, Any],
    r2: Any, p0: Any, window: dict[str, torch.Tensor]
) -> dict[str, Any]:
    model = _fresh_model(ce, technical_model, seed, rounds)
    model.load_state_dict(k_preflight._clone_tree(bundle["model_state"]), strict=True)
    model = model.to(dtype=torch.float64)
    model.train()
    records: list[dict[str, torch.Tensor]] = []

    def capture_fc2(_module: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        records.append({"input": inputs[0], "output": output})

    handle = model.blocks[0].fc2.register_forward_hook(capture_fc2)
    state = bundle["initial_recurrent_state"].detach().to(dtype=torch.float64)
    teacher_logits = window["teacher_logits"].detach().to(dtype=torch.float64)
    next_state, student_logits, readout_states = r2._reference_forward(model, window["inputs"], state, p0)
    loss_terms = quality.r1_masked_token_mean_loss(
        student_logits, teacher_logits, window["targets"], window["valid_mask"]
    )
    loss_terms["total"].backward()
    handle.remove()

    expected_calls = rounds * int(window["inputs"].shape[1])
    if len(records) != expected_calls:
        raise ValueError(f"K{rounds} expected {expected_calls} fc2 invocations, got {len(records)}")
    round_inputs: list[list[torch.Tensor]] = [[] for _ in range(rounds)]
    round_grad_outputs: list[list[torch.Tensor]] = [[] for _ in range(rounds)]
    for call_index, item in enumerate(records):
        round_index = call_index % rounds
        if item["output"].grad is None:
            raise RuntimeError("FP64 fc2 output gradient was not retained")
        round_inputs[round_index].append(item["input"].detach().reshape(-1, item["input"].shape[-1]))
        round_grad_outputs[round_index].append(item["output"].grad.detach().reshape(-1, item["output"].shape[-1]))
    del records

    round_gradients: list[torch.Tensor] = []
    for round_index in range(rounds):
        all_inputs = torch.cat(round_inputs[round_index], dim=0)
        all_grad_outputs = torch.cat(round_grad_outputs[round_index], dim=0)
        round_gradients.append(all_grad_outputs.transpose(0, 1) @ all_inputs)
        del all_inputs, all_grad_outputs
        round_inputs[round_index].clear()
        round_grad_outputs[round_index].clear()
    round_gradient_stack = torch.stack(round_gradients, dim=0)
    total_from_rounds = round_gradient_stack.sum(dim=0)
    total_autograd = model.blocks[0].fc2.weight.grad.detach()
    norm_squared = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm_squared += torch.sum(parameter.grad.detach() * parameter.grad.detach())
    fp64_norm = torch.sqrt(norm_squared)
    clip_coefficient = min(1.0, float(ce.CLIP_NORM) / (float(fp64_norm.item()) + 1.0e-6))
    return {
        "round_gradients": round_gradient_stack.detach().cpu().numpy().copy(),
        "total_gradient": total_from_rounds.detach().cpu().numpy().copy(),
        "direct_total_gradient": total_autograd.detach().cpu().numpy().copy(),
        "round_sum_direct_max_abs_error": float((total_from_rounds - total_autograd).abs().max().item()),
        "fp64_loss_ce": float(loss_terms["ce"].item()),
        "fp64_loss_kl": float(loss_terms["kl"].item()),
        "fp64_loss_total": float(loss_terms["total"].item()),
        "fp64_clip_norm": float(fp64_norm.item()),
        "fp64_clip_coefficient": clip_coefficient,
        "fp64_readout_shapes": {"next_state": list(next_state.shape), "logits": list(student_logits.shape), "readout": list(readout_states.shape)},
    }


def _histogram(cancellation: np.ndarray) -> dict[str, int]:
    values = np.asarray(cancellation, dtype=np.float64).reshape(-1)
    infinity = np.isinf(values)
    finite = values[np.isfinite(values)]
    counts: dict[str, int] = {}
    lower = 0.0
    for upper, label in zip(BIN_EDGES, BIN_LABELS[:-2]):
        counts[label] = int(np.count_nonzero((finite >= lower) & (finite < upper)))
        lower = upper
    counts[">=10^6"] = int(np.count_nonzero(finite >= BIN_EDGES[-1]))
    counts["infinity"] = int(np.count_nonzero(infinity))
    return counts


def _stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    subset = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if subset.size == 0:
        return {"count": 0, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(subset.size),
        "median": float(np.median(subset)),
        "mean": float(np.mean(subset)),
        "p95": float(np.quantile(subset, 0.95)),
        "max": float(np.max(subset)),
    }


def _bin_masks(cancellation: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(cancellation, dtype=np.float64)
    finite = np.isfinite(values)
    masks: dict[str, np.ndarray] = {}
    lower = 0.0
    for upper, label in zip(BIN_EDGES, BIN_LABELS[:-2]):
        masks[label] = finite & (values >= lower) & (values < upper)
        lower = upper
    masks[">=10^6"] = finite & (values >= BIN_EDGES[-1])
    masks["infinity"] = np.isinf(values)
    return masks


def _run_one(
    *, seed: int, rounds: int, output_dir: Path, init_dir: Path,
    manifest: dict[str, Any], train_manifest: dict[str, Any], docs: list[dict[str, Any]],
    payload: Any, teacher_weight: torch.Tensor, teacher_bias: torch.Tensor | None,
    r2: Any, p0: Any, ce: Any, bridge: Any, local_gates: Any, technical_model: Any,
    loss_source: dict[str, Any],
) -> dict[str, Any]:
    bundle, init_path, init_sha = _make_bundle(
        ce=ce, technical_model=technical_model, seed=seed, rounds=rounds, init_dir=init_dir
    )
    positions = [int(value) for value in train_manifest["pairs"][0]["document_indices"]]
    pair_keys = train_manifest["pairs"][0]["document_keys"]
    source = torch.tensor([docs[position]["tokens"] for position in positions], dtype=torch.long)
    teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, 0)
    window = {
        "inputs": source[:, :WINDOW_TOKENS],
        "targets": source[:, 1 : WINDOW_TOKENS + 1],
        "teacher_logits": teacher_logits,
        "valid_mask": torch.ones((EXPECTED_BATCH, WINDOW_TOKENS), dtype=torch.bool),
    }

    fp32_routes: dict[str, dict[str, Any]] = {}
    for backend in ("pytorch", "native"):
        fp32_routes[backend] = _run_fp32_origin(
            backend=backend,
            seed=seed,
            rounds=rounds,
            bundle=bundle,
            window=window,
            source=source,
            positions=positions,
            pair_keys=pair_keys,
            manifest=manifest,
            loss_source=loss_source,
            ce=ce,
            r2=r2,
            p0=p0,
            bridge=bridge,
            local_gates=local_gates,
            technical_model=technical_model,
        )

    fp64 = _fp64_round_gradients(
        ce=ce,
        technical_model=technical_model,
        seed=seed,
        rounds=rounds,
        bundle=bundle,
        r2=r2,
        p0=p0,
        window=window,
    )
    g_round = np.asarray(fp64["round_gradients"], dtype=np.float64)
    g_total64 = np.asarray(fp64["total_gradient"], dtype=np.float64)
    g_pytorch = fp32_routes["pytorch"]["preclip_gradient"].numpy().astype(np.float64)
    g_native = fp32_routes["native"]["preclip_gradient"].numpy().astype(np.float64)
    g_post_pytorch = fp32_routes["pytorch"]["postclip_gradient"].numpy().astype(np.float64)
    g_post_native = fp32_routes["native"]["postclip_gradient"].numpy().astype(np.float64)
    m_pytorch = fp32_routes["pytorch"]["exp_avg"].numpy().astype(np.float64)
    v_pytorch = fp32_routes["pytorch"]["exp_avg_sq"].numpy().astype(np.float64)
    m_native = fp32_routes["native"]["exp_avg"].numpy().astype(np.float64)
    v_native = fp32_routes["native"]["exp_avg_sq"].numpy().astype(np.float64)
    w_old = bundle["model_state"]["blocks.0.fc2.weight"].numpy().astype(np.float64)
    w_pytorch = fp32_routes["pytorch"]["weight_after_adamw"].numpy().astype(np.float64)
    w_native = fp32_routes["native"]["weight_after_adamw"].numpy().astype(np.float64)

    A = np.abs(g_round).sum(axis=0)
    denominator = np.abs(g_total64)
    cancellation = np.full(denominator.shape, np.inf, dtype=np.float64)
    nonzero = denominator != 0.0
    cancellation[nonzero] = A[nonzero] / denominator[nonzero]
    grad_error_native = np.abs(g_native - g_total64)
    grad_error_pytorch = np.abs(g_pytorch - g_total64)

    clip_coefficient64 = float(fp64["fp64_clip_coefficient"])
    g_clipped64 = g_total64 * clip_coefficient64
    beta1, beta2 = ce.ADAMW_BETAS
    lr, eps, weight_decay = float(ce.BASE_LR), float(ce.ADAMW_EPS), float(ce.WEIGHT_DECAY)
    m64 = (1.0 - beta1) * g_clipped64
    v64 = (1.0 - beta2) * g_clipped64 * g_clipped64
    mhat64 = m64 / (1.0 - beta1)
    vhat64 = v64 / (1.0 - beta2)
    w64 = w_old * (1.0 - lr * weight_decay) - lr * mhat64 / (np.sqrt(vhat64) + eps)
    weight_error_native = np.abs(w_native - w64)
    weight_error_pytorch = np.abs(w_pytorch - w64)

    atol = float(local_gates.ATOL)
    rtol = float(local_gates.RTOL)
    def gate(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        return np.isfinite(candidate) & (np.abs(candidate - reference) <= atol + rtol * np.abs(reference))

    gate_masks = {
        "gradient_preclip": gate(g_pytorch, g_native),
        "gradient_postclip": gate(g_post_pytorch, g_post_native),
        "exp_avg": gate(m_pytorch, m_native),
        "exp_avg_sq": gate(v_pytorch, v_native),
        "weight_post_adamw": gate(w_pytorch, w_native),
    }
    cancellation_masks = _bin_masks(cancellation)
    c_hist = _histogram(cancellation)
    per_bin: dict[str, Any] = {}
    for label, mask in cancellation_masks.items():
        per_bin[label] = {
            "count": int(np.count_nonzero(mask)),
            "E_grad_native": _stats(grad_error_native, mask),
            "E_grad_pytorch": _stats(grad_error_pytorch, mask),
            "E_weight_native": _stats(weight_error_native, mask),
            "E_weight_pytorch": _stats(weight_error_pytorch, mask),
            "primary_gate_failures": {
                stage: int(np.count_nonzero(mask & ~values)) for stage, values in gate_masks.items()
            },
        }

    summary = {
        "K": rounds,
        "seed": seed,
        "common_init": {"path": str(init_path), "sha256": init_sha, "model_sha256": bundle["model_state_sha256"]},
        "source_pair_sha256": quality._tensor_hash(source),
        "teacher_logits_sha256": quality._tensor_hash(teacher_logits),
        "fp64_round_sum_direct_max_abs_error": fp64["round_sum_direct_max_abs_error"],
        "fp64_loss_ce": fp64["fp64_loss_ce"],
        "fp64_loss_kl": fp64["fp64_loss_kl"],
        "fp64_loss_total": fp64["fp64_loss_total"],
        "fp64_clip_coefficient": clip_coefficient64,
        "fp32_clip_coefficients": {backend: fp32_routes[backend]["clip_coefficient"] for backend in ("pytorch", "native")},
        "tensor_element_count": TARGET_ELEMENT_COUNT,
        "cancellation_histogram": c_hist,
        "exact_zero_denominator_count": c_hist["infinity"],
        "C_max_finite": None if not np.isfinite(cancellation).any() else float(np.max(cancellation[np.isfinite(cancellation)])),
        "C_percentiles_finite": {
            str(q): float(np.quantile(cancellation[np.isfinite(cancellation)], q / 100.0))
            for q in (50, 90, 95, 99, 99.9)
            if np.isfinite(cancellation).any()
        },
        "all_element_errors": {
            "E_grad_native": _stats(grad_error_native, np.ones_like(cancellation, dtype=bool)),
            "E_grad_pytorch": _stats(grad_error_pytorch, np.ones_like(cancellation, dtype=bool)),
            "E_weight_native": _stats(weight_error_native, np.ones_like(cancellation, dtype=bool)),
            "E_weight_pytorch": _stats(weight_error_pytorch, np.ones_like(cancellation, dtype=bool)),
        },
        "primary_gate_fail_counts": {
            stage: int(np.count_nonzero(~values)) for stage, values in gate_masks.items()
        },
        "per_cancellation_bin": per_bin,
        "route_scalars": {
            backend: {
                "losses": fp32_routes[backend]["losses"],
                "clip_norm": fp32_routes[backend]["clip_norm"],
                "clip_coefficient": fp32_routes[backend]["clip_coefficient"],
                "model_sha256_after_adamw": fp32_routes[backend]["model_state_sha256"],
                "optimizer_sha256_after_adamw": fp32_routes[backend]["optimizer_state_sha256"],
            }
            for backend in ("pytorch", "native")
        },
        "route_gate_matches_by_element": True,
        "native_backend": "be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc / workers=4",
        "oracle_dtype": "float64; reference-then-F initialization cast to float64; same W0 tokens and cached teacher logits",
    }

    artifact_dir = output_dir / "per_element"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"K{rounds}_seed_{seed}.npz"
    artifact_tmp = artifact_path.with_suffix(".npz.tmp")
    with artifact_tmp.open("wb") as artifact_stream:
        np.savez_compressed(
            artifact_stream,
        g_r_fp64=g_round,
        g_total_fp64=g_total64,
        g_total_fp64_direct=fp64["direct_total_gradient"],
        g_pytorch_preclip=g_pytorch,
        g_native_preclip=g_native,
        g_pytorch_postclip=g_post_pytorch,
        g_native_postclip=g_post_native,
        exp_avg_pytorch=m_pytorch,
        exp_avg_sq_pytorch=v_pytorch,
        exp_avg_native=m_native,
        exp_avg_sq_native=v_native,
        w_old=w_old,
        w_pytorch_post_adamw=w_pytorch,
        w_native_post_adamw=w_native,
        w_fp64_post_adamw=w64,
        cancellation_C=cancellation,
        A_fp64_round_abs_sum=A,
        E_grad_native=grad_error_native,
        E_grad_pytorch=grad_error_pytorch,
        E_weight_native=weight_error_native,
        E_weight_pytorch=weight_error_pytorch,
        **{f"gate_{name}": mask for name, mask in gate_masks.items()},
        )
    os.replace(artifact_tmp, artifact_path)
    summary["per_element_artifact"] = str(artifact_path)
    summary["per_element_artifact_sha256"] = _sha256_file(artifact_path)
    return summary


def _aggregate_by_depth(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for rounds in DEPTHS:
        subset = [row for row in run_rows if int(row["K"]) == rounds]
        histogram = {label: sum(int(row["cancellation_histogram"][label]) for row in subset) for label in BIN_LABELS}
        bins: dict[str, Any] = {}
        merged: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "C", "E_grad_native", "E_grad_pytorch", "E_weight_native", "E_weight_pytorch",
                "gate_gradient_preclip", "gate_gradient_postclip", "gate_exp_avg", "gate_exp_avg_sq", "gate_weight_post_adamw",
            )
        }
        for row in subset:
            with np.load(row["per_element_artifact"], allow_pickle=False) as artifact:
                for key in merged:
                    artifact_key = "cancellation_C" if key == "C" else key
                    merged[key].append(np.asarray(artifact[artifact_key]).reshape(-1))
        concatenated = {
            key: np.concatenate(values) if values else np.asarray([], dtype=np.float64)
            for key, values in merged.items()
        }
        masks = _bin_masks(concatenated["C"])
        for label, mask in masks.items():
            bins[label] = {
                metric: _stats(concatenated[metric], mask)
                for metric in ("E_grad_native", "E_grad_pytorch", "E_weight_native", "E_weight_pytorch")
            }
            bins[label]["element_count"] = int(np.count_nonzero(mask))
            bins[label]["primary_gate_failures"] = {
                stage.removeprefix("gate_"): int(np.count_nonzero(mask & ~concatenated[stage].astype(bool)))
                for stage in merged
                if stage.startswith("gate_")
            }
        high_c_mask = masks["10^4-10^5"] | masks["10^5-10^6"] | masks[">=10^6"] | masks["infinity"]
        high_c_gate_fails = {
            stage.removeprefix("gate_"): int(np.count_nonzero(high_c_mask & ~concatenated[stage].astype(bool)))
            for stage in merged
            if stage.startswith("gate_")
        }
        monotonic_native_bins = [
            bins[label]["E_grad_native"]["median"]
            for label in BIN_LABELS[:-1]
            if bins[label]["E_grad_native"]["count"] >= 100
        ]
        result[str(rounds)] = {
            "seed_count": len(subset),
            "histogram_element_counts": histogram,
            "total_element_count": sum(histogram.values()),
            "primary_gate_failures_total": {
                stage: sum(int(row["primary_gate_fail_counts"][stage]) for row in subset)
                for stage in subset[0]["primary_gate_fail_counts"]
            } if subset else {},
            "primary_gate_failures_high_C": high_c_gate_fails,
            "by_C_bin": bins,
            "E_grad_native_median_non_decreasing_across_populated_bins": bool(
                len(monotonic_native_bins) >= 2
                and all(left <= right for left, right in zip(monotonic_native_bins, monotonic_native_bins[1:]))
            ),
            "per_seed": {
                str(row["seed"]): {
                    "histogram": row["cancellation_histogram"],
                    "max_C_finite": row["C_max_finite"],
                    "primary_gate_fail_counts": row["primary_gate_fail_counts"],
                    "E_grad_native_median": row["all_element_errors"]["E_grad_native"]["median"],
                    "E_weight_native_median": row["all_element_errors"]["E_weight_native"]["median"],
                }
                for row in subset
            },
        }
    return result


def _decide_category(aggregate: dict[str, Any]) -> dict[str, Any]:
    high_bins = ("10^4-10^5", "10^5-10^6", ">=10^6", "infinity")
    k2_rows = aggregate["2"]["per_seed"]
    k2_high_fail_seeds = [
        seed for seed, row in k2_rows.items()
        if row["primary_gate_fail_counts"].get("weight_post_adamw", 0) > 0
        and sum(row["histogram"][label] for label in high_bins) > 0
    ]
    k2_max = max((float(row["max_C_finite"] or 0.0) for row in k2_rows.values()), default=0.0)
    other_max = max(
        (
            float(row["max_C_finite"] or 0.0)
            for k in ("1", "4", "6")
            for row in aggregate[k]["per_seed"].values()
        ),
        default=0.0,
    )
    k2_reaches_high = k2_max >= 1.0e4
    others_reach_k2_high = other_max >= 1.0e4
    comparable_other_high_bins = any(
        sum(aggregate[k]["histogram_element_counts"][label] for label in high_bins) > 0
        for k in ("4", "6")
    )
    k2_repeated_high_failures = len(k2_high_fail_seeds) >= 2
    other_comparable_no_weight_fails = all(
        aggregate[k]["primary_gate_failures_high_C"].get("weight_post_adamw", 0) == 0
        for k in ("4", "6")
        if sum(aggregate[k]["histogram_element_counts"][label] for label in high_bins) > 0
    )
    if k2_repeated_high_failures and comparable_other_high_bins and other_comparable_no_weight_fails:
        category = "A_K2_REPEATED_HIGH_C_FAILS_REOPEN_K2_LOCAL_REDUCTION"
    elif k2_reaches_high and not others_reach_k2_high:
        category = "C_K2_UNIQUE_EXTREME_C_REQUIRES_SYNTHETIC_CANCELLATION_FIXTURES"
    elif len(k2_high_fail_seeds) == 1 and "20260913" in k2_high_fail_seeds and comparable_other_high_bins:
        category = "D_SEED_20260913_ISOLATED_WITH_COMPARABLE_C_ELSEWHERE_NO_KERNEL_CHANGE_FROM_ONE_EDGE_CASE"
    elif all(aggregate[k]["E_grad_native_median_non_decreasing_across_populated_bins"] for k in ("2", "4", "6")) and comparable_other_high_bins:
        category = "B_SHARED_HIGH_C_DEGRADATION_ACROSS_DEPTHS_GENERAL_SHARED_REDUCTION"
    else:
        category = "NO_SINGLE_PREDECLARED_CATEGORY_RESOLVES_ALL_EVIDENCE_HOLD"
    return {
        "selected_category": category,
        "K2_high_C_primary_fail_seeds": k2_high_fail_seeds,
        "K2_max_C_finite": k2_max,
        "other_K_max_C_finite": other_max,
        "K2_reaches_C_ge_1e4": k2_reaches_high,
        "other_K_reaches_C_ge_1e4": others_reach_k2_high,
        "other_K4_K6_have_elements_in_high_C_bins": comparable_other_high_bins,
        "interpretation_is_descriptive_only": True,
        "kernel_reopened": False,
    }


def run_audit(output_dir: Path, *, resume: bool = False, max_runs: int | None = None) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"shared-gradient audit output exists; use --resume: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    manifest = quality._load_qualification_manifest()
    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    r1, _expanded = modules
    runtime_policy = smoke._validate_runtime_policy(ce, p0)
    if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("shared-gradient audit uses fixed torch 4/1")
    if _sha256_file(quality.BE376_DLL) != EXPECTED_DLL_SHA256:
        raise ValueError("shared-gradient audit selected DLL hash mismatch")
    if _sha256_file(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()) != EXPECTED_LOSS_SHA256:
        raise ValueError("shared-gradient audit canonical loss hash mismatch")

    technical_path = SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415

    train_manifest = json.loads(quality.TRAIN_MANIFEST.read_text(encoding="utf-8"))
    docs, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    source_ids = {
        "canonical_loss": str(Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()),
        "backend_quality_runner": str(Path(quality.__file__).resolve()),
        "causal_smoke_gates": str(Path(smoke.__file__).resolve()),
        "native_bridge": str(Path(bridge.__file__).resolve()),
        "native_r2_runner": str(Path(r2.__file__).resolve()),
        "technical_architecture": str(technical_path.resolve()),
        "audit_runner": str(Path(__file__).resolve()),
    }
    source_hashes = {key: _sha256_file(Path(value)) for key, value in source_ids.items()}
    loss_source = {
        "callable": quality.r1_masked_token_mean_loss.__name__,
        "path": source_ids["canonical_loss"],
        "sha256": source_hashes["canonical_loss"],
    }
    local_gates = smoke._load_gate_module(quality)
    train_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in train_manifest["documents"]]
    actual_keys = [(row["full_text_sha256"], row["retained_513_token_sha256"]) for row in docs]
    if len(docs) != 602 or actual_keys != train_keys:
        raise ValueError("shared-gradient audit training document order differs from frozen manifest")
    positions = [int(value) for value in train_manifest["pairs"][0]["document_indices"]]
    pair_keys = train_manifest["pairs"][0]["document_keys"]
    source = torch.tensor([docs[index]["tokens"] for index in positions], dtype=torch.long)
    teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, 0)
    if quality._tensor_hash(source) != "1ff5c2ff2811190775690c000f444a1d358116b8d96f7267b1d92225f54be435":
        raise ValueError("shared-gradient audit first pair differs from the sealed W0 source")
    window = {
        "inputs": source[:, :WINDOW_TOKENS],
        "targets": source[:, 1 : WINDOW_TOKENS + 1],
        "teacher_logits": teacher_logits,
        "valid_mask": torch.ones((EXPECTED_BATCH, WINDOW_TOKENS), dtype=torch.bool),
    }
    init_dir = output_dir / "common_initializations"
    init_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "audit_progress.json"
    if resume:
        if not progress_path.is_file():
            raise FileNotFoundError("resume requested but audit_progress.json is missing")
        report = json.loads(progress_path.read_text(encoding="utf-8"))
        if report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("audit progress manifest identity changed")
        prior_hashes = report.get("source_hashes", {})
        for name, digest in source_hashes.items():
            if name != "audit_runner" and prior_hashes.get(name) != digest:
                raise ValueError(f"pinned source changed while resuming shared-gradient audit: {name}")
        history = list(report.get("audit_runner_source_hash_history", [prior_hashes.get("audit_runner")]))
        if source_hashes["audit_runner"] not in history:
            history.append(source_hashes["audit_runner"])
        report["audit_runner_source_hash_history"] = history
        report["source_hashes_current_invocation"] = source_hashes
        completed = {(int(row["K"]), int(row["seed"])) for row in report.get("runs", [])}
    else:
        report = {
            "schema": "omega-shared-grad-cancellation-audit-v1",
            "unit": "OMEGA-SHARED-GRAD-CANCELLATION-AUDIT",
            "status": "IN_PROGRESS",
            "qualification_manifest_sha256": manifest["manifest_sha256"],
            "training_manifest_sha256": manifest["training_manifest"]["manifest_sha256"],
            "loss_contract": quality.LOSS_CONTRACT,
            "loss_source": loss_source,
            "native_dll": {"path": str(quality.BE376_DLL), "sha256": quality.BE376_SHA256},
            "teacher_cache_sha256": manifest["hidden_cache"]["cache_file"]["sha256"],
            "teacher_cache_manifest_sha256": manifest["hidden_cache"]["manifest"]["sha256"],
            "runtime_policy": runtime_policy,
            "torch": {"version": str(torch.__version__), "intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
            "cpu": {"processor": platform.processor(), "platform": platform.platform()},
            "depths": list(DEPTHS),
            "seeds": list(SEEDS),
            "origin": "per-seed/per-K common initialization; W0/update0 only; first predeclared training pair",
            "technical_updates": 0,
            "scientific_updates": 0,
            "validation_or_test_loaded": False,
            "teacher_model_forward": False,
            "cached_teacher_logits_used": True,
            "kernel_modified": False,
            "gates_changed": False,
            "cancellation_bins": list(BIN_LABELS),
            "source_ids": source_ids,
            "source_hashes": source_hashes,
            "audit_runner_source_hash_history": [source_hashes["audit_runner"]],
            "runs": [],
            "output_dir": str(output_dir),
        }
        completed = set()
        quality._write_json(progress_path, report)

    run_count = 0
    for seed in SEEDS:
        for rounds in DEPTHS:
            if (rounds, seed) in completed:
                continue
            if max_runs is not None and run_count >= max_runs:
                break
            available = int(psutil.virtual_memory().available)
            if available < 8 * 1024**3:
                raise MemoryError(f"FP64 audit requires >=8 GiB available; got {available}")
            summary = _run_one(
                seed=seed,
                rounds=rounds,
                output_dir=output_dir,
                init_dir=init_dir,
                manifest=manifest,
                train_manifest=train_manifest,
                docs=docs,
                payload=payload,
                teacher_weight=teacher_weight,
                teacher_bias=teacher_bias,
                r2=r2,
                p0=p0,
                ce=ce,
                bridge=bridge,
                local_gates=local_gates,
                technical_model=technical_model,
                loss_source=loss_source,
            )
            summary["audit_runner_sha256"] = source_hashes["audit_runner"]
            report["runs"].append(summary)
            report["technical_updates"] += 3  # PyTorch W0, production-native W0, diagnostic-native W0
            run_count += 1
            quality._write_json(progress_path, report)
        if max_runs is not None and run_count >= max_runs:
            break

    completed = {(int(row["K"]), int(row["seed"])) for row in report["runs"]}
    if completed == {(k, seed) for k in DEPTHS for seed in SEEDS}:
        report["aggregate_by_K"] = _aggregate_by_depth(report["runs"])
        report["decision_table"] = _decide_category(report["aggregate_by_K"])
        report["status"] = "AUDIT_COMPLETE"
        report["report_self_sha256"] = quality._canonical_hash(report)
        quality._write_json(output_dir / "shared_grad_cancellation_audit_report.json", report)
        quality._write_json(progress_path, report)
    else:
        report["status"] = "IN_PROGRESS"
        quality._write_json(progress_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-shared-grad-cancellation-audit-go", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_shared_grad_cancellation_audit_go:
        parser.error("shared-gradient audit requires the explicit MD/283 GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"shared_grad_cancellation_audit_{stamp}"
    try:
        report = run_audit(output_dir, resume=args.resume, max_runs=args.max_runs)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "audit_failure.json",
                {"status": "FAILED_RUNTIME", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "output_dir": str(output_dir)},
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_runs": len(report.get("runs", [])),
                "target_runs": TARGET_UPDATES,
                "technical_updates": report.get("technical_updates", 0),
                "decision": report.get("decision_table"),
                "report": str(output_dir / "shared_grad_cancellation_audit_report.json") if report["status"] == "AUDIT_COMPLETE" else str(output_dir / "audit_progress.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "AUDIT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

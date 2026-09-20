"""End-to-end P0-oracle comparison for the temporary P2-R0 torch bridge.

The reference route is P0's pure PyTorch production route.  The candidate route
replaces only recurrent execution with ``OmegaRecurrentFunction``; embedding,
prelude token projection, readout, vocabulary loss, clipping, and AdamW remain
the same PyTorch route.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


HERE = Path(__file__).resolve().parent
CAMPAIGN_ROOT = HERE.parent
P0_DIR = CAMPAIGN_ROOT / "omega_native_runtime_p0"
CE_DIR = CAMPAIGN_ROOT / "omega_ce_only_baseline"
for path in (P0_DIR, CE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_omega_native_runtime_p0 as p0  # noqa: E402
import run_omega_ce_only_baseline as ce  # noqa: E402
import omega_recurrent_bridge as bridge  # noqa: E402


ATOL = 1.0e-5
RTOL = 1.0e-4
FP32_UNIT_ROUNDOFF = 5.9604644775390625e-8
WINDOW_TOKENS = 256
DEFAULT_UPDATES = 6


class RealExecutionAuthorizationError(RuntimeError):
    pass


class BridgeGateError(RuntimeError):
    pass


def require_real_authorization(confirmed: bool) -> None:
    if not confirmed:
        raise RealExecutionAuthorizationError("R1 bridge validation requires --confirm-real-execution")


def _native_recurrent(model: nn.Module, token_ids: Tensor, previous_state: Tensor) -> tuple[Tensor, Tensor]:
    dimension = int(model.dimension)
    token_part = F.linear(model.embedding(token_ids), model.prelude.weight[:, :dimension], model.prelude.bias)
    block = model.blocks[0]
    return bridge.apply(
        token_part,
        previous_state,
        model.prelude.weight[:, dimension:].contiguous(),
        model.prelude_norm_weight,
        block.qkv.weight,
        block.qkv.bias,
        block.out.weight,
        block.out.bias,
        block.fc1.weight,
        block.fc1.bias,
        block.fc2.weight,
        block.fc2.bias,
        block.norm_weight,
        model.depth_embedding.weight,
        model.gate_logits,
        int(model.rounds),
    )


def _native_forward(model: nn.Module, token_ids: Tensor, previous_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    next_state, readout_states = _native_recurrent(model, token_ids, previous_state)
    student_logits = model.logits_from_projected(model.project(readout_states))
    return next_state, student_logits, readout_states


def _reference_forward(model: nn.Module, token_ids: Tensor, previous_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    next_state, student_logits, boundaries, _ = p0._student_forward_timed(model, token_ids, previous_state)
    return next_state, student_logits, boundaries["recurrent_readout_boundary"]


def _load_p0_inputs(manifest_path: Path, cache_file: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], Any, Tensor, Tensor | None]:
    p0.integration.verify_provenance(manifest_path, cache_file=cache_file)
    manifest, resolved_cache = p0.hidden.hidden_cache_load_manifest(manifest_path)
    payload, _, _ = p0.hidden.preload_hidden_cache(resolved_cache, manifest)
    teacher_weight, teacher_bias = p0.hidden.load_lm_head(manifest_path.parent, manifest["lm_head"])
    _, documents, _ = p0.hidden.base.load_frozen_train_documents()
    return documents, manifest, payload, teacher_weight, teacher_bias


def _teacher_logits(payload: Any, teacher_weight: Tensor, teacher_bias: Tensor | None, positions: Sequence[int], window: int) -> Tensor:
    hidden_states = torch.from_numpy(payload[window, list(positions)].copy()).float()
    return p0.hidden.hidden_to_logits(hidden_states, teacher_weight, teacher_bias)


def _tensor_gate(
    reference: Tensor,
    candidate: Tensor,
    *,
    name: str,
    sum_abs_contributions: Tensor | None = None,
    contribution_counts: Tensor | None = None,
    allow_accumulation_fallback: bool = True,
    oracle_reference: Tensor | None = None,
    oracle_levels: Tensor | None = None,
) -> dict[str, Any]:
    left = reference.detach().cpu().contiguous()
    right = candidate.detach().cpu().contiguous()
    if tuple(left.shape) != tuple(right.shape):
        return {"name": name, "pass": False, "reason": "shape_mismatch", "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}
    difference = (right - left).abs()
    tolerance = ATOL + RTOL * left.abs()
    normal = torch.isfinite(right) & (difference <= tolerance)
    failed = ~normal
    fallback = torch.zeros_like(normal, dtype=torch.bool)
    fallback_unavailable_count = int(failed.sum().item()) if sum_abs_contributions is None or contribution_counts is None else 0
    fallback_fail_count = 0
    max_kappa = 0.0
    max_eta = 0.0
    max_gamma = 0.0
    max_eta_over_2gamma = 0.0
    oracle_fallback = torch.zeros_like(normal, dtype=torch.bool)
    max_e_n = 0.0
    max_e_t = 0.0
    max_gamma_h_a = 0.0
    failure_diagnostics: list[dict[str, Any]] = []
    if sum_abs_contributions is not None and contribution_counts is not None:
        contribution = sum_abs_contributions.detach().cpu().contiguous()
        counts = contribution_counts.detach().cpu().contiguous()
        if tuple(contribution.shape) != tuple(left.shape):
            raise BridgeGateError(f"sum_abs_contributions shape mismatch for {name}: {tuple(contribution.shape)} != {tuple(left.shape)}")
        if tuple(counts.shape) != tuple(left.shape):
            raise BridgeGateError(f"contribution_counts shape mismatch for {name}: {tuple(counts.shape)} != {tuple(left.shape)}")
        counts_float = counts.to(dtype=torch.float64)
        kappa = contribution / left.abs().clamp_min(1.0e-30)
        eta = difference / contribution.clamp_min(1.0e-30)
        gamma = (counts_float * FP32_UNIT_ROUNDOFF) / (1.0 - counts_float * FP32_UNIT_ROUNDOFF)
        bound_valid = (counts > 0) & (counts_float * FP32_UNIT_ROUNDOFF < 1.0)
        bound = 2.0 * gamma
        fallback = failed & allow_accumulation_fallback & bound_valid & (eta.to(dtype=torch.float64) <= bound)
        fallback_fail_count = int((failed & ~fallback).sum().item())
        if failed.any():
            max_kappa = float(kappa[failed].max().item())
            max_eta = float(eta[failed].max().item())
            max_gamma = float(gamma[failed].max().item())
            max_eta_over_2gamma = float((eta.to(dtype=torch.float64) / bound.clamp_min(1.0e-30))[failed].max().item())
            for index in failed.nonzero(as_tuple=False):
                flat_index = tuple(int(value) for value in index.tolist())
                n = int(counts[flat_index].item())
                eta_value = float(eta[flat_index].item())
                gamma_value = float(gamma[flat_index].item())
                ratio = eta_value / (2.0 * gamma_value) if gamma_value > 0.0 else float("inf")
                oracle_pass = False
                oracle_diagnostic: dict[str, Any] = {}
                if oracle_reference is not None and oracle_levels is not None:
                    oracle = oracle_reference[flat_index].to(dtype=torch.float64)
                    native = right[flat_index].to(dtype=torch.float64)
                    torch_reference = left[flat_index].to(dtype=torch.float64)
                    level = int(oracle_levels[flat_index].item())
                    level_product = level * FP32_UNIT_ROUNDOFF
                    gamma_h = level_product / (1.0 - level_product) if level_product < 1.0 else float("inf")
                    e_n = float((native - oracle).abs().item())
                    e_t = float((torch_reference - oracle).abs().item())
                    gamma_h_a = gamma_h * float(contribution[flat_index].item())
                    oracle_pass = e_n <= gamma_h_a and e_n < e_t
                    oracle_fallback[flat_index] = oracle_pass
                    max_e_n = max(max_e_n, e_n)
                    max_e_t = max(max_e_t, e_t)
                    max_gamma_h_a = max(max_gamma_h_a, gamma_h_a)
                    oracle_diagnostic = {
                        "g64": float(oracle.item()),
                        "E_N": e_n,
                        "E_T": e_t,
                        "h": level,
                        "gamma_h_A": gamma_h_a,
                        "classification": "oracle_superior" if oracle_pass else "oracle_fail",
                    }
                failure_diagnostics.append({
                    "index": list(flat_index),
                    "abs_error": float(difference[flat_index].item()),
                    "relative_error": float((difference / left.abs().clamp_min(1.0e-30))[flat_index].item()),
                    "n": n,
                    "sum_abs": float(contribution[flat_index].item()),
                    "kappa": float(kappa[flat_index].item()),
                    "eta": eta_value,
                    "gamma_n": gamma_value,
                    "eta_over_2gamma": ratio,
                    "classification": "oracle_superior" if oracle_pass else (
                        "primary_only" if not allow_accumulation_fallback else (
                            "within_count_bound" if bool(fallback[flat_index].item()) else (
                                "count_unavailable" if n == 0 else "outside_count_bound"
                            )
                        )
                    ),
                    **oracle_diagnostic,
                })
    report: dict[str, Any] = {
        "name": name,
        "shape": list(left.shape),
        "max_abs_error": float(difference.max().item()) if difference.numel() else 0.0,
        "max_relative_error": float((difference / left.abs().clamp_min(1.0e-30)).max().item()) if difference.numel() else 0.0,
        "normal_gate_fail_count": int(failed.sum().item()),
        "cancellation_fallback_count": int(fallback.sum().item()),
        "fallback_unavailable_count": fallback_unavailable_count,
        "fallback_fail_count": fallback_fail_count,
        "max_kappa": max_kappa,
        "max_eta": max_eta,
        "max_gamma_n": max_gamma,
        "max_eta_over_2gamma": max_eta_over_2gamma,
        "oracle_fallback_count": int(oracle_fallback.sum().item()),
        "oracle_fail_count": int((failed & ~oracle_fallback).sum().item()) if oracle_reference is not None else 0,
        "max_E_N": max_e_n,
        "max_E_T": max_e_t,
        "max_gamma_h_A": max_gamma_h_a,
        "failure_diagnostics": failure_diagnostics,
        "pass": bool(torch.all(normal | fallback | oracle_fallback).item()),
    }
    return report


def _parameter_grads(model: nn.Module) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise BridgeGateError(f"missing gradient: {name}")
        result[name] = parameter.grad.detach().clone()
    return result


def _optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        result[name] = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
    return result


def _compare_update(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    update: int,
    k: int,
) -> dict[str, Any]:
    direct_tensors: list[tuple[str, Tensor, Tensor]] = [
        ("loss.ce", reference["losses"]["ce"], candidate["losses"]["ce"]),
        ("loss.kl", reference["losses"]["kl"], candidate["losses"]["kl"]),
        ("loss.total", reference["losses"]["total"], candidate["losses"]["total"]),
        ("readout_states", reference["readout_states"], candidate["readout_states"]),
        ("next_state", reference["next_state"], candidate["next_state"]),
    ]
    reports = [_tensor_gate(left, right, name=name) for name, left, right in direct_tensors]
    sum_names = {
        "prelude_norm_weight": "prelude_norm_weight",
        "gate_logits": "gate_logits",
        "blocks.0.qkv.weight": "block_qkv_weight",
        "blocks.0.qkv.bias": "block_qkv_bias",
        "blocks.0.out.weight": "block_out_weight",
        "blocks.0.out.bias": "block_out_bias",
        "blocks.0.fc1.weight": "block_fc1_weight",
        "blocks.0.fc1.bias": "block_fc1_bias",
        "blocks.0.fc2.weight": "block_fc2_weight",
        "blocks.0.fc2.bias": "block_fc2_bias",
        "blocks.0.norm_weight": "block_norm_weight",
        "depth_embedding.weight": "depth_embedding",
    }
    for name in reference["grads"]:
        sum_name = sum_names.get(name)
        sum_abs = candidate["sum_abs_contributions"].get(sum_name) if sum_name is not None else None
        counts = candidate["contribution_counts"].get(sum_name) if sum_name is not None else None
        reports.append(_tensor_gate(
            reference["grads"][name],
            candidate["grads"][name],
            name=f"grad.{name}",
            sum_abs_contributions=sum_abs,
            contribution_counts=counts,
            allow_accumulation_fallback=name != "depth_embedding.weight",
            oracle_reference=candidate["fp64_d_depth_embedding"] if name == "depth_embedding.weight" else None,
            oracle_levels=candidate["depth_max_level"] if name == "depth_embedding.weight" else None,
        ))
    reports.extend(_tensor_gate(left, right, name=name) for name, left, right in (
        [(f"parameter.{name}", reference["parameters"][name], candidate["parameters"][name]) for name in reference["parameters"]]
        + [
        (f"optimizer.{name}.{slot}", reference["optimizer"][name][slot], candidate["optimizer"][name][slot])
        for name in reference["optimizer"]
        for slot in reference["optimizer"][name]
        if torch.is_tensor(reference["optimizer"][name][slot])
        ]
    ))
    scalar_reports = [
        _tensor_gate(torch.tensor(reference["clip_norm"]), torch.tensor(candidate["clip_norm"]), name="clip_norm"),
    ]
    reports.extend(scalar_reports)
    passed = all(bool(report["pass"]) for report in reports)
    result = {"update": update, "K": k, "pass": passed, "tensors": reports}
    if not passed:
        raise BridgeGateError(json.dumps(result, indent=2, sort_keys=True))
    return result


def _run_route_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    previous_state: Tensor,
    teacher_logits: Tensor,
    *,
    native: bool,
) -> tuple[dict[str, Any], Tensor]:
    optimizer.zero_grad(set_to_none=True)
    if native:
        next_state, student_logits, readout_states = _native_forward(model, inputs, previous_state)
    else:
        next_state, student_logits, readout_states = _reference_forward(model, inputs, previous_state)
    losses = p0.hidden.base.distillation_loss(student_logits, teacher_logits, targets)
    losses["total"].backward()
    gradients = _parameter_grads(model)
    clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ce.CLIP_NORM).item())
    optimizer.step()
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_state = _optimizer_state(model, optimizer)
    return {
        "losses": {name: value.detach().clone() for name, value in losses.items()},
        "grads": gradients,
        "clip_norm": clip_norm,
        "parameters": parameters,
        "optimizer": optimizer_state,
        "next_state": next_state.detach().clone(),
        "readout_states": readout_states.detach().clone(),
        "sum_abs_contributions": bridge.last_sum_abs_contributions() if native else {},
        "contribution_counts": bridge.last_contribution_counts() if native else {},
        "fp64_d_depth_embedding": bridge.last_fp64_d_depth_embedding() if native else None,
        "depth_max_level": bridge.last_depth_max_level() if native else None,
    }, next_state.detach().clone()


def run_comparison(
    *,
    k: int,
    updates: int,
    manifest_path: Path,
    cache_file: Path,
    dll_path: Path | None,
    progress_log: Path | None,
) -> dict[str, Any]:
    if k not in (1, 4):
        raise ValueError("K must be 1 or 4")
    if updates <= 0:
        raise ValueError("updates must be positive")
    if dll_path is not None:
        bridge.configure_library(dll_path)
    documents, _, payload, teacher_weight, teacher_bias = _load_p0_inputs(manifest_path, cache_file)
    reference_model = ce.fresh_model(20260913, k)
    candidate_model = ce.fresh_model(20260913, k)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS, eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY)
    candidate_optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=ce.BASE_LR, betas=ce.ADAMW_BETAS, eps=ce.ADAMW_EPS, weight_decay=ce.WEIGHT_DECAY)
    frozen, _, _ = p0.hidden.base.load_frozen_train_documents()
    reference_state = reference_model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
    candidate_state = candidate_model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
    reports: list[dict[str, Any]] = []
    for update in range(updates):
        pair_index = update // 2
        window = update % 2
        positions = [int(index) for index in frozen["cyclic_pairs"]["pairs"][pair_index]["document_indices"]]
        source = torch.tensor([documents[position]["tokens"] for position in positions], dtype=torch.long)
        offset = window * WINDOW_TOKENS
        inputs = source[:, offset : offset + WINDOW_TOKENS]
        targets = source[:, offset + 1 : offset + WINDOW_TOKENS + 1]
        if window == 0:
            reference_state = reference_model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
            candidate_state = candidate_model.initial_state(p0.PHYSICAL_BATCH, device=torch.device("cpu"))
        teacher_logits = _teacher_logits(payload, teacher_weight, teacher_bias, positions, window)
        heartbeat_stop = threading.Event()
        update_started = time.monotonic()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(30.0):
                event = {
                    "K": k,
                    "update": update,
                    "event": "heartbeat",
                    "elapsed_seconds": round(time.monotonic() - update_started, 1),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if progress_log is not None:
                    with progress_log.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event, sort_keys=True) + "\n")
                        stream.flush()
                print(json.dumps(event, sort_keys=True), flush=True)

        heartbeat_thread = threading.Thread(target=heartbeat, name=f"r1-heartbeat-k{k}-u{update}", daemon=True)
        heartbeat_thread.start()
        try:
            reference_result, reference_state = _run_route_step(reference_model, reference_optimizer, inputs, targets, reference_state, teacher_logits, native=False)
            candidate_result, candidate_state = _run_route_step(candidate_model, candidate_optimizer, inputs, targets, candidate_state, teacher_logits, native=True)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
        comparison = _compare_update(reference_result, candidate_result, update=update, k=k)
        reports.append(comparison)
        if progress_log is not None:
            summary = {
                "K": k,
                "update": update,
                "pass": comparison["pass"],
                "max_abs_error": max(float(report["max_abs_error"]) for report in comparison["tensors"]),
                "max_E_N": max(float(report.get("max_E_N", 0.0)) for report in comparison["tensors"]),
                "max_E_T": max(float(report.get("max_E_T", 0.0)) for report in comparison["tensors"]),
                "max_gamma_h_A": max(float(report.get("max_gamma_h_A", 0.0)) for report in comparison["tensors"]),
                "oracle_fallback_count": sum(int(report.get("oracle_fallback_count", 0)) for report in comparison["tensors"]),
                "oracle_fail_count": sum(int(report.get("oracle_fail_count", 0)) for report in comparison["tensors"]),
            }
            with progress_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, sort_keys=True) + "\n")
                stream.flush()
            print(json.dumps(summary, sort_keys=True), flush=True)
    return {"K": k, "updates": updates, "status": "PASS", "comparisons": reports}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-real-execution", action="store_true")
    parser.add_argument("--k", type=int, choices=(1, 4), action="append", default=None)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--manifest", type=Path, default=p0.SEALED_MANIFEST)
    parser.add_argument("--cache-file", type=Path, default=p0.DEFAULT_CACHE_FILE)
    parser.add_argument("--dll", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "r1_bridge_report.json")
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args(argv)
    require_real_authorization(args.confirm_real_execution)
    combinations = args.k or [1, 4]
    report: dict[str, Any] = {"campaign_id": "OMEGA-NATIVE-RUNTIME-R1-BRIDGE", "status": "PASS", "combinations": {}}
    for k in combinations:
        report["combinations"][f"K{k}"] = run_comparison(
            k=k,
            updates=args.updates,
            manifest_path=args.manifest,
            cache_file=args.cache_file,
            dll_path=args.dll,
            progress_log=args.progress_log or args.output.with_suffix(".progress.jsonl"),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"campaign_id": report["campaign_id"], "status": report["status"], "combinations": list(report["combinations"]), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

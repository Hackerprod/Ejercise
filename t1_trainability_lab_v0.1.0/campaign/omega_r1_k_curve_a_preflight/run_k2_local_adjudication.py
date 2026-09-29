"""FP64 AdamW adjudication of the single K2 W0 post-step parameter residual."""

from __future__ import annotations

import argparse
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
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_backend_quality_qualification as quality  # noqa: E402
import run_loss_contract_causal_smoke as smoke  # noqa: E402
import run_k2_k6_preflight as preflight  # noqa: E402


SEED = 20260913
K = 2
TARGET = (33, 219)
WINDOW_TOKENS = 256
EXPECTED_BATCH = 8
PREFLIGHT_REPORT = (
    HERE
    / "results"
    / "k2_k6_preflight_20260927T121900Z"
    / "k2_k6_preflight_report.json"
)
EXPECTED_LOSS_SHA256 = "4f456775993c60dc58c63a160ecd292c37c5b53518e933cca64d209698927c88"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _gate(reference: torch.Tensor, candidate: torch.Tensor, local_gates: Any, name: str) -> dict[str, Any]:
    return local_gates._tensor_gate(reference, candidate, name=name)


def _fp64_reference(
    *,
    ce: Any,
    technical_model: Any,
    r2: Any,
    p0: Any,
    bundle: dict[str, Any],
    window: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model = preflight._fresh_model(ce, technical_model, SEED, K)
    model.load_state_dict(preflight._clone_tree(bundle["model_state"]), strict=True)
    model = model.to(dtype=torch.float64)
    model.train()
    contribution_records: list[dict[str, torch.Tensor]] = []

    def capture_fc2(_module: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        contribution_records.append({"input": inputs[0], "output": output})

    handle = model.blocks[0].fc2.register_forward_hook(capture_fc2)
    state = bundle["initial_recurrent_state"].detach().to(dtype=torch.float64)
    targets = window["targets"]
    mask = window["valid_mask"]
    teacher_logits64 = window["teacher_logits"].detach().to(dtype=torch.float64)
    next_state, student_logits, readout_states = r2._reference_forward(model, window["inputs"], state, p0)
    loss_terms = quality.r1_masked_token_mean_loss(student_logits, teacher_logits64, targets, mask)
    loss_terms["total"].backward()
    handle.remove()

    expected_calls = K * int(window["inputs"].shape[1])
    if len(contribution_records) != expected_calls:
        raise RuntimeError(f"expected {expected_calls} position×round fc2 calls in FP64, got {len(contribution_records)}")
    round_grad_scalars = [0.0 for _ in range(K)]
    for call_index, row in enumerate(contribution_records):
        grad_output = row["output"].grad
        if grad_output is None:
            raise RuntimeError("FP64 fc2 output gradient was not retained")
        round_index = call_index % K  # recurrent rounds are the inner loop at each token position
        product_terms = grad_output.detach()[..., TARGET[0]] * row["input"].detach()[..., TARGET[1]]
        round_grad_scalars[round_index] += float(product_terms.to(dtype=torch.float64).sum().item())

    total_gradient = model.blocks[0].fc2.weight.grad.detach()
    parameter_count = 0
    norm_squares = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm_squares += torch.sum(parameter.grad.detach() * parameter.grad.detach())
            parameter_count += 1
    fp64_clip_norm = torch.sqrt(norm_squares)
    clip_coef = min(1.0, float(ce.CLIP_NORM) / (float(fp64_clip_norm.item()) + 1.0e-6))
    g_r = round_grad_scalars
    g_total = _one(total_gradient[TARGET])
    g_from_rounds = sum(g_r)
    sum_abs_fp64 = sum(abs(value) for value in g_r)
    cancellation = sum_abs_fp64 / abs(g_from_rounds) if g_from_rounds != 0.0 else float("inf")
    g_clip64 = g_total * clip_coef
    beta1, beta2 = ce.ADAMW_BETAS
    step = 1
    m64 = (1.0 - beta1) * g_clip64
    v64 = (1.0 - beta2) * g_clip64 * g_clip64
    m_hat64 = m64 / (1.0 - beta1**step)
    v_hat64 = v64 / (1.0 - beta2**step)
    weight_old = float(bundle["model_state"]["blocks.0.fc2.weight"][TARGET].item())
    lr = float(ce.BASE_LR)
    weight_decay = float(ce.WEIGHT_DECAY)
    eps = float(ce.ADAMW_EPS)
    weight_new64 = weight_old * (1.0 - lr * weight_decay) - lr * m_hat64 / (v_hat64**0.5 + eps)

    return {
        "model_dtype": "float64",
        "loss_total_fp64": _one(loss_terms["total"]),
        "forward_output_shapes": {
            "next_state": list(next_state.shape),
            "student_logits": list(student_logits.shape),
            "readout_states": list(readout_states.shape),
        },
        "fc2_position_round_call_count": len(contribution_records),
        "g_r0_fp64": g_r[0],
        "g_r1_fp64": g_r[1],
        "g_total_fp64_direct_autograd": g_total,
        "g_total_fp64_from_round_sum": g_from_rounds,
        "round_sum_vs_direct_abs_error": abs(g_total - g_from_rounds),
        "A_fp64_abs_round_contributions": sum_abs_fp64,
        "cancellation_factor_C": cancellation,
        "full_fp64_global_clip_norm": float(fp64_clip_norm.item()),
        "fp64_clip_coefficient": clip_coef,
        "g_clipped_fp64": g_clip64,
        "adamw_step": step,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "lr": lr,
        "weight_decay": weight_decay,
        "exp_avg_fp64": m64,
        "exp_avg_sq_fp64": v64,
        "m_hat_fp64": m_hat64,
        "v_hat_fp64": v_hat64,
        "sqrt_v_hat_fp64": v_hat64**0.5,
        "denominator_fp64": v_hat64**0.5 + eps,
        "w_old": weight_old,
        "w_new_fp64": weight_new64,
        "w_delta_fp64": weight_new64 - weight_old,
        "parameters_with_grad": parameter_count,
    }


def run_adjudication(preflight_report_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"K2 adjudication output is immutable: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight_report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    unsigned = dict(preflight_report)
    signature = unsigned.pop("report_self_sha256", None)
    if not signature or signature != quality._canonical_hash(unsigned):
        raise ValueError("source K2/K6 preflight report self-hash mismatch")
    if preflight_report["per_depth"]["2"]["common_origin"]["window0"]["native_vs_pytorch_gate"]["pass"] is not False:
        raise ValueError("the registered K2 W0 primary failure is absent from the source preflight")
    manifest = quality._load_qualification_manifest()
    if preflight_report.get("qualification_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("K2 preflight/qualification manifest mismatch")
    loss_source_path = Path(quality.r1_masked_token_mean_loss.__code__.co_filename).resolve()
    loss_source_sha = _sha256_file(loss_source_path)
    if loss_source_sha != EXPECTED_LOSS_SHA256:
        raise ValueError("canonical R1 loss source hash changed")

    r2, p0, ce, bridge, modules = quality._load_real_dependencies()
    if str(P2R0) not in sys.path:
        sys.path.insert(0, str(P2R0))
    import omega_recurrent_bridge as diagnostic_bridge  # noqa: PLC0415

    local_gates = smoke._load_gate_module(quality)
    runtime_policy = smoke._validate_runtime_policy(ce, p0)
    r1, _expanded = modules
    tech_path = preflight.SCRIPTS / "run_omega_core_lm_0_r1_training_technical_preflight.py"
    import run_omega_core_lm_0_r1_training_technical_preflight as technical_model  # noqa: PLC0415

    train_manifest = json.loads(quality.TRAIN_MANIFEST.read_text(encoding="utf-8"))
    documents, payload, teacher_weight, teacher_bias = r2._load_inputs(
        p0,
        Path(manifest["hidden_cache"]["manifest"]["path"]),
        Path(manifest["hidden_cache"]["cache_file"]["path"]),
    )
    positions = [int(value) for value in train_manifest["pairs"][0]["document_indices"]]
    pair_keys = train_manifest["pairs"][0]["document_keys"]
    source = torch.tensor([documents[index]["tokens"] for index in positions], dtype=torch.long)
    if quality._tensor_hash(source) != preflight_report["data"]["source_pair_sha256"]:
        raise ValueError("K2 adjudication source batch differs from the original common-origin preflight")
    teacher_logits = r2._teacher_logits(p0, payload, teacher_weight, teacher_bias, positions, 0)
    if quality._tensor_hash(teacher_logits) != preflight_report["data"]["teacher_logits_sha256_by_window"]["0"]:
        raise ValueError("K2 adjudication teacher logits differ from the saved preflight operands")
    window = {
        "inputs": source[:, :WINDOW_TOKENS],
        "targets": source[:, 1 : WINDOW_TOKENS + 1],
        "teacher_logits": teacher_logits,
        "valid_mask": torch.ones((EXPECTED_BATCH, WINDOW_TOKENS), dtype=torch.bool),
    }

    init_meta = preflight_report["common_initializations"][str(K)]
    init_path = Path(init_meta["path"])
    if _sha256_file(init_path) != init_meta["sha256"]:
        raise ValueError("K2 common-origin initialization bundle hash mismatch")
    bundle = torch.load(init_path, map_location="cpu", weights_only=False)
    if int(bundle["seed"]) != SEED or int(bundle["K"]) != K:
        raise ValueError("K2 common initialization identity mismatch")

    captured: dict[str, Any] = {}
    active: dict[str, Any] = {}
    original_clip = torch.nn.utils.clip_grad_norm_
    original_reference = r2._reference_forward
    original_native = r2._native_forward

    def capture_clipped_gradients(parameters: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_clip(parameters, *args, **kwargs)
        current = active.get("model")
        if current is not None:
            active["postclip_gradients"] = {
                name: parameter.grad.detach().clone()
                for name, parameter in current.named_parameters()
                if parameter.grad is not None
            }
        return result

    def capture_reference_forward(model: Any, token_ids: torch.Tensor, previous_state: torch.Tensor, p0_module: Any) -> Any:
        result = original_reference(model, token_ids, previous_state, p0_module)
        if active:
            active["forward"] = {
                "next_state": result[0].detach().clone(),
                "student_logits": result[1].detach().clone(),
                "readout_states": result[2].detach().clone(),
            }
        return result

    def capture_native_forward(model: Any, token_ids: torch.Tensor, previous_state: torch.Tensor, state_part_weight: Any, bridge_module: Any) -> Any:
        result = original_native(model, token_ids, previous_state, state_part_weight, bridge_module)
        if active:
            active["forward"] = {
                "next_state": result[0].detach().clone(),
                "student_logits": result[1].detach().clone(),
                "readout_states": result[2].detach().clone(),
            }
        return result

    torch.nn.utils.clip_grad_norm_ = capture_clipped_gradients
    r2._reference_forward = capture_reference_forward
    r2._native_forward = capture_native_forward
    routes: dict[str, dict[str, Any]] = {}
    try:
        for backend in ("pytorch", "native"):
            model = preflight._fresh_model(ce, technical_model, SEED, K)
            model.load_state_dict(preflight._clone_tree(bundle["model_state"]), strict=True)
            if quality._value_hash(model.state_dict()) != bundle["model_state_sha256"]:
                raise ValueError(f"{backend} K2 W0 model differs from the original common initialization")
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
                raise AssertionError(f"{backend} K2 W0 did not start with the original fresh AdamW")
            state = bundle["initial_recurrent_state"].detach().clone()
            state_part_weight = None
            if backend == "native":
                bridge.configure_library(quality.BE376_DLL)
                if not bridge.runtime_abi_available():
                    raise RuntimeError("certified native runtime ABI unavailable")
                bridge.configure_runtime(4)
                state_part_weight = r2.prepare_native_model(model)
            active.clear()
            active["backend"] = backend
            active["model"] = model
            event, next_state, snapshot = smoke._capture_route_update(
                backend=backend,
                rounds=K,
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
                loss_source={"callable": quality.r1_masked_token_mean_loss.__name__, "path": str(loss_source_path), "sha256": loss_source_sha},
                manifest=manifest,
            )
            routes[backend] = {
                "event": event,
                "forward": active["forward"],
                "preclip_gradient": snapshot["grads"]["blocks.0.fc2.weight"].detach().clone(),
                "postclip_gradient": active["postclip_gradients"]["blocks.0.fc2.weight"],
                "clip_norm": snapshot["clip_norm"],
                "clip_coefficient": snapshot["clip_coefficient"],
                "optimizer": preflight._clone_tree(snapshot["optimizer"]["blocks.0.fc2.weight"]),
                "weight_after_adamw": snapshot["parameters"]["blocks.0.fc2.weight"].detach().clone(),
                "next_state": next_state.detach().clone(),
                "losses": preflight._clone_tree(snapshot["losses"]),
            }
            del model, optimizer
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        r2._reference_forward = original_reference
        r2._native_forward = original_native

    # Repeat only native K2/W0 in the already-existing diagnostic bridge mode,
    # which exports the per-element sum-of-absolute-contributions/count arrays
    # needed by the frozen FP64 accumulation bound. This is not a gate change.
    diagnostic_model = preflight._fresh_model(ce, technical_model, SEED, K)
    diagnostic_model.load_state_dict(preflight._clone_tree(bundle["model_state"]), strict=True)
    torch.set_rng_state(bundle["torch_rng_state"].clone())
    random.setstate(bundle["python_rng_state"])
    diagnostic_model.train()
    diagnostic_optimizer = torch.optim.AdamW(
        diagnostic_model.parameters(),
        lr=ce.BASE_LR,
        betas=ce.ADAMW_BETAS,
        eps=ce.ADAMW_EPS,
        weight_decay=ce.WEIGHT_DECAY,
    )
    diagnostic_state = bundle["initial_recurrent_state"].detach().clone()
    diagnostic_bridge.configure_library(quality.BE376_DLL)
    if not diagnostic_bridge.runtime_abi_available():
        raise RuntimeError("diagnostic bridge runtime ABI is unavailable")
    diagnostic_bridge.configure_runtime(4)
    diagnostic_state_weight = r2.prepare_native_model(diagnostic_model)
    diagnostic_active: dict[str, Any] = {"backend": "native_diagnostic", "model": diagnostic_model}
    active.clear()
    active.update(diagnostic_active)
    diagnostic_event, diagnostic_next_state, diagnostic_snapshot = smoke._capture_route_update(
        backend="native",
        rounds=K,
        update=0,
        model=diagnostic_model,
        optimizer=diagnostic_optimizer,
        inputs=window["inputs"],
        targets=window["targets"],
        teacher_logits=window["teacher_logits"],
        valid_mask=window["valid_mask"],
        state=diagnostic_state,
        r2=r2,
        p0=p0,
        ce=ce,
        bridge=diagnostic_bridge,
        local_gates=local_gates,
        state_part_weight=diagnostic_state_weight,
        source=source,
        pair_positions=positions,
        pair_keys=pair_keys,
        loss_source={"callable": quality.r1_masked_token_mean_loss.__name__, "path": str(loss_source_path), "sha256": loss_source_sha},
        manifest=manifest,
    )
    diagnostic_sum_abs = diagnostic_bridge.last_sum_abs_contributions()["block_fc2_weight"]
    diagnostic_counts = diagnostic_bridge.last_contribution_counts()["block_fc2_weight"]
    diagnostic_depth_levels = diagnostic_bridge.last_depth_max_level()
    production_native_gradient = routes["native"]["preclip_gradient"]
    diagnostic_native_gradient = diagnostic_snapshot["grads"]["blocks.0.fc2.weight"].detach().clone()
    diagnostic_gradient_matches_production_exactly = torch.equal(
        diagnostic_native_gradient, production_native_gradient
    )
    routes["native_diagnostic"] = {
        "preclip_gradient": diagnostic_native_gradient,
        "sum_abs_contributions": diagnostic_sum_abs.detach().clone(),
        "contribution_counts": diagnostic_counts.detach().clone(),
        "depth_max_level": diagnostic_depth_levels,
        "gradient_matches_production_native_exactly": diagnostic_gradient_matches_production_exactly,
        "event": diagnostic_event,
        "next_state": diagnostic_next_state.detach().clone(),
    }
    del diagnostic_model, diagnostic_optimizer

    index = TARGET
    reference = routes["pytorch"]
    native = routes["native"]
    old_weight = bundle["model_state"]["blocks.0.fc2.weight"].detach()
    target_old_weight = _one(old_weight[index])
    g_pytorch = _one(reference["preclip_gradient"][index])
    g_native = _one(native["preclip_gradient"][index])
    g_post_pytorch = _one(reference["postclip_gradient"][index])
    g_post_native = _one(native["postclip_gradient"][index])
    m_pytorch = _one(reference["optimizer"]["exp_avg"][index])
    v_pytorch = _one(reference["optimizer"]["exp_avg_sq"][index])
    m_native = _one(native["optimizer"]["exp_avg"][index])
    v_native = _one(native["optimizer"]["exp_avg_sq"][index])
    w_pytorch = _one(reference["weight_after_adamw"][index])
    w_native = _one(native["weight_after_adamw"][index])

    def stage_pair(left: torch.Tensor, right: torch.Tensor, name: str) -> dict[str, Any]:
        result = _gate(left, right, local_gates, name)
        return result

    stage_gates = {
        "forward_readout_states": stage_pair(reference["forward"]["readout_states"], native["forward"]["readout_states"], "K2.W0.readout_states"),
        "forward_student_logits": stage_pair(reference["forward"]["student_logits"], native["forward"]["student_logits"], "K2.W0.student_logits"),
        "forward_next_state": stage_pair(reference["forward"]["next_state"], native["forward"]["next_state"], "K2.W0.next_state"),
        "canonical_total_loss": stage_pair(reference["losses"]["total"], native["losses"]["total"], "K2.W0.loss.total"),
        "fc2_gradient_preclip": stage_pair(reference["preclip_gradient"], native["preclip_gradient"], "K2.W0.grad.fc2.weight.preclip"),
        "clip_pre_norm": stage_pair(torch.tensor(reference["clip_norm"]), torch.tensor(native["clip_norm"]), "K2.W0.clip.pre_norm"),
        "clip_coefficient": stage_pair(torch.tensor(reference["clip_coefficient"]), torch.tensor(native["clip_coefficient"]), "K2.W0.clip.coefficient"),
        "fc2_gradient_postclip": stage_pair(reference["postclip_gradient"], native["postclip_gradient"], "K2.W0.grad.fc2.weight.postclip"),
        "fc2_exp_avg": stage_pair(reference["optimizer"]["exp_avg"], native["optimizer"]["exp_avg"], "K2.W0.optimizer.fc2.exp_avg"),
        "fc2_exp_avg_sq": stage_pair(reference["optimizer"]["exp_avg_sq"], native["optimizer"]["exp_avg_sq"], "K2.W0.optimizer.fc2.exp_avg_sq"),
        "fc2_weight_post_adamw": stage_pair(reference["weight_after_adamw"], native["weight_after_adamw"], "K2.W0.parameter.fc2.weight.post_adamw"),
    }
    ordered_stages = list(stage_gates.items())
    first_failed_stage = next((name for name, gate in ordered_stages if not gate["pass"]), None)

    # Independent FP64 recomputation from the exact common initialization,
    # token/teacher tensors, and canonical loss; no native weights or DLL changes.
    fp64 = _fp64_reference(
        ce=ce,
        technical_model=technical_model,
        r2=r2,
        p0=p0,
        bundle=bundle,
        window=window,
    )
    g64 = fp64["g_total_fp64_direct_autograd"]
    g64_roundsum = fp64["g_total_fp64_from_round_sum"]
    e_native_gradient = abs(g_native - g64)
    e_torch_gradient = abs(g_pytorch - g64)
    native_sum_abs = _one(routes["native_diagnostic"]["sum_abs_contributions"][index])
    native_contribution_count = int(routes["native_diagnostic"]["contribution_counts"][index].item())
    unit_roundoff = float(local_gates.FP32_UNIT_ROUNDOFF)
    h_product = native_contribution_count * unit_roundoff
    gamma_h = h_product / (1.0 - h_product) if h_product < 1.0 else float("inf")
    gamma_h_A = gamma_h * native_sum_abs
    oracle_gradient_bound_pass = diagnostic_gradient_matches_production_exactly and e_native_gradient <= gamma_h_A
    oracle_gradient_superior = e_native_gradient < e_torch_gradient

    # Reconstruct one AdamW update in FP64 using the shared PyTorch clipping
    # coefficient (which itself passed its frozen numerical gate).
    common_clip_coefficient = float(reference["clip_coefficient"])
    g_clip64 = g64 * common_clip_coefficient
    beta1, beta2 = ce.ADAMW_BETAS
    lr, eps, weight_decay = float(ce.BASE_LR), float(ce.ADAMW_EPS), float(ce.WEIGHT_DECAY)
    step = 1
    exp_avg64 = (1.0 - beta1) * g_clip64
    exp_avg_sq64 = (1.0 - beta2) * g_clip64 * g_clip64
    m_hat64 = exp_avg64 / (1.0 - beta1**step)
    v_hat64 = exp_avg_sq64 / (1.0 - beta2**step)
    w_new64 = target_old_weight * (1.0 - lr * weight_decay) - lr * m_hat64 / (v_hat64**0.5 + eps)
    e_native_weight = abs(w_native - w_new64)
    e_torch_weight = abs(w_pytorch - w_new64)
    adamw_oracle_superior = e_native_weight < e_torch_weight

    # Verify that the formula reconstructs each backend's observed first-step
    # update from its own stored moments, without imposing a new tolerance.
    def formula_from_observed_moments(route: dict[str, Any]) -> dict[str, float]:
        m = route["optimizer"]["exp_avg"][index].detach().double()
        v = route["optimizer"]["exp_avg_sq"][index].detach().double()
        mhat = float((m / (1.0 - beta1**step)).item())
        vhat = float((v / (1.0 - beta2**step)).item())
        reconstructed = target_old_weight * (1.0 - lr * weight_decay) - lr * mhat / (vhat**0.5 + eps)
        actual = _one(route["weight_after_adamw"][index])
        return {"m_hat": mhat, "v_hat": vhat, "w_new_formula_fp64": reconstructed, "w_new_actual_fp32": actual, "abs_reconstruction_residual": abs(actual - reconstructed)}

    native_formula_check = formula_from_observed_moments(native)
    pytorch_formula_check = formula_from_observed_moments(reference)
    if not diagnostic_gradient_matches_production_exactly:
        classification = "INCONCLUSIVE_HOLD"
    elif oracle_gradient_bound_pass and oracle_gradient_superior and adamw_oracle_superior:
        classification = "PASS_ORACLE_SUPERIOR"
    elif not oracle_gradient_bound_pass:
        classification = "ORACLE_NATIVE_REDUCTION_BOUND_FAIL_HOLD"
    elif not oracle_gradient_superior or not adamw_oracle_superior:
        classification = "INCONCLUSIVE_HOLD"
    else:
        classification = "INCONCLUSIVE_HOLD"

    report = {
        "schema": "omega-k2-local-adjudication-v1",
        "unit": "OMEGA-K2-LOCAL-ADJUDICATION",
        "status": classification,
        "classification": classification,
        "source_preflight_report": str(preflight_report_path),
        "source_preflight_report_sha256": _sha256_file(preflight_report_path),
        "qualification_manifest_sha256": manifest["manifest_sha256"],
        "loss_contract": quality.LOSS_CONTRACT,
        "loss_source": {"path": str(loss_source_path), "sha256": loss_source_sha},
        "seed": SEED,
        "K": K,
        "backend": "PyTorch vs native be37623d",
        "update": 0,
        "window": 0,
        "origin": "exact K2 common initialization from original preflight; same pair 0 and W0 teacher logits",
        "scientific_update": False,
        "technical_updates_replayed": 3,
        "runtime": {
            "native_workers": 4,
            "torch_intraop": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "cpu": platform.processor(),
        },
        "primary_failure_preserved": {
            "stage": "parameter.blocks.0.fc2.weight.update0 post-AdamW",
            "index": list(TARGET),
            "reference_pytorch": -0.01675686426460743,
            "candidate_native": -0.016741327941417694,
            "abs_difference": 1.5536323189735413e-05,
            "existing_tolerance": 1.167568643722916e-05,
            "primary_gate_reclassified_as_pass": False,
        },
        "stage_gates_in_predeclared_order": stage_gates,
        "first_failed_stage": first_failed_stage,
        "fc2_weight_scalar_trace": {
            "index": list(TARGET),
            "w_old_common": target_old_weight,
            "gradient_preclip": {"pytorch": g_pytorch, "native": g_native, "abs_difference": abs(g_native - g_pytorch)},
            "clip_norm": {"pytorch": reference["clip_norm"], "native": native["clip_norm"]},
            "clip_coefficient": {"pytorch": reference["clip_coefficient"], "native": native["clip_coefficient"]},
            "gradient_postclip": {"pytorch": g_post_pytorch, "native": g_post_native, "abs_difference": abs(g_post_native - g_post_pytorch)},
            "exp_avg": {"pytorch": m_pytorch, "native": m_native, "abs_difference": abs(m_native - m_pytorch)},
            "exp_avg_sq": {"pytorch": v_pytorch, "native": v_native, "abs_difference": abs(v_native - v_pytorch)},
            "weight_post_adamw": {"pytorch": w_pytorch, "native": w_native, "abs_difference": abs(w_native - w_pytorch)},
        },
        "fp64_oracle": {
            **fp64,
            "gradient_errors": {
                "E_N_native_vs_fp64": e_native_gradient,
                "E_T_pytorch_vs_fp64": e_torch_gradient,
            },
            "native_accumulation_bound": {
                "diagnostic_bridge_gradient_matches_production_native_exactly": diagnostic_gradient_matches_production_exactly,
                "h_from_observed_native_contribution_count": native_contribution_count,
                "unit_roundoff": unit_roundoff,
                "gamma_h": gamma_h,
                "A_native_sum_abs_contributions": native_sum_abs,
                "gamma_h_A": gamma_h_A,
                "E_N_le_gamma_h_A": oracle_gradient_bound_pass,
                "E_N_lt_E_T": oracle_gradient_superior,
            },
            "direct_adamw_operation": {
                "formula": "w_new = w_old*(1-lr*weight_decay) - lr*m_hat/(sqrt(v_hat)+eps)",
                "clip_coefficient_common_input": common_clip_coefficient,
                "g_clipped_fp64": g_clip64,
                "exp_avg_fp64": exp_avg64,
                "exp_avg_sq_fp64": exp_avg_sq64,
                "m_hat_fp64": m_hat64,
                "v_hat_fp64": v_hat64,
                "sqrt_v_hat_fp64": v_hat64**0.5,
                "denominator_fp64": v_hat64**0.5 + eps,
                "w_new_fp64": w_new64,
                "E_N_native_weight_vs_fp64": e_native_weight,
                "E_T_pytorch_weight_vs_fp64": e_torch_weight,
                "native_weight_closer_to_fp64": adamw_oracle_superior,
                "formula_from_observed_native_moments": native_formula_check,
                "formula_from_observed_pytorch_moments": pytorch_formula_check,
            },
            "cancellation_measurement": {
                "g_r0_fp64": fp64["g_r0_fp64"],
                "g_r1_fp64": fp64["g_r1_fp64"],
                "g_total_fp64": fp64["g_total_fp64_from_round_sum"],
                "A_fp64": fp64["A_fp64_abs_round_contributions"],
                "C": fp64["cancellation_factor_C"],
            },
        },
        "tolerances_changed": False,
        "kernel_modified": False,
        "sentinel_k2_started": False,
        "blocks_a_b_started": False,
        "output_dir": str(output_dir),
    }
    report["report_self_sha256"] = quality._canonical_hash(report)
    quality._write_json(output_dir / "k2_local_adjudication_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-k2-local-adjudication-go", action="store_true")
    parser.add_argument("--preflight-report", type=Path, default=PREFLIGHT_REPORT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.confirm_k2_local_adjudication_go:
        parser.error("K2 local adjudication requires the explicit MD/281 GO")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = args.output_dir.resolve() if args.output_dir is not None else HERE / "results" / f"k2_local_adjudication_{stamp}"
    try:
        report = run_adjudication(args.preflight_report.resolve(), output_dir)
    except Exception as error:
        if output_dir.exists():
            quality._write_json(
                output_dir / "adjudication_failure.json",
                {
                    "status": "FAILED_RUNTIME",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "sentinel_k2_started": False,
                    "output_dir": str(output_dir),
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "first_failed_stage": report["first_failed_stage"],
                "oracle": report["fp64_oracle"]["native_accumulation_bound"],
                "cancellation": report["fp64_oracle"]["cancellation_measurement"],
                "adamw": report["fp64_oracle"]["direct_adamw_operation"],
                "report": str(output_dir / "k2_local_adjudication_report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS_ORACLE_SUPERIOR" else 2


if __name__ == "__main__":
    raise SystemExit(main())

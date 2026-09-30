"""V2-0 core adapters and D1-D8 cell bodies for the d512 pilot."""

from __future__ import annotations

import copy
import math
import os
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor

from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES
from omega_v2.ledger import state_dict_sha256
from omega_v2.variants import R4Shared, U4Untied, clone_value_and_storage_report
from omega_v2_2a_cuda_preflight.core import cuda_rounds, shared_cuda_rounds
from omega_v2_2a_cuda_preflight.variants import make_cpu_normal

from .config import (
    D,
    D1_K_VALUES,
    D6_BACKWARD_K_VALUES,
    D6_FORWARD_K_VALUES,
    D7_BETAS,
    D7_EPS,
    D7_FINAL_LOSS_RATIO,
    D7_LR,
    D7_UPDATES,
    D7_WEIGHT_DECAY,
    K4,
    M,
    TRAINABILITY_BATCH,
    CONFORMANCE_BATCH,
)
from .ledger import recompute_d512_ledger
from .metrics import (
    all_finite,
    d6_invariance_record,
    d6_invariance_snapshot,
    d1_correctness_metrics,
    d2_structural_parity,
    d3_d512_gates,
    finite_optimizer_state,
    optimizer_state_canonical_sha256,
    storage_conformance,
    tensor_raw_sha256,
)


class StructuralCorruption(RuntimeError):
    """A parameter/storage/schema corruption that blocks safe continuation."""


class NonFinitePilotState(RuntimeError):
    """A non-finite state prevents the next registered operation."""


def configure_cuda_reference() -> None:
    """Apply the same deterministic FP32 CUDA switches used by V2-2A/D3Q."""
    from omega_v2_2a_cuda_preflight.core import configure_v2_2a_execution

    configure_v2_2a_execution()
    torch.set_num_threads(1)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("V2-2B requires deterministic algorithms ON")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("V2-2B requires TF32 OFF")
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("V2-2B requires CUDA AMP OFF")


def _make_pair(seed: int) -> tuple[R4Shared, U4Untied, dict[str, Any]]:
    r4 = R4Shared.seeded(D, seed, dtype=torch.float32)
    u4 = U4Untied.from_shared(r4)
    clone_report = clone_value_and_storage_report(r4, u4)
    if not clone_report["initial_values_bitwise_equal"]:
        raise StructuralCorruption("U4 initial values are not bitwise clones of R4")
    return r4, u4, clone_report


def _cpu_normal(shape: tuple[int, ...], seed: int) -> Tensor:
    return make_cpu_normal(shape, seed)


def _copy_model_to_cuda(model):
    return copy.deepcopy(model).to(device="cuda")


def _model_weight_copy_equal(cpu_block: ContractualCoreBlock, cuda_block: ContractualCoreBlock) -> bool:
    return all(torch.equal(getattr(cpu_block, family), getattr(cuda_block, family).detach().cpu()) for family in MATRIX_FAMILIES)


def _u4_weight_copy_equal(cpu_model: U4Untied, cuda_model: U4Untied) -> bool:
    return all(
        _model_weight_copy_equal(cpu_model.blocks[index], cuda_model.blocks[index])
        for index in range(K4)
    )


def _copy_report(r4_cpu: R4Shared, u4_cpu: U4Untied, r4_cuda: R4Shared, u4_cuda: U4Untied) -> dict[str, Any]:
    rows = []
    for family in MATRIX_FAMILIES:
        r4_equal = torch.equal(
            getattr(r4_cpu.recurrent.block, family),
            getattr(r4_cuda.recurrent.block, family).detach().cpu(),
        )
        u4_equal = all(
            torch.equal(
                getattr(u4_cpu.blocks[index], family),
                getattr(u4_cuda.blocks[index], family).detach().cpu(),
            )
            for index in range(K4)
        )
        rows.append({"family": family, "R4_cpu_to_cuda_equal": r4_equal, "U4_cpu_to_cuda_equal": u4_equal})
    return {"rows": rows, "all_bitwise_equal": all(row["R4_cpu_to_cuda_equal"] and row["U4_cpu_to_cuda_equal"] for row in rows)}


def _cpu_forward(block: ContractualCoreBlock, x: Tensor, k: int, *, traces: bool) -> tuple[Tensor, list[Tensor]]:
    current = x
    outputs: list[Tensor] = []
    for _ in range(k):
        current = block.step(current)
        if traces:
            outputs.append(current.detach().clone())
    return current, outputs


def run_d1_cell(plan, k: int, mark_boundary=None) -> dict[str, Any]:
    if k not in D1_K_VALUES:
        raise ValueError("D1 accepts only its frozen K values")
    r4_cpu = R4Shared.seeded(D, plan.weight_seed, dtype=torch.float32)
    x_cpu = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.input_seed)
    with torch.no_grad():
        y_cpu, trace_cpu = _cpu_forward(r4_cpu.recurrent.block, x_cpu, k, traces=(k == K4))
    r4_cuda = _copy_model_to_cuda(r4_cpu)
    x_cuda = x_cpu.to(device="cuda", dtype=torch.float32)
    copy_checks = {
        "weights_bitwise_equal": all(
            torch.equal(getattr(r4_cpu.recurrent.block, family), getattr(r4_cuda.recurrent.block, family).detach().cpu())
            for family in MATRIX_FAMILIES
        ),
        "input_bitwise_equal": torch.equal(x_cpu, x_cuda.cpu()),
    }
    if mark_boundary is not None:
        mark_boundary()
    with torch.no_grad():
        y_cuda, trace_cuda = shared_cuda_rounds(r4_cuda.recurrent.block, x_cuda, k, return_trace=True)
    compared = []
    if k == 1:
        compared.append({"output": "K1_final", **d1_correctness_metrics(y_cuda, y_cpu)})
    else:
        for index, (cuda_value, cpu_value) in enumerate(zip(trace_cuda, trace_cpu), start=1):
            compared.append({f"output": f"K4_round_{index}", **d1_correctness_metrics(cuda_value, cpu_value)})
        compared.append({"output": "K4_final", **d1_correctness_metrics(y_cuda, y_cpu)})
    bundle = {
        "cpu": {"final": y_cpu.detach().cpu().contiguous(), **{f"trace_{i+1}": t.cpu().contiguous() for i, t in enumerate(trace_cpu)}},
        "cuda": {"final": y_cuda.detach().cpu().contiguous(), **{f"trace_{i+1}": t.detach().cpu().contiguous() for i, t in enumerate(trace_cuda)}},
    }
    raw_tensor_hashes = {
        f"cpu_final": tensor_raw_sha256(f"D1/{plan.master_seed}/K{k}/cpu_final", y_cpu),
        f"cuda_final": tensor_raw_sha256(f"D1/{plan.master_seed}/K{k}/cuda_final", y_cuda),
    }
    if k == K4:
        raw_tensor_hashes.update({
            f"cpu_trace_{index+1}": tensor_raw_sha256(f"D1/{plan.master_seed}/K{k}/cpu_trace_{index+1}", value)
            for index, value in enumerate(trace_cpu)
        })
        raw_tensor_hashes.update({
            f"cuda_trace_{index+1}": tensor_raw_sha256(f"D1/{plan.master_seed}/K{k}/cuda_trace_{index+1}", value)
            for index, value in enumerate(trace_cuda)
        })
    return {
        "gate": "D1",
        "seed_plan": plan.as_dict(),
        "K": k,
        "outputs": compared,
        "copy_checks": copy_checks,
        "raw_tensor_sha256": raw_tensor_hashes,
        "pass": all(row["finite"] and row["elementwise_gate"]["pass"] and row["normwise_gate"]["pass"] for row in compared) and all(copy_checks.values()),
        "tensor_bundle": bundle,
    }


def run_d2_cell(plan, mark_boundary=None) -> dict[str, Any]:
    r4_cpu, u4_cpu, clone_report = _make_pair(plan.weight_seed)
    x_cpu = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.input_seed)
    r4_cuda = _copy_model_to_cuda(r4_cpu)
    u4_cuda = _copy_model_to_cuda(u4_cpu)
    x_cuda = x_cpu.to(device="cuda", dtype=torch.float32)
    copy_checks = {
        "R4_weights_bitwise_equal": _model_weight_copy_equal(r4_cpu.recurrent.block, r4_cuda.recurrent.block),
        "U4_weights_bitwise_equal": _u4_weight_copy_equal(u4_cpu, u4_cuda),
        "x_bitwise_equal": torch.equal(x_cpu, x_cuda.cpu()),
    }
    if not all(copy_checks.values()):
        raise StructuralCorruption("D2 initial R4/U4 copies or fixed input are not bitwise equal")
    if mark_boundary is not None:
        mark_boundary()
    with torch.no_grad():
        r4_final, r4_trace = shared_cuda_rounds(r4_cuda.recurrent.block, x_cuda, K4, return_trace=True)
        u4_final, u4_trace = cuda_rounds(u4_cuda.block_for_round, x_cuda, K4, return_trace=True)
    parity = d2_structural_parity(
        clone_report=clone_report,
        round_traces_r4=r4_trace,
        round_traces_u4=u4_trace,
        final_r4=r4_final,
        final_u4=u4_final,
    )
    bundle = {
        "R4_trace": {f"round_{i+1}": value.detach().cpu().contiguous() for i, value in enumerate(r4_trace)},
        "U4_trace": {f"round_{i+1}": value.detach().cpu().contiguous() for i, value in enumerate(u4_trace)},
        "R4_final": r4_final.detach().cpu().contiguous(),
        "U4_final": u4_final.detach().cpu().contiguous(),
    }
    clone_hashes = {
        family: {
            "R4": tensor_raw_sha256(f"D2/{plan.master_seed}/{family}/R4_initial", getattr(r4_cpu.recurrent.block, family)),
            **{
                f"U4_{index}": tensor_raw_sha256(
                    f"D2/{plan.master_seed}/{family}/U4_{index}_initial",
                    getattr(u4_cpu.blocks[index], family),
                )
                for index in range(K4)
            },
        }
        for family in MATRIX_FAMILIES
    }
    trace_hashes = {
        name: tensor_raw_sha256(f"D2/{plan.master_seed}/{name}", tensor)
        for name, tensor in {
            **{f"R4_trace_{i+1}": tensor for i, tensor in enumerate(r4_trace)},
            **{f"U4_trace_{i+1}": tensor for i, tensor in enumerate(u4_trace)},
            "R4_final": r4_final,
            "U4_final": u4_final,
        }.items()
    }
    return {"gate": "D2", "seed_plan": plan.as_dict(), "structural": parity, "copy_checks": copy_checks, "clone_hashes": clone_hashes, "trace_raw_hashes": trace_hashes, "pass": bool(parity["pass"] and all(copy_checks.values())), "tensor_bundle": bundle}


def run_d3_cell(plan, mark_boundary=None) -> dict[str, Any]:
    r4_cpu, u4_cpu, _ = _make_pair(plan.weight_seed)
    x_cpu = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.input_seed)
    w_cpu = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.loss_w_seed)
    r4_cuda = _copy_model_to_cuda(r4_cpu)
    u4_cuda = _copy_model_to_cuda(u4_cpu)
    x_cuda = x_cpu.to(device="cuda", dtype=torch.float32)
    w_cuda = w_cpu.to(device="cuda", dtype=torch.float32)
    copy_checks = {
        "R4_weights_bitwise_equal": _model_weight_copy_equal(r4_cpu.recurrent.block, r4_cuda.recurrent.block),
        "U4_weights_bitwise_equal": _u4_weight_copy_equal(u4_cpu, u4_cuda),
        "x_bitwise_equal": torch.equal(x_cpu, x_cuda.cpu()),
        "w_bitwise_equal": torch.equal(w_cpu, w_cuda.cpu()),
    }
    if not all(copy_checks.values()):
        raise StructuralCorruption("D3 R4/U4 weights or x/w were not copied bitwise to CUDA")
    if mark_boundary is not None:
        mark_boundary()
    with torch.enable_grad():
        y_r4 = shared_cuda_rounds(r4_cuda.recurrent.block, x_cuda, K4)
        y_u4 = cuda_rounds(u4_cuda.block_for_round, x_cuda, K4)
        loss_r4 = torch.sum(y_r4 * w_cuda)
        loss_u4 = torch.sum(y_u4 * w_cuda)
        grads_r4 = torch.autograd.grad(loss_r4, tuple(getattr(r4_cuda.recurrent.block, family) for family in MATRIX_FAMILIES))
        params_u4 = [getattr(u4_cuda.blocks[r], family) for r in range(K4) for family in MATRIX_FAMILIES]
        grads_u4 = torch.autograd.grad(loss_u4, tuple(params_u4))
    families = {}
    bundle_families = {}
    for family_index, family in enumerate(MATRIX_FAMILIES):
        g_r32 = grads_r4[family_index].detach().clone()
        g_u32 = [grads_u4[r * len(MATRIX_FAMILIES) + family_index].detach().clone() for r in range(K4)]
        gate = d3_d512_gates(g_r32, g_u32)
        family_bundle = {
            "gR_fp32": g_r32,
            **{f"gU{r}_fp32": value for r, value in enumerate(g_u32)},
            **{f"{key}_fp32": value for key, value in gate["sums_fp32"].items()},
            "gR64_primary": gate["gR64"],
            "S64_primary": gate["S64"],
        }
        bundle_families[family] = family_bundle
        families[family] = {key: value for key, value in gate.items() if key not in ("sums_fp32", "gR64", "S64")}
    return {
        "gate": "D3",
        "seed_plan": plan.as_dict(),
        "families": families,
        "copy_checks": copy_checks,
        "pass": all(item["pass"] for item in families.values()),
        "tensor_bundle": bundle_families,
    }


def run_d4_cpu(plan) -> dict[str, Any]:
    r4, u4, clone_report = _make_pair(plan.weight_seed)
    storage = storage_conformance(r4, u4, MATRIX_FAMILIES)
    r4_schema = [(name, str(value.dtype), tuple(value.shape)) for name, value in r4.state_dict().items()]
    u4_schema = [(name, str(value.dtype), tuple(value.shape)) for name, value in u4.state_dict().items()]
    expected_r4_names = {f"recurrent.block.{family}" for family in MATRIX_FAMILIES}
    expected_u4_names = {f"blocks.{index}.{family}" for index in range(K4) for family in MATRIX_FAMILIES}
    schema_conformant = set(name for name, _, _ in r4_schema) == expected_r4_names and set(name for name, _, _ in u4_schema) == expected_u4_names
    return {
        "gate": "D4",
        "seed_plan": plan.as_dict(),
        "clone_report": clone_report,
        "storage": storage,
        "R4_state_dict_schema": r4_schema,
        "U4_state_dict_schema": u4_schema,
        "state_dict_schema_conformant": schema_conformant,
        "pass": bool(storage["pass"] and schema_conformant),
    }


def run_d6_cell(plan, mode: str, k: int, mark_boundary=None) -> dict[str, Any]:
    if mode not in ("forward", "backward"):
        raise ValueError("D6 mode must be forward or backward")
    if mode == "forward" and k not in D6_FORWARD_K_VALUES:
        raise ValueError("D6 forward K outside frozen set")
    if mode == "backward" and k not in D6_BACKWARD_K_VALUES:
        raise ValueError("D6 backward K outside frozen set")
    r4_cpu = R4Shared.seeded(D, plan.weight_seed, dtype=torch.float32)
    x_cpu = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.input_seed)
    w_cpu = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.loss_w_seed) if mode == "backward" else None
    r4_cuda = _copy_model_to_cuda(r4_cpu)
    x_cuda = x_cpu.to(device="cuda", dtype=torch.float32)
    w_cuda = w_cpu.to(device="cuda", dtype=torch.float32) if w_cpu is not None else None
    copy_checks = {
        "R4_weights_bitwise_equal": _model_weight_copy_equal(r4_cpu.recurrent.block, r4_cuda.recurrent.block),
        "x_bitwise_equal": torch.equal(x_cpu, x_cuda.cpu()),
    }
    if w_cpu is not None and w_cuda is not None:
        copy_checks["w_bitwise_equal"] = torch.equal(w_cpu, w_cuda.cpu())
    if not all(copy_checks.values()):
        raise StructuralCorruption("D6 R4 weights or x/w were not copied bitwise to CUDA")
    if mode == "forward":
        equality_before = d6_invariance_snapshot(r4_cuda)
        if mark_boundary is not None:
            mark_boundary()
        output = shared_cuda_rounds(r4_cuda.recurrent.block, x_cuda, k)
        equality_after = d6_invariance_snapshot(r4_cuda)
        result: dict[str, Any] = {"output_finite": all_finite(output)}
    else:
        x_for_grad = x_cuda.detach().requires_grad_(True)
        assert w_cuda is not None
        equality_before = d6_invariance_snapshot(r4_cuda)
        if mark_boundary is not None:
            mark_boundary()
        output_for_grad = shared_cuda_rounds(r4_cuda.recurrent.block, x_for_grad, k)
        loss = torch.sum(output_for_grad * w_cuda)
        grads = torch.autograd.grad(loss, (x_for_grad, *tuple(r4_cuda.parameters())))
        equality_after = d6_invariance_snapshot(r4_cuda)
        result = {
            "output_finite": all_finite(output_for_grad),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "input_gradient_finite": all_finite(grads[0]),
            "parameter_gradients_finite": all_finite(*grads[1:]),
            "loss": float(loss.detach().item()),
            "input_gradient": grads[0].detach().cpu().contiguous(),
            "parameter_gradients": [grad.detach().cpu().contiguous() for grad in grads[1:]],
        }
    equality = d6_invariance_record(equality_before, equality_after)
    equality_pass = equality["pass"]
    finite = all(value for key, value in result.items() if key.endswith("_finite"))
    return {
        "gate": "D6",
        "seed_plan": plan.as_dict(),
        "mode": mode,
        "K": k,
        "copy_checks": copy_checks,
        "equality": equality,
        "equality_pass": equality_pass,
        "finite": bool(finite),
        "pass": bool(equality_pass and finite),
        "tensor_bundle": {key: value for key, value in result.items() if isinstance(value, (Tensor, list))},
        "metrics": {key: value for key, value in result.items() if not isinstance(value, (Tensor, list))},
    }


def d7_optimizer_loop(forward, target: Tensor, parameters, optimizer, mark_boundary=None) -> dict[str, Any]:
    """Run the frozen D7 20 updates plus the non-grad L20 forward (21 total forwards)."""
    step_rows = []
    l0 = None
    for step in range(1, D7_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        if step == 1 and mark_boundary is not None:
            mark_boundary()
        output = forward()
        loss = torch.mean((output - target) ** 2)
        if step == 1:
            l0 = float(loss.detach().item())
        output_finite = all_finite(output)
        loss_finite = bool(torch.isfinite(loss).item())
        if not output_finite or not loss_finite:
            raise NonFinitePilotState(f"D7 non-finite output/loss at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in parameters]
        gradients_finite = all(
            gradient is not None and bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
        )
        if not gradients_finite:
            raise NonFinitePilotState(f"D7 non-finite/missing gradients at step {step}")
        optimizer.step()
        parameters_finite = all_finite(*(parameter.detach() for parameter in parameters))
        optimizer_finiteness = finite_optimizer_state(optimizer)
        if not parameters_finite or not optimizer_finiteness["all_optimizer_state_finite"]:
            raise NonFinitePilotState(f"D7 non-finite parameters/Adam state at step {step}")
        step_rows.append({
            "step": step,
            "loss": float(loss.detach().item()),
            "output_finite": output_finite,
            "loss_finite": loss_finite,
            "gradients_finite": gradients_finite,
            "parameters_finite": parameters_finite,
            "exp_avg_finite": optimizer_finiteness["exp_avg_finite"],
            "exp_avg_sq_finite": optimizer_finiteness["exp_avg_sq_finite"],
        })
    with torch.no_grad():
        final_output = forward()
        l20 = float(torch.mean((final_output - target) ** 2).item())
    return {
        "L0": float(l0),
        "L20": l20,
        "step_finiteness": step_rows,
        "forward_count": len(step_rows) + 1,
        "optimizer_step_count": len(step_rows),
    }


def run_d7_cell(plan, variant: str, mark_boundary=None) -> dict[str, Any]:
    if variant not in ("R4", "U4"):
        raise ValueError("D7 variant must be R4 or U4")
    r4_cpu, u4_cpu, clone_report = _make_pair(plan.weight_seed)
    x_cpu = _cpu_normal((TRAINABILITY_BATCH, M, D), plan.input_seed)
    target_cpu = _cpu_normal((TRAINABILITY_BATCH, M, D), plan.target_seed)
    model_cpu = r4_cpu if variant == "R4" else u4_cpu
    model = _copy_model_to_cuda(model_cpu)
    x = x_cpu.to(device="cuda", dtype=torch.float32)
    target = target_cpu.to(device="cuda", dtype=torch.float32)
    if variant == "R4":
        weights_copy_equal = _model_weight_copy_equal(r4_cpu.recurrent.block, model.recurrent.block)
    else:
        weights_copy_equal = _u4_weight_copy_equal(u4_cpu, model)
    copy_checks = {"weights_bitwise_equal": weights_copy_equal, "x_bitwise_equal": torch.equal(x_cpu, x.cpu()), "target_bitwise_equal": torch.equal(target_cpu, target.cpu())}
    if not all(copy_checks.values()):
        raise StructuralCorruption(f"D7 {variant} weights/x/target were not copied bitwise to CUDA")
    parameters = tuple(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=D7_LR,
        betas=D7_BETAS,
        eps=D7_EPS,
        weight_decay=D7_WEIGHT_DECAY,
    )
    def forward():
        if variant == "R4":
            return shared_cuda_rounds(model.recurrent.block, x, K4)
        return cuda_rounds(model.block_for_round, x, K4)

    trainability = d7_optimizer_loop(forward, target, parameters, optimizer, mark_boundary)
    l0_value = trainability["L0"]
    l20 = trainability["L20"]
    step_rows = trainability["step_finiteness"]
    ratio = l20 / l0_value if l0_value != 0 else (0.0 if l20 == 0 else math.inf)
    return {
        "gate": "D7",
        "seed_plan": plan.as_dict(),
        "variant": variant,
        "updates": D7_UPDATES,
        "L0": l0_value,
        "L20": l20,
        "L20_over_L0": ratio,
        "forward_count": trainability["forward_count"],
        "optimizer_step_count": trainability["optimizer_step_count"],
        "loss_reduction_pass": bool(l20 <= D7_FINAL_LOSS_RATIO * l0_value),
        "step_finiteness": step_rows,
        "all_steps_finite": all(row["output_finite"] and row["loss_finite"] and row["gradients_finite"] and row["parameters_finite"] and row["exp_avg_finite"] and row["exp_avg_sq_finite"] for row in step_rows),
        "initial_R4_U4_clone_report": clone_report,
        "copy_checks": copy_checks,
        "final_parameter_canonical_sha256": state_dict_sha256(model),
        "final_optimizer_state_canonical_sha256": optimizer_state_canonical_sha256(optimizer, model),
    }


def run_calibration_qa_cpu(master_seed: int) -> dict[str, Any]:
    from .config import make_seed_plan

    plan = make_seed_plan(master_seed, mode="qa")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ledger = recompute_d512_ledger()
    r4, u4, clone_report = _make_pair(plan.weight_seed)
    x = _cpu_normal((CONFORMANCE_BATCH, M, D), plan.input_seed)
    with torch.no_grad():
        y_r4, trace_r4 = r4(x, return_trace=True)
        y_u4, trace_u4 = u4(x, return_trace=True)
    trace_equal = [bool(torch.equal(trace_r4[i], trace_u4[i])) for i in range(K4)]
    final_equal = bool(torch.equal(y_r4, y_u4))
    d4 = storage_conformance(r4, u4, MATRIX_FAMILIES)
    # CPU-only shape/equation wiring check; it is explicitly not a CUDA conformance claim.
    with torch.no_grad():
        d1_k1_r4, _ = _cpu_forward(r4.recurrent.block, x, 1, traces=False)
        d1_k1_repeat, _ = _cpu_forward(r4.recurrent.block, x, 1, traces=False)
        d1_k4_r4, d1_k4_trace = _cpu_forward(r4.recurrent.block, x, K4, traces=True)
        d1_k4_repeat, d1_k4_repeat_trace = _cpu_forward(r4.recurrent.block, x, K4, traces=True)
    d1_cpu_wiring = bool(
        torch.equal(d1_k1_r4, d1_k1_repeat)
        and torch.equal(d1_k4_r4, d1_k4_repeat)
        and all(torch.equal(a, b) for a, b in zip(d1_k4_trace, d1_k4_repeat_trace))
    )
    qa_pass = bool(ledger["exact_match"] and clone_report["initial_values_bitwise_equal"] and all(trace_equal) and final_equal and d4["pass"] and d1_cpu_wiring)
    return {
        "schema": "omega-v2-2b-calibration-cpu-qa-v1",
        "classification": "CALIBRATION_QA_ONLY",
        "V2_2B_verdict": None,
        "seed_plan": plan.as_dict(),
        "D2_structural": {"clone_report": clone_report, "trace_equal": trace_equal, "final_equal": final_equal},
        "D4_storage": d4,
        "D5_ledger": ledger,
        "D1_CPU_wiring_check_NON_SCIENTIFIC": d1_cpu_wiring,
        "cpu_reference_operations": "D512 CPU forward/plumbing only; no CPU performance claim",
        "cuda_kernels_launched": 0,
        "heldout_seed_values_materialized": False,
        "pass": qa_pass,
    }

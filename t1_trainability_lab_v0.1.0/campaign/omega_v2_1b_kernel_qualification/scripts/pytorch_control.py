from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time


def set_process_affinity(logical_processor_ids: list[int]) -> tuple[int, int]:
    if not logical_processor_ids or any(index < 0 or index >= 64 for index in logical_processor_ids):
        raise RuntimeError("PyTorch control requires four group-0 P-core logical processor IDs")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_process = kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_process_mask = kernel32.GetProcessAffinityMask
    get_process_mask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    set_process_mask = kernel32.SetProcessAffinityMask
    set_process_mask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    process = get_process()
    old_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not get_process_mask(process, ctypes.byref(old_mask), ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    requested = sum(1 << index for index in logical_processor_ids)
    if requested & ~system_mask.value:
        raise RuntimeError("selected CPU-set logical IDs are outside the active group-0 process affinity mask")
    if not set_process_mask(process, ctypes.c_size_t(requested)):
        raise ctypes.WinError(ctypes.get_last_error())
    observed = ctypes.c_size_t()
    if not get_process_mask(process, ctypes.byref(observed), ctypes.byref(system_mask)) or observed.value != requested:
        raise RuntimeError("Windows did not preserve the PyTorch process affinity mask")
    return int(old_mask.value), requested


def qpc_frequency() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    value = ctypes.c_longlong()
    if not kernel32.QueryPerformanceFrequency(ctypes.byref(value)) or value.value <= 0:
        raise RuntimeError("QueryPerformanceFrequency failed in PyTorch control process")
    return int(value.value)


def qpc_ticks() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    value = ctypes.c_longlong()
    if not kernel32.QueryPerformanceCounter(ctypes.byref(value)):
        raise RuntimeError("QueryPerformanceCounter failed in PyTorch control process")
    return int(value.value)


def fixed_state(torch, m: int, d: int):
    indices = torch.arange(m * d, dtype=torch.int64, device="cpu")
    values = torch.remainder(indices * 37 + (d + m), 127) - 63
    return (values.to(torch.float32) * (1.0 / 256.0)).reshape(1, m, d)


def max_error(reference, candidate) -> dict[str, float]:
    absolute = (reference - candidate).abs()
    relative = absolute / reference.abs().clamp_min(1e-12)
    return {"max_abs": float(absolute.max().item()), "max_rel": float(relative.max().item())}


def measure(block, initial_state, rounds: int, frequency: int, torch) -> dict[str, object]:
    def forward():
        state = initial_state
        for _ in range(rounds):
            state = block.step(state)
        return state

    with torch.inference_mode():
        for _ in range(10):
            warm = forward()
        if not bool(torch.isfinite(warm).all().item()):
            raise RuntimeError("PyTorch KQ warmup returned a non-finite state")
        samples = []
        final = warm
        for _ in range(31):
            start = qpc_ticks()
            final = forward()
            stop = qpc_ticks()
            if stop <= start:
                raise RuntimeError("PyTorch KQ QPC interval was non-positive")
            samples.append((stop - start) / frequency)
        ordered = sorted(samples)
        median_seconds = ordered[len(ordered) // 2]
        if not bool(torch.isfinite(final).all().item()):
            raise RuntimeError("PyTorch KQ measured output returned non-finite values")
        checksum = hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest()
    return {
        "warmups": 10,
        "repetitions": 31,
        "median_seconds": median_seconds,
        "sample_seconds": samples,
        "output_sha256": checksum,
        "output": final.detach().cpu(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-processor-ids", required=True)
    parser.add_argument("--dequantized-weights", required=True)
    parser.add_argument("--native-output-m4", required=True)
    parser.add_argument("--native-output-m16", required=True)
    parser.add_argument("--native-output-m8-k4", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    logical_ids = [int(value) for value in args.logical_processor_ids.split(",") if value]
    old_affinity, active_affinity = set_process_affinity(logical_ids)

    repo_root = Path(__file__).resolve().parents[4]
    v2_0_unit = repo_root / "t1_trainability_lab_v0.1.0" / "campaign" / "omega_v2_0_conformance"
    sys.path.insert(0, str(v2_0_unit))
    import torch
    from omega_v2.core import ContractualCoreBlock, MATRIX_FAMILIES

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)

    seed = 20260929
    source_block = ContractualCoreBlock(512, dtype=torch.float32, seed=seed).eval()
    source_hash = hashlib.sha256()
    for name in MATRIX_FAMILIES:
        source_hash.update(name.encode("utf-8"))
        source_hash.update(b"\0")
        source_hash.update(getattr(source_block, name).detach().contiguous().numpy().tobytes())

    dequant_bytes = Path(args.dequantized_weights).read_bytes()
    dequant_tensor = torch.frombuffer(bytearray(dequant_bytes), dtype=torch.float32)
    dequant_block = ContractualCoreBlock(512, dtype=torch.float32, seed=seed).eval()
    with torch.no_grad():
        offset = 0
        for name in MATRIX_FAMILIES:
            parameter = getattr(dequant_block, name)
            count = parameter.numel()
            parameter.copy_(dequant_tensor[offset : offset + count].reshape_as(parameter))
            offset += count
        if offset != dequant_tensor.numel():
            raise RuntimeError("dequantized Q4 tensor stream has an unexpected float count")

    frequency = qpc_frequency()
    original_state_m4 = fixed_state(torch, 4, 512)
    original_state_m16 = fixed_state(torch, 16, 512)
    original_state_m8 = fixed_state(torch, 8, 512)
    source_m4 = measure(source_block, original_state_m4, 1, frequency, torch)
    source_m16 = measure(source_block, original_state_m16, 1, frequency, torch)
    source_m8_k4 = measure(source_block, original_state_m8, 4, frequency, torch)
    dequant_m4 = measure(dequant_block, original_state_m4, 1, frequency, torch)
    dequant_m16 = measure(dequant_block, original_state_m16, 1, frequency, torch)
    dequant_m8_k4 = measure(dequant_block, original_state_m8, 4, frequency, torch)

    native_outputs = {}
    for key, path in (
        ("m4_k1", args.native_output_m4),
        ("m16_k1", args.native_output_m16),
        ("m8_k4", args.native_output_m8_k4),
    ):
        raw = Path(path).read_bytes()
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        m = {"m4_k1": 4, "m16_k1": 16, "m8_k4": 8}[key]
        rounds = 4 if key == "m8_k4" else 1
        native_outputs[key] = tensor.reshape(1, m, 512)
        dequant_reference = {"m4_k1": dequant_m4, "m16_k1": dequant_m16, "m8_k4": dequant_m8_k4}[key]["output"]
        fp32_reference = {"m4_k1": source_m4, "m16_k1": source_m16, "m8_k4": source_m8_k4}[key]["output"]
        native_outputs[key] = {
            "rounds": rounds,
            "kernel_error_vs_dequant_fp32": max_error(dequant_reference, native_outputs[key]),
            "quantization_error_vs_original_fp32": max_error(fp32_reference, dequant_reference),
        }

    payload = {
        "schema": "omega-v2-1b-pytorch-control-v1",
        "source": {
            "class": "omega_v2.core.ContractualCoreBlock",
            "path": str((v2_0_unit / "omega_v2" / "core.py").resolve()),
            "seed": seed,
            "source_weight_value_sha256": source_hash.hexdigest(),
        },
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "eval": True,
        "grad_enabled": torch.is_grad_enabled(),
        "process_affinity": {"previous_mask": old_affinity, "active_mask": active_affinity, "logical_processor_ids": logical_ids},
        "qpc_frequency": frequency,
        "S_native_cells": {
            "d512_m8_K4": {
                "native_median_seconds": None,
                "pytorch_fp32_original_median_seconds": source_m8_k4["median_seconds"],
                "pytorch_fp32_dequant_q4_median_seconds_diagnostic": dequant_m8_k4["median_seconds"],
                "pytorch_original_samples_seconds": source_m8_k4["sample_seconds"],
                "pytorch_dequant_samples_seconds": dequant_m8_k4["sample_seconds"],
                "dequant_vs_original_median_delta_fraction": abs(dequant_m8_k4["median_seconds"] - source_m8_k4["median_seconds"]) / source_m8_k4["median_seconds"],
            }
        },
        "correctness_side_checks": native_outputs,
        "pytorch_output_hashes": {
            "source_m4_k1": source_m4["output_sha256"],
            "source_m16_k1": source_m16["output_sha256"],
            "source_m8_k4": source_m8_k4["output_sha256"],
            "dequant_m4_k1": dequant_m4["output_sha256"],
            "dequant_m16_k1": dequant_m16["output_sha256"],
            "dequant_m8_k4": dequant_m8_k4["output_sha256"],
        },
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

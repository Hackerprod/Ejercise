"""ENGINEERING_DIAGNOSTIC (MD/308-309): memory/correctness/OOM/timing of the V2-0 core on the local GTX 1650 SUPER.

No optimizer, FP32, no quality verdict. Not a G0/G1/G2 run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

V20 = Path(__file__).resolve().parents[1] / "omega_v2_0_conformance"
sys.path.insert(0, str(V20))
from omega_v2.core import ContractualCoreBlock, rms_norm  # noqa: E402

BUDGET_BYTES = int(3.0 * 1024**3)
MAX_TEST_SECONDS = 30 * 60
OUT = Path(__file__).resolve().parent / "results"


def gpu_step(block: ContractualCoreBlock, state: torch.Tensor) -> torch.Tensor:
    """Same equations as ContractualCoreBlock.step (which is CPU-only guarded)."""
    d = block.d
    n = rms_norm(state)
    q = n @ block.W_Q.transpose(0, 1)
    k = n @ block.W_K.transpose(0, 1)
    v = n @ block.W_V.transpose(0, 1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    att = torch.softmax(scores, dim=-1) @ v
    hidden = state + att @ block.W_O.transpose(0, 1)
    m_in = rms_norm(hidden)
    gate = m_in @ block.W_gate.transpose(0, 1)
    up = m_in @ block.W_up.transpose(0, 1)
    return hidden + (F.silu(gate) * up) @ block.W_down.transpose(0, 1)


def rounds(block, state, K):
    cur = state
    for _ in range(K):
        cur = gpu_step(block, cur)
    return cur


def macs_per_round(d, m):
    return 16 * m * d * d + 2 * m * m * d


def correctness(d, m, K, seed=20260929):
    block = ContractualCoreBlock(d, seed=seed)
    g = torch.Generator().manual_seed(7)
    state = torch.randn(2, m, d, generator=g)
    with torch.no_grad():
        ref = state
        for _ in range(K):
            ref = block.step(ref)
    gblock = ContractualCoreBlock(d, seed=seed).cuda()
    with torch.no_grad():
        out = rounds(gblock, state.cuda(), K).cpu()
    diff = (out - ref).abs()
    return {"d": d, "m": m, "K": K, "max_abs": float(diff.max()), "max_rel": float((diff / ref.abs().clamp_min(1e-6)).max()),
            "pass": bool(float(diff.max()) < 1e-3)}


def probe(d, m, K, B, warm=1, reps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row = {"d": d, "m": m, "K": K, "B": B}
    try:
        block = ContractualCoreBlock(d).cuda()
        state = torch.randn(B, m, d, device="cuda", requires_grad=True)
        times = []
        for i in range(warm + reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = rounds(block, state, K)
            loss = out.pow(2).mean()
            loss.backward()
            for p in block.parameters():
                p.grad = None
            state.grad = None
            torch.cuda.synchronize()
            if i >= warm:
                times.append(time.perf_counter() - t0)
        med = sorted(times)[len(times) // 2]
        row.update(status="OK", peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, fwd_bwd_median_s=med,
                   gmac_per_s_fwd_bwd=3 * B * K * macs_per_round(d, m) / med / 1e9)
    except torch.OutOfMemoryError:
        row.update(status="OOM_CAPACITY_DIAGNOSTIC", peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    finally:
        for name in ("block", "state", "out", "loss"):
            locals().pop(name, None)
        torch.cuda.empty_cache()
    return row


def main():
    assert torch.cuda.is_available()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, BUDGET_BYTES / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    result = {
        "phase": "ENGINEERING_DIAGNOSTIC", "hardware": "GTX_1650_SUPER_4GB", "status": "AUTHORIZED_DIAGNOSTIC",
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "budget_bytes": BUDGET_BYTES,
        "dtype": "float32", "optimizer": "none", "source_core": "omega_v2_0_conformance ContractualCoreBlock (same equations, GPU step)",
        "correctness": [correctness(d, m, K) for d, m, K in ((64, 4, 2), (256, 8, 4), (512, 8, 4))],
        "param_check": {d: sum(p.numel() for p in ContractualCoreBlock(d).parameters()) for d in (256, 512, 640)},
        "sweep": [],
    }
    for d in (256, 512):
        for m in (4, 8, 16):
            for K in (1, 4, 8):
                for B in (1, 8, 32, 128, 512, 2048):
                    if time.perf_counter() - t_start > MAX_TEST_SECONDS:
                        result["truncated_by_time"] = True
                        break
                    row = probe(d, m, K, B)
                    result["sweep"].append(row)
                    print(json.dumps(row), flush=True)
                    if row["status"] != "OK":
                        break
    result["wall_seconds"] = time.perf_counter() - t_start
    path = OUT / "gpu_memory_diag.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("sha256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

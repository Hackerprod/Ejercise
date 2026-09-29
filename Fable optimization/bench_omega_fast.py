"""Equivalencia numérica + benchmark forward+backward: referencia del repo vs omega_fast.

Ejecutar desde la raíz del repo:
    python bench_omega_fast.py --lab t1_trainability_lab_v0.1.0 [--T 256] [--B 8] [--device cuda] [--compile]
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--lab", default="t1_trainability_lab_v0.1.0")
parser.add_argument("--T", type=int, default=64)
parser.add_argument("--B", type=int, default=8)
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--V", type=int, default=50257)
parser.add_argument("--device", default="cpu")
parser.add_argument("--threads", type=int, default=0)
parser.add_argument("--compile", action="store_true", help="torch.compile del paso secuencial (GPU: reduce-overhead)")
parser.add_argument("--skip-campaign-mode", action="store_true")
args = parser.parse_args()

lab = Path(args.lab).resolve()
sys.path.insert(0, str(lab)); sys.path.insert(0, str(lab / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
from run_omega_core_lm_0_r1_training_technical_preflight import OmegaCoreLM0R1Technical, distillation_loss  # noqa: E402
from omega_fast import OmegaCoreLMFast, distillation_loss_lse, teacher_targets  # noqa: E402

if args.threads:
    torch.set_num_threads(args.threads)
dev = torch.device(args.device)
torch.manual_seed(20260913)
B, T, V, D, S, K = args.B, args.T, args.V, 128, 8, args.K

ref = OmegaCoreLM0R1Technical(vocab_size=V, dimension=D, slots=S, rounds=K, variant="shared").to(dev)
fast = OmegaCoreLMFast.from_reference(ref)
tokens = torch.randint(0, V, (B, T + 1), device=dev)
inp, tgt = tokens[:, :T], tokens[:, 1:]
teacher_logits = torch.randn(B, T, V, device=dev) * 3
mask = torch.ones_like(tgt, dtype=torch.bool)
teacher_p, teacher_ne = teacher_targets(teacher_logits)  # una vez (cacheable con el profesor)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


def step_ref_batched():
    ref.zero_grad(set_to_none=True)
    st = ref.initial_state(B, device=dev)
    _, logits = ref.forward_window(inp, st)
    loss = distillation_loss(logits, teacher_logits, tgt, mask)["total"]
    loss.backward()
    return loss


def step_ref_campaign():  # exactamente como _batch_loss del runner: 8 docs con batch 1
    ref.zero_grad(set_to_none=True)
    losses = []
    for i in range(B):
        st = ref.initial_state(1, device=dev)
        _, logits = ref.forward_window(inp[i:i + 1], st)
        losses.append(distillation_loss(logits, teacher_logits[i:i + 1], tgt[i:i + 1], mask[i:i + 1])["total"])
    loss = torch.stack(losses).mean(); loss.backward(); return loss


recur = fast.recur_states
if args.compile:
    recur = torch.compile(fast.recur_states, mode="reduce-overhead" if dev.type == "cuda" else "default", dynamic=False)


def step_fast():
    fast.zero_grad(set_to_none=True)
    st = fast.initial_state(B, device=dev)
    _, states = recur(inp, st)
    loss = distillation_loss_lse(fast, states, teacher_p, teacher_ne, tgt, mask)["total"]
    loss.backward()
    return loss


# ---------------------------------------------------------------- equivalencia
l_ref = step_ref_batched(); g_ref = {n: p.grad.clone() for n, p in ref.named_parameters()}
l_fast = step_fast()
print(f"loss ref={l_ref.item():.6f}  fast={l_fast.item():.6f}  |Δ|={abs(l_ref.item()-l_fast.item()):.2e}")
# comparar gradientes de los pesos comunes (QKV fusionado → concatenar los del ref)
def grad_pairs():
    b = ref.blocks[0]; fb = fast.blocks[0]
    yield "embedding", g_ref["embedding.weight"], fast.embedding.weight.grad
    yield "prelude.w", g_ref["prelude.weight"], fast.prelude.weight.grad
    yield "qkv.w", torch.cat((g_ref["blocks.0.slot_mix.query.weight"], g_ref["blocks.0.slot_mix.key.weight"], g_ref["blocks.0.slot_mix.value.weight"])), fb.qkv.weight.grad
    yield "fc1.w", g_ref["blocks.0.core.network.0.weight"], fb.fc1.weight.grad
    yield "fc1.b", g_ref["blocks.0.core.network.0.bias"], fb.fc1.bias.grad
    yield "gate_logits", g_ref["gate_logits"], fast.gate_logits.grad
    yield "depth", g_ref["depth_embedding.weight"], fast.depth_embedding.weight.grad
    yield "out_proj", g_ref["output_projection.weight"], fast.output_projection.weight.grad
worst = 0.0
for name, a, b_ in grad_pairs():
    rel = ((a - b_).abs().max() / a.abs().max().clamp_min(1e-12)).item(); worst = max(worst, rel)
    print(f"  grad {name:12s} max|Δ|={(a-b_).abs().max().item():.2e}  rel={rel:.2e}")
print(f"peor error relativo de gradiente: {worst:.2e}  ({'OK' if worst < 1e-4 else 'REVISAR'})")


# ---------------------------------------------------------------- benchmark
def bench(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(); return (time.perf_counter() - t0) / reps

print(f"\ndevice={dev} threads={torch.get_num_threads()} B={B} T={T} K={K} V={V} (forward+backward por update)")
if not args.skip_campaign_mode:
    t = bench(step_ref_campaign); print(f"ref  (modo runner: {B}×batch1) : {t:8.3f} s  -> {B*T/t:8.0f} tok/s")
t_ref = bench(step_ref_batched); print(f"ref  (batch {B})               : {t_ref:8.3f} s  -> {B*T/t_ref:8.0f} tok/s")
t_fast = bench(step_fast);       print(f"fast (batch {B}{', compile' if args.compile else ''})      : {t_fast:8.3f} s  -> {B*T/t_fast:8.0f} tok/s   x{t_ref/t_fast:.2f} vs ref batched")

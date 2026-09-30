# OMEGA_HARDWARE_BASELINE_V3 (draft v0.1, 2026-09-30)

Authority: Sol MD/315 (re-baseline), MD/314, MD/316. Status: DRAFT pending Sol ratification. Does not alter any preregistered experiment retroactively.

## 1. Hierarchy

| Role | Platform / profile |
|---|---|
| PRIMARY_RESEARCH_PLATFORM | i7-13700F + GTX 1650 SUPER (+ RunPod after explicit GO) |
| PRIMARY_CPU_PHYSICAL_PROFILE | 8 P-cores / 8 threads (1 thread per physical P-core, no SMT sibling) |
| THROUGHPUT_PROFILES | 8P/16T; 8P + 8E = 24 logical threads |
| LEGACY_DEPLOYMENT_PROFILE | Ryzen AI 5 330, 4 physical cores (device no longer owned; not measurable) |
| CPU Q4 native runtime | SECONDARY_ENGINEERING_TRACK until it again beats PyTorch/reference |

24T is NOT a primary residency gate (mixes P/E/SMT); it is for system throughput.

## 2. Measured host facts (this machine, 2026-09-30)

| Item | Value | Source |
|---|---|---|
| CPU | 13th Gen Intel Core i7-13700F, 16 cores / 24 threads (8P with SMT + 8E) | Win32_Processor |
| L2 total | 24 MiB (reported 24576 KB) = 8 x 2 MiB (P) + 2 x 4 MiB (E clusters) | Win32_Processor; per-core split from Intel spec, verify with GetLogicalProcessorInformationEx |
| L3 / LLC | 30 MiB shared | Win32_Processor |
| Base clock (reported) | 2.1 GHz (boost higher; not pinned) | Win32_Processor |
| RAM | 31.8 GiB (2 x 16 GiB DDR5, 5200 MT/s) | CIM |
| GPU | GTX 1650 SUPER, 4096 MiB, cc 7.5, driver 617.14 | nvidia-smi |
| GPU torch | .venv_cuda torch 2.11.0+cu128 | gpu_memory_diag |
| Disk (D:) | ~1.47 TB free | PSDrive |
| OS | Windows 11 Enterprise LTSC 2024 (10.0.26100) | CIM |
| Timed V2-1c CPU-set IDs (P-cores) | 256..270 range; V2-1c selected [266,260,256,264] | V2-1c H0 |

## 3. What is host-specific (not universal)

Ryzen-only, historical: "4 physical threads", d512 = L2 candidate, d640 = L2 frontier, 8 MB L3 assumptions, rho <= 0.50 as universal gate, fixed cache knees.
Not obsolete: shared recurrent core, K-flex, matrix workspace m, 16d^2 block, external memory later, R4 vs U4.
Thesis wording (MD/315): recurrence/sharing = general architectural thesis; cache residency = host-dependent deployment hypothesis.

## 4. Outstanding derivations (in order, per MD/315)

1. Seal V2-1c (terminal V2_1C_INVALID_PREFLIGHT, MD/316).
2. Complete OMEGA_EVIDENCE_TRANSFER_REGISTRY.md.
3. CPU non-gate diagnostic: 1P/2P/4P/8P, 8P/16T, E-cores, full 24T.
4. Re-derive T0 with this host's own cache knees (P-core L2 2 MiB, LLC ~30 MiB); explore d1024–d2048 for crossing.
5. Open V2-2 GPU (PyTorch/CUDA R4 vs U4, d256 then d512) only on Sol's explicit decision (GPU_HOLD until then).

## 5. Open issues

- Correctness comparator for a full-size Q4 kernel is ill-conditioned near zero (MD/316); any future physical unit needs a new preregistered instrument. Not part of this baseline.
- Result validity of any i7 rho is host-specific; never claim Ryzen transfer.

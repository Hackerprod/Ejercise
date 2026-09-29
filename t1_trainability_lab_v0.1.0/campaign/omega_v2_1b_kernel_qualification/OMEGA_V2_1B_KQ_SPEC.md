# OMEGA-V2-1b — Kernel Qualification only

Authority: `MD/306.md` and `MD/307.md`. This is a separate qualification unit. It must not modify or overwrite V2-1 attempt_01 or attempt_02, and it must not perform any A/B/C residency sweep.

## Frozen boundaries

- Maximum two candidate kernels. Candidate selection uses only KQ correctness, `E_Q4`, `E_FULL(4)`, `E_FULL(16)`, and `S_native`.
- Persistent native weights remain signed symmetric Q4, group size 32, FP16 scales, no zero-point. No persistent FP32/Q8/dequantized weight copy is allowed.
- Candidate scratch is at most 64 KiB per worker. Candidate 1 uses a temporary output-row dequant tile of `4*640*sizeof(float) = 10,240` bytes, reused across all slots in that row tile; the tile is stack scratch and is overwritten for every row.
- The old V2-1 source/binary/results remain unchanged. This unit compiles into its own build directory and writes only under `results/omega_v2_1b_kernel_qualification/`.
- KQ performs no A/B/C measurement and does not read or calculate `rho_resident` for candidate selection.

## KQ measurements

Use the four CPU-set IDs frozen by attempt_02, one hardware thread per P-core, fixed affinity, persistent worker pool, Windows QPC, MSVC Release with the V2-1 AVX2/FMA flags. Each KQ cell uses 10 warmups and 31 measured samples.

1. `P_FMA`: measured FP32 AVX2/FMA L1-resident throughput on the same four P-cores; aggregate MAC/s median. Per worker, input+weight+output tile footprint must fit the measured L1D.
2. `P_H0(4)` and `P_H0(16)`: candidate Q4 linear W_Q throughput on d512, same packed weights, scales, sharding, selected cores, and candidate inner kernel. Weights are pre-touched outside timing.
3. `P_FULL(4)` and `P_FULL(16)`: one resident full-block K=1 path at d512. This is labelled `KQ_FULL_RESIDENT`, never A/B/C. Also measure the S_native control at d512,m8,K4.
4. `E_Q4=P_H0(16)/P_FMA`; require `>=0.25`. `E_FULL(m)=P_FULL(m)/P_H0(m)`; require both m4 and m16 `>=0.60`.
5. `S_native=median(T_PyTorch_FP32)/median(T_native_Q4)` at d512,m8,K4; require `>=1.20`. PyTorch uses V2-0 `ContractualCoreBlock` FP32 source weights before Q4; the native path uses Q4 quantized from those same source tensors. The dequantized-Q4 FP32 PyTorch side-check is correctness-only and separately reports kernel and quantization error.
6. Run the d32,m4,K2 vectorized-vs-scalar Q4 check (`max_abs<=1e-5`, `max_rel<=1e-4`) and d512 full-block finite/repeatable-checksum correctness before interpreting timing.
7. Report, never gate, an L1/L2/LLC/DRAM roofline/machine-balance diagnostic, physical Q4 AI, and m1 diagnostic. m1 cannot rescue or alter the m4 residency threshold.

## Candidate state machine

- Candidate 1 passes all five gates: freeze candidate 1; candidate 2 is prohibited; exactly one attempt_03 becomes allowed.
- Candidate 1 fails correctness or any scientific roofline threshold: candidate 2 may be implemented and evaluated KQ-only.
- Candidate 1 passes correctness and scientific thresholds but misses `S_native`: candidate 2 may be evaluated KQ-only; do not freeze candidate 1.
- After candidate 2, either freeze only if all five gates pass and then run at most one attempt_03, or stop as `KERNEL_SCIENTIFICALLY_QUALIFIED / PROJECT_NATIVE_SPEED_GATE_FAIL` or `V2_1B_KERNEL_QUALIFICATION_FAIL` as specified by MD/306-307.
- A frozen candidate is committed before its executable is built and hashed; no source changes occur between the seal and its single attempt_03.

## KQ result status

KQ always leaves `CONFORMANCE_HOLD`, GPU `HOLD`, and T3 `HOLD`. No training, language scoring, GPU execution, LN, or T3 activity is permitted.

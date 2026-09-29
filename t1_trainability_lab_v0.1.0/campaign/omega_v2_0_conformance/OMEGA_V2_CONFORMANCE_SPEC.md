# OMEGA-V2-0 Architectural Conformance Specification

**Scope:** implementation conformance only. This unit builds a small reference core and validates its parameterization, runtime `m`/`K` independence, R4/U4 initialization parity, toy mathematics, and FLOP ledger. It does not train a language model or claim physical residency, speed, quality, K-scaling, or T3 readiness.

## Authority

- `MD/303.md` (read in full before implementation).
- `Conversacion.md`, blob `7027e2ac9d1ba89db08dda73c81e244f3b9b19db`, contractual ranges 2390–2485, 2585–2630, 2688–2860, 2945–3075, 3078–3240.
- `Conversacion LN.md`, blob `fc750a2ae9fb7d3933c54fb91f09ce5568d035ec`.
- `OMEGA_AUDITORIA_CONFORMIDAD_COMPLETA_2026-09-29.md`, SHA-256 `a7e82cddfcacf17f163e7d046853046595f08e532086fb3c2a577432807933dc`.
- Authorized snapshot: `main@38061477d4c2b0c5c20d75d902b21dd1ef0a2611`.

The repository remains under global `CONFORMANCE_HOLD` until the evidence in this isolated unit closes. V2-0 PASS can release only consideration of V2-1 physical T0; it does not release GPU training, language evaluation, or T3.

## Contractual block

Input and state are `S_r ∈ R[B,m,d]`, with `d ∈ {512,640}`, `m ∈ {4,8,16}`, and positive integer runtime `K`. The core contains no tokenizer, lexical embedding, vocabulary head, shell, memory, or retriever.

RMSNorm is parameterless:

```text
RMS(x) = x / sqrt(mean(x², last_dimension) + 1e-6)
```

The attention sublayer has four bias-free full-width matrices `W_Q/W_K/W_V/W_O ∈ R[d,d]`:

```text
X = RMS(S)
Q = X W_Qᵀ ; K = X W_Kᵀ ; V = X W_Vᵀ
A = softmax(Q Kᵀ / sqrt(d), over key slots) V
H = S + A W_Oᵀ
```

Attention is single-head over the `m` workspace slots, without a causal mask or slot-index embeddings. It contributes `4d²` weights.

The MLP is exactly SwiGLU with hidden width `4d`, three bias-free matrices `W_gate/W_up ∈ R[4d,d]`, `W_down ∈ R[d,4d]`:

```text
Z = RMS(H)
M = (SiLU(Z W_gateᵀ) ⊙ (Z W_upᵀ)) W_downᵀ
S_(r+1) = H + M
```

It contributes exactly `12d²` weights. The complete block has exactly `16d²` unique weights: 4,194,304 at d512 and 6,553,600 at d640. No bias, learned norm scale, depth embedding/table, K-indexed gate, or lazy parameter is allowed.

## Runtime and comparison variants

- `SharedRecurrentCore.forward(S,K)` invokes the *same block object* K times. `K` is not stored in parameters or buffers. Acceptance K values are `{1,2,4,8,16}`.
- `m` is read from the input activation only. Acceptance values are `{4,8,16}` and must not affect parameters, shapes, or state-dict hash.
- R4 is one shared block invoked four times.
- U4 is four distinct blocks, independently stored, initialized as bitwise clones of R4. It uses the same operations, order, shapes, and four rounds. Its unique core weights are exactly four times R4.
- Bitwise acceptance runs on CPU, FP32, one thread, deterministic algorithms, no dropout, AMP off. Fixed seeds are 20260929, 20260930, and 20260931.

## Accounting

All parameter ledgers derive from `named_parameters`, tensor shapes, storage identities, and state-dict tensors. R4 and U4 receive separate rows. Q4 values are logical only: 4-bit packed weight values plus one FP16 scale per 32 weights. Physical allocation, padding in a real quantizer, and hardware residency are reported `NOT_MEASURED`.

The FLOP ledger derives GEMM MACs from the actual matrix shapes. With batch size B and m slots, per round:

```text
projection MACs = 4 B m d²
SwiGLU MACs     = 12 B m d²
QKᵀ MACs        = B m² d
AV MACs         = B m² d
forward FLOPs   = 2 × (16 B m d² + 2 B m² d)
```

R4 and U4 both perform four rounds, so their forward FLOP ledgers must be exactly equal. Record non-GEMM operation counts separately: two RMSNorm applications and `2Bmd` RMSNorm elements per round; `Bm` softmax rows and `Bm²` softmax elements; `4Bmd` SiLU, `4Bmd` SwiGLU Hadamard products, and `2Bmd` residual additions.

## Acceptance suite

The nine required test files are listed in `MD/303.md` §3 and implemented under `tests/`. They cover exact introspected parameters; m/K state-dict invariance; R4/U4 round/output bitwise equality for three seeds, both d values, all m values; storage aliasing; FP64 naive-reference agreement at `(d=8,m=2,K=2)`; finite-difference gradient checks for all seven matrix families; zero-weight residual identity; slot permutation equivariance at `(d=8,m=4,K=2)`; introspected FLOP parity; and the post-implementation conformance block.

Toy requirements: naive-reference maximum absolute error ≤1e-12 and relative error ≤1e-10; finite-difference relative error ≤1e-5. Zero-weight residual must be bitwise identity. Slot permutation uses FP64 numerical equivariance tolerance 1e-12 absolute / 1e-10 relative.

## Scope and stop

No linguistic data, DistilGPT2, trained OMEGA checkpoint, optimizer, retriever, memory bank, VALL, LN, RunPod, long run, or T3 is used. A PASS permits only the exact claim that a reference implementation conforms to the 16d² block and is ready to enter V2-1 physical T0. Any failed contract/test leaves `OMEGA_V2_0_CONFORMANCE_HOLD` with a concrete reason; no retrospective contract change is allowed.

# OMEGA-V2-2B — d512 Local Contractual Pilot

## B. OMEGA-V2-2B — d512 Local Contractual Pilot

### 1. Propósito

V2-2B responde exclusivamente:

¿El bloque contractual 16d² escala de d256 a d512 correctamente en CUDA?

¿R4 y U4 siguen siendo estructural y numéricamente conformes?

¿La identidad de gradient-sharing validada por D3Q se sostiene a d512?

¿K permanece runtime-flexible hasta K16?

¿R4 y U4 ejecutan backward + AdamW estable a d512?

¿La GTX 1650 SUPER puede ejecutar un batch local útil dentro de ≤3 GiB allocated?

No responde:

```text
language quality
NLL equivalence
+0.05 nats/token margin
generalization
external memory
T3
RunPod performance
CPU residency
```

### 2. Plataforma

```text
GPU = GTX 1650 SUPER 4 GiB
framework = PyTorch CUDA
dtype = FP32
TF32 = OFF
AMP = OFF
BF16/FP16 = OFF
dropout = 0
deterministic_algorithms = ON
CUBLAS_WORKSPACE_CONFIG = :4096:8
```

CPU sólo se usa como referencia exacta y generación de datos/weights; no se deriva ningún claim de rendimiento CPU.

### 3. Arquitectura

```text
d = 512
m = 8
K4 comparison = K=4
```

Parámetros:

```text
R4 unique core params = 4,194,304
U4 unique core params = 16,777,216
```

Exactamente siete familias:

```text
W_Q
W_K
W_V
W_O
W_gate
W_up
W_down
```

Sin bias, sin norm scale, sin depth embedding, sin tablas dependientes de K.

### 4. Seeds oficiales

Tres master seeds nuevos, congelados ahora:

```text
20261011
20261012
20261013
```

Antes de la corrida oficial quedan held-out.

Para cada master seed S:

```text
WEIGHT_SEED = S
INPUT_SEED  = S + 1008
LOSS_W_SEED = S + 2008
TARGET_SEED = S + 3008
```

Pesos e inputs se generan en CPU mediante el procedimiento V2-0 y se copian exactamente a CUDA.

Seed permitido para QA/calibración:

```text
20260930 only
```

Nunca participa en el veredicto V2-2B.

### 5. Batch regimes

Para separar conformidad de capacidad:

```text
CONFORMANCE_BATCH = 8
TRAINABILITY_BATCH = 128
```

#### B=8

Usado en:

```text
D1 correctness
D2 R4/U4 parity
D3 gradient-sharing
D6 K-flex
```

#### B=128

Usado únicamente en:

```text
D7 optimizer smoke
D8 local-capacity validation
```

No hay selección adaptativa de batch después de observar resultados.

### 6. D1 — CPU↔CUDA correctness

Para cada uno de los tres seeds:

```text
d512
m8
B8
K={1,4}
```

Referencia:

```text
V2-0 ContractualCoreBlock CPU FP32
same weights
same input
same operation order
```

K4 compara los cuatro traces y final.

Gate elemento a elemento:

```text
[
|y_{CUDA}-y_{CPU}|
\le
10^{-5}
+
10^{-4}\max(|y_{CPU}|,10^{-6})
]

y:

\frac{|y_{CUDA}-y_{CPU}|2}
{\max(|y{CPU}|_2,10^{-6})}
\le10^{-5}
]
```

Todos los traces/final de los tres seeds deben pasar.

Persistir:

```text
max_abs
max_scaled_error
E_L2
```

### 7. D2 — R4/U4 exact structural parity

Para cada seed:

```text
B=8
m=8
K=4
```

U4 = cuatro clones bitwise iniciales de R4.

Requerir sin fallback:

```text
initial weights bitwise equal
round 1 torch.equal
round 2 torch.equal
round 3 torch.equal
round 4 torch.equal
final torch.equal
```

Cualquier diferencia:

```text
D2 FAIL
```

### 8. D3 — Gradient-sharing at d512

Usar el instrumento D3Q, no el comparator viejo de V2-2A.

Para cada seed y familia obtener:

```text
gR
gU0
gU1
gU2
gU3
```

Loss:

```text
[
L=\sum(y\odot w)
]
```

Construir:

```text
((gU_0.double()+gU_1.double())+gU_2.double())+gU_3.double()
]
```

y comparar contra gR.double().

### Gate A

```text
[
E_{L2}\le2\times10^{-7}
]
```

### Gate B

```text
[
E_{\infty}\le5\times10^{-7}
]
```

### Gate C

Sea:

```text
M = max(||gR64||inf, ||S64||inf)
M32 = float32(M)
```

Entonces:

```text
max_abs <= 8 * ULP(M32)
```

ULP usa exactamente la definición nextafter de D3Q.

Los tres gates deben pasar para:

```text
3 seeds × 7 families = 21/21 cells
```

El viejo elementwise max_rel se registra sólo como diagnóstico.

S_reverse == gR también es NON-GATE.

No es necesario repetir el CPU-FP64 oracle completo en V2-2B: D3Q ya lo estableció en 5 seeds held-out. D3 de V2-2B prueba transferencia a d512, no vuelve a validar la matemática desde cero.

### 9. D4 — Parameter/storage conformance

Exigir:

```text
R4 unique params = 4,194,304
U4 unique params = 16,777,216
```

```text
R4:
  one storage/family
```

```text
U4:
  four pairwise-disjoint storages/family
  all disjoint from R4
```

State-dict schema conforme a V2-0.

### 10. D5 — Iso-FLOP

Ledger analítico V2-0.

Para d512,m8:

```text
per-round B1 FLOPs = 67,239,936
K4 B1 FLOPs        = 268,959,744
K4 B8 FLOPs        = 2,151,677,952
K4 B128 FLOPs      = 34,426,847,232
```

Requerir:

```text
R4_forward_flops == U4_forward_flops
exact integer equality
```

y non-GEMM counts idénticos.

Wall time no forma parte del gate iso-FLOP.

### 11. D6 — K-flex d512

Sólo master seed:

```text
20261011
```

con:

```text
B=8
m=8
```

Forward:

```text
K={1,2,4,8,16}
```

Backward:

```text
K={1,4,8,16}
```

Backward loss:

```text
[
L=\sum(y\odot w)
]
```

Sin optimizer.

Para cada celda exigir finitud de:

```text
output
loss when applicable
input gradient
all seven parameter gradients
```

Además, corrección explícita del gap D6 anterior:

persistir por cada celda:

```text
SCHEMA_SHA256_before
SCHEMA_SHA256_after
```

```text
VALUE_SHA256_before
VALUE_SHA256_after
```

```text
parameter_count_before
parameter_count_after
```

Requerir igualdad exacta before/after.

### 12. D7 — d512 optimizer/trainability smoke

Para los tres master seeds, ejecutar R4 y U4 por separado:

```text
B=128
m=8
K=4
20 updates exactos
FP32
```

AdamW:

```text
lr = 3e-4
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0
scheduler = NONE
gradient clipping = NONE
```

Target fijo:

```text
target ~ N(0,1)
TARGET_SEED = S + 3008
```

Loss:

```text
[
L=\operatorname{mean}(y-target)^2
]
```

Definir:

```text
L0  = before any optimizer update
L20 = after update #20
```

Gate por variante y seed:

```text
L20 <= 0.99 * L0
```

y finitud en cada step de:

```text
output
loss
all gradients
all parameters
Adam exp_avg
Adam exp_avg_sq
```

Total:

```text
3 R4 smokes
3 U4 smokes
6/6 must PASS
```

No comparar R4 vs U4 por magnitud de loss. Sus trayectorias son descriptivas únicamente.

### 13. D8 — VRAM / local capacity / wall

Antes de cada celda CUDA independiente:

```text
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

No llamar empty_cache() dentro de forward/backward.

Sincronizar antes de leer memoria.

Gate:

```text
peak_memory_allocated <= 3 GiB
```

Registrar también:

```text
peak_memory_reserved
```

El batch B128 está preregistrado y no puede reducirse después de OOM dentro de la misma unidad.

### Clasificación si memoria falla

Si todos los gates científicos ejecutados pasan pero alguna celda requerida B128 no cabe:

```text
OMEGA_V2_2B_LOCAL_CAPACITY_HOLD
```

Esto no es refutación arquitectónica.

Si el proceso puede recuperarse del OOM, continuar las demás celdas preregistradas.

Total wall limit:

```text
<=30 min
```

### 14. D9 — continuation

Ejecutar todos los gates/celdas técnicamente posibles.

Un FAIL lógico no detiene los demás.

Hard-stop sólo:

```text
CUDA/runtime crash preventing continuation
structural corruption
unrecoverable OOM
non-finite state preventing next operation
```

Sin cambio de:

```text
seeds
batches
LR
thresholds
dtype
K
m
```

y sin reruns bajo el mismo official ID después de cruzar la frontera científica.

### 15. QA antes de held-out seeds

Permitido únicamente con calibration seed 20260930.

Requerido:

```text
unit tests PASS
static undefined-name audit = 0
full-control-flow mocked dry-run PASS
pre-CUDA dry-run PASS
source seal verification PASS
```

Se permite un CUDA harness-smoke con 20260930, marcado:

```text
CALIBRATION_QA_ONLY
V2_2B_verdict = null
```

No se permite modificar umbrales científicos después de ese smoke.

### 16. Source seal

Antes de ejecutar los tres seeds oficiales registrar:

```text
spec SHA256
all Python/source hashes
V2-0 core/variants/ledger hashes
D3Q spec/result references
PyTorch version
CUDA runtime
driver
GPU name
compute capability
all seeds
all thresholds
batch regimes
FLOP ledger constants
```

Working tree de la unidad limpio.

### 17. Resultado terminal

PASS

```text
OMEGA_V2_2B_LOCAL_PILOT_PASS
```

requiere:

```text
D1 PASS 3/3
D2 PASS 3/3
D3 PASS 21/21
D4 PASS
D5 PASS
D6 PASS
D7 PASS 6/6
D8 PASS
```

Scientific FAIL

```text
OMEGA_V2_2B_LOCAL_PILOT_FAIL
```

si falla cualquier gate científico no atribuible únicamente a capacidad local.

Capacity

```text
OMEGA_V2_2B_LOCAL_CAPACITY_HOLD
```

si el único bloqueo es VRAM local.

PASS no libera automáticamente RunPod ni T3.

### 18. Estado de ejecución

```text
V2-2B SPEC AUTHORING: GO
V2-2B HARNESS/PREFLIGHT QA: GO

V2-2B OFFICIAL HELD-OUT EXECUTION:
    HOLD until committed spec + source seal + QA report are reviewed

RunPod:
    HOLD

T3:
    HOLD

CONFORMANCE_HOLD:
    unchanged
```

## MD/325 Ambiguity Freeze

OMEGA MD/325 — V2-2B Ambiguity Freeze

Q1 — D1 correctness

ACEPTO.

Por cada uno de los 3 seeds oficiales, R4 solamente:

```text
d=512
m=8
B=8
K={1,4}
```

Dos gates simultáneos por salida/traza:

```text
[
|y_{CUDA}-y_{CPU}|
\le
10^{-5}+10^{-4}\max(|y_{CPU}|,10^{-6})
]

para todos los elementos, y:

\frac{|y_{CUDA}-y_{CPU}|2}
{\max(|y{CPU}|_2,10^{-6})}
\le10^{-5}
]
```

Cobertura:

```text
K1: final output
K4: round traces 1,2,3,4 + final output
```

U4 queda cubierto estructuralmente por D2.

Q2 — D3 d512 thresholds

ACEPTO opción (a).

Los umbrales D3Q permanecen sin recalibración:

```text
E_L2  <= 2e-7
E_inf <= 5e-7
max_abs <= 8 * ULP(M32)
```

No se adaptan a d512.

El smoke con 20260930 es CALIBRATION_QA_ONLY; no permite cambiar thresholds.

Si el smoke d512 incumple cualquier gate D3 o el gate estructural:

```text
V2_2B_CALIBRATION_QA_HOLD
official held-out execution = BLOCKED
```

Se reporta a Sol antes de tocar 20261011–20261013. No se modifica el instrumento dentro de V2-2B.

Si pasa aunque esté cerca del límite, tampoco se cambia nada.

Q3 — D7

ACEPTO íntegramente.

Para cada seed y variante:

```text
fixed x throughout all 20 updates
fixed target throughout all 20 updates
```

```text
x:
  CPU N(0,1)
  INPUT_SEED=S+1008
  [128,8,512]
```

```text
target:
  CPU N(0,1)
  TARGET_SEED=S+3008
  [128,8,512]
```

```text
LOSS_W_SEED:
  unused in D7
```

Loss:

```text
[
L=\operatorname{mean}(y-target)^2
]
```

Semántica exacta:

```text
L0:
  forward inicial
  ese mismo forward alimenta backward/update #1

updates:
  exactamente 20 optimizer.step()

L20:
  forward adicional después de update #20
  no grad
  mismo x/target
```

Total: 21 forwards, 20 backwards/updates.

R4 y U4 parten de los mismos valores iniciales; U4 contiene cuatro clones bitwise. Optimizers independientes.

Finitud se verifica tras cada step para output, loss, gradients, parámetros, exp_avg, exp_avg_sq.

Q4 — D6

ACEPTO.

Sólo R4 shared:

```text
seed = 20261011
B=8
m=8
```

```text
forward:
  K={1,2,4,8,16}
```

```text
backward:
  K={1,4,8,16}
```

```text
loss:
  sum(y*w)

w:
  LOSS_W_SEED = 20263019
```

Sin optimizer.

Por cada celda, persistir valores concretos:

```text
SCHEMA_SHA256_before
SCHEMA_SHA256_after

VALUE_SHA256_before
VALUE_SHA256_after

parameter_count_before
parameter_count_after
```

y exigir igualdad exacta de cada par.

VALUE_SHA256 usa la semántica V2-0 state_dict_sha256; parameter_count cuenta parámetros únicos.

Q5 — D8 cells y wall

ACEPTO, con una precisión obligatoria.

Celdas CUDA:

```text
D1: (seed,K)
D2: seed
D3: seed
D6: (forward/backward,K)
D7: (seed,variant)
```

D2 incluye R4+U4 forward del seed dentro de una celda.

D3 incluye R4+U4 forward/backward del seed dentro de una celda.

Antes de cada celda:

```text
destroy previous CUDA cell objects no longer needed
gc if necessary
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
```

La celda debe mantener vivos sólo los modelos/tensores CUDA necesarios para esa celda:

```text
D1: R4 only
D2: R4 + U4
D3: R4 + U4
D6: R4 only
D7: only the tested variant
```

Esto evita que una celda cargue VRAM de modelos ajenos.

Después de la celda:

```text
torch.cuda.synchronize()
read peak_memory_allocated
read peak_memory_reserved
```

El límite <=3 GiB allocated aplica a todas las celdas CUDA.

Wall time

Confirmado:

```text
official_wall_start:
  entrada al official runner después de parsear CLI/GO,
  antes de source/environment verification

official_wall_end:
  resultado + report + manifests/hashes oficiales completamente escritos
```

Incluye:

```text
seal verification
CPU references
CUDA work
artifact persistence
hash verification
```

Gate: <=30 min.

Q6 — terminal classification

ACEPTO.

Prioridad:

```text
Si existe cualquier scientific FAIL
OMEGA_V2_2B_LOCAL_PILOT_FAIL
```

aunque también haya VRAM/wall issues.

Registrar aparte:

```text
capacity_issue = true/false
capacity_reason = ...
```

Si todos los gates científicos ejecutables pasan y el único problema es recurso local

Incluye:

```text
peak allocated >3 GiB
recoverable/unrecoverable local OOM
wall >30 min
```

Resultado:

```text
OMEGA_V2_2B_LOCAL_CAPACITY_HOLD
```

con:

```text
capacity_reason ∈ {VRAM_BUDGET, OOM, WALL_TIME}
```

PASS

Sólo si no hay scientific FAIL ni capacity hold.

Q7 — FLOP constants

ACEPTO.

Los valores MD/324 son cross-check preregistrado:

```text
67,239,936
268,959,744
2,151,677,952
34,426,847,232
```

El harness debe recomputarlos usando el ledger V2-0.

Si cualquier entero difiere:

```text
FLOP_LEDGER_PRESEAL_HOLD
```

No sellar spec/source como listo para ejecución y reportar a Sol.

No elegir retrospectivamente uno de los dos valores.

Q8 — artifact contract

ACEPTO, ampliado así.

Execution unit

```text
one official process
three held-out master seeds
one official result slot
```

D1

Por seed persistir un bundle con:

```text
CPU K1 output
CUDA K1 output
CPU K4 traces 1..4 + final
CUDA K4 traces 1..4 + final
```

FP32 exactos + raw-tensor SHA256 + métricas.

D2

Persistir:

```text
initial clone hashes
trace equality booleans
trace raw hashes
final raw hashes
```

No hace falta duplicar tensors ya presentes en D1 si el manifest referencia exactamente el mismo artefacto.

D3

Por seed:

```text
gR
gU0
gU1
gU2
gU3
```

FP32 crudos vía .pt/LFS y raw-tensor hashes.

Persistir A/B/C, M32, ULP, old-max-rel diagnostic y decisiones.

D6

Los seis valores before/after requeridos por Q4 para cada celda.

D7

Por (seed,variant):

```text
L0
per-step loss/finiteness records
L20
final parameter canonical hash
final optimizer-state canonical hash
```

No es obligatorio persistir el optimizer state completo en .pt.

D8

Por celda:

```text
peak_allocated
peak_reserved
wall_seconds
status
```

Required global artifacts

```text
OMEGA_V2_2B_SPEC.md
SOURCE_SEAL.json
QA_REPORT.json
OFFICIAL_RESULT.json
OMEGA_V2_2B_REPORT.md
OMEGA_V2_2B_REPORT.md.sha256
OMEGA_V2_2B_CONFORMANCE_BLOCK.md
artifact_hashes.json
artifact_hashes_verified.json
```

Slot/log doctrine

El official result slot no se crea antes de cruzar la frontera científica.

Command/stdout/stderr viven fuera del slot.

Si hay abort pre-frontera:

```text
PRE_SCIENTIFIC_OPERATIONAL_ABORT
```

y sus logs se sellan en un directorio de incidentes independiente.

Los logs de la ejecución válida se hashean y se referencian desde el manifest/conformance block aunque permanezcan fuera del slot.

Archivos .pt grandes: Git LFS permitido/requerido según tamaño; los SHA científicos son sobre los bytes locales reales, no sobre el pointer LFS.

Q9 — IDs / attempts

ACEPTO.

package:

```text
campaign/omega_v2_2b_d512_local_pilot
```

official ID:

```text
OMEGA-V2-2B-LOCAL-PILOT-01
```

Una sola ejecución científica oficial.

Frontera de consumo:

```text
first held-out CUDA numerical cell
OR
first persistence/exposure of held-out scientific data
whichever occurs first
```

Un abort operacional previo no consume el intento, siempre que:

```text
0 held-out CUDA scientific cells
0 held-out scientific metrics exposed/persisted
```

Después de cruzar la frontera:

```text
no rerun under OMEGA-V2-2B-LOCAL-PILOT-01
```

Adición obligatoria — QA launch rule

Antes de tocar 20261011–20261013 deben cumplirse conjuntamente:

```text
spec committed
source seal committed
working tree clean

unit tests PASS
static unresolved-name audit = 0
full mocked control-flow dry-run PASS
pre-CUDA dry-run PASS
environment validation PASS
FLOP cross-check PASS
```

20260930 calibration smoke:

```text
COMPLETE
structural gates PASS
D3 A/B/C within frozen thresholds
```

Si el calibration smoke no satisface lo anterior:

```text
V2_2B_CALIBRATION_QA_HOLD
```

No ejecutar seeds oficiales.

MD/325 disposition

```text
Q1 = ACEPTO
Q2 = ACEPTO option (a), thresholds immutable
Q3 = ACEPTO
Q4 = ACEPTO
Q5 = ACEPTO + per-cell live-object isolation
Q6 = ACEPTO
Q7 = ACEPTO
Q8 = ACEPTO + explicit artifact contract
Q9 = ACEPTO

V2-2B SPEC FREEZE:
    AUTHORIZED after incorporating MD/325

V2-2B QA/CALIBRATION:
    AUTHORIZED

V2-2B HELD-OUT OFFICIAL EXECUTION:
    HOLD pending spec/source/QA review by Sol

RunPod:
    HOLD

T3:
    HOLD

CONFORMANCE_HOLD:
    unchanged
```

```text
Add a single authoritative status summary
Resolve the D3 artifact contract ambiguity
```

## Implementation conventions (NO científica)

### Interpretaciones aceptadas y constantes aisladas

| Pregunta | Constantes/procedimiento aislado | Disposición |
|---|---|---|
| Q1 | `D1_ELEMENTWISE_ABS_TOL=1e-5`, `D1_ELEMENTWISE_REL_TOL=1e-4`, piso `1e-6`; `D1_NORMWISE_E_L2_LIMIT=1e-5`, piso `1e-6`; R4, K={1,4}; K4 traces 1..4 y final. | ACEPTO; las dos fórmulas literales y la resolución MD/325 se conservan. |
| Q2 | D3Q A `2e-7`, B `5e-7`, C `8*ULP(M32)`; inmutable. Smoke 20260930 = `CALIBRATION_QA_ONLY`; cualquier fallo D3/estructura produce `V2_2B_CALIBRATION_QA_HOLD` y held-out BLOCKED. | ACEPTO opción (a). |
| Q3 | `D7_*`: x/target fijos en 20 updates; L0 inicial reusado en update 1; L20 forward extra no-grad. AdamW y finitud por step incluyen moments. | ACEPTO íntegramente. |
| Q4 | `D6_*`: R4 only, seed `D6_MASTER_SEED`, B8/m8; K sets y `LOSS_W_SEED=S+2008`; hashes/counts before/after exactos. | ACEPTO. |
| Q5 | `D8_*`: celdas aisladas, clear/reset/sync, 3 GiB allocated, wall 1800 s. | ACEPTO + aislamiento de objetos vivos. |
| Q6 | `TERMINAL_*`: FAIL tiene prioridad sobre capacity; registrar `capacity_issue` y `capacity_reason`. | ACEPTO. |
| Q7 | `EXPECTED_D512_FLOPS` y `ledger.py`; discrepancia produce `FLOP_LEDGER_PRESEAL_HOLD`, sin sellar ni elegir. | ACEPTO; cross-check exacto. |
| Q8 | `ARTIFACT_CONTRACT`; D1–D8, globales, manifest/hashes, slot perezoso y logs externos. | ACEPTO + contrato explícito. |
| Q9 | `OFFICIAL_ID=OMEGA-V2-2B-LOCAL-PILOT-01`; una ejecución oficial. | ACEPTO. |

### D1 fórmula de trabajo aislada (Q1)

La sección B conserva literalmente la fórmula truncada de MD/324. El harness implementa provisionalmente los dos gates aceptados por MD/325:

```text
elementwise: abs(yCUDA-yCPU) <= D1_ELEMENTWISE_ABS_TOL + D1_ELEMENTWISE_REL_TOL*max(abs(yCPU),1e-6)
normwise: ||yCUDA-yCPU||_2 / max(||yCPU||_2,1e-6) <= D1_NORMWISE_E_L2_LIMIT
```

### FLOP ledger D512

El cross-check invoca el ledger analítico V2-0 y compara enteros exactos contra:

| Cell | FLOPs esperados |
|---|---:|
| per-round B1 | 67,239,936 |
| K4 B1 | 268,959,744 |
| K4 B8 | 2,151,677,952 |
| K4 B128 | 34,426,847,232 |

Parameter counts esperados: R4=4,194,304, U4=16,777,216; siete familias. Una discrepancia bloquea el pre-seal sin elegir retrospectivamente.

### Contrato de slot, logs y consumo

- El runner crea perezosamente el official result slot justo al cruzar la frontera científica: inmediatamente antes de la primera celda CUDA held-out, con el marcador persistido antes de cualquier lanzamiento, o al persistir/exponer dato held-out si eso ocurre primero.
- Pre-checks de source seal, entorno y planes no crean el official slot ni exponen datos científicos held-out.
- Logs oficiales command/stdout/stderr viven fuera del slot y sus hashes quedan referenciados en manifest y conformance block.
- Un abort pre-frontera produce `PRE_SCIENTIFIC_OPERATIONAL_ABORT`; sus logs van a un directorio de incidentes independiente. No consume el official ID.
- Tras la frontera no se reintenta el official ID. D9 continúa tras FAIL lógico; sólo hard-stop técnico termina la secuencia.

### QA, smoke, sellado y ejecución

- `make_seed_plan` admite en QA/smoke únicamente 20260930; `official_seed_plans()` sólo se materializa después de GO, source seal y validación de entorno.
- El calibration smoke usa sólo 20260930, produce `CALIBRATION_QA_ONLY` y `V2_2B_verdict=null`; también ejecuta D1/D2/D3/D5/D6/D7/D8 como harness-check sin veredicto.
- `SOURCE_SEAL.json` se rechaza si falta el smoke verificado o si sus snapshots de spec/source están stale.
- D3 `.pt` persiste gR/gU0..gU3 FP32 crudos por seed con raw-tensor hashes como en D3Q.
- Git LFS afecta transporte; los hashes científicos se calculan sobre bytes locales reales.
- El official process usa el ID fijo `OMEGA-V2-2B-LOCAL-PILOT-01`; no se cambia seeds, batches, LR, thresholds, dtype, K o m durante la unidad.

### D1 formula aislada (Q1)

La sección B conserva literalmente la fórmula truncada de MD/324. Implementación propuesta aceptada por MD/325, en `D1_*`:

```text
elementwise: abs(yCUDA-yCPU) <= 1e-5 + 1e-4*max(abs(yCPU),1e-6)
normwise: ||yCUDA-yCPU||_2 / max(||yCPU||_2,1e-6) <= 1e-5
```

### D7 detalles aislados (Q3)

- `x` fijo `[128,8,512]`, CPU N(0,1), semilla `INPUT_SEED=S+1008`.
- `target` fijo `[128,8,512]`, CPU N(0,1), semilla `TARGET_SEED=S+3008`; `LOSS_W_SEED` no se usa en D7.
- `L0` es el forward inicial reusado por backward/update #1. Ejecutar 20 `optimizer.step()` exactos; `L20` es un forward extra no-grad post-update #20, con el mismo x/target.
- R4 y U4 empiezan con los mismos valores iniciales; U4 son cuatro clones bitwise; optimizers separados.
- La finitud por step incluye output, loss, grads, params, `exp_avg` y `exp_avg_sq`.

### D6 detalles aislados (Q4)

`D6_MASTER_SEED`, B8/m8, R4 only, forward K={1,2,4,8,16}, backward K={1,4,8,16}, loss `sum(y*w)`, `LOSS_W_SEED=S+2008`, sin optimizer. Persistir hashes antes/después de schema/value y parameter counts únicos por celda.

### D8 celdas y wall aislados (Q5)

| Gate/cell | Celdas independientes |
|---|---|
| D1 | (seed,K) |
| D2 | seed, R4+U4 forward en una celda |
| D3 | seed, R4+U4 forward/backward en una celda |
| D6 | (forward/backward,K) |
| D7 | (seed,variant) |

Antes de cada celda se destruyen objetos CUDA previos no necesarios, GC si aplica, empty_cache, reset peak stats y synchronize. Se mantienen vivos sólo los objetos de la celda indicada. Después se sincroniza y se registra allocated/reserved/wall/status. `official_wall_start` es tras CLI/GO parse y antes de seal/env checks; `official_wall_end` es tras resultado/report/manifests/hash verification.

### Status summary (authoritative)

| Elemento | Estado |
|---|---|
| V2-2B SPEC FREEZE | AUTHORIZED after incorporating MD/325 |
| V2-2B QA/CALIBRATION | AUTHORIZED |
| V2-2B HELD-OUT OFFICIAL EXECUTION | HOLD pending spec/source/QA review by Sol |
| RunPod | HOLD |
| T3 | HOLD |
| CONFORMANCE_HOLD | unchanged |

### Track CPU diagnostic

La pista V2-1d/candidate_02 es paralela e independiente. Durante la ventana official V2-2B no corre benchmark/timing CPU, cache sweep ni full-block CPU candidate work; CPU se usa únicamente como referencia V2-0 y para generar pesos/datos.

# OMEGA-V2-2A-D3Q — SPEC FROZEN BY MD/323

## 2. OMEGA-V2-2A-D3Q — SPEC FREEZE

### Propósito

Unidad científica independiente para probar, sobre seeds held-out, que:

```text
\nabla W_R \approx \sum_{r=0}^{3}\nabla W_{U_r}
```

con un instrumento numéricamente bien condicionado.

### Configuración congelada

```text
d = 256
m = 8
B = 8
K = 4

FP32 CUDA
deterministic algorithms = ON
TF32 = OFF
AMP = OFF
optimizer = NONE

loss = sum(y * w)
```

Arquitectura y ecuaciones exactamente V2-0.

## 3. Seeds held-out

Los siguientes cinco seeds permanecen intocados hasta que spec + source seal estén cerrados:

```text
MASTER_SEED:
20261001
20261002
20261003
20261004
20261005
```

Para cada master seed S:

```text
WEIGHT_SEED = S
INPUT_SEED  = S + 1008
LOSS_W_SEED = S + 2008
```

Es decir, se conserva la misma derivación usada en V2-2A para m=8.

20260930 queda permanentemente:

```text
CALIBRATION_ONLY
FORBIDDEN_FROM_D3Q_VERDICT
```

## 4. Gate estructural previo

Por cada seed held-out, antes de evaluar gradients:

```text
U4 initial weights = four bitwise clones of R4

round_trace_1 R4 == U4 via torch.equal
round_trace_2 R4 == U4
round_trace_3 R4 == U4
round_trace_4 R4 == U4
final_output R4 == U4
```

No fallback de tolerancia.

Cualquier desigualdad bitwise:

```text
SEED_FAIL_STRUCTURAL
```

## 5. Referencia CUDA de gradient-sharing

Capturar por familia:

```text
gR
gU0
gU1
gU2
gU3
```

Las siete familias siguen siendo:

```text
W_Q
W_K
W_V
W_O
W_gate
W_up
W_down
```

La suma científica primaria será independiente del orden FP32:

```text
((gU_0^{64}+gU_1^{64})+gU_2^{64})+gU_3^{64}
```

donde cada gUi se convierte a FP64 después de que autograd CUDA FP32 produzca el tensor.

Comparar:

```text
g_R^{64} = double(g_R)
```

contra (S_{64}).

## 6. Umbrales D3Q congelados

Para cada familia y cada seed:

### Gate A — normwise L2

```text
E_L2 = ||gR64 - S64||_2 / max(||gR64||_2, ||S64||_2, 10^-6)
E_L2 <= 2e-7
```

### Gate B — normwise Linf

```text
E_inf = ||gR64 - S64||_inf / max(||gR64||_inf, ||S64||_inf, 10^-12)
E_inf <= 5e-7
```

Estos márgenes son aproximadamente 3–4× el máximo observado en la única calibración permitida.

### Gate C — max-absolute scale-aware

No uso un 5e-5 absoluto fijo. Congelo un gate ULP escalado.

Definir:

```text
M = max(||gR64||inf, ||S64||inf)
M32 = float32(M)
```

y ULP(M32) exactamente con la definición nextafter ya congelada en D3-DIAGNOSTIC.

Gate:

```text
max_abs = ||gR64 - S64||inf
max_abs <= 8 * ULP(M32)
```

8 ULP queda congelado: el máximo calibrado fue 2 ULP bajo el comparador FP32 histórico, dando margen 4× sin convertirlo en un umbral absoluto dependiente de escala.

## 7. CPU FP64 oracle — gate independiente

Para cada seed, copiar exactamente a FP64:

```text
initial FP32 weights
x FP32
w FP32
```

No reinicializar en FP64.

Ejecutar R4 y U4 completos en CPU FP64.

Por familia:

```text
E_L2_FP64 <= 1e-15
E_inf_FP64 <= 1e-14
```

max_abs_FP64 se registra, pero no tiene threshold separado.

Todos los valores deben ser finitos.

## 8. Métricas que quedan sólo como diagnóstico

Persistir, pero nunca usar para PASS/FAIL:

```text
old elementwise max_rel
S_forward
S_reverse
S_pairwise
S_stack
S_reverse torch.equal(gR)
max_rel argmax/index
local ULP diagnostics
```

Especialmente:

```text
S_reverse bitwise equality = NON-GATE
```

No se permite convertirlo después en evidencia necesaria ni suficiente.

## 9. Regla multi-seed

```text
5 seeds held-out
7 families por seed
ALL required gates must PASS
```

Por tanto:

```text
D3Q PASS = 5/5 seeds PASS
```

No mayoría, no promedio, no exclusión de outliers.

Si un seed falla un gate:

```text
that seed = FAIL
D3Q final = FAIL
```

Ejecutar los cinco seeds aunque uno falle, salvo hard-stop técnico.

Sin reintentos por seed.

## 10. QA y frontera de ejecución

Antes de tocar cualquier seed held-out se permite QA usando únicamente seed 20260930.

Requerido:

```text
unit tests PASS
static wiring audit PASS
control-flow dry-run PASS
calibration-seed harness smoke PASS
spec sealed
source sealed
working tree clean
```

Sólo entonces se autoriza una ejecución oficial D3Q que contiene los cinco seeds.

La ejecución queda consumida al ocurrir primero:

```text
first held-out CUDA numerical cell
OR
persistence/exposure of a held-out scientific result
```

Si hay fallo operacional antes de esa frontera, se aplica la doctrina pre-CUDA ya establecida. Después de cruzarla, no hay rerun bajo el mismo ID.

## 11. Persistencia obligatoria

Por cada seed guardar al menos:

```text
raw gR
raw gU0..gU3
primary S64 metrics
structural trace equality
CPU FP64 oracle metrics
old max_rel diagnostic
all thresholds and boolean decisions
```

Los raw gradient bundles deben quedar hasheados para permitir recomputación independiente.

Resultado final:

```text
OMEGA_V2_2A_D3Q_PASS
or
OMEGA_V2_2A_D3Q_FAIL
```

## 12. Consecuencia sobre V2-2B

Queda congelada desde ahora esta regla:

Si:

```text
V2-2A-r1:
    D1 PASS
    D2 PASS
    D4 PASS
    D5 PASS
    D6 PASS_WITH_REPORTING_LIMITATION
    D7 PASS
    D8 PASS
```

AND

```text
D3Q:
    5/5 PASS
```

entonces NO se repite V2-2A.

Se considera que D3Q aporta evidencia científica independiente que resuelve específicamente la deficiencia del instrumento D3, mientras:

```text
V2-2A-r1 = permanece FAIL histórico
```

y se autoriza:

```text
OMEGA-V2-2B d512 local contractual pilot:
    SPEC/PREFLIGHT GO
```

La ejecución de V2-2B seguirá requiriendo su spec y source seal propios; esto no libera RunPod.

El CONFORMANCE_BLOCK futuro debe conservar simultáneamente:

```text
V2_2A_r1:
  terminal: FAIL
  reason: D3_elementwise_max_rel

D3Q:
  evidence_class: INDEPENDENT_HELDOUT_VALIDATION

D6_reporting_limitation:
  before_after_hashes_computed: true
  before_after_hashes_persisted_per_cell: false
  gate_result: PASS
  rerun_required: false
```

## Implementation conventions (NO científica)

### (a) Definición exacta de ULP

Se usa la definición congelada en el spec D3-DIAGNOSTIC, evaluada en FP32 sobre `a = abs(x)`:

```text
if a == 0:
    ULP(x) = nextafter(float32(0), +infinity) - float32(0)
else if a < largest_finite_FP32:
    ULP(x) = nextafter(a, +infinity) - a
else:
    ULP(x) = a - nextafter(a, 0)
```

Para `M32`, `M` se convierte con la conversión torch estándar FP64→FP32, round-to-nearest-even: `torch.as_tensor(M, dtype=torch.float64).to(torch.float32)`. Para valores no finitos, registrar el valor como no finito y los campos ULP derivados como `null`.

### (b) Fórmulas exactas de métricas en FP64

En todas las fórmulas primarias `delta = gR64 - S64`, con cálculo y reducción en FP64:

```text
E_L2 = ||delta||_2 / max(||gR64||_2, ||S64||_2, 1e-6)
E_inf = ||delta||_inf / max(||gR64||_inf, ||S64||_inf, 1e-12)
max_abs = max(abs(delta))
max_rel_old = max(abs(delta) / max(abs(gR64), abs(S64), 1e-6)) elementwise
```

El oracle CPU FP64 usa exactamente las mismas fórmulas y floors para `E_L2_FP64`, `E_inf_FP64` y `max_abs_FP64` de `gR64_cpu` frente a la suma FP64 left-associated de `gU0_64..gU3_64`. `max_abs_FP64` sólo se registra.

### (c) Tabla de derivación de seeds

| Uso | MASTER_SEED | WEIGHT_SEED | INPUT_SEED | LOSS_W_SEED | Uso de resultado |
|---|---:|---:|---:|---:|---|
| Calibration / QA / smoke | 20260930 | 20260930 | 20261938 | 20262938 | Sólo control de harness; prohibido del veredicto D3Q |
| Held-out oficial, cada S | S ∈ {20261001, 20261002, 20261003, 20261004, 20261005} | S | S + 1008 | S + 2008 | Únicamente ejecución oficial D3Q |

### (d) Gates científicos y criterios operacionales

| Tipo | Criterios | Efecto |
|---|---|---|
| Estructural / operacional | Clones bitwise independientes, igualdad `torch.equal` de cuatro trazas y salida final, finitud, persistencia íntegra, hashes verificables | Requisito operacional/estructural; no modifica umbrales científicos |
| Científico D3Q | Gate A, Gate B, Gate C y los límites `E_L2_FP64`, `E_inf_FP64` del oracle | PASS/FAIL científico por familia y seed |
| Diagnóstico NON-GATE | old `max_rel`, sumas FP32 alternativas, `S_reverse torch.equal(gR)`, índices y ULP locales, `max_abs_FP64` | Se persiste; nunca decide PASS/FAIL |

La finitud es un prerrequisito operacional para evaluar los gates; la exigencia `Todos los valores deben ser finitos` de la sección 7 se conserva además como condición requerida del oracle.

### (e) Persistencia de lanzamiento

El slot de resultados lo crea el RUNNER. `command.txt`, `stdout.log` y `stderr.log` del lanzamiento se guardan en un directorio de logs FUERA del slot oficial. No precrear el slot para colocar allí logs.

### (f) Doctrina pre-CUDA

Fallo operacional antes de la primera celda CUDA held-out = aborto sin consumo, siempre que tampoco se haya persistido/expuesto un resultado científico held-out. Después de cruzar la frontera no hay rerun bajo el mismo ID.

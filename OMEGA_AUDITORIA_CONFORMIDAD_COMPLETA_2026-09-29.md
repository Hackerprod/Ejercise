# OMEGA — Auditoría completa de conformidad arquitectónica y plan de corrección V2

**Fecha:** 2026-09-29  
**Autor de auditoría:** GPT-5.6 Sol  
**Repositorio:** `Hackerprod/Ejercise`  
**Snapshot auditado:** `main` observado en `375149498d0a888f7d0e720e77567be8b6cbcb13`  
**Solicitud formal:** `AUDIT_REQUEST_DESVIACIONES.md`  
**Contrato arquitectónico primario:** `Conversacion.md` (blob Git `7027e2ac9d1ba89db08dda73c81e244f3b9b19db`)  
**Protocolo lingüístico pre-T3:** `Conversacion LN.md` (blob Git `fc750a2ae9fb7d3933c54fb91f09ce5568d035ec`)  

---

## 0. Veredicto ejecutivo

La campaña OMEGA **no ha refutado la arquitectura original**, pero tampoco la ha validado todavía. La mayor parte de la evidencia lingüística reciente corresponde a un **surrogate de scoping** (`d=128, m=8`) cuya forma y escala se alejaron del núcleo concreto fijado en `Conversacion.md` (`d=512` como candidato principal y `d=640` como frontera física del Ryzen original). Ese surrogate produjo resultados científicos reales y útiles —recurrencia causal, efecto de K, cobertura de datos, backend CPU—, pero varias conclusiones adquirieron una autoridad arquitectónica mayor que la permitida por su alcance.

El hallazgo principal de esta auditoría es doble:

1. **La línea física original sí fue parcialmente ejecutada.** T0-M tiene evidencia fuerte de matrixización y T0-R tiene mediciones A/B/C; por tanto es falso decir que “T0 nunca ocurrió”. Sin embargo, el programa T0 completo **no quedó cerrado** y el control causal de residencia de T0-R no quedó limpio.
2. **La línea lingüística T2 que dominó la campaña no es una instancia conforme del núcleo final.** No sólo se redujo `d=512→128`; el bloque actual usa una forma aproximada `12d²+10d`, mientras la tabla contractual `d512≈4.19M` / `d640≈6.55M` deriva de un bloque `16d²`. Cambiar simplemente `dimension=512` en el modelo actual produciría otro surrogate, no el núcleo contractual.

En consecuencia se establece un **OMEGA_CONFORMANCE_BLOCK** obligatorio. Hasta que se satisfaga:

- no se autoriza T3;
- no se interpreta un PASS/FAIL lingüístico de `d128` como veredicto sobre OMEGA conforme;
- no se llama “núcleo de 7M” al modelo actual;
- no se trata `K4` como profundidad operativa final;
- no se transfiere `be376...` ni `QUALIFIED_AT_2000` a `d512/d640` sin revalidación.

La corrección propuesta es una **V2 conforme**, primero en `d=512`, después `d=640`, con `m={4,8,16}`, core recurrente realmente compartido, K definido de forma que no dependa de tablas finitas por ronda, y una comparación obligatoria **shared R4 vs untied U4 a igual FLOPs**. El núcleo físico se re-gatea en la i7 actual antes de gastar GPU. El entrenamiento lingüístico se hace en GPU. El protocolo `MINIMUM-LANGUAGE-COMPETENCE` se reutiliza como instrumento, pero el gate arquitectónico se aplica a la V2 conforme.

---

# 1. Corrección de una confusión importante: 7M no era el mínimo contractual de pesos recurrentes

Una parte de la conversación reciente confundió parámetros con bytes.

`Conversacion.md` alrededor de las líneas **1284–1300** recomienda, en un ejemplo físico anterior, comenzar con **7–9 MB reales** de working set en vez de ocupar el 100% de la L2 nominal. Eso es un presupuesto de **bytes**, no una exigencia de “7M parámetros recurrentes”.

El contrato concreto posterior para el Ryzen AI 5 330 aparece en `Conversacion.md` **2585–2630**:

| d | pesos únicos del bloque contractual | Q4 + escalas | régimen esperado original |
|---:|---:|---:|---|
| 384 | 2.36M | 1.27 MiB | L2 holgada |
| **512** | **4.19M** | **2.25 MiB** | **candidato principal L2** |
| **640** | **6.55M** | **3.52 MiB** | **frontera L2** |
| 768 | 9.44M | 5.06 MiB | L3 |
| 1024 | 16.78M | 9.00 MiB | DRAM |

Por tanto la afirmación corregida es:

> **El núcleo contractual inicial es de ~4.19M pesos recurrentes únicos en d512; d640 (~6.55M) es el punto de frontera física que debía probarse. No existe un contrato de “mínimo 7M pesos recurrentes”.**

Además, esa etiqueta “frontera L2” pertenecía al hardware Ryzen original. La campaña ahora corre en un **i7-13700F**; las dimensiones `d512/d640` siguen siendo puntos contractuales, pero su residencia real debe **volver a medirse**. No se asume que la rodilla de caché sea idéntica.

---

# 2. Hallazgo nuevo de esta auditoría: la forma del bloque también se desvió

El contrato de `Conversacion.md` **2585–2630** calcula el bloque como:

\[
P_{core}=4d^2+12d^2=16d^2
\]

El primer término corresponde a Q/K/V/out y el segundo a la MLP contractual tipo SwiGLU/expansión descrita allí.

El código lingüístico actual (`omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py`) implementa, por bloque:

- QKV: `3d²`
- out: `d²`
- FC1: `4d²`
- FC2: `4d²`
- biases + norm: `10d`

es decir:

\[
P_{FastWorkspaceUpdateBlock}=12d^2+10d.
\]

Para `d=128`:

\[
12(128)^2+10(128)=197,888
\]

que coincide con el código y con la corrección del usuario.

### Si sólo escaláramos el código actual

| d | bloque actual escalado `12d²+10d` | bloque contractual `16d²` | diferencia estructural |
|---:|---:|---:|---:|
| 128 | 197,888 | 262,144 | -24.5% aprox. |
| 512 | 3,150,848 | 4,194,304 | -24.9% aprox. |
| 640 | 4,921,600 | 6,553,600 | -24.9% aprox. |

**Conclusión:** `dimension=512` sobre el código actual **NO** produce el núcleo `d512≈4.19M` de `Conversacion.md`.

Esto crea una decisión de conformidad que debe ser explícita antes de V2:

- **Ruta V2-CONFORMANT (recomendada):** restaurar la forma de bloque que justifica `16d²` y medirla.
- **Ruta V2-AMENDED:** declarar formalmente que el contrato se modifica a la forma `12d²`, justificarlo y recalcular todo el presupuesto físico.

Este informe propone **V2-CONFORMANT** porque el usuario pidió volver a la línea original, no redefinirla silenciosamente otra vez.

---

# 3. Mapa contractual T0→T3

## 3.1 T0 — Gate físico

**Contrato — `Conversacion.md` 3078–3240 y 2688–2945**

T0 debía ocurrir antes del entrenamiento lingüístico y usar:

- bloque sintético/realista;
- A = pesos compartidos residentes;
- B = pesos distintos por profundidad;
- C = mismos pesos con expulsión artificial;
- d512 como candidato principal;
- barridos m/K;
- gate de residencia;
- gate de matrixización.

Criterio de residencia contractual (`Conversacion.md` 2770–2860):

\[
\rho_{resident}=c_A(K)/c_B(K)
\]

para d512, m pequeño, K8:

- `ρ ≤ 0.50` = gate mínimo;
- `ρ ≤ 0.25` = evidencia fuerte;
- `ρ≈1` = sin ventaja útil.

Criterio de matrixización:

- `MAC/s(m=8 o 16) / MAC/s(m=1) ≥ 1.5` mínimo;
- `≥2.0` fuerte.

### Lo ejecutado

**T0-M** sí se ejecutó. `t0m_phase3.summary.txt` registra `PASS_STRONG` y 4,240 procesos medidos. Los G8/G16 se encuentran ampliamente por encima del gate en numerosos puntos.

**T0-R** también se ejecutó: 240 filas válidas, A/B/C, afinidad física, AVX2 y un techo DRAM medido de 32.9295 GB/s.

### Problema de T0-R

El harness real construye:

```text
A/C: 1 bloque
B: depth bloques
```

De modo que B no es un control “no residente” limpio para todas las profundidades. En depth pequeño todavía puede vivir parcial o totalmente en caché; el sweep histórico no garantiza que el working set B exceda el nivel rápido en todos los puntos.

`B_over_measured_dram` es además **bandwidth efectivo derivado de MAC/s**, no un contador físico de tráfico DRAM.

### Re-clasificación

```text
T0-M_MATRIXIZATION:
    VALID / PASS_STRONG_IN_MEASURED_D512_SCOPE

T0-R_RESIDENCY:
    DEGRADED / SUBSTANTIAL_EVIDENCE_BUT_CAUSAL_CONTROL_NOT_CLEAN

BRIDGE-1:
    FAIL_RECORDED_AGAINST_HISTORICAL_REFERENCE
    reference validity insufficient for architecture-wide verdict

T0_END_TO_END:
    NOT_CLOSED
```

---

## 3.2 T1 — Trainability sintética

**Contrato — `Conversacion.md` 3078–3240**

T1 debía usar un modelo diminuto `d=64–128, m=4–8, K=2–4` para tareas de copiar, recuperar clave, componer hechos, actualizar slots, READ/THINK y evitar colapso.

Su función era únicamente:

> demostrar entrenabilidad del mecanismo, **no capacidad lingüística**.

### Re-clasificación

Los múltiples resultados toy/sintéticos siguen siendo evidencia válida en su alcance. Nada de esta auditoría los invalida.

```text
T1_SYNTHETIC_TRAINABILITY:
    VALID_WITH_SCOPE
    does_not_establish_language
    does_not_establish_d512_conformance
```

---

## 3.3 T2 — Destilación del núcleo sin banco

**Contrato — `Conversacion.md` 3078–3240**

T2 debía:

- usar maestro Transformer congelado;
- entrenar núcleo sin banco;
- combinar CE/KL y opcionalmente alineación de estados;
- currículo K2→K4→K6;
- correr `m=4,8,16` por separado;
- posteriormente scheduled sampling y TBPTT corto;
- entrenar lenguaje en GPU, no en la laptop CPU.

El contrato físico/arquitectónico inmediatamente anterior fija como primer subconjunto probable (`Conversacion.md` 2688–2770):

```text
d = 512
m ∈ {4,8,16}
K ∈ {4,6,8}
```

### Lo ejecutado

La campaña lingüística real convergió en el surrogate:

```text
d = 128
m = 8
K = 1/4/6 en campañas principales
```

con bloque actual de **197,888 pesos recurrentes**, no 4.19M.

### Desviaciones T2 principales

1. **d512→d128** sin puente experimental de escala.
2. **forma 16d²→12d²+10d**.
3. **m-curve no ejecutada lingüísticamente**.
4. **K quedó condicionado por `depth_embedding[K,D]` y `gate_logits[K,D]`**, dificultando la tesis de profundidad flexible.
5. **cabeza/vocab completo atado domina el ledger paramétrico**.
6. **selective head contractual no fue implementada**.
7. scheduled sampling nunca se volvió receta productiva.
8. TBPTT16 se probó y cerró DIRECTIONAL-STOP; eso no equivale al plan T2 completo de entrenamiento de estado.

---

## 3.4 T3 — Memoria externa

**Contrato — `Conversacion.md` 3078–3240**

T3 debía introducir:

- banco congelado creado offline;
- retriever pequeño;
- núcleo inicialmente congelado;
- query/top-k/escritura en slots;
- detección de recuperación insuficiente;
- posterior descongelado parcial.

Además `Conversacion.md` 2945–3075 exige comparación factorial:

| Arquitectura | sin banco | mismo banco |
|---|---|---|
| densa | D0 | D1 |
| recurrente | R0 | R1 |

para separar recurrencia, memoria y su interacción.

### Estado

T3 todavía no comenzó y debe permanecer en HOLD.

```text
T3:
    HOLD_BY_CONFORMANCE_AUDIT
```

---

# 4. Tabla de desviaciones arquitectónicas

| Tema | Contrato / líneas `Conversacion.md` | Ejecución real | Auditoría | Qué conclusiones afecta |
|---|---|---|---|---|
| Tamaño core | 2585–2630: d512=4.19M; d640=6.55M | d128; bloque 197,888 | **DESVIACIÓN MAYOR** | toda conclusión de capacidad/escala |
| Forma del core | 2585–2630: `16d²` | `12d²+10d` | **DESVIACIÓN MAYOR** | no basta escalar dimension |
| m/workspace | 2390–2485, 2688–2750: m barrido y Km constante | T0 físico sí; T2 lenguaje m=8 | **T2 INCOMPLETO** | no sabemos si workspace compensa saturación K |
| K/profundidad | 1770–1790: profundidad+halting; 2460s: K composición | K fijo por checkpoint + tablas por ronda | **DESVIACIÓN DE FLEXIBILIDAD** | TTR-A no prueba tesis completa |
| K inferencia | profundidad dinámica era objetivo conceptual | TAIL-REPEAT sobre tablas finitas | **ADAPTACIÓN EXPLORATORIA** | negativo válido sólo para policy V1 |
| L2 residencia | 2770–2945 gates A/B/C | T0-M fuerte; T0-R control imperfecto | **NO CERRADO** | claims CPU-native físicos |
| vocab/head | 1770–1790: selective head evita dominio del vocab | tied 50,257×128, full projection | **DESVIACIÓN** | bytes/token y ledger de parámetros |
| parámetros | 2945–3075 exige core/shell/memory/index separados | “7M model” usado informalmente | **CORREGIDO AHORA** | comparaciones de tamaño |
| datos | no fijaba old-513 como dogma | Coverage-C encontró mejor policy | **ADAPTACIÓN JUSTIFICADA** | nueva policy reusable |
| gate LN | faltaba gate absoluto pre-T3 | protocolo Sol–Opus creado | **CORRECCIÓN NECESARIA** | T3 sigue HOLD |

---

# 5. Re-clasificación de resultados recientes

Se usan estas etiquetas:

- **VALID**: sigue sosteniendo la misma afirmación.
- **VALID_WITH_SCOPE**: resultado real, pero sólo para surrogate/configuración ejecutada.
- **DEGRADED**: evidencia real cuya interpretación arquitectónica debe reducirse.
- **SUPERSEDED**: decisión operativa reemplazada.
- **NOT_TRANSFERABLE**: no puede trasladarse a V2 sin nueva prueba.

## 5.1 Backend `QUALIFIED_AT_2000`

Resultado real:

- native vs PyTorch equivalente en d128/K1/K4 a 2k;
- performance ~20%+ mejor bajo la pérdida canónica;
- `strict_trajectory_proximity=FAIL_RECORDED` preservado.

Reclasificación:

```text
BACKEND_QUALIFIED_AT_2000:
    VALID_WITH_SCOPE
    scope = current d128 FastWorkspaceUpdateBlock

TRANSFER_TO_D512_D640:
    NOT_TRANSFERABLE
```

Se reutilizan harness, ABI, técnicas de optimización, metodología y tests; no el binario `be376...` como backend V2 certificado.

---

## 5.2 `CAUSAL-STATE-USE-ESTABLISHED`

Reset/shuffle/gate-init demostraron que el estado transportado y la adaptación pequeña de gates son funcionalmente útiles en d128.

```text
STATE_CAUSAL_USE_D128:
    VALID

STATE_CAUSAL_USE_V2:
    NOT_YET_ESTABLISHED
```

Es evidencia de que la idea recurrente funciona en el surrogate, no de que `m=8/d512` sea óptimo.

---

## 5.3 K-CURVE-A / `SATURATION-BEYOND-K4`

Resultado a d128, document-balanced, update 2000:

- K1→K4: mejora establecida.
- K4→K6: IC incluye cero.

Esto sigue siendo correcto para ese estimando.

La curva histórica old-513 muestra además que el efecto K1-K4 cambia con el presupuesto y no es monótono.

Reclasificación:

```text
K4_GT_K1_AT_2K_D128:
    VALID

SATURATION_BEYOND_K4_AT_2K_D128:
    VALID_WITH_SCOPE

K4_AS_FINAL_OPERATING_DEPTH:
    SUPERSEDED / NOT_ESTABLISHED

K_EFFECT_STABLE_ACROSS_BUDGET:
    NOT_ESTABLISHED
```

La variación con update es **effect modification**, no una invalidación del IC fijado en update 2000.

---

## 5.4 Coverage A/B/B2/C

Coverage-C aisló cobertura de weighting documental y mostró beneficio en VALL token-weighted y document-macro.

```text
DOCUMENT_BALANCED_MULTICHUNK:
    VALID_DATA_POLICY_RESULT
```

Puede reutilizarse como receta de datos para V2, tras asegurar que manifests/tokenizer son equivalentes.

No valida arquitectura d128 ni d512; valida una política de exposición de datos.

---

## 5.5 TTR-A / `NO_TEST_TIME_SCALING_WITH_TAIL_REPEAT`

TAIL-REPEAT degradó K1/K4 y fue neutro sólo en K6→K8 antes de degradar.

La arquitectura V1 tenía:

```text
depth_embedding[K,D]
gate_logits[K,D]
```

por lo que más K requería una policy inventada para rondas nuevas.

Reclasificación:

```text
TAIL_REPEAT_V1:
    VALID_NEGATIVE_RESULT

ORIGINAL_ARBITRARY_K_THESIS:
    NOT_TESTED_CLEANLY
```

TTR-A no refuta un V2 diseñado para arbitrary-K por construcción.

---

## 5.6 Generación autorregresiva histórica

Old-513 @2k produjo loops severos.

```text
OLD513_FREE_GENERATION:
    VALID_STRONG_NEGATIVE_EVIDENCE
```

Pero no se trasladó al checkpoint Coverage-C actual.

El protocolo LN actual permanece válido como instrumento, pero por esta auditoría:

```text
D128_LN_RESULT_IF_EXECUTED:
    BASELINE_DIAGNOSTIC_ONLY
    NOT_ARCHITECTURE_CONFORMANCE_VERDICT
```

La **Fase B automática** del instrumento puede completarse; no se autoriza usar un PASS de d128 para liberar T3.

---

## 5.7 Fable / kernel nativo

El ciclo Fable demostró que es posible construir un backend CPU mejor que PyTorch para la configuración d128 medida.

Eso es un activo de ingeniería importante.

```text
FABLE_OPTIMIZATION_METHODS:
    REUSABLE

BE376_BINARY_AS_V2_BACKEND:
    NOT_TRANSFERABLE
```

---

# 6. Qué queda realmente demostrado hasta hoy

### Demostrado

1. Weight sharing/recurrent state puede entrenarse en escala toy y d128.
2. El estado persistente d128 se usa causalmente.
3. A d128/2k, K4 supera K1; K6 no demuestra ganancia adicional.
4. La cobertura document-balanced mejora generalización frente a old-513.
5. Tail-repeat V1 no produce test-time scaling.
6. El backend nativo d128 puede superar PyTorch en tiempo manteniendo calidad a 2k.
7. T0-M demuestra matrixización fuerte en su scope sintético d512.

### No demostrado

1. Que el núcleo contractual d512/d640 aprenda lenguaje.
2. Que el bloque contractual 16d² tenga las mismas conclusiones que el bloque Fast 12d².
3. Que shared d512 preserve calidad frente a untied a igual FLOPs.
4. Que más K en **un mismo V2 flexible** compense menor número de pesos.
5. Que m=8 sea óptimo lingüísticamente.
6. Que el núcleo V2 permanezca realmente residente en la i7 actual.
7. Que la arquitectura complete el gate mínimo de lenguaje.
8. Que una memoria externa beneficie específicamente a la recurrencia más que a un baseline denso con el mismo banco.

---

# 7. Plan corregido V2 — objetivo

## 7.1 Identidad V2

### V2-512 — candidato principal

```text
d = 512
m = 8 inicialmente
core único compartido
P_core contractual ≈ 4.19M pesos recurrentes
Q4+scales contractual ≈ 2.25 MiB
```

### V2-640 — frontera de anchura

```text
d = 640
m = 8 inicialmente
P_core contractual ≈ 6.55M
Q4+scales contractual ≈ 3.52 MiB
```

`d640` fue “frontera L2” en el hardware Ryzen original; en i7 debe medirse de nuevo.

## 7.2 Forma del bloque

V2-CONFORMANT debe implementar la forma que produce `16d²`.

No se permite:

```text
cambiar d=128→512 en FastWorkspaceUpdateBlock
+
seguir llamándolo núcleo contractual
```

sin una enmienda explícita al contrato.

## 7.3 Ledger de parámetros obligatorio

Cada build V2 reportará por separado:

```text
P_core
P_shell
P_retriever
P_memory_learned
B_memory_static
B_index
B_quantized_core_physical
```

El embedding/head nunca vuelve a sumarse al core para anunciar “X M de núcleo”.

---

# 8. V2 y K: restaurar la tesis de cómputo flexible

El diseño actual usa tablas de longitud K. Eso hace que el número de rondas sea parte del shape paramétrico.

V2 debe tener una ruta **K-flexible por construcción**.

## 8.1 V2-KFLEX primaria

Primera variante recomendada:

```text
un único bloque recurrente
un único gate vector compartido entre rondas
sin depth_embedding aprendido por índice de ronda
```

Cada vuelta aplica literalmente el mismo operador parametrizado.

Si se necesita señal de profundidad, sólo se añade en una unidad posterior una función extrapolable explícita; nunca una tabla finita que vuelva a hardcodear K silenciosamente.

## 8.2 Entrenamiento

Restaurar el currículo contractual:

```text
K2 → K4 → K6
```

seguido por una **fase variable-K** para estabilizar un mismo checkpoint:

```text
K_train sampled from {2,4,6,8}
```

misma loss principal; no crear un modelo diferente por K.

Esto es una extensión de conformidad necesaria para probar la tesis de cómputo dinámico/halting.

## 8.3 Barrido de inferencia

Sobre el mismo checkpoint:

```text
Kinfer ∈ {1,2,4,6,8,12,16}
```

- 1–8: dentro o cerca del soporte entrenado.
- 12/16: extrapolación real de compute.

### Métrica primaria

\[
\Delta_K=NLL(K_{ref})-NLL(K_{infer})
\]

positivo = compute adicional mejora.

### Test-time-compute scaling SUPPORT

Predeclarar:

```text
algún K > max(K_train):
    mean ΔK >= 0.02 nats
    AND IC90% lower bound > 0
    AND automatic degeneration no empeora materialmente
```

Si ningún K extrapolado cumple, la tesis de **test-time compute scaling** no queda demostrada para ese training.

Si todos los K extrapolados degradan con IC completo, se clasifica:

```text
TEST_TIME_COMPUTE_SCALING_NOT_SUPPORTED_FOR_V2_RECIPE
```

sin destruir las demás partes de OMEGA.

## 8.4 Halting

Sólo se diseña halting una vez que una misma V2 muestre una curva útil/saturante con K.

Halting no puede usarse para ocultar una curva donde más iteraciones degradan sistemáticamente.

---

# 9. Prueba obligatoria: Shared R4 vs Untied U4 a igual FLOPs

Esta es una de las pruebas centrales que faltó en la línea lingüística.

## 9.1 Construcción

Mismo:

```text
d = 512
m = 8
K = 4
shell
data
tokenizer
teacher
loss
batch
optimizer
training tokens
```

### R4

```text
1 bloque recurrente
reutilizado 4 veces
P_core_unique ≈ 4.19M
```

### U4

```text
4 bloques independientes
cada uno ≈4.19M
P_core_unique ≈16.78M
```

Los dos ejecutan las **mismas cuatro transformaciones de forma y el mismo número de MACs/FLOPs por token**. La diferencia es si los pesos son compartidos.

En entrenamiento, U4 tiene más parámetros/optimizer state; eso se reporta aparte. El conteo de FLOPs forward/backward debe verificarse, no inferirse únicamente por fórmula.

## 9.2 Inicialización pareada

En update0:

```text
U4.block0 == U4.block1 == U4.block2 == U4.block3 == R4.block
```

bitwise, antes del primer step.

Shell también idéntica.

Así el desvío posterior es causado por sharing vs untied, no por una inicialización distinta.

## 9.3 Gate científico

Definir:

\[
\Delta_{share}=NLL_{R4}-NLL_{U4}
\]

positivo = untied mejor.

Margen de ingeniería predeclarado:

```text
0.05 nats/token
```

(~5.1% en ratio de perplexity, `exp(0.05)`).

Clasificación con cinco seeds en el endpoint maduro:

```text
SHARING_QUALITY_PRESERVED:
    upper CI90(Δshare) <= +0.05

SHARING_HAS_MEANINGFUL_QUALITY_COST:
    lower CI90(Δshare) > +0.05

INCONCLUSIVE:
    intervalo cruza +0.05
```

No se ejecutan cinco seeds desde el comienzo: primero piloto direccional y cost gate.

## 9.4 Gate físico paralelo

En CPU, R4/U4 se comparan además a igual FLOPs:

- latency;
- bytes DRAM si contador disponible;
- working set;
- cache misses si disponibles;
- A/B/C causal.

La tesis necesita simultáneamente:

```text
sharing preserva calidad razonablemente
+
sharing produce ventaja física medible
```

Si sólo se obtiene una de las dos, la tesis central no está cerrada.

---

# 10. V2 T0 físico en la i7-13700F

La i7 nueva invalida cualquier transferencia automática de la rodilla Ryzen.

## 10.1 Preflight de hardware

Registrar mediante APIs locales/CPUID:

- núcleos físicos y lógicos;
- topología P/E;
- L1/L2/L3 por core/cluster;
- afinidad real;
- ISA disponible;
- frecuencia efectiva;
- bandwidth de DRAM medido.

No fijar shards por “número de cores” antes de medir rendimiento individual.

## 10.2 Puntos obligatorios

```text
d = 512, 640
m = 4,8,16
K = 1,4,8
A = shared
B = untied/distinct
C = shared + eviction
```

El subset mínimo para GO GPU es:

```text
d512, m4/m8, K1/K8, A/B/C
```

## 10.3 Gates heredados

Mantener, salvo que una unidad formal los revise antes de ver resultados:

```text
rho_resident <= 0.50       mínimo
rho_resident <= 0.25       fuerte
matrixization gain >=1.5   mínimo
>=2.0                      fuerte
C debe aproximarse a B, no a A
```

## 10.4 Stop físico

Si `d512` no pasa residencia mínima y no existe una causa instrumental corregible:

```text
V2_GPU_TRAINING = HOLD
```

El principio original era precisamente fallar barato en hardware antes de gastar entrenamiento grande.

`d640` puede fallar residencia y seguir siendo útil como **frontera**; no necesita desplazar d512.

---

# 11. Presupuesto CPU vs GPU

## 11.1 CPU local — i7-13700F / 32 GB

Usos autorizados:

- T0 físico;
- correctness/native parity;
- profiling;
- KN5/unigram;
- inferencia y LN;
- K-sweeps;
- cuantización/runtime;
- tests toy pequeños.

No usar CPU para entrenar campañas lingüísticas V2 largas.

### Presupuesto operativo CPU

Por unidad física:

```text
preflight + sweep corto: <= 2 h wall
sweep completo T0-V2:    <= 8 h wall
LN/inference campaign:    <= 8 h wall por checkpoint family
```

Superar esos valores requiere justificar por qué el resultado no puede obtenerse en GPU o con una reducción predeclarada del sweep.

Estos son **límites de gobernanza**, no thresholds científicos.

## 11.2 RunPod GPU

Toda V2 lingüística se entrena en GPU.

No se preautoriza una campaña enorme. Cada configuración pasa por escalones.

### G0 — timing/correctness

```text
<= 200 optimizer updates
```

Objetivos:

- finitud;
- memoria;
- sec/update;
- exact data/loss contract;
- projected cost.

### G1 — smoke de aprendizaje

```text
500 updates
≈ 1.024M target presentations
1 seed
```

### G2 — directional pilot

```text
2000 updates
≈ 4.096M target presentations
1 seed
```

### G3 — madurez inicial

```text
4000 / 8000 updates
1–2 seeds
```

Sólo si sigue mejorando de forma útil y el costo proyectado es razonable.

### G4 — extensión máxima antes de revisión arquitectónica

```text
16000 / 32000 updates
```

No automática.

Antes de cada escalón:

\[
GPU\_hours=\frac{sec/update \times updates \times runs}{3600}
\]

El reporte debe incluir el coste estimado **antes** del GO.

Reglas de gobernanza recomendadas:

```text
>8 GPU-h proyectadas para una sola configuración/etapa:
    review explícito

>40 GPU-h acumuladas para una campaña multi-config:
    review explícito del usuario/Sol
```

No son gates científicos; evitan repetir el problema de gastar semanas antes de responder la pregunta barata.

---

# 12. Criterios de parada V2

## STOP-0 — Conformance

Cualquier unidad cuyo `OMEGA_CONFORMANCE_BLOCK` no cuadre:

```text
CONFORMANCE_HOLD
no interpretar resultados
```

## STOP-1 — Hardware

Si d512 no demuestra ventaja física mínima de sharing/residencia:

```text
STOP GPU scale-up
```

hasta corregir instrumento o arquitectura.

## STOP-2 — Trainability

NaN/Inf, no reducción de loss/NLL, o divergencia fuerte en G1/G2:

```text
STOP candidate
```

sin cinco seeds.

## STOP-3 — Sharing

Si en endpoint maduro:

```text
lower CI90(NLL_R4-NLL_U4) > +0.05
```

weight sharing tiene un coste de calidad material.

Antes de abandonar la tesis, puede probarse explícitamente si **más K de R compartido compensa ese gap a un presupuesto de FLOPs reportado**.

No se permite ocultar el coste comparando configuraciones con compute no declarado.

## STOP-4 — K compensation

Si el mismo checkpoint V2 flexible no mejora con K extra y los K extrapolados degradan consistentemente:

```text
TEST_TIME_COMPUTE_SCALING_NOT_SUPPORTED
```

La afirmación “K compensa tamaño en inferencia” queda rechazada para esa receta.

## STOP-5 — m/workspace

Si `m=8/16` no compra calidad ni eficiencia respecto a m4 en el presupuesto maduro:

```text
MATRIX_WORKSPACE_ML_BENEFIT_NOT_ESTABLISHED
```

No confundir con el PASS físico de T0-M.

## STOP-6 — Language

T3 no se libera hasta:

```text
MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED
```

sobre un candidato **conforme** o sobre una unidad que declare explícitamente por qué otro candidato sustituye al contrato.

---

# 13. Orden experimental V2 mínimo

## V2-0 — Conformance implementation

1. Implementar bloque 16d² d512.
2. Implementar d640.
3. Ledger core/shell.
4. Eliminar hard-cap K por tablas finitas en la variante K-flex.
5. Tests bitwise/toy.

**Cero entrenamiento largo.**

## V2-1 — T0 físico i7

A/B/C + m/K + d512/d640.

Si d512 falla, HOLD.

## V2-2 — GPU trainability d512

R4 y U4:

- G0 200;
- G1 500;
- G2 2000.

No cinco seeds todavía.

## V2-3 — Shared vs untied

Si ambos aprenden:

- llevar R4/U4 al mismo endpoint maduro;
- multi-seed sólo tras cost gate;
- aplicar equivalence margin 0.05.

## V2-4 — K-flex

Entrenar una sola R compartida con currículo + fase variable-K.

Inferencia Kinfer 1…16.

## V2-5 — m curve

En la profundidad/presupuesto seleccionados:

```text
m=4,8,16
```

## V2-6 — d640

Sólo después de saber qué ocurre en d512, salvo el T0 físico que sí corre desde el principio.

Pregunta:

> ¿más width compra calidad suficiente para justificar cruzar la rodilla física?

## V2-7 — LN mínimo

Aplicar el banco/protocolo ya construido.

La Fase B de calibración del instrumento puede completarse antes; no necesita entrenar OMEGA.

## V2-8 — T3

Sólo si V2 pasa lenguaje mínimo y existen evidencias suficientes de sharing/hardware.

T3 debe usar el factorial D0/D1/R0/R1 contractual.

---

# 14. Reutilizar vs descartar

## Reutilizar

### Ciencia/metodología

- document-balanced multichunk;
- manifest/provenance fail-closed;
- loss canónica corregida;
- hidden teacher cache;
- metodología de paired seeds;
- state causal diagnostics;
- LN Sol–Opus;
- absolute calibration unigram/KN5/teacher;
- generation degeneration detector;
- human blind protocol;
- cost gates;
- K-curve harness concept;
- TTR harness concept.

### Ingeniería

- técnicas Fable;
- C ABI/runtime design;
- workspace ownership;
- persistent threadpool lessons;
- AVX kernels como referencias;
- profiler/auditoría PyTorch;
- i7 concurrency topology methodology.

## No transferir automáticamente

- DLL `be376...`;
- `QUALIFIED_AT_2000` a d512/d640;
- `K4 selected` como profundidad V2 final;
- `SATURATION-BEYOND-K4` como ley arquitectónica;
- TAIL-REPEAT negative result como refutación de arbitrary-K;
- T0 Ryzen cache knees a i7;
- d128 NLL/generation como veredicto del core contractual.

## Descartar como afirmación

```text
“OMEGA ya validó el núcleo final”
“el núcleo es de 7M porque el modelo tiene 7M”
“K4 es el K óptimo de OMEGA”
“TTR-A demostró que más K no funciona en OMEGA”
“T0 demostró end-to-end residencia del núcleo real”
```

---

# 15. Responsabilidades por la deriva

Esta sección es de proceso, no de culpabilización personal.

## Arquitecto — Sol

Errores de gobernanza arquitectónica:

1. Aceptó `d128` como surrogate de scoping, pero no obligó a una unidad explícita `SURROGATE→CONTRACT` antes de permitir que sus resultados eligieran profundidad operativa.
2. Permitió que el bloque Fast (`12d²`) reemplazara en la práctica al bloque contractual (`16d²`) sin registrar una enmienda arquitectónica.
3. Autorizó una larga optimización del backend antes de revalidar la conformidad del modelo optimizado con el núcleo final.
4. Dio demasiado peso operacional a K-curve d128 y demasiado poco a la m-curve contractual pendiente.
5. El gate de competencia lingüística absoluta se introdujo demasiado tarde.
6. La generación degenerada histórica quedó correctamente como documentary-only, pero no se abrió inmediatamente una unidad preregistrada que respondiera la pregunta lingüística que ese resultado hacía urgente.

## Juez/orquestador

Errores de revisión:

1. Verificó muy bien hashes, gates locales y reproducibilidad, pero no mantuvo un **gate de conformidad arquitectónica global** por unidad.
2. Permitió que cada resultado local válido se acumulara hacia un relato global que no estaba demostrado.
3. No exigió sistemáticamente el ledger `P_core/P_shell/memory/index`.
4. No bloqueó la transferencia de evidencia física D512 hacia un surrogate D128 ni la transferencia inversa.
5. No reabrió `m` cuando K saturó, aunque el contrato lo exigía conceptualmente.

## Implementador/opencode

La mayoría de la deriva conceptual **no es responsabilidad del implementador**: implementó contratos autorizados y, en numerosos puntos, hizo HOLD correctamente.

Incidentes operativos reales —working tree/candidate provenance, reporting, harnesses— fueron detectados y corregidos durante la campaña. Deben seguir bajo fail-closed, pero no explican el desvío arquitectónico principal.

## Usuario

El usuario fue quien reintrodujo explícitamente la tesis original cuando detectó la discrepancia tamaño/K. No se asigna responsabilidad metodológica por la deriva al usuario.

---

# 16. OMEGA_CONFORMANCE_BLOCK — obligatorio desde ahora

Toda nueva unidad debe comenzar con este bloque y persistirlo en el manifest/report.

```yaml
OMEGA_CONFORMANCE_BLOCK:
  authority:
    repository: Hackerprod/Ejercise
    conversacion_md_blob: 7027e2ac9d1ba89db08dda73c81e244f3b9b19db
    contract_line_ranges:
      - 2390-2485   # K*m / A-B-C / matrixization
      - 2585-2630   # d / P_core / cache regime
      - 2688-2860   # H1 / variants / residency gates
      - 2945-3075   # parameter ledger / factorial comparisons
      - 3078-3240   # T0-T3 program

  phase: T0|T1|T2|T3|ENGINEERING

  contract_target:
    d: null
    m: null
    K_train: null
    K_infer_policy: null
    recurrent_block_formula: null
    P_core_unique: null
    P_shell: null
    P_retriever: 0
    P_memory_learned: 0
    B_memory_static: 0
    B_index: 0
    quantized_core_bytes: null
    sharing: shared|untied|na
    data_recipe: null

  actual_candidate:
    d: null
    m: null
    K_train: null
    K_infer_policy: null
    recurrent_block_formula: null
    P_core_unique: null
    P_shell: null
    quantized_core_bytes: null
    sharing: null

  deviations: []
  authorized_deviation_ids: []

  claim_scope:
    allowed: []
    forbidden: []

  status: CONFORMANT|AUTHORIZED_DEVIATION|CONFORMANCE_HOLD
```

## Regla fail-closed

Si existe cualquier discrepancia entre `contract_target` y `actual_candidate` y no hay un `authorized_deviation_id` previo a resultados:

```text
status = CONFORMANCE_HOLD
```

Consecuencias:

- se puede conservar evidencia técnica;
- no se interpreta como resultado arquitectónico OMEGA;
- no puede liberar la siguiente fase;
- no puede usarse para cambiar el contrato retrospectivamente.

## Casilla obligatoria en todo GO de Sol/juez

Antes de autorizar una unidad:

```text
[ ] d conforme
[ ] forma del bloque conforme
[ ] m conforme o desviación declarada
[ ] rol de K conforme
[ ] sharing conforme
[ ] ledger core/shell/memory completo
[ ] hardware claim coincide con hardware realmente medido
[ ] scope científico declarado
[ ] siguiente fase que puede/no puede liberar declarada
```

Sin todas las casillas resueltas: **NO-GO**.

---

# 17. OMEGA_CONFORMANCE_BLOCK global actual

```text
OMEGA_CONFORMANCE_BLOCK_GLOBAL

Original physical/scientific line:
    d512 primary / d640 frontier
    core formula 16d²
    m={4,8,16}
    shared recurrent core
    K as compute/depth dimension
    external bank only at T3

Current d128 line:
    d128
    core 197,888 recurrent params
    ~6.43M tied lexical embedding
    m8
    finite per-round depth/gate tables
    no external bank

Status:
    CONFORMANCE_HOLD

Reason:
    scale + block-form + m-curve + K-flex + physical residency
    not jointly reconciled

Allowed while HOLD:
    Phase-B LN instrument calibration without trained OMEGA
    documentation/audit
    V2 implementation
    hardware preflights
    correctness tests

Forbidden while HOLD:
    T3
    treating d128 LN as architecture verdict
    new long d128 training campaigns
    deployment claims for original OMEGA
```

---

# 18. Estado del protocolo LN

La revisión Sol–Opus produjo un instrumento metodológicamente útil y reutilizable.

Su estado debe separarse de la conformidad del candidato:

```text
LN_INSTRUMENT:
    VALID / FROZEN METHODOLOGY

D128_CANDIDATE_AS_T3_GATE:
    BLOCKED_BY_CONFORMANCE

V2_CANDIDATE_AS_T3_GATE:
    intended target
```

La Fase B automática puede completar:

- unigram;
- MKN5;
- DistilGPT2;
- L0/L1 instrument calibration;
- degeneration reference/control;
- manifests humanos.

No necesita observar OMEGA entrenado.

El panel humano sigue siendo un requisito posterior del protocolo principal.

---

# 19. Qué debe hacer el proyecto inmediatamente

Orden mínimo recomendado:

1. **Cerrar únicamente Phase B automática de LN**; no ejecutar trained d128 LN como gate de T3.
2. Crear `OMEGA_V2_CONFORMANCE_SPEC.md` con bloque `16d²`, d512/d640, ledger y K-flex.
3. Implementar V2 core aislado, todavía sin training largo.
4. Correr T0-V2 en i7 con A/B/C real y shared R4/untied U4 equal-FLOPs.
5. Sólo si d512 pasa gate físico, abrir GPU G0/G1.
6. Ejecutar shared R4 vs untied U4.
7. Entrenar V2-KFLEX y probar K en inferencia.
8. Reintroducir m={4,8,16} en presupuesto maduro.
9. Aplicar LN mínimo al candidato conforme.
10. Sólo entonces discutir T3.

---

# 20. Criterio final de éxito de V2 antes de T3

T3 puede discutirse únicamente si existe evidencia conjunta de:

```text
PHYSICAL:
    d512 passes residence/matrixization gate on current CPU

SHARING:
    shared R preserves acceptable quality vs untied U at equal FLOPs

COMPUTE:
    recurrent K provides measurable useful composition
    and, for the strong original claim, test-time K scaling is demonstrated

WORKSPACE:
    m curve is measured linguistically, not only physically

LANGUAGE:
    MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED

ACCOUNTING:
    core/shell/memory/index reported separately
```

Si el lenguaje pasa pero test-time scaling no, puede existir un recurrente útil, pero **no** se sostiene la versión fuerte de la tesis “un núcleo pequeño compensa tamaño mediante más K en inferencia”.

Si sharing da ventaja física pero un coste de calidad grande, la tesis tampoco está cerrada.

Si d512 no ofrece residencia en la i7, debe redimensionarse el core desde medición; no se conserva d512 por dogma.

---

# 21. Conclusión

La campaña no debe desecharse. Produjo activos valiosos:

- metodología científica fuerte;
- backend CPU real;
- mecanismos causales de estado;
- datos sobre K;
- mejor política de datos;
- instrumento lingüístico preregistrado;
- una enorme cantidad de tooling reproducible.

El error fue de **conformidad acumulativa**: un surrogate útil (`d128`) pasó gradualmente de ser instrumento de scoping a actuar como representante de una arquitectura cuyo núcleo contractual era más ancho, tenía otra forma paramétrica y exigía una interacción K×m/hardware que nunca se cerró conjuntamente.

La corrección no es volver a cero. Es insertar ahora el puente que faltó:

\[
\boxed{
T0\ físico\ conforme
\rightarrow
V2\ d512/d640
\rightarrow
shared\ vs\ untied
\rightarrow
K\ flexible
\rightarrow
m\ curve
\rightarrow
LN
\rightarrow
T3
}
\]

Y desde este momento ninguna unidad nueva puede omitir `OMEGA_CONFORMANCE_BLOCK`.

---

# Apéndice A — mapa de líneas contractuales de `Conversacion.md`

| Líneas | Contenido relevante |
|---:|---|
| 1284–1300 | uso conservador de capacidad L2; ejemplo de 7–9 MB físicos |
| 1770–1790 | profundidad recurrente, banco factual, residencia, halting, cabeza selectiva |
| 2390–2485 | barrido `Km`, A/B/C, matrixización, tesis K×m |
| 2585–2630 | fórmula `16d²`; d384/d512/d640/d768/d1024 y bytes Q4 |
| 2688–2770 | H1 realista; d/m/K; barrido a FLOPs constantes; A/B |
| 2770–2860 | C expulsado; gate `ρresident`; matrixización |
| 2860–2945 | rodillas de caché y metodología física Windows |
| 2945–3075 | ledger de parámetros/memoria; factorial D0/D1/R0/R1; nociones de igualdad |
| 3078–3240 | estrategia GPU y programa T0/T1/T2/T3 |

---

# Apéndice B — clasificación resumida de evidencia

| Evidencia | Estado tras auditoría |
|---|---|
| T0-M matrixization D512 | VALID / PASS_STRONG scope físico |
| T0-R residency | DEGRADED / control causal no cerrado |
| Bridge-1 | FAIL_RECORDED histórico, no veredicto de residencia |
| T1 toy | VALID_WITH_SCOPE |
| d128 backend qualification | VALID_WITH_SCOPE |
| d128 causal state | VALID_WITH_SCOPE |
| d128 K4>K1 @2k | VALID |
| d128 K4→K6 saturation @2k | VALID_WITH_SCOPE |
| K4 operating depth global | NOT_ESTABLISHED |
| Coverage-C document-balanced | VALID data-policy result |
| TTR-A tail-repeat | VALID negative result para V1 policy |
| arbitrary-K original thesis | NOT CLEANLY TESTED |
| Fable techniques | REUSABLE engineering asset |
| be376 DLL → V2 | NOT_TRANSFERABLE |
| LN methodology | VALID instrument |
| d128 LN → T3 gate | BLOCKED_BY_CONFORMANCE |
| T3 | HOLD |

---

# Apéndice C — fuentes principales

- `AUDIT_REQUEST_DESVIACIONES.md`
- `Conversacion.md`
- `Conversacion LN.md`
- `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py`
- `Trash/cpu-native-arch/sweep-output/t0m-phase3-median10-rerun/t0m_phase3.summary.txt`
- `Trash/cpu-native-arch/sweep-output/t0r-int8-sharded/t0r_int8_sharded.summary.txt`
- `Trash/cpu-native-arch/archive/t0r/t0r-provenance-manifest.md`
- `Trash/cpu-native-arch/sweep-output/t0m-bridge1-residency-d1472/summary.txt`
- `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_autoregressive_generation_audit/CONTRACT.md`
- `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_autoregressive_generation_audit/results/autoregressive_generation_audit/generation_results.json`
- `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_pilot_freeze/SCIENTIFIC_PILOT_FREEZE.md`
- campañas recientes verificadas por el usuario: backend qualification, state causal diagnostics, K-CURVE-A, Coverage A/B/C, TTR-A.

**Nota de provenance:** los resultados recientes grandes no están todos versionados en GitHub porque `results/` fue excluido en `6b61c04`; cuando este informe usa sus cifras, las clasifica como artifacts locales verificados por el usuario, no como contenido leído de GitHub.

# Contexto completo — arquitectura CPU-native recurrente (para onboarding de agente externo)

Documento de traspaso, no es un spec de trabajo. Objetivo: que un tercer agente entienda de dónde viene todo esto, qué se validó, qué falta, y las reglas de proceso vigentes, sin tener que reconstruir la sesión.

## 0. Roles y comunicación

- **Usuario**: dueño del proyecto, decide alcance/autorizaciones, único canal de contacto con Sol (vía interfaz web, copia/pega manual).
- **Claude (esta sesión, registrado como `claude_bd6c2c15`, session "Architecture" en agent-orchestrator)**: líder técnico y validador. Delega implementación a opencode, valida CADA resultado que opencode reporta antes de aceptarlo (regla explícita del usuario, ver sección 6), redacta specs citables, arma reportes para Sol.
- **opencode** (registrado como `opencode_7421b4ce`, mismo session): ejecutor. Implementa, corre benchmarks/entrenamientos, reporta con números crudos, pregunta antes de decidir cualquier cosa no cubierta textualmente por un spec.
- **Sol**: autor/crítico de la arquitectura, interactúa solo con el usuario vía web. Sus mensajes llegan pegados por el usuario en el chat; NUNCA se le escribe directo (ni por agent_send ni por ningún canal automático) — toda comunicación hacia Sol la redacta Claude como texto plano y el usuario la pega manualmente.
- **Máquina de validación**: única, sin hardware de respaldo — laptop del usuario, AMD Ryzen AI 5 330 (1× núcleo Zen5 "Classic" + 3× Zen5c "Compact", 4 físicos/8 lógicos SMT, L2 ~1MiB/núcleo, L3 8MiB compartida, DDR5-5600 **single-channel**, TDP STAPM=28W/PL1=35W/PL2=40W). Windows 11, MSVC "VS18" Build Tools (toolset v180, NO hay VS2022/v143 instalado).

## 1. Origen y arquitectura propuesta

Todo nace de `Conversacion.md` (diálogo Sol/Claude Opus, ahora archivado en `cpu_native_recurrence_lab_v0.4.1/legacy/` y en `cpu-native-arch/MD/`): una arquitectura de LLM "CPU-native" que explota la jerarquía de memoria de CPU en vez de imitar GPU:

- **Weight sharing en profundidad**: el mismo núcleo de pesos se reutiliza en las R rondas de recurrencia, manteniéndolo residente en L2 en vez de releerlo de DRAM cada ronda.
- **Matrixización por slots**: en vez de un solo vector de estado (GEMV), varios estados latentes en paralelo (S slots) convierten la operación en small-GEMM, amortizando cada carga de peso entre varios slots.
- **Arquitectura READ→THINK→CHECK→EMIT**: recuperar de memoria externa, K iteraciones del núcleo compartido, halting, cabeza de salida.
- **No-objetivo explícito**: la recursión NO aumenta la capacidad factual del modelo (lección Ouro/LoopLM: "repetir no almacena"), solo ayuda a componer conocimiento ya presente.

## 2. Track A — MRDL (pausado, no confundir con esto)

Antes de este trabajo hubo una investigación separada sobre "MRDL" (Modelo Relacional Disperso de Lenguaje) donde actué como juez/validador, encontré 6 bugs, y until una propuesta de Sol ("TFA", Transporte Fásico Anclado) que pasó verificación matemática pero no significancia estadística. **Esa línea está pausada por decisión del usuario** ("dejaremos a un lado mrdl"). No es relevante para el trabajo actual salvo como antecedente de que Sol ya había mostrado ida y vuelta rigurosa de crítica/corrección antes.

## 3. Primera ronda de validación física — T0-R y T0-M aislados (probes ad-hoc)

Con probes C++ escritos a mano (`cpu-native-arch/int8_probe.cpp`, `t0m_int8_probe.cpp`) en esta misma laptop:

- **T0-R (residencia)**: PASS limpio. Kernel int8 AVX2 real (dot-product, `_mm256_maddubs_epi16`+`_mm256_madd_epi16`), sharding proporcional por calibración real de núcleo, orden alternado entre repeticiones, 10 corridas independientes. Separación A(pesos compartidos)/B(pesos distintos por ronda) de **2.5–2.9x**, reproducible.
- **T0-M (matrixización)**: PASS_STRONG. Barrido completo 4 tamaños × 5 profundidades × 2 variantes. G(S)=MAC/s_fused(S)/MAC/s_fused(1): **G8 máx 3.89, G16 máx 3.61**.

Ambos verificados independientemente desde CSV crudos, con lecciones metodológicas duras aprendidas en el camino (ver sección 5 — la mayoría de esas lecciones se repitieron y quedaron finalmente resueltas en el rediseño de la sección 7).

## 4. Intento de recurrencia real — colapso y diagnóstico (invalidado)

Se intentó extender a recurrencia real (`t0m_recurrence_probe.cpp`, D=512, S y R variables). Resultado inicial: aparente colapso (A/B≈0.94, G16≈1.41) — casi se reportó como "la recurrencia mata el mecanismo". **Antes de enviarlo a Sol, Claude retractó la conclusión** tras la propia auditoría de código, y Sol (segunda crítica, `Respuesta_2.md`) confirmó **5 desvíos reales** que invalidaban esa medición:

1. Kernel de recurrencia distinto al fused validado en T0-M (reducción horizontal cada 16 columnas + `checked_add_i64`, no acumulador persistente).
2. Transición completa ejecutada en el hilo coordinador (no fijado a núcleo), serializada entre barreras — 4 workers esperando.
3. A y B contaminados desde la ronda 0 (`weight_seed ^= ... * (round+1)`, con `round+1` ya distinto en round=0).
4. D=512 no reproduce el régimen de memoria de T0-R (256 KiB de bloque, 64 KiB/núcleo con 4 shards iguales — muy por debajo del régimen 384–768 KiB validado).
5. Instrumentación por ronda (reserve/push_back de vectores de estadísticas) dentro de la ventana cronometrada.

Más: sharding igual (128,128,128,128) en vez de proporcional; self-test de Bclone comparando dirección del objeto vector en vez de `.data()` del buffer.

## 5. Desvío de esta sesión (descartado) — Bclone en `int8_probe.cpp`

Se pasaron **varias horas** esta sesión persiguiendo una idea propia (no del MD): agregar una variante "Bclone" (bloques físicamente distintos, contenido byte-idéntico) directo al binario original `int8_probe.exe`, con filas congeladas, para aislar "efecto de dirección" de "efecto de contenido". Se encontraron y diagnosticaron varias anomalías reales en el camino (útiles como aprendizaje metodológico, no como resultado):

- **Ruido térmico real y grande**: el mismo test exacto, corrido back-to-back, dio 57.85 GMAC/s vs 39.24 GMAC/s (32% de diferencia) con la máquina caliente tras horas de uso continuo; en frío, dos corridas idénticas dieron 70.65 vs 71.51 GMAC/s (1.2% de diferencia). Confirmado con WMI (aunque `CurrentClockSpeed` de Windows no es informativo, siempre reporta el nominal fijo).
- Un misterio de timing (Bclone con wall-clock ~26x el kernel_elapsed) que se rastreó hasta el **warmup loop** corriendo dentro de la ventana no cronometrada pero SÍ dentro del wall-clock del coordinador — un hallazgo real, pero sobre un experimento que finalmente se determinó **no estar justificado por ningún MD** (Respuesta_1 no menciona Bclone; Respuesta_2 lo define solo como parte del Bridge en D=1472, con el kernel fused de T0-M, no con `int8_probe.cpp` a D=512).

**El usuario ordenó frenar todo y auditar el código contra los MD de forma exhaustiva**, con números exactos, sin más validación a memoria. Resultado de esa auditoría: **los 9 desvíos de la sección 4 seguían sin corregir**, tal cual estaban cuando se escribió la crítica de Sol — nunca se habían tocado. Se escribió `cpu-native-arch/MD/Spec_Correcciones_Puentes.md` (spec de corrección, cada punto con cita textual de Respuesta_1/Respuesta_2), pero quedó **superado/archivado** por el punto 6.

## 6. Pivote grande — Sol construye CNRL desde cero

El usuario le pidió directamente a Sol que armara un proyecto nuevo para evitar desvíos. Resultado: **`cpu_native_recurrence_lab_v0.4.0`** (CNRL, "CPU Native Recurrence Lab"), un proyecto C++/CMake completo construido desde cero, que **corrige explícitamente los ~28 desvíos documentados en su propio `docs/DEVIATION_LEDGER.md`**, incluyendo todos los de la sección 4/5 de este resumen:

- Kernel prevalidado una sola vez (`run_kernel_unchecked`), mismo registro para T0-R/T0-M/T0-RM.
- Transición ejecutada por workers fijados a núcleos físicos, nunca por el coordinador.
- Estado doble-buffered (`state_a`/`state_b`).
- `clflush` (variante `cold`) estrictamente fuera del timer.
- Autotests en ejecutable separado (`cnrl_tests`), nunca en el hot path.
- Conteo de MAC/bytes calculado una vez en la librería y **recomputado independientemente** por `scripts/analyze_results.py --strict-structure` (rechaza estructuralmente cualquier discrepancia).
- Distinción explícita GMAC/s vs GB/s (`one_pass_weight_gb_per_second`), para no confundirlas cuando S>1.
- AVX2 tile 4 como baseline auditado; tile 8 documentado como estrés con riesgo de spill.
- Sharding proporcional Zen5/Zen5c vía `cnrl_calibrate` (3 pasadas: directa/inversa/rotada, mediana), granularidad de una fila (no 64).
- Generación de pesos **stateless por (seed, fila_global, columna, ronda)** — invariante al sharding, con test específico.
- Orden de sweep (tamaños/profundidades/slots/variantes) rotado/revertido entre repeticiones, no siempre ascendente.
- Fases frozen y recurrente **intercaladas por celda (S,R)**, no en bloques térmicos separados.
- Variantes de peso: `shared` (A), `clone` (Bclone: bytes idénticos, direcciones físicas distintas, verificado por hash+puntero), `untied` (contenido distinto, nunca usado como control de residencia), `cold` (C, clflush fuera del timer).
- Transiciones: `frozen`, `fixed-point` (mínima local), `group-rms` (por shard, dependiente del layout), `global-rms` (portable, reducción global pequeña).

Documentación completa en `cpu_native_recurrence_lab_v0.4.1/docs/`: `EXPERIMENT_CONTRACT.md` (fórmulas y umbrales), `DEVIATION_LEDGER.md` (tabla de desvíos históricos), `AUDIT_CHECKLIST.md`, `WINDOWS_RUNBOOK.md` (procedimiento paso a paso para esta laptop), `DELIVERY_AUDIT.md`.

## 7. Primera validación bare-metal de CNRL v0.4.0 (en esta laptop, Windows)

Build validado en Linux (GCC/Clang, ASan/UBSan) por Sol antes de entregarlo; **nunca antes corrido en Windows/este hardware**. Se corrió el runbook completo:

- **Build**: sin VS2022/v143 disponible, se adaptó a CMake+Ninja+`vcvarsall.bat x64` con el toolchain VS18/v180 ya instalado (excepción autorizada por el usuario, documentada como tal). Flags verificados directo en `build.ninja`: `/O2 /arch:AVX2 /W4 /WX`. CTest 3/3 PASS.
- **Topología real** (`cnrl_topology.exe --json`, no asumida): 4 físicos confirmados — core0 = Zen5 (efficiency_class=1) en lógicos [0,1]; cores 1-3 = Zen5c (efficiency_class=0) en [2,3],[4,5],[6,7]. Confirma que "CPUs 0,2,4,6" era correcto.
- **Bandwidth**: `cnrl_bandwidth` read-only = 16.513 GB/s. Cruzado contra el viejo `stream_bandwidth.cpp` (32.9295 GB/s, pero ese cuenta memcpy read+write → normalizado a solo-lectura da 16.465 GB/s) — 0.29% de diferencia entre dos benchmarks independientes. Se adopta 16.513 GB/s como techo de referencia del proyecto CNRL.
- **Calibración** (3 pasadas, mediana): CPU0(Zen5)=42.9 GMAC/s, CPU2=30.4, CPU4=33.7, CPU6=33.7 GMAC/s.
- **T0-R aislado — PASS**: a R=16, tamaños 384/512/640/768 KiB/núcleo: A/B = 1.95x/2.26x/2.80x/2.68x, con `min(A)>max(B)` (separación total, sin superposición) en varias celdas de R=8/16. R=1/R=4 casi paridad (esperado: no hay profundidad suficiente todavía).
- **T0-M aislado — PASS_STRONG**: G(16) de `clone` llega a **5.94x** en R=16/512KiB (muy sobre el umbral fuerte de 2.0x).
- **T0-RM (recurrencia real, D=1472, transiciones fixed-point/group-rms/global-rms)**: en S=1,R=8 las **tres** transiciones dan 1.96–2.24x de separación shared/Bclone — **la ventaja de residencia sobrevive a recurrencia real con transición real**, primer resultado limpio de toda la sesión en esta pregunta. En S=8/16 la separación cae cerca de 1x, coherente con lo que el propio contrato anticipa (B deja de ser DRAM-bound por reutilización intra-ronda entre slots — no es una falla).
- **Hallazgo reportado a Sol**: `fixed-point` en `cnrl_transition_bench` clipeaba ~93-94% de las celdas de forma consistente en TODAS las combinaciones D×S probadas — `projection_shift=12` mal calibrado.
- **Hallazgo reportado a Sol**: la auditoría de ensamblado de MSVC marcó `fused4` como FAIL, pero inspección directa del dump confirmó que era **falso positivo** — los accesos a stack de XMM6-15 estaban solo en el prólogo/epílogo obligatorio del ABI de Windows x64 (una vez por llamada), el loop caliente en sí no tenía ningún acceso a stack. El kernel estaba bien; el script de auditoría tenía el bug.
- **Gap abierto en ese momento**: no había forma de correr T0-R/T0-M aislados a D=1472 exacto (los scripts tenían `--D 512` hardcodeado) para comparar contra el puente frozen bit a bit.

Validación estructural: `analyze_results.py --strict-structure` corrió 3 veces, 1274 filas totales, PASS completo, sin filas inválidas ni divergencia de checksums shared/clone. Alerta no bloqueante: 186 condiciones con CV externo >10% (ruido térmico, consistente con lo observado en la sección 5).

## 8. Respuesta de Sol y v0.4.1

Sol aceptó la validación, corrigió la interpretación del clipping (el 93-94% era de 1000 cadenas independientes cada una saturando igual con el mismo output sintético, no un fenómeno de recurrencia real; en una prueba equivalente propia a D=1472/R=8 dio ~25% con shift=12, bajando a 0% con shift=14) y lanzó **v0.4.1**:

- Build portable (detecta toolchain solo, ya no fijo a VS2022/v143 — la adaptación de esta laptop queda oficialmente soportada).
- `run_exact_bridges.ps1`: puentes exactos D=1472 para comparar standalone vs frozen embebido bit a bit.
- Auditoría de ensamblado corregida (excluye prólogo/epílogo del ABI, sigue exigiendo `vpmovsxbw`/`vpmaddwd` y rechaza spill real dentro del loop).
- `cnrl_transition_bench` distingue `chain_length=1` (costo aislado) de `chain_length=R` (deriva numérica en recurrencia real).
- Default de `fixed-point projection_shift`: 12→14 (12 queda RECHAZADO por saturación).

**Verificación independiente en esta laptop (no se aceptó de palabra)**:
- SHA-256 del ZIP y `MANIFEST.sha256` interno (84/84) verificados exactos.
- Build limpio, CTest 4/4 PASS.
- Shift=14 CONFIRMADO en este hardware: shift12≈25.1% clipping (coincide con la predicción de Sol), shift13≈2.9-3.3%, shift14=0% mediano, shift15=0%.
- T0-RM con shift14: S1=2.260x, S8=1.059x, S16=1.068x.
- Auditoría de ensamblado: `fused4` PASS limpio, `fused8` da warning esperado (documentado, no bloqueante), `transitions` PASS.
- **Gap cerrado**: `run_exact_bridges.ps1` comparó standalone T0-R/T0-M a D=1472 contra el frozen embebido en T0-RM al mismo D/S/R=8 — **checksums bit-idénticos** (output_checksum, state_checksum, round_sink, weight_hash_signature) entre ambas mediciones. El throughput varía hasta 23.6% entre ambas (ruido térmico ya documentado), pero la identidad de cómputo queda probada, no inferida.

## 9. T0 — cierre formal (Sol)

Sol cerró el ledger:

| Componente | Estado |
|---|---|
| T0-R (residencia) | **PASS** |
| T0-M (matrixización) | **PASS_STRONG** |
| T0-RM (recurrencia real) | **PASS** |
| Fixed-point shift 12 | RECHAZADO |
| Fixed-point shift 14 | VALIDADO |
| Group-RMS / Global-RMS | VALIDADO, 0 clipping |
| Puente frozen D=1472 | IDENTIDAD BIT A BIT |
| Kernel fused4 | VALIDADO |
| Kernel fused8 | Solo estrés, riesgo de spill conocido |
| Mecanismo lingüístico | Todavía no probado |

Corrección de Sol sobre nuestros números: 2.260x/1.059x/1.068x (S1/S8/S16) son ratios **shared/Bclone**, NO son `G_RM(S)=MAC/s_recurrente(S)/MAC/s_recurrente(S=1)`. Esa métrica **no está calculada** en `analyze_results.py` ni en `analysis.md` — queda pendiente, pero Sol confirmó explícitamente que **no bloquea T1**.

Conclusión física: `shared/Bclone = 2.260×` a D=1472,S=1,R=8 con transición real entre rondas — la hipótesis física de residencia+matrixización ya no es especulación, está demostrada en hardware real.

## 10. T1 — autorizado, en curso

Pregunta que T1 debe responder: **¿puede un núcleo pequeño con pesos compartidos y estado matricial aprender computación recurrente útil, estable y generalizable?** No incluye lenguaje, tokenizer, LM head, memoria externa ni cuantización todavía — cada cosa se prueba por separado.

Spec completo (de Sol, estructurado por Claude) en `T1_Spec_Trainability.md`. Resumen:

**Arquitectura (T1-A)**: `X^(r) ∈ R^(S×D)`. `U^(r) = SlotMix(X^(r))` (self-attention single-head escalada S×S). `X^(r+1) = RMSNorm[X^(r) + g_r⊙F_θ(U^(r),e_r)]`, con `F_θ` = MLP compartido (Linear(D,4D)→GELU→Linear(4D,D)), `e_r` = embedding de profundidad, `g_r` = gate aprendido (sigmoid, init≈0.1). D∈{64,128} (solo D=64 usado por ahora), S∈{1,4,8}, R∈{1,2,4,6,8}. 4 baselines: `single` (1 núcleo, 1 ronda), `shared` (1 núcleo, R rondas), `untied` (R núcleos, R rondas), `vector-state` (shared, S=1).

**Implementado y verificado** (`t1_trainability_lab_v0.1.0/`): entorno aislado (`uv venv --python 3.14`, `torch==2.11.0+cpu` fijado), arquitectura (`t1_trainability/model.py`: `RMSNorm`, `SlotMix`, `CoreMLP`, `RecurrentCore`), 5 tareas sintéticas con generadores deterministas y datasets (10k/2k/2k train/val/test cada una): associative recall, multi-hop, variable binding con distractores, actualización secuencial de estado, generalización de longitud (train 1-3 hops, test OOD 4-6 hops).

**6 gates fijados ANTES de entrenar** (no se relajan después de ver resultados): (1) estabilidad en ≥5 semillas sin NaN/Inf/colapso; (2) `shared R=4` debe superar claramente a `shared R=1` en multi-hop (≥15pp); (3) `shared` debe conservar ≥90% de la exactitud de `untied`; (4) slots no colapsados (coseno medio <0.90, rango efectivo >0.5×S, ningún slot muerto en >95% de ejemplos); (5) generalización a 4-6 hops debe superar claramente el azar; (6) dependencia real entre rondas (estados cambian, permutar rondas degrada resultado, embedding de ronda tiene efecto medible).

**Historia de la campaña de entrenamiento (en curso)**:
1. Primera campaña completa (100 runs, D=64, grid mínimo de 8 configuraciones estructurales × 5 tareas, seeds escalonadas [101,202,303,404,505] solo en combos gate-críticos) corrió limpia (sin NaN/Inf/crash).
2. **Pero antes de interpretar los gates**, auditoría propia (de opencode, sin que Claude lo forzara) encontró **dos rondas de leaks reales en los generadores de datos**:
   - `multi_hop`/`length_generalization`: las relaciones se serializaban en orden de cadena, así que el target siempre era el último token de la secuencia — un modelo `single`/`shared R=1` (sin recurrencia real) llegaba a 1.000 de exactitud, prueba de shortcut trivial, no de composición de hops.
   - `variable_binding`: dos leaks — (a) el atributo target siempre era el primer `ATTR` después de `ASSIGN` (los distractores siempre después); (b) el `query_token` del OutputReader se construía directo como `OBJECT:<target_object>`, saltándose la resolución de variable por completo.
3. Fix aplicado: barajar hechos REL/ATTR antes de serializar (multi-hop/length-gen), usar `query_token='VAR:X'` no resuelto (variable_binding). `associative_recall` y `sequential_update` auditados y confirmados limpios, sin necesidad de cambios.
4. Verificación barata post-fix: heurísticas triviales (última relación, primer atributo) corridas contra los datasets regenerados — resultados pegados al azar esperado en las 3 tareas corregidas (diferencias de 1-3 puntos porcentuales), sin señal de un tercer leak.
5. **Estado actual**: rerun acotado de 60 corridas (solo las 3 tareas afectadas: multi_hop 28, length_generalization 16, variable_binding 16; `associative_recall`/`sequential_update` NO se repiten, sus resultados de la campaña de 100 corridas siguen siendo válidos) lanzado en background, estimado 2-4h (techo conservador <6h), timing basado en piloto real cronometrado, no adivinado.

**Todavía sin resultado**: no hay números de los 6 gates todavía — la campaña corregida está corriendo.

## 11. Reglas de proceso vigentes (importante para cualquier agente que se sume)

1. **Nada se implementa sin cita textual** de un spec/documento fuente (Respuesta_1.md, Respuesta_2.md, EXPERIMENT_CONTRACT.md, DEVIATION_LEDGER.md, AUDIT_CHECKLIST.md, T1_Spec_Trainability.md, o el mensaje original de Sol). Si algo no está cubierto, se pregunta antes de decidir — nunca se improvisa "lo más razonable".
2. **Claude valida cada resultado de opencode antes de aceptarlo** — no se relaya nada como válido sin releer la cita correspondiente y, cuando es factible, verificar el código/dato directamente (diff, grep, recomputar a mano). Regla explícita del usuario tras un patrón repetido de desvíos.
3. **Disciplina de commits**: commit funcional en cada hito validado, no solo al final — para poder diffear/bisectar en vez de reconstruir de memoria. Cada commit de esta sesión referencia qué se validó y con qué evidencia.
4. **Comunicación con Sol**: solo texto plano en el chat, para que el usuario lo pegue manualmente en la interfaz web de Sol. Nunca `agent_send` directo a Sol.
5. **No se acepta ningún resultado "por autoridad"** — ni de opencode, ni de Sol, ni de Claude mismo. Todo reclamo de rendimiento/corrección se verifica con datos crudos, checksums, o recomputación independiente antes de pasar a la siguiente etapa. Ejemplos aplicados: se rechazó una primera interpretación propia de "la recurrencia colapsa" hasta verificar el código; se verificó el SHA-256 y manifest de cada release de Sol antes de tocarlo; se pidió confirmación en hardware real del shift=14 que Sol solo había probado en Linux.
6. **Ruido de máquina es real y grande en esta laptop** — variaciones de hasta 30% run-a-run por calentamiento en sesiones largas de benchmarking continuo. Cualquier resultado nuevo debe considerar esto antes de interpretarse como señal.

## 12. Mapa de archivos relevantes

```text
Ejercise/                                  (raíz del repo, git)
├── cpu-native-arch/                       (LEGADO — probes ad-hoc, ya no se usa)
│   └── MD/Spec_Correcciones_Puentes.md    (superado, ver sección 5-6)
├── cpu_native_recurrence_lab_v0.4.0/      (CNRL, primera entrega de Sol — T0)
├── cpu_native_recurrence_lab_v0.4.1/      (CNRL, versión activa — T0 cerrado acá)
│   └── docs/{EXPERIMENT_CONTRACT,DEVIATION_LEDGER,AUDIT_CHECKLIST,WINDOWS_RUNBOOK,DELIVERY_AUDIT,Q4_STATUS}.md
├── T1_Spec_Trainability.md                (spec de T1, estructurado de Sol)
├── t1_trainability_lab_v0.1.0/            (proyecto activo — T1, en curso)
│   ├── t1_trainability/{model.py,data.py,adapters.py}
│   ├── datasets/                          (5 tareas, JSONL + manifest.json)
│   └── campaign/runs/...                  (resultados por task/variant/config/seed)
└── HANDOFF_CONTEXT_2026-09-04.md          (este documento)
```

## 13. Próximo paso inmediato

Esperar a que termine el rerun de 60 corridas de T1 (en background). Cuando termine: consolidar los 6 gates con números crudos por semilla, decidir si T1 pasa/falla/parcial, y recién ahí reportar a Sol (nunca antes de tener los números reales).

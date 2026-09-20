# Handoff — Pausa de campaña OMEGA para enfocarnos en Prioridad 0 / Truncated BPTT / Prioridad 1

**Fecha**: 2026-09-19
**Por qué existe este documento**: pedido explícito del usuario (2026-09-18) — durante el proceso de decidir Prioridad 0/1, tanto el judge (yo) como Sol (arquitecta, vía ChatGPT) acumulamos mucho contexto lateral (preguntas de hardware, investigación de optimización, verificación de fuentes externas) que puede diluirse o perderse entre sesiones. Este documento es el punto de referencia único para retomar sin tener que releer toda la conversación.

**Regla de oro para quien retome esto**: no asumas que nada de lo de abajo sigue vigente sin releer el código/estado actual primero. Esto es una fotografía de 2026-09-19, no una fuente de verdad viva.

---

## 1. Dónde quedó la campaña (estado cerrado, no se toca)

### Cadena de unidades cerradas, en orden
SCOPE-B (PROMISING-B) → SCOPE-C 3ª seed (PROMISING-C) → ER32-DESIGN (PASS_DESIGN) → ER32-INTEGRATION-AND-COST-GATE (PASS) → ER32-QUALITY-SCOPING-A (QUALITY-MIXED-A) → AUTOREGRESSIVE-GENERATION-AUDIT (AUDIT_COMPLETE) → ER64-SCALING-GATE Stage 1 (PASS) → Stage 2 cost-gate (PASS) → U1 SVD-diagnostic (DIAGNOSTIC_COMPLETE) → U2 expanded-validation (SECONDARY_EVIDENCE_COMPLETE) → U3 cache-survival-audit (MEASUREMENT_COMPLETE) → **ER64-QUALITY-SCOPING-A (CLOSED / QUALITY-NO-GO-A, cierre final de esta sesión)**.

Todo esto está documentado en Addenda 200-219 de `T1.5_Spec_MIX_O.md`, con sus MD/168.md-184.md correspondientes (respuestas verbatim de Sol). No hace falta releerlos todos — el resumen relevante para retomar está abajo.

### El resultado final real (ER64-QUALITY-SCOPING-A)
4 combinaciones reales (seed13/14 × K1/K4, 2000 updates cada una), sin crashes, todo verificado (hashes de checkpoints, clasificación reproducida con la función real del módulo). Commits `12d46ea` (resultados) + `476c027` (Addendum 219, respuesta de Sol).

**Clasificación: `QUALITY-NO-GO-A`.**

| | δ_K1 | δ_K4 | Δ_ER64 (K1−K4, +=K4 mejor) |
|---|---|---|---|
| seed13 | 0.174372 | 0.193084 | +0.001014 |
| seed14 | 0.178340 | 0.245505 | −0.006229 |
| **media** | **0.176356** | **0.219294** | **−0.002607** |

**Lectura correcta de Sol (no simplificar a "ER64 fracasó")**:
- Calidad absoluta mejoró MUCHO vs ER32: δ_K1 −57.5% (0.4149→0.1764), δ_K4 −43.4% (0.3872→0.2193). Duplicar el rango sí compró calidad — no alcanzó el techo de 0.10, pero se acercó bastante más que ER32.
- La ventaja recurrente K4>K1 que sobrevivía en ER32 (positiva en ambas seeds) **se diluye a rank64** — Δ_ER64 oscila alrededor de cero en todos los checkpoints medidos, nunca consistente.
- **Hallazgo científico nuevo, sin explicar todavía**: identidad `Δ_ER64 = Δ_R1 + (δ_K1 − δ_K4)`. En rank32 la compresión perjudicaba MÁS a K1 que a K4 (amplificando la ventaja recurrente). En rank64 pasa lo contrario — perjudica más a K4 que a K1, cancelando la ventaja que R1 ya tenía. Hipótesis abierta: ¿K4 compensa mejor bajo restricción severa y esa ventaja se diluye al relajarla? ¿O es dinámica de optimización distinta entre K1/K4 dentro de la parametrización factorizada? Sol NO lo resolvió, lo dejó anotado como "pregunta científica propia" — no es un próximo paso autorizado, es una observación para el futuro.
- Validación secundaria (60 docs) reproduce el patrón casi exacto — no es accidente de los 8 documentos originales.

**Sol NO autorizó continuar** (`ER64 Quality-B/C = HOLD`) — el NO-GO cumplió su función: frenar antes de gastar más updates. No hay una "siguiente unidad de scoping" pendiente de lanzar.

### Qué NO se tocó, no hace falta revisar
Todos los archivos históricos (R1, F, A/B/C, ER32) permanecen intactos — verificado repetidamente durante la sesión, ningún resultado congelado fue modificado.

---

## 2. Por qué se pausa acá (motivo del usuario, no mío)

Cita textual del usuario (2026-09-18): *"Estos test simples nos están tomando demasiado tiempo... estamos sumando muchísimo tiempo en pasos cortos en lugar de parar y enfocar todos los esfuerzos en optimizar y reducir tiempos, yendo al fondo del mecanismo de entrenamiento e idear vías, porque en adelante las pruebas serán más rigurosas y nos echaremos meses de paso en paso."*

Perfil de costo REAL medido (ER64 cost-gate, config F, por update, batch=8):
- `teacher_forward_seconds`: 1.951s (**42.3%** del total) — el componente más grande
- `backward_seconds`: 1.438s (**31.2%**)
- `vocab_projection_loss_seconds`: 0.729s (**15.8%**)
- `student_recurrence_forward_seconds`: 0.227s (4.9%) — el núcleo recurrente en sí ya es barato
- resto: ~0.8%
- total: 4.611s/update

Decisión explícita del usuario: **antes de lanzar cualquier unidad de scoping nueva** (rank/K adicionales, memoria externa, lo que sea), atacar de raíz por qué cada paso cuesta tantas horas. Todo lo que Sol pudiera sugerir como "siguiente paso lógico" después de este cierre queda **documentado pero NO perseguido** hasta que Prioridad 0 y el siguiente-en-cola (truncated BPTT) den resultado.

---

## 3. Prioridad 0 — ¿el teacher hace falta? (lo primero a hacer)

### Contexto — por qué existe el teacher, no es una muleta autoimpuesta
Verificado directamente en `Conversacion.md` (documento fundacional):
- Línea 3090: *"No se entrenará desde cero un modelo lingüístico significativo en el Ryzen AI 5 330."*
- Línea 3101: *"El entrenamiento lingüístico se hará en GPU mediante inicialización parcial y destilación desde un maestro denso."*
- Línea 3170 (fase T2): *"Se utiliza un Transformer denso preentrenado como maestro congelado."*

Es decir: el plan ORIGINAL nunca previó entrenamiento lingüístico real en esta laptop CPU — las corridas actuales (SCOPE-A/B/C, ER32/ER64) son *scoping* barato para comparar arquitecturas (mismo protocolo entre variantes), no la producción real. El teacher es diseño fundacional, no invención de la campaña.

Precedente independiente que refuerza que el teacher probablemente sí importa a esta escala: `sanoTTS` (proyecto TTS real, 294K-2.3M params, 721★) usa exactamente la misma receta — destilación desde un teacher más grande — para lograr resultados buenos con modelos diminutos y corpus chico.

### La pregunta exacta
¿Es la destilación (CE+KL contra DistilGPT2) necesaria para que R1 aprenda lenguaje coherente a esta escala (602 documentos, corpus revisitado ~13× por corrida), o CE puro contra el corpus (como cualquier LLM normal) da resultados comparables?

### Diseño AUTORIZADO por Sol (Addendum 220, MD/185.md, commit `b0e2909`) — ya no es borrador
Unidad `OMEGA-CE-ONLY-BASELINE`, aislada, solo R1 (no ER32/ER64). Único cambio científico: `L=0.5CE+0.5T²KL → L=CE completo` (no 0.5·CE — receta convencional completa, no gradiente reducido). Todo lo demás idéntico a R1 (D=128, S=8, K1/K4, seeds, AdamW, CPU FP32 4/1, B=8, T=256, chunk=512).

**Corrección clave de Sol al borrador original**: no alcanza comparar calidad a igual número de updates — hay que comparar también a igual COSTO de entrenamiento, porque CE-only más barato podría correr más updates por el mismo presupuesto de tiempo. Por eso el diseño real tiene 2 fases + rama automática:

- **Gate de correctness previo**: cada modelo CE-only debe ser bit-idéntico al checkpoint R1 en `update=0` (torch.equal + hash), oráculo de inicialización únicamente.
- **Fase 0 (cost probe, 24 updates descartables)**: DISTILL-K1/K4 + CE-only-K1/K4, 6 updates cada uno (2 warmup+4 measured), mismo protocolo del cost-gate. Calcula `q_cost` real (no asumir 42.3%) y predeclara `U_equal_cost = 2·⌊(2000/q_cost)/2⌋` ANTES de ver calidad.
- **Fase A (8000 updates reales, 4 combinaciones)**: `δ^CE_K,i(2000)≤0.10` → `CE-ONLY-NONINFERIOR-A`/`INFERIOR-A`/`MIXED-A`.
- **Rama automática equal-cost**: si Fase A no da NONINFERIOR, continúa automáticamente hasta `U_equal_cost` sin tocar hiperparámetros → `CE-ONLY-COST-NONINFERIOR`/`DISTILLATION-COST-ADVANTAGE`/`COST-MIXED`.
- **Generation audit** (documentario, sin gate): 32 generaciones, mismo protocolo congelado, comparadas contra las históricas del commit `ca8f4ad` (R1 con teacher).

Detalle completo del contrato en Addendum 220 de `T1.5_Spec_MIX_O.md` — no reinventar, ya está especificado exacto.

### Qué demostraría cada resultado
- CE-only pierde incluso a igual costo → "evidencia fuerte de que el teacher está económicamente justificado en esta receta" (Sol) — pero no prueba que TODO OMEGA necesite DistilGPT2 inevitablemente (no se probó LR tuning, scheduled sampling, más datos).
- CE-only alcanza calidad a updates o costo iguales → razón fuerte para sacar el teacher del camino futuro. Sería el hallazgo más grande de la campaña.

### Estado (2026-09-19): **CERRADA — `OMEGA-CE-ONLY-BASELINE = CLOSED / MIXED TRADEOFF`** (Addendum 224, MD/189.md). Fase 0 + Fase A (8000 updates reales) + rama equal-cost automática (4/4 combinaciones) + generation audit documentario, todo ejecutado real y verificado independiente por el judge (self-hash, tests, agregados recomputados desde JSON crudo).

**Resultado real**: equal-updates@2000 = MIXED (3/4≤0.10, K1 2/2 noninferior). Equal-cost congelado: K1=DISTILLATION-COST-ADVANTAGE (falla 2/2, presupuesto validado por timing largo), K4=COST-MIXED (una seed mejora fuerte, otra empeora — sensibilidad a seed, presupuesto conocido conservador). Generation audit: CE-only menos repetitivo que R1+teacher en las 4 métricas congeladas, en ambos presupuestos — pero sin poder afirmar "más coherente" (modos de fallo distintos, ver caso de artefactos `@-@`).

**Veredicto de Sol**: "el teacher no está demostrado como necesario para aprender comportamiento lingüístico... distillation SÍ mejora robustez de NLL held-out tardío, pero NO muestra ventaja correspondiente en generación." CE-only es receta viable. NO se cierra como "el teacher es necesario" — sería perder el hallazgo principal.

**Hallazgo nuevo, no perseguido todavía** (anotado en IDEAS_FUTURAS.md): NLL held-out y dinámica autoregresiva libre se DESACOPLAN con entrenamiento CE-only prolongado — empeora NLL, mejora diversidad/estabilidad de generación. Sol lo conecta con el scheduled sampling del diseño T2 original (brecha teacher-forced vs free-running). Reactivado como candidato futuro, no autorizado como unidad todavía.

**Siguiente paso real, per instrucción del usuario ("Stop temporal")**: NO perseguir las sugerencias de Sol sobre esta nueva brecha NLL/generación todavía — pasar a truncated BPTT (§4), siguiente en la cola ya acordada.

---

## 4. Siguiente en la cola — Truncated BPTT (justo después de Prioridad 0, no es Prioridad 1)

### Verificado en el código (2026-09-18)
En `run_er64_quality_scoping_a.py` (mismo patrón en R1/ER32): `.detach()` del estado SOLO se aplica entre ventanas (window 0→1), nunca dentro de una ventana. `loss.backward()` es una sola llamada — el backward de una ventana completa (256-512 tokens × K rondas) se computa de punta a punta en un solo grafo continuo, sin truncar.

### La técnica
Ya prevista en `Conversacion.md` línea 3210-3215 (plan fundacional T2): *"se generan trazas del maestro offline; se entrena el estado siguiente usando el estado objetivo anterior; luego se introduce scheduled sampling; finalmente se realizan unrolls cortos de 8, 16 y 32 tokens con truncated BPTT."* Es la técnica RNN clásica para acotar el costo del backward (31.2% del total) en secuencias largas — cortar el grafo en trozos más chicos con estado detached entre ellos.

### Por qué va después de Prioridad 0 específicamente
Con CE puro (sin KL), el backward se simplifica — es un momento limpio para medir el impacto de truncar sin la complejidad extra de la pérdida de destilación mezclada. Pedido explícito del usuario: "otro speed-up significativo" a probar justo después de simplificar la pérdida.

### Qué falta para saber si ayuda
No hay forma de saber si el balance neto ahorra tiempo sin medir — trozos más chicos de backward individual, pero más pasadas por ventana (overhead extra por chunk). Requiere desglosar `backward_seconds` por componente (hoy es un timer único, no sabemos si el costo grande viene del núcleo o de la proyección a vocabulario) y/o medir directamente con y sin truncamiento.

### Estado (2026-09-20): **CERRADA — `OMEGA-TBPTT-DIRECTIONAL-PROBE = CLOSED / DIRECTIONAL-STOP`** (Addendum 227, MD/192.md). Probado real (horizon=16, K4, seed13, 0→2000 updates): técnicamente positivo (R_t=0.891, ~11% menos wall-clock, RSS −62.8%) pero calidad se cae de forma creciente y clara (δ16 llega a +0.379 vs ceiling +0.10) — el NLL se estanca cerca de update 1000 y retrocede, no es un caso de "necesita más updates". Sweep formal 8/16/32 NO autorizado con esta evidencia. Consecuencia directa: el runtime nativo (Prioridad 2) YA NO necesita implementar `bptt_horizon` configurable como feature — solo dejar la costura arquitectónica para no cerrar la puerta. Alcance: solo refuta K4/horizon=16, K1 y horizon=32 quedan sin probar.

**Siguiente paso real**: Prioridad 1 (teacher logit caching, confirmado por el usuario como scope aislado, "solo eso, nada más todavía").

---

## 5. Prioridad 1 — Teacher logit caching (confirmado por el usuario como aparte, no junto con lo de arriba)

### La idea
El teacher forward es 42.3% del costo — pero OMEGA recorre 602 documentos por índice modular sobre 2000-8000 updates, así que cada documento se revisita del orden de ~13 veces por corrida (2000 updates ÷ 2 = 1000 pares × 8 docs/par = 8000 slots de documento ÷ 602 docs ≈ 13.3 revisitas). **Sin verificar todavía el patrón exacto de traversal en el código** — hay que confirmarlo antes de proponerlo a Sol.

Si es correcto: cachear el forward del teacher la PRIMERA vez que se ve cada (documento, ventana) y reusar en las ~12 revisitas siguientes eliminaría la gran mayoría del 42.3% después de la primera pasada por el corpus.

### Investigación GitHub real (verificada, no descartada por estrellas/mantenimiento — regla explícita del usuario)
- **`CompactifAI/Full-Chunked-KL-Loss`** (arXiv:2608.03796) — 19★, código real. Implementa caching offline de logits top-K del teacher. Diseñado para CUDA (portar lógica, no kernel). La variante "chunked" de la pérdida en sí AGREGA cómputo a cambio de memoria — solo la parte de CACHING es relevante acá, no la parte "chunked".
- **`akhilkedia/RandomSamplingKD`** (ACL 2025 Oral) — crítica formal a que el caching top-K naive da estimación SESGADA del teacher. Sin código (promesa vacía desde jul-2025). Vale como advertencia conceptual, no como algo a usar.

### Estado (2026-09-20): **CERRADA END-TO-END — `PRIORIDAD 1 = CLOSED/SELECTED`** (Addendum 232, MD/198.md; integración real Addendum del mismo día). Patrón de revisitas confirmado real (no estimado): 602 docs, 1000 pares, 13-14 apariciones/documento por ciclo de 1000 pares.

**Camino real recorrido**: (1) `OMEGA-TEACHER-LOGIT-CACHE` (logits crudos completos, 57.7GiB) → `CLOSED/NO-GO` real — 33-34% MÁS LENTO, no más rápido. Causa: el drive donde vive el repo (`D:`) resultó ser un HDD externo por USB (129 MB/s medido), no el NVMe interno (`C:`, 1.2-4.5GB/s medido) — el caché de 57.7GiB no cabía en RAM ni en el NVMe (solo 41.7GB libres entonces), terminó en el HDD lento sin que nadie lo marcara como variable. (2) Sol reinterpretó el NO-GO como específico del HDD, no una conclusión general, y autorizó `OMEGA-TEACHER-HIDDEN-CACHE-PROBE`: cachear el hidden state pre-LM-head (`[256,768]`, 0.88GiB, 65.4x más chico) en vez de logits completos — matemáticamente exacto (`logits=lm_head(hidden)`), cabe en RAM, construido en el NVMe. Resultado real: `PASS_STRONG` — R_K1=0.653, R_K4=0.724, R_joint=0.692, ~31% más rápido, break-even <0.1 corridas. (3) `OMEGA-HIDDEN-CACHE-PRODUCTION-INTEGRATION` (gate final, no investigación): provenance fail-closed, ausencia real del teacher confirmada, checkpoint/resume desde proceso fresco, oráculo bit-exacto — `INTEGRATION_PASS` real en K1 y K4.

**Implementación seleccionada, lista para usarse en futuras campañas que compartan teacher/corpus/context policy**: caché de hidden state FP32 en `C:\omega_cache\teacher_hidden.fp32` (NVMe), preload completo a RAM antes de entrenar, LM head original ejecutado online sobre el hidden cacheado.

---

## 6. Todo lo demás — queda en Stop temporal, documentado, NO se persigue

### 6.1. Fusión AdamW/backward (DeepSpeed cpu_adam pattern)
`microsoft/DeepSpeed`, `csrc/adam/cpu_adam_impl.cpp` — verificado real: fusiona AdamW completo en un solo paso AVX512+OpenMP. Real, producción (ZeRO-Offload), transferible conceptualmente a modelo chico. No es Prioridad 0/1/truncated-BPTT — queda pausado.

### 6.2. Recurrencia con pesos compartidos (Universal Transformer / looped) — investigación académica
`rkstgr/LoopLM`, `andreamad8/Universal-Transformer-Pytorch`, `locuslab/deq` (Deep Equilibrium Models) — todos reales, verificados, pero enfocados en GPU, sin optimización de CPU documentada. Gap explícito: nadie ataca "mantener el bloque de pesos del núcleo caliente en caché durante el loop de K rondas" de forma directa.

### 6.3. Cache-residency / CAT pseudo-locking — NO ataca el dolor actual, doble bloqueo práctico
Investigación muy bien verificada (código real de `arch/x86/kernel/cpu/resctrl/pseudo_lock.c` del kernel Linux, y `intel/intel-cmt-cat` PSEUDO_LOCK example — ambos confirmados byte por byte). Pero:
- **NO ataca el costo actual**: el forward recurrente del núcleo YA es barato (4.9%, 0.227s de 4.61s) — U3 midió un efecto de INFERENCIA (generación token-por-token), no de entrenamiento.
- **Bloqueo 1**: Intel CAT es tecnología de Intel — nuestro Ryzen AI 5 330 es AMD, soporte incierto en consumer/mobile (más común en EPYC datacenter).
- **Bloqueo 2**: la implementación demostrada es específica del kernel Linux (`mount -t resctrl`) — la laptop de validación corre Windows 11.
- El i7 futuro SÍ es Intel — pero no todos los i7 traen RDT/CAT, hay que verificarlo en el modelo específico cuando exista, y seguiría bloqueado si se sigue en Windows.
- `baidu-research/persistent-rnn` y `arXiv 2606.25353` ("Cache-Resident LLM Inference in GB-Scale Last-Level Caches") quedan marcados como los candidatos más fuertes para retomar esto en el futuro (relevante para INFERENCIA/producción, no para acelerar las corridas actuales de entrenamiento).

### 6.4. Curva de K (K2, K8, etc.) — solo si Prioridad 0/1 validan
Solo tenemos 2 puntos (K1, K4) — valida DIRECCIÓN (K4>K1 en R1/ER32, pero NO en ER64 per el cierre de hoy), no la FORMA de la curva. Cada K nuevo necesita su PROPIO baseline R1 entrenado a ese K (no es gratis, mismo costo relativo que K1/K4 ya gastado). Dada la interacción rank×K descubierta hoy (§1), esta pregunta se vuelve más interesante todavía — pero sigue siendo una unidad completa nueva, a diseñar con Sol después de 0/1.

### 6.5. TTS side-project del usuario (VPS, fuera del pipeline Sol/judge)
El usuario quiere probar el núcleo recurrente de OMEGA (K1 vs K4) en un dominio de TTS (texto→audio), corpus chico, en la VPS de Contabo, completamente aparte de esta campaña — sin pasar por Sol/judge. Referencia real encontrada: `Ampixa/sanoTTS` (721★, activo, 294K-2.3M params, corre en microcontroladores de $3, usa destilación desde un teacher — mismo patrón que OMEGA). Razonamiento del usuario: TTS tiene señal de aprendizaje más limpia que LM abierto, podría validar la hipótesis de profundidad recurrente (K4>K1) más rápido/barato que seguir peleando con lenguaje. Requiere reemplazar el head de entrada/salida (fonemas/mel-spectrograma en vez de tokenizer GPT-2 + readout 50257) — no es un swap trivial, pero el núcleo recurrente en sí es reusable. Es responsabilidad del usuario, no mía — anotado acá solo para que no se pierda la idea.

### 6.6. Deep research findings menores (mencionados, no accionables todavía)
- ARM way-locking (Cortex-R, L2C-310) — real, pero no aplica, no es hardware ARM.
- TI DSP SRAM/TCM — real, no aplica, no es ese hardware.
- `yandex/faster-rnnlm`, `asappresearch/sru` — kernels C++ reales con pre-alojo de estado y GEMM fusionado entre timesteps — técnica portable, referencia útil para cuando se ataque el núcleo recurrente en sí (no es prioridad ahora porque el núcleo ya es barato).

---

## 7. Cómo retomar esto

1. **Primero**: confirmar con el usuario que sigue queriendo este orden (Prioridad 0 → truncated BPTT → Prioridad 1 → todo lo demás en Stop).
2. Proponerle a Sol el diseño de `OMEGA-CE-ONLY-BASELINE` (Prioridad 0) — usar §3 de este documento como base, no reinventar desde cero.
3. Una vez Sol autorice el diseño, implementación en una unidad aislada nueva (no tocar R1/F/A/B/C/ER32/ER64 históricos), tests primero, sin ejecución real hasta revisión del judge.
4. Verificación del judge con el mismo rigor de siempre (hashes, recomputar números, reproducir con el código real, nunca confiar en resúmenes de opencode sin verificar).
5. Todo lo de `IDEAS_FUTURAS.md` (secciones "Reducir el costo real por update", "Optimización del propio bucle recurrente", "Mantener el bloque de pesos residente en caché", "Truncated BPTT", "PRIORIDAD 0") sigue siendo la fuente de detalle técnico completo — este handoff es el resumen ejecutivo, no lo reemplaza.

---

## 8. Archivos clave para no perder el hilo

- `T1.5_Spec_MIX_O.md` — Addenda 200-219 (esta sesión). Addendum 219 es el cierre real más reciente.
- `IDEAS_FUTURAS.md` — toda la investigación de optimización detallada, con rigor de verificación explícito en cada entrada.
- `MD/168.md` a `MD/184.md` — respuestas verbatim de Sol de esta sesión.
- `Conversacion.md` — documento fundacional; líneas 2496-2498, 3086-3101, 3168-3227 son las citadas en este handoff (rol del teacher, T3/memoria modular, "laptop = validación de inferencia no entrenamiento principal").
- `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er64_quality_scoping_a/` — el runner y resultados del cierre de hoy.
- Commits clave de esta sesión: `773f0dc` (fix real U3), `b9774d8` (resultado U3), `543a1f5` (runner Quality-A), `12d46ea` (resultado real Quality-A), `476c027` (Addendum 219, respuesta de Sol).

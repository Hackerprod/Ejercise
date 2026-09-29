# OMEGA-MINIMUM-LANGUAGE-COMPETENCE — bitácora del diseño adversarial

Dinámica: Sol propone, Opus audita/critica, Sol ajusta según la crítica, y se repite hasta que ambos coincidan en un protocolo congelado. Opus NO propone un diseño propio competidor — solo critica las propuestas de Sol. Cada ronda se guarda acá **verbatim**, sin resumir.

Contexto previo relevante (no repetido en cada ronda): K-CURVE-A cerró SATURATION-BEYOND-K4 (K4 seleccionado). Coverage-A/B/C cerraron con DOCUMENT_BALANCED_MULTICHUNK como receta de datos seleccionada. OMEGA-TEST-TIME-RECURRENCE-SCALING-A cerró NO_TEST_TIME_SCALING_WITH_TAIL_REPEAT. Hallazgo arquitectónico verificado contra el código real (`t1_trainability_lab_v0.1.0/campaign/omega_ce_only_baseline/run_omega_ce_only_baseline.py`): `DIMENSION=128`, `SLOTS=8`, `TOKENIZER_VOCAB=50257` — el embedding atado (50257×128≈6.43M) es ~92% de los "~7M parámetros" citados hasta ahora; el bloque recurrente compartido en sí (prelude+bloque compartido+readout+gates+depth-embedding) es solo ~0.6M. El modelo de referencia PyTorch está en `t1_trainability_lab_v0.1.0/t1_trainability/model.py`, clase `RecurrentCore`.

---

## Ronda 1 — Propuesta de Sol v0.1 (Paso A, sellada, MD/297.md)

Texto completo, verbatim:

> Mi propuesta independiente queda así. La diseñé para que pueda congelarse antes de ver una sola generación nueva; MiniMind calibra la escala de la prueba, pero no define el PASS de OMEGA.
>
> OMEGA-MINIMUM-LANGUAGE-COMPETENCE — Protocolo independiente v0.1
>
> ## 0. Pregunta científica
>
> La unidad debe responder únicamente: ¿El núcleo OMEGA actual, a ~7M parámetros y con la mejor receta científicamente seleccionada hasta ahora, ha adquirido competencia lingüística real y coherencia corta suficientes para justificar pasar a T3?
>
> No intenta demostrar razonamiento general, capacidad de asistente, conocimiento factual amplio ni paridad con un modelo de ~65M. Tampoco permite rescatar un resultado malo aumentando K en inferencia: se utiliza el horizonte nominal entrenado.
>
> ## 1. Modelo primario
>
> OMEGA seleccionado: arquitectura R1/F actual; D=128, slots=8; K=4; backend de evaluación PyTorch común; inferencia con exactamente K=4, sin TAIL-REPEAT; entrenamiento DOCUMENT_BALANCED_MULTICHUNK; loss canónica CE+KL ya validada; checkpoints update=2000; las cinco seeds: 20260913, 20260914, 20260915, 20260916, 20260917. Se evalúan las cinco. No se escoge retrospectivamente la seed que genere mejor texto.
>
> ### Controles
>
> **Control negativo:** los cinco checkpoints update=0 correspondientes. Su función es demostrar que la prueba detecta aprendizaje real y no propiedades accidentales del tokenizer, prompts o inicialización.
>
> **Referencia calibradora:** MiniMind local ~65M. MiniMind no constituye el threshold de PASS; no necesita ser igualado; sirve para saber cómo se ve en estos mismos instrumentos un pequeño LM ya conocido por producir lenguaje coherente. Preferencia: utilizar un checkpoint base causal LM. Si el disponible es chat/SFT, se registra explícitamente y se trata como CALIBRATOR_NON_ISOMORPHIC; no se convierte silenciosamente en comparación directa. No utilizar chat templates para OMEGA.
>
> ## 2. Preflight
>
> Antes de obtener outputs: congelar código, hashes y checkpoints; congelar el banco completo de prompts; verificar que ningún prompt completo aparezca literalmente en el train corpus; no tocar el test split histórico; verificar que el evaluator no modifica pesos; asegurar inferencia FP32; usar el mismo tokenizer que corresponde a cada modelo; no utilizar teacher ni hidden-cache; no ajustar thresholds después de observar resultados; guardar logits/scores crudos necesarios para reproducir todas las métricas.
>
> Si un prompt aparece literalmente en training, debe reemplazarse antes de cualquier inferencia, volver a sellar el banco completo y documentarlo.
>
> ## 3. Estructura de la prueba
>
> La unidad tiene tres capas:
>
> ```
> L0 — BASIC LANGUAGE COMPETENCE
>      contrastive / automática
>
> L1 — SHORT COHERENCE
>      contrastive / automática
>
> G  — FREE GENERATION
>      degeneración automática + evaluación humana ciega
> ```
>
> No hay razonamiento complejo todavía.
>
> ## 4. L0 — BASIC LANGUAGE COMPETENCE
>
> ### 4.1 Banco
>
> 96 ítems contrastivos, 24 por categoría.
>
> **A. Sintaxis local**
> Prompt: "The box of old photographs"
> Preferred: " was stored in the attic."
> Foil: " were stored in the attic."
>
> **B. Plausibilidad semántica**
> Prompt: "Mira poured the hot tea into the"
> Preferred: " cup on the table."
> Foil: " blanket on the table."
>
> **C. Referencia local**
> Prompt: "Jon placed the key beside the book. Later he picked up the"
> Preferred: " key before leaving."
> Foil: " weather before leaving."
>
> **D. Continuidad semántica inmediata**
> Prompt: "The room was completely dark, so Lena switched on the"
> Preferred: " lamp beside the bed."
> Foil: " sandwich beside the bed."
>
> Los ítems deben utilizar vocabulario cotidiano y minimizar conocimiento factual externo.
>
> ### 4.2 Scoring
>
> Para cada continuación candidata: S(c|p) = (1/N_c) · Σ log P(c_i|p,c_<i), donde N_c es el número de tokens de esa continuación bajo el tokenizer del modelo evaluado. El modelo acierta si S_preferred > S_foil. Empate exacto = 0.5. Se calcula accuracy por seed.
>
> ### 4.3 Gate L0
>
> Con las cinco accuracies entrenadas: CI90%(accuracy_L0) usando la misma convención t sobre seeds que hemos utilizado en la campaña. L0_ESTABLISHED requiere lower_bound_CI90 > 0.60.
>
> Además, calcular por seed: accuracy^trained_s − accuracy^init_s. Y requerir CI90(mean ΔL0) completamente > +0.10. Esto evita declarar competencia únicamente porque el banco resulta fácil para un modelo no entrenado.
>
> ## 5. L1 — SHORT COHERENCE
>
> 96 ítems adicionales, también contrastivos.
>
> **A. Persistencia de entidad/atributo**
> "Mara owns a red bicycle. Theo owns a blue bicycle. Mara rode to the park. When she arrived, her bicycle was"
> Preferred: " still red."
> Foil: " suddenly blue."
>
> **B. Orden temporal**
> "Leo put the bread in the toaster. A minute later the toast popped up. After that, Leo"
> Preferred: " removed the toast."
> Foil: " put the untouched bread in for the first time."
>
> **C. Causalidad simple**
> "The glass fell from the counter onto the hard floor. A moment later, Nina found"
> Preferred: " pieces of broken glass."
> Foil: " the glass floating above the counter."
>
> **D. Consistencia de situación**
> "Omar left his umbrella at home before walking outside. Heavy rain started halfway to the station. By the time Omar arrived,"
> Preferred: " his clothes were wet."
> Foil: " his clothes had stayed perfectly dry under his umbrella."
>
> No usar acertijos, matemáticas ni conocimiento cultural.
>
> **Gate L1:** por las cinco seeds, lower_bound_CI90(accuracy_L1) > 0.55. Y aprendizaje respecto a inicialización: ΔL1_s = accuracy^trained_s − accuracy^init_s, con CI90(mean ΔL1) completamente > +0.05. L1 es deliberadamente más difícil que L0.
>
> ## 6. Curva de adquisición
>
> Los 192 ítems L0+L1 deben evaluarse también, sin generación, en update 0, 500, 1000, 1500, 2000, para las cinco seeds, si esos checkpoints están disponibles. Esto no cambia los gates anteriores. Sirve para distinguir falta de entrenamiento, plateau, o posible límite de capacidad. Registrar especialmente accuracy_2000 − accuracy_1000.
>
> ## 7. Generación libre
>
> La evaluación contrastiva puede demostrar que el modelo asigna mejores probabilidades a continuaciones coherentes, pero no demuestra que genere texto coherente libremente. Por eso hace falta una prueba separada.
>
> ### 7.1 Banco humano
>
> 12 prompts congelados. Todos son comienzos de prosa, no instrucciones.
>
> 1. "Mara had a red notebook and a blue pen. She left the notebook on the kitchen table before going outside. When she returned,"
> 2. "The power went out during the storm. The apartment became dark, and Leo reached into the drawer where he kept a flashlight. A moment later,"
> 3. "Nina promised to meet Omar at the station at six. She arrived ten minutes early and waited beside the ticket machine. At six o'clock,"
> 4. "Daniel planted three small tomato plants behind the house. For several weeks he watered them every morning. By the middle of summer,"
> 5. "Sara put the cake in the oven and set a timer. She began washing the dishes while she waited. When the timer rang,"
> 6. "A small dog followed Elena home from the park. The dog had no collar, so Elena gave it some water and called the local shelter. While she waited,"
> 7. "Marcus borrowed a history book from the library on Monday. The book was due back in two weeks. He finished reading it that weekend and"
> 8. "The road through the mountains was covered with snow. The driver slowed the car and turned on the headlights. Around the next bend,"
> 9. "Julia kept two boxes in the closet. The green box contained old photographs, while the black box contained letters. Looking for a photograph, she opened"
> 10. "A loud noise came from the kitchen during the night. Amir walked downstairs carefully and switched on the light. Near the window,"
> 11. "The children built a tower from wooden blocks. It became taller than the chair beside them. When one child accidentally bumped the table,"
> 12. "Emma filled a glass with cold water and placed it beside her computer. After working for an hour, she reached toward the desk and"
>
> Estos textos deben pasar el chequeo de no coincidencia literal con training antes de congelarse definitivamente.
>
> ## 8. Sampling
>
> Para cada uno de los 12 prompts y cada una de las cinco seeds OMEGA: max_new_tokens = 96, temperature = 0.8, top_p = 0.95, top_k = disabled/0, repetition_penalty = 1.0, beam_search = false. No introducir repetition penalty ni stop heuristics que oculten loops. Sólo EOS normal del modelo puede terminar anticipadamente. RNG predeclarado: generation_seed = 910000 + 1000·replicate_index + prompt_index.
>
> Resultado: 12 prompts × 5 OMEGA seeds = 60 outputs. También: 12 outputs de MiniMind con seeds predeclaradas; 12 outputs de OMEGA update0 seed 20260913 como control humano negativo. Total para evaluación humana: 84 outputs. Adicionalmente puede generarse una versión greedy para diagnóstico automático, pero no sustituye el protocolo primario.
>
> ## 9. Métricas automáticas de generación
>
> Sobre los 96 tokens generados o hasta EOS: Distinct-2 = #bigrams únicos/#bigrams. Distinct-4 = #4-grams únicos/#4-grams. Repetition-4 = 1−distinct4. Maximum 4-gram recurrence = número máximo de apariciones de cualquier 4-gram exacto. Early termination = generación <24 tokens. Prompt-copy = fracción de 8-grams generados que aparecen exactamente en el prompt.
>
> **Catastrophic degeneration:** un output se marca como degenerado si max_4gram_count ≥ 4 OR repeat4 > 0.30. Gate automático: catastrophic_degeneration_rate ≤ 10% sobre los 60 outputs OMEGA entrenados. Las demás métricas se reportan descriptivamente; no se crea después un threshold nuevo según los resultados.
>
> ## 10. Evaluación humana ciega
>
> Esta es co-primaria, no decorativa. No usaría un promedio donde un buen Distinct-N pudiera compensar texto incoherente. Por tanto, no existe un "30% automático + 70% humano" que permita compensación cruzada. El gate humano es obligatorio.
>
> **Raters.** Ideal: 3 evaluadores humanos independientes. Los outputs se mezclan y anonimizan completamente: sin modelo; sin seed; sin condición; orden aleatorio distinto por evaluador. El prompt sí se muestra.
>
> **Escalas.** Cada output recibe 1–5 en:
> - **G — Grammaticality:** 1 = prácticamente ilegible, 3 = lenguaje mayormente interpretable, 5 = prosa fluida y bien formada.
> - **R — Relevance:** 1 = abandona/contradice inmediatamente el prompt, 3 = relación reconocible, 5 = continuación natural del contexto.
> - **C — Internal coherence:** 1 = secuencia incoherente, 3 = coherencia local básica, 5 = eventos y afirmaciones consistentes.
> - **E — Entity/causal continuity:** 1 = pierde o contradice entidades/relaciones esenciales, 3 = conserva parcialmente la situación, 5 = mantiene claramente entidades, atributos y causalidad.
> - **N — Non-degeneration:** 1 = loops/gibberish grave, 3 = algunos artefactos pero usable, 5 = generación natural sin degeneración.
>
> Para cada output se toma la mediana de los tres raters por dimensión. HumanComposite: H = (G+R+C+E+N)/5.
>
> **Gate humano.** Agrupar los 60 outputs OMEGA entrenados por prompt para el bootstrap, de modo que las cinco seeds del mismo prompt no sean tratadas como 5 prompts independientes. Requerimientos: lower_bound_bootstrap90(H) ≥ 3.0; y cada una de las 5 OMEGA seeds: mean H ≥ 2.75; y catastrophic_human_failure ≤ 10%, donde catastrophic human failure significa median(C) < 2 OR median(N) < 2.
>
> Reportar acuerdo inter-rater —preferiblemente Krippendorff α ordinal— pero no cambiar el threshold si α resulta bajo. Si α < 0.4, el resultado humano se etiqueta adicionalmente LOW_RATER_AGREEMENT y no se utiliza por sí solo para declarar FAIL; se requiere revisión del instrumento.
>
> ## 11. MiniMind
>
> MiniMind corre: los 96 ítems L0; los 96 ítems L1; los mismos 12 prompts generativos. Sus resultados se muestran al lado de OMEGA. Pero OMEGA PASS ≠ OMEGA ≥ MiniMind. MiniMind sólo responde: ¿qué valores producen estos instrumentos cuando los aplicamos a un pequeño LM que ya sabemos que puede generar lenguaje reconocible? No se modifican gates de OMEGA según sus resultados. Si MiniMind falla sorprendentemente el banco, se investiga si el checkpoint utilizado es chat/SFT o si el instrumento está mal calibrado. No se baja automáticamente el gate de OMEGA.
>
> ## 12. Clasificación final
>
> **A. READY_FOR_T3 / MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED.** Sólo si pasan simultáneamente: L0_ESTABLISHED AND L1_ESTABLISHED AND AUTOMATIC_NONDEGENERATION_PASS AND HUMAN_BLIND_PASS. Esto significa: OMEGA 7M muestra lenguaje funcional y coherencia corta suficiente para justificar añadir memoria externa. No significa razonamiento general demostrado.
>
> **B. LANGUAGE_PRESENT_BUT_SHORT_COHERENCE_NOT_ESTABLISHED.** Si L0 pasa pero L1 o humano falla. Entonces mirar la curva de adquisición. Si LateGain_L1 ≥ +0.02, la interpretación primaria será TRAINING_NOT_CONVERGED. Siguiente trabajo: entrenamiento/learning-rate schedule/presupuesto antes de escalar. Si LateGain_L1 < +0.02 y MiniMind muestra una separación clara en L1/humano: SCALE_PROBE_JUSTIFIED. No saltaría directamente a 65M. Primera escala sugerida: ~20–30M manteniendo la arquitectura y receta para obtener un segundo punto de capacidad.
>
> **C. LANGUAGE_FOUNDATION_NOT_ESTABLISHED.** Si L0 falla. No pasar a T3. Tampoco escalar automáticamente. Primero determinar si: la pérdida/optimización está funcionando; la curva L0 sigue aumentando; existe degeneración de generación; el modelo aprendió sólo estadísticas débiles. Si L0 está claramente mejor que init pero todavía creciendo, investigar entrenamiento. Si L0 está estancado cerca del control pese a NLL razonable, investigar arquitectura/objetivo antes de gastar cómputo en escala.
>
> ## 13. Qué justificaría escalar
>
> Escalar no se decide porque 7M sea menor que 65M. Se justifica cuando vemos: basic language = present; short coherence = insuficiente; learning curve = near plateau; training pipeline = funcionando; MiniMind calibration = demuestra que el instrumento puede medir la capacidad ausente. Eso sería evidencia de un posible cuello de botella de capacidad. En ese caso: 7M → ~20–30M primero. Sólo después tendría sentido decidir si acercarse a ~65M.
>
> ## 14. Qué justificaría pasar a T3
>
> T3 recibe GO si READY_FOR_T3 / MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED. No exigiría que 7M iguale MiniMind. Tampoco exigiría todavía razonamiento complejo. La razón es conceptual: memoria externa debe añadirse a un núcleo que ya demuestra que puede transformar contexto en lenguaje coherente; no debe usarse para intentar fabricar esa competencia desde cero.
>
> ## 15. Qué NO pertenece a esta unidad
>
> No incluir: preguntas factual-heavy; MMLU; GSM8K; instrucciones tipo asistente; chain-of-thought; benchmark de conocimiento; T3/memoria externa; test-time K extra; CYCLE; fine-tuning conversacional; selección manual de generaciones "bonitas". Razonamiento debe ser una unidad posterior.
>
> ## 16. Principio de decisión
>
> La unidad debe poder producir una respuesta incómoda. Si OMEGA falla, no se cambia: sampling; prompts; thresholds; seed; K; longitud; scoring; para rescatarlo. Y si pasa, tampoco se eleva el gate retrospectivamente porque MiniMind se vea mucho mejor. La pregunta es únicamente: ¿Existe ya un LM pequeño funcional dentro de OMEGA que justifique construir T3 encima? Ese es el gate.
>
> Un punto que espero que Opus ataque es el nivel de los thresholds 0.60/0.55 y el gate humano 3.0; los fijé antes de ver resultados, pero no pretendo que sean incuestionables. También quiero que audite si 12 prompts humanos bastan y si el criterio para distinguir "entrenar más" de "escalar" está suficientemente identificado.
>
> No lanzaría ninguna inferencia de esta unidad hasta tener su propuesta independiente y hacer el cruce adversarial.

---

## Ronda 1 — Propuesta independiente original de Opus (Paso C, previa a la aclaración del usuario sobre la dinámica — Opus escribió esto ANTES de saber que su rol iba a ser solo de crítico, sin ver la propuesta de Sol)

Texto completo, verbatim:

> # Protocol proposal: does the OMEGA K4 core have real language competence? (Gate before T3)
>
> Independent architect proposal. This is written from first principles. I read two source files only to pin down the architecture (see §0), and I deliberately did not read any MD/ or Sol material.
>
> ---
>
> ## 0. Two facts from the code that change the framing
>
> The brief describes the model as a "~7M shared MLP core". The reference implementation says something more specific (`omega_core_lm_0_r1_fable_optimization_proposal/omega_fast.py`, and `omega_core_lm_0_autoregressive_generation_audit/run_autoregressive_generation_audit.py:150`, which uses `dimension=128, slots=8`).
>
> 1. **It is a pure recurrent model with a fixed-size state.** The "attention" (`SlotMix`) runs across the 8 slots of the state, not across tokens. All information about the past passes through a single state of 8×128 = 1,024 floats, updated once per token by K shared rounds. So functionally it is an RNN with a K-round shared-weight transition. It has no token-level attention and no KV cache.
> 2. **Most of the "7M parameters" is the tied vocabulary embedding, not the recurrent core.** 50,257 × 128 ≈ 6.43M parameters. The prelude, the shared block and the output projection together are only about 0.6M. (There is also a factorized ER32 vocab variant; which variant the selected K4 checkpoint uses must be stated explicitly.) Two consequences follow:
>    - A large, well-trained tied embedding plus a weak recurrence can reach decent teacher-forced NLL while behaving roughly like a smoothed bigram/trigram model. **The most important floor is therefore an n-gram model trained on the same tokens, not MiniMind.**
>    - For T3, the question is not "does it write nice prose". It is **"can a 1,024-float recurrent state absorb information from its context and use it later?"** Whatever T3's memory interface turns out to be, the retrieved content must pass through that state (or be read by the slot-mix). If the state cannot carry in-context information, T3 has nothing to plug into.
>
> A side note: a directory named `omega_core_lm_0_autoregressive_generation_audit` already exists with a generation script. The brief says no generation has been produced. Either way, that script must pass the L0 harness-validity tests below before any output from it is trusted.
>
> ---
>
> ## 1. What to measure: five levels (L0–L4)
>
> I call them levels L0–L4 to avoid clashing with the roadmap's "T3". Each level gates the ones after it. Levels L1–L4 are always interpreted **relative to a fixed ladder of reference arms** (§4), never in absolute terms. A 7M model trained on about 2–3M tokens of Wikipedia will read as poor text in absolute terms no matter what. An absolute "is it coherent?" gate would almost certainly fail and tell us nothing.
>
> ### L0 — Harness validity (blocking; if it fails, no conclusions are drawn)
>
> - **Incremental-equals-batch equivalence.** Feed 20 held-out 1,024-token sequences through the generation path one token at a time, carrying the state, crossing the 256 window boundaries exactly as training does. The logits must match the batched `forward_window` teacher-forced logits to within the same tolerance used for the native/PyTorch cross-validation. The cheapest way to wrongly kill a model is a state-handoff bug in the generator (a state reset per window, a wrong detach, a zero state instead of the carried one).
> - **Scoring self-consistency.** Mean NLL of the harness on the held-out set must reproduce the already-reported K4 held-out NLL.
> - **Same harness, known models.** Run the teacher (DistilGPT2) and the n-gram model through the identical sampling code. The teacher must produce visibly fluent text. This proves the sampler, detokenizer and prompt handling are correct.
> - **Determinism.** A fixed seed gives byte-identical output.
>
> ### L1 — Non-degeneracy under free-running generation (automatic)
>
> This checks whether the model can feed on its own outputs without collapsing:
> - repetition loops;
> - vocabulary collapse;
> - drift that grows with position (exposure bias);
> - breakage at the 256/513 boundaries it was trained on.
>
> ### L2 — Local grammatical competence (automatic, likelihood-based, no decoding choices)
>
> Scored with **BLiMP** minimal pairs: 67 paradigms, subsampled to 200 pairs each ≈ 13.4k pairs. This is the cleanest syntax probe available at this scale. It depends on neither the decoding settings nor raters, and published n-gram and small-LM numbers exist for sanity-checking. Report the overall score plus per-phenomenon results, especially agreement and filler-gap. Those long-dependency phenomena are exactly what an n-gram cannot do and a working recurrence should.
>
> ### L3 — Short-span coherence of free text (blind rated, plus automatic support)
>
> - Does each sentence follow from the previous one?
> - Does the text stay on the prompt's topic across about 100 words?
> - Does it avoid contradicting itself?
>
> ### L4 — In-context information use (automatic, likelihood-based; the T3-relevant level)
>
> 1. **Induction/copy probe.** Take a random sequence of 50 tokens (drawn from the 5k most frequent tokens) and repeat it after a gap of G filler tokens, with G ∈ {0, 150, 350}. G=350 forces the second copy across a 256 window boundary. Metric: the drop in NLL on the second copy relative to the first. Use natural-text repeats as well as random-token ones.
> 2. **Counterbalanced entity binding.** 200 templated items, each with a swapped twin. An item is scored correct only if the model prefers the right answer in *both* versions. This removes name-frequency bias; chance is 25%. Example pair:
>    - "Anna and Mark went to the market. Anna bought apples and Mark bought bread. When they got home, the apples were put away by" → compare P(" Anna") vs P(" Mark").
>    - Twin: "…Mark bought apples and Anna bought bread. … the apples were put away by" → the correct answer is now " Mark".
>
>    A second template uses a fact/number: "The Harlow Bridge was completed in 1874, and a second bridge was added in 1911. [~40 filler tokens] The Harlow Bridge was completed in" → compare " 1874" vs " 1911"; the twin swaps the dates.
> 3. **Context-benefit Δ (a direct pre-T3 proxy).** Target: a 64-token held-out Wikipedia passage. Prefix it with either (a) a 200-token passage from the *same article* (relevant) or (b) a 200-token passage from a random other article (irrelevant, same length). Δ = NLL(target | irrelevant) − NLL(target | relevant). This measures, in miniature, exactly what T3 is supposed to exploit: better prediction when relevant text is supplied. If Δ≈0 for OMEGA while the teacher shows a clear Δ, the state does not carry usable content, and a memory bank feeding this core is unlikely to help.
>
> **Reasoning and instruction-following: do not test.** This is a base LM with no instruction data, trained on a few hundred Wikipedia documents. The expected score is chance, and a chance result carries no information about architecture or training. The only "reasoning-like" item I would keep is trivial in-context pattern continuation, scored by likelihood as part of L4 (e.g. "Monday, Tuesday, Wednesday," → " Thursday"; "3, 6, 9, 12," → " 15").
>
> ---
>
> ## 2. Generation protocol
>
> ### Checkpoints
>
> - **Primary:** K4, multichunk recipe, 2,000 updates. **All 5 seeds** are used for the automatic metrics, since seed variance is part of the result. For blind rating, use **one pre-registered seed: the seed with the median held-out NLL**, chosen before any text is looked at.
> - **Internal ablation:** K1 checkpoint(s) from the **same recipe**. If the K sweep was run under the old-513 recipe, say so and treat K1-vs-K4 as confounded. The question here is whether recurrence depth buys generation quality, not just NLL.
> - **Optional:** a CE-only OMEGA checkpoint, if one exists (the repo has an `omega_ce_only_baseline` campaign). It tests whether distillation helps free-running text.
>
> ### Prompts: 140 total, fixed and pre-registered before generation
>
> - **100 in-domain prompts** from the held-out split (WikiText-2 test/validation documents that were never trained on): the first 32–48 tokens of a paragraph, cut at a sentence or clause boundary. Stratify: 40 biography/history, 30 geography/places/structures, 30 culture/media/science. The true continuation is kept as the human reference arm.
> - **40 hand-written generic prompts** in Wikipedia register, but on topics that are not in the training documents. These test generalization rather than recall. Examples:
>   - "The river flows north through a narrow valley before it reaches"
>   - "In 1998, the band released their second album, which"
>   - "The town is known for its annual festival, held every summer in"
>   - "She was born in a small village near the coast and later moved to"
>   - "The building was designed by a local architect and completed in"
>   - "During the war, the regiment was stationed in"
>   - "The species is found mainly in tropical forests, where it feeds on"
>   - "After the election, the new government announced that"
>   - "The game received mixed reviews from critics, who praised its"
>   - "The first bridge across the river was built of wood, but"
> - **Formatting:** WikiText has artifacts (" @-@ ", " @,@ ", "= = Heading = ="). Run **one shared detokenizer** over all arms, including the human reference, before rating. Tell raters that headings may appear.
>
> ### Decoding (identical across all arms, no repetition penalty or other anti-degeneration hacks)
>
> Repetition penalties would hide exactly the pathology we are trying to detect.
>
> | Setting | Purpose | Samples per prompt | Length |
> |---|---|---|---|
> | **Primary: nucleus top-p=0.9, T=1.0** | Standard for small base LMs (Holtzman et al.); what raters see | 3 | 128 tokens |
> | Pure ancestral sampling, T=1.0 | Faithful view of the learned distribution; used for MAUVE and the drift curve | 3 | 512 tokens (+ a 1,024-token stress subset of 20 prompts) |
> | Greedy | Loop detection: onset position and loop rate | 1 | 256 tokens |
> | Sensitivity: top-p 0.9 at T=0.7 | Reported only, not used for the gate | 1 | 128 tokens |
>
> Why these values:
> - T=1.0 nucleus keeps the model's own distribution and removes only the tail.
> - Greedy exposes whether the model's argmax dynamics fall into attractors. That matters in particular for a recurrent state that might converge to a fixed point.
> - 128 tokens (~90 words) is long enough for 3–5 sentences of coherence and short enough to rate reliably.
> - 512 and 1,024 tokens deliberately cross the 256 BPTT boundary and the 513 training horizon, to check that state carry works in free-running mode.
>
> Compute is trivial for OMEGA and n-gram generation on the i7. Scoring with a ~0.5–1B model on CPU for about 1.5M tokens takes on the order of an hour.
>
> ---
>
> ## 3. Metrics
>
> ### Automatic metrics (L1, L2, L4, plus L3 support)
>
> 1. **Degeneracy:**
>    - rep-4: the fraction of 4-grams in a continuation that duplicate an earlier 4-gram in the same continuation;
>    - **loop rate**: the share of samples containing a span of 3 or more tokens repeated at least 3 times consecutively, plus the greedy loop-onset position;
>    - distinct-1 and distinct-2 across all samples;
>    - unigram entropy of the generated tokens against the reference text.
> 2. **Generative perplexity under an independent scorer.** Do *not* score with DistilGPT2: the student was distilled toward it, so teacher-scoring flatters the student. Use a different family, e.g. Pythia-410M/1B or Qwen2.5-0.5B, run on CPU. **Always report it next to entropy/distinct-n**, because low generative perplexity can be gamed by repetition.
> 3. **Drift curve.** Mean scorer NLL per 32-token bin over positions 1–512 of the T=1 samples, with the 256 boundary marked. The pathology signature is an OMEGA slope clearly steeper than the teacher's own slope, or a step at 256.
> 4. **MAUVE** (T=1 samples against the human continuations, ~420 samples per arm; scorer features). Secondary only, because the sample count is modest.
> 5. **Memorization check.** Only a few hundred training documents exist, so fluent-looking text could be copied. Report the fraction of generated 8-grams that occur verbatim in the training corpus and the longest common substring with it, for every arm. Human text is the reference level.
> 6. **BLiMP accuracy** (L2) and the **three L4 probes** (induction NLL drop at each gap, strict entity-binding accuracy, context-benefit Δ).
> 7. **Data-limitation diagnostic.** Train-vs-held-out NLL gap for K4, computed from existing numbers. It decides whether "scale up" should mean more parameters or more data (§5).
>
> ### Are automatic metrics trusted for coherence? No.
>
> They are reliable for degeneracy, distribution mismatch, syntax (BLiMP) and context use (L4). The gates for those levels are automatic. For L3 coherence, automatic metrics are only supporting evidence, and **blind rating decides L3**.
>
> ### Blind rating protocol (L3)
>
> **Raters.** Be realistic: the project has one human and some AI agents.
> - Primary raters: **two LLM judges from different model families** (Claude and GPT), each in a fresh context and blind to arm identity.
> - The human rates a stratified random **25% subset**.
> - Report agreement: Spearman for the scales, κ for the pairwise choices. If human–LLM agreement on the pairwise outcome is below κ=0.4, the human ratings alone decide, and the rated subset is enlarged.
>
> **What raters see.**
> - Prompt plus 128-token continuation, detokenized, with the arm hidden and the order shuffled.
> - Rated pool: 60 prompts (40 in-domain + 20 generic) × 1 sample × arms {human reference, teacher, OMEGA-K4, OMEGA-K1, 5-gram, matched dense control if trained, MiniMind if admitted per §4}.
> - **Calibration controls mixed in:** 10 word-shuffled human continuations, which are known to be bad.
>
> **What raters are asked.**
> 1. **Absolute scores, 1–5**, each with a written rubric and anchor examples:
>    - grammaticality (sentence by sentence);
>    - local coherence (does each sentence follow the previous one?);
>    - topicality (is it still about the prompt's subject?);
>    - consistency (no self-contradiction);
>    - an overall binary "degenerate?" flag.
> 2. **Pairwise forced choice**, "Which continuation is the more coherent piece of English text?", for OMEGA-K4 vs each of: 5-gram, K1, dense control, teacher. Use 60 prompt-pairs per comparison and randomize left/right. Pairwise choices are more reliable than absolute scales for weak systems.
>
> **Rater validity check.** The ratings count only if the raters order human > teacher > 5-gram > shuffled with non-overlapping bootstrap CIs. If they cannot order the anchors, they cannot rate OMEGA either.
>
> ### Statistics
>
> - The unit of analysis is the prompt; every comparison is paired on the same prompts.
> - Use a hierarchical bootstrap: prompts, and model seeds where there are several. Report 95% CIs.
> - **Pre-register exactly five primary endpoints**; everything else is descriptive:
>   - (P1) loop rate at the primary decoding;
>   - (P2) BLiMP accuracy, OMEGA-K4 minus 5-gram;
>   - (P3) pairwise win rate of OMEGA-K4 vs 5-gram;
>   - (P4) context-benefit Δ;
>   - (P5) induction NLL drop at G=350.
> - Apply a Holm correction across the five.
>
> ---
>
> ## 4. Reference ladder, and what to do with MiniMind
>
> Every OMEGA number is read by its position between a floor and a ceiling, using the same tokenizer, prompts, decoding and harness:
>
> | Arm | Role | Cost |
> |---|---|---|
> | **Kneser–Ney 5-gram** on the exact training token stream (GPT-2 BPE) | **Floor.** "Learned nothing beyond local statistics." Matters most because ~90% of OMEGA's parameters are the embedding | Minutes, no neural training |
> | **Human continuation** | Top anchor for rating and memorization levels | Free |
> | **DistilGPT2 (teacher)** | Ceiling. Same tokenizer; shows what the distillation target looks like when free-running | Free |
> | **OMEGA-K1** (same recipe) | Does recurrence depth matter for generation? | Free (exists) |
> | **Matched dense control** | Discriminates "the OMEGA core is broken" from "this parameter/data budget is just this weak" | One new training run, 3 seeds (see below) |
> | MiniMind | Optional, outside the decision rule | Smoke test first |
>
> ### MiniMind: demote it; do not let it enter the decision rule
>
> Scale (~9×) and base-vs-chat mismatch are real problems, but two further problems make it close to useless as a gate:
> - **Tokenizer.** MiniMind uses its own small vocabulary (reportedly ~6.4k tokens), so NLLs are not comparable. The only valid likelihood comparison would be bits-per-byte on identical raw text.
> - **Training language.** To my knowledge, MiniMind's training data is predominantly Chinese. Its English continuation ability may be poor, in which case a "comparison" measures its data, not OMEGA.
>
> Procedure:
> 1. Establish whether the checkpoint is pretrain-only or SFT/chat.
> 2. Run a 10-prompt English continuation smoke test through its native tokenizer.
> 3. Admit it **only** if it is a base (or usable-as-base) checkpoint *and* its English continuations are fluent. If admitted, it goes into the blind rating pool as a descriptive anchor ("a known-coherent small dense LM"), with bits-per-byte reported, and its results appear in the report but not in any gate.
> 4. If it is chat-only, or not fluent in English, drop it without regret.
>
> The better "what does a coherent small model look like" anchor is the teacher, which shares tokenizer and domain.
>
> ### New training: one run, argued as necessary
>
> **A matched dense control:**
> - a small causal Transformer (or, even more tightly matched, a single-layer GRU/LSTM with the same 1,024-float state);
> - the **same tied embedding size**, same non-embedding parameter count (~0.6M), same data, same multichunk recipe, same distillation loss, same 2,000 updates;
> - 3 seeds.
>
> Why it is necessary: without it, the decision "(b) scale" vs "(c) the training/architecture is broken" cannot be made. A poor OMEGA result is otherwise uninterpretable, because we would not know whether this budget *ever* produces coherent text. Its cost is roughly one arm of the K sweep that was already run.
>
> If it truly cannot be afforded, the protocol still runs using the floor/ceiling ladder, but outcome (c) then carries the caveat "not separated from budget limits".
>
> ---
>
> ## 5. Decision rule
>
> Evaluate in order; stop at the first match.
>
> **(D) Harness invalid.** L0 fails. Fix the harness. Nothing about competence may be concluded.
>
> **(C) Go back to the training objective or procedure.** Triggered by any one of the following:
> - **Free-running pathology despite good teacher-forced NLL:**
>   - loop rate at the primary decoding above 20% of samples, or more than 2× the dense control/teacher; or
>   - a drift-curve slope or 256-boundary step significantly worse than the dense control.
>
>   Interpretation: exposure bias or a state-handoff/state-dynamics problem (e.g. the state converging to an attractor). Investigate before anything else: state norms over long rollouts, scheduled sampling or a small free-running fine-tune, the handling of detach and state carry at the boundary.
> - **No advantage over the n-gram floor.** BLiMP not significantly above the 5-gram (P2 CI includes 0), *and* pairwise win rate vs 5-gram not significantly above 50% (P3).
>
>   Interpretation: the recurrence adds nothing beyond what the embedding table encodes. Check whether the state is being used at all, e.g. with a state-ablation NLL test that resets the state every N tokens.
> - **Significantly worse than the matched dense control** on P2 or P3.
>
>   Interpretation: the problem is specific to OMEGA, not to the budget.
>
> **(B′) Add data before adding parameters** (the option missing from the brief). The model is healthy (no (C) trigger, beats the floor), but the train/held-out gap is large, or held-out NLL is still improving at 2,000 updates. The current regime is then data-limited: a few hundred documents is tiny for a 50k-vocab embedding. Scaling parameters first would just overfit more. Next step: more tokens (e.g. WikiText-103 scale) with the same 7M model, then re-run this protocol. It is cheap on CPU and changes no architecture.
>
> **(B) Scale the core before T3.** The model is healthy, beats the floor, is not worse than the dense control, and is not strongly data-limited, **but L4 fails**:
> - induction drop at G=350 below 25% of the teacher's (P5), or
> - context-benefit Δ CI including 0 (P4).
>
> Interpretation: the 1,024-float state cannot carry usable content, which is exactly T3's prerequisite. Scale the **state** (slots × D), not just K, and re-test L4.
>
> **(A) Proceed to T3.** All of the following:
> - L0 passes;
> - no (C) trigger;
> - P2 and P3 significantly beat the 5-gram floor (CIs above 0 and above 50% respectively);
> - OMEGA is not significantly worse than the dense control;
> - L4 shows **positive, CI-bounded** context benefit (P4) *and* copy ability across the 256 boundary (P5 at least 25% of the teacher's drop).
>
> Absolute text quality is not required to be good. I expect the text to be locally grammatical-ish and globally weak, and that is acceptable at this size.
>
> **(E) K4 does not beat K1 in generation even though it beats it in NLL** (checked alongside any outcome). Record it as a finding: the NLL benefit of depth does not transfer to free-running generation, so recurrent depth should not be credited for generation competence. It does not block T3 by itself, but K becomes an open question again when T3 is designed.
>
> ---
>
> ## 6. Where I would push back on the framing
>
> 1. **"Does it have language competence?" is the wrong gate question for T3; "can it use supplied context?" is the right one.** A 7M model with ~0.6M non-embedding parameters trained on WikiText-2-scale data will not produce convincing text under any objective. Gating on absolute coherence would stop the project for a reason that is really data and scale, and it would not predict whether T3 works. What *does* predict T3 is L4: does prediction improve when relevant information is placed in the context, and does that information survive in the state? That is why L4 is part of the proceed criterion.
> 2. **The "T3 might mask a broken core" worry is better solved by a permanent control than by a one-shot gate.** Every T3 evaluation should include (i) memory disabled and (ii) memory filled with irrelevant or shuffled entries. If T3 gains vanish under the shuffled-memory control, or the core without memory regresses, you find out immediately. This protocol's L1/L4 battery should become that standing regression suite.
> 3. **The "7M core" label is misleading and should be corrected in project documents.** The shared recurrent block plus prelude and readout is ~0.6M. The tied embedding is ~6.4M, and at inference the V×D output projection (~25 MB fp32) dominates per-token compute and memory traffic, more than the core does. This has direct consequences for:
>    - **the cache-residency thesis:** the core fits in L2, but the unembedding does not fit in L2 and barely fits in L3;
>    - **"scale the core" decisions:** they should scale state/core width, not count embedding parameters;
>    - **every comparison to other models:** report embedding and non-embedding parameters separately.
> 4. **Teacher-forced NLL is not uninformative.** Of everything listed here, generation adds mainly the exposure-bias/state-dynamics check (L1) and the human coherence read (L3). The whole unit should be bounded to roughly 1–2 weeks. It must not become a new open-ended campaign.
> 5. **The existing autoregressive-generation-audit directory** should be reconciled with the claim that "no generation has ever been produced". If outputs from it exist, they are useful pre-data. They should still not be used to tune the protocol, which should be frozen before anyone looks at them.
>
> ---
>
> ## 7. Execution order (all CPU)
>
> 1. **Freeze first:** the prompt lists (both sets), decoding settings, the five primary endpoints, the thresholds in §5, and the rubric and anchors. Hash them.
> 2. L0 harness validation, using or fixing the existing generation-audit script.
> 3. Build the 5-gram model. Start the matched dense control training (3 seeds) in the background.
> 4. Automatic runs: L1 generation and metrics, L2 BLiMP, L4 probes, for all arms.
> 5. Blind rating packet (L3): LLM judges plus the human's 25% subset, then the anchor-validity check.
> 6. Apply the §5 rule mechanically, and write up the result, including any outcome that is inconvenient.

---

## Ronda 1 — Crítica de Opus a la propuesta de Sol v0.1 (Paso D)

Texto completo, verbatim:

> # Critique of Sol's proposal "OMEGA-MINIMUM-LANGUAGE-COMPETENCE v0.1"
>
> Section numbers below refer to Sol's proposal unless marked otherwise. "Mine" means my own proposal, levels L0–L4. The critique is adversarial as requested. Where she is right and I was wrong, I say so plainly.
>
> ---
>
> ## 1. Headline verdict
>
> The procedural side of her proposal is sound, in places stronger than mine:
> - pre-freezing everything, including a train-set contamination check on the prompts;
> - evaluating all five seeds;
> - an acquisition curve across checkpoints;
> - a principle that the human gate cannot be offset by good automatic scores;
> - an explicit list of things that do not belong in this unit.
>
> **Her measuring instruments, however, have a construct-validity problem that could make the gate pass or fail for the wrong reasons.** Three defects combine:
> 1. The contrastive items can largely be solved without using the context.
> 2. The only negative control is a random-init model, which any trained model beats trivially.
> 3. All prompts are everyday narrative, while the model was trained only on encyclopedic Wikipedia text.
>
> As written, a Kneser–Ney n-gram could plausibly pass L0, and a model that genuinely learned Wikipedia-register language could fail the human gate. These are fixable, and the fixes are cheap.
>
> ---
>
> ## 2. The embedding/core split: her framing does mislead the design
>
> This is now verified (D=128, S=8, V=50257): ~6.43M of the parameters are tied embedding and ~0.6M are the recurrent core. Her proposal says "~7M" throughout (§0, §12A, §13), and this matters in three concrete places.
>
> ### 2a. §12B and §13: "scale ~20–30M keeping architecture and recipe" is mostly buying vocabulary rows
>
> - With the current architecture, parameters ≈ 50,257·D + ~36·D² (block 12D², prelude 2S·D², readout S·D², with S=8).
> - 25M total means D≈390, of which ~19.6M is embedding and ~5.5M is core.
> - About 80% of the "scale-up" would go to the lookup table, not to the recurrent computation whose capacity the scale-probe is meant to test.
> - It also triples the per-token V×D unembedding cost, which already dominates inference compute and memory traffic, not the "cache-resident" core.
>
> If the question is "is there a capacity bottleneck in the core", the scale-probe must grow the core and state (D_core, slots) while holding the vocabulary cost roughly fixed. The repo already has the tool for this: the factorized-vocabulary ER32 variant (`FactorizedVocabulary`). A second reading of "keeping the architecture" would scale slots S alone, since the state is the bottleneck.
>
> ### 2b. §11 and §12B: MiniMind is much further away than "9×"
>
> - If MiniMind uses its reported ~6.4k-token vocabulary, its embedding is a few million parameters and almost all of its ~65M is non-embedding.
> - Non-embedding parameters are therefore ~60M against OMEGA's ~0.6M: roughly **100×**, not 9×.
> - Her §12B rule is "LateGain < 0.02 **and** MiniMind shows clear separation → SCALE_PROBE_JUSTIFIED". The second condition is effectively guaranteed at a ~100× compute gap, so it adds no information and the rule collapses to "LateGain < 0.02 → scale".
> - §13 claims the MiniMind calibration "demonstrates the instrument can measure the missing capacity". It demonstrates only that a ~100× larger model trained on far more (and different) data scores higher, which nobody doubted.
>
> ### 2c. Why the split matters for what the gate is testing
>
> With ~90% of parameters in the embedding table, the most likely failure mode is **"a smoothed n-gram in disguise"**. Her negative control (update-0, §1) cannot detect that failure mode. A trained unigram model would already beat update-0 by a wide margin. See §3.
>
> ---
>
> ## 3. The contrastive banks (§4, §5): right idea, weak instrument
>
> She is right that likelihood-based forced choice is a clean, decoding-independent measurement. My L2 (BLiMP) and L4 probes use the same principle. Her L1 also covers something BLiMP does not: discourse-level coherence (entity persistence, temporal order, causality). I left that to human rating only, and an automatic handle on it is valuable. That idea should survive into the joint protocol. The implementation has five problems.
>
> ### 3a. Most example items can be solved without the context
>
> - **§4.1 D:** "switched on the" → " lamp" vs " sandwich". A bigram/trigram solves this.
> - **§4.1 C:** labelled "local reference", but "picked up the" → " key" vs " weather" is decided by selectional restriction (you do not pick up weather), not by reference to the earlier sentence. A real reference test would use a foil that is equally pickable and present in context ("picked up the book").
> - **§5 C:** " the glass floating above the counter" is implausible in any context.
> - **§5 D:** the foil "stayed perfectly dry under his umbrella" is detectable locally.
> - **§5 B:** the foil is a long, odd construction whose improbability does not depend on the story.
>
> **Fix: counterbalanced twins.**
> - Each item gets a twin in which the context is minimally edited so the foil becomes correct. Example: "Mara owns a **blue** bicycle. Theo owns a **red** bicycle…" → " still blue" is now preferred.
> - Credit goes only to pairs where both twins are correct. Chance is 25%, and context-free continuation priors score at most 50% by construction.
> - This is the design I proposed for entity binding in my L4. It should become the standard for every L1 category, and for L0-C.
>
> ### 3b. Required diagnostic: context-ablated scoring
>
> - Score every item a second time with the prompt truncated to its last 3 tokens, and once more with an empty prompt.
> - Items still solved under truncation measure lexical priors, not context use.
> - Report accuracy on the context-necessary subset separately; ideally the gate applies to that subset.
>
> ### 3c. The negative control must be an n-gram floor, not only update-0
>
> - Update-0 (§1, §4.3 ΔL0 > +0.10, §5 ΔL1 > +0.05) shows only that training did something. A random-init model is near 50% on any balanced bank, and any trained model beats it.
> - Add a unigram and a KN 5-gram, both trained on the exact training token stream, to the L0/L1 banks. It costs minutes on CPU.
> - **If the 5-gram clears L0 > 0.60, the L0 gate has no evidential value for "language competence beyond local statistics", whatever OMEGA scores.** I consider this likely for her categories A and D.
> - Keep update-0 as a pipeline sanity check. It is not a baseline.
>
> ### 3d. Scoring rule (§4.2)
>
> - Mean per-token log-prob over the whole continuation, including a shared suffix like " on the table.", dilutes the discriminating span. It also interacts with length differences between preferred and foil, which her L1 examples have in abundance.
> - Better: score the **summed log-prob over the differing span only**, with shared suffixes excluded and lengths matched by construction.
> - Report mean-normalized scores as a secondary.
> - Cross-tokenizer comparisons (MiniMind) should use per-byte or per-word normalization, never per-token. Per-token means are not comparable across different tokenizers.
>
> ### 3e. Statistics: the CI ignores the dominant variance source (§4.3)
>
> - A t-CI90 over 5 seeds that are all scored on the **same 96 items** measures seed variance only.
> - With 96 items at p≈0.65, the item-sampling standard error is ≈0.049 for binary scoring, larger than the likely between-seed spread.
> - Her CI will therefore be overconfident about what the model would score on a different draw of items, which is the quantity the gate is actually about.
> - **Fix:** a two-way bootstrap that resamples items and seeds. Once items are included, 24 items per category is too few for per-category claims. Either report categories descriptively only, or raise the banks to ≥48 twin-pairs per level.
>
> ### 3f. Register mismatch
>
> - §4.1 asks for "everyday vocabulary", and §7.1 uses everyday narratives (toast, umbrellas, notebooks).
> - OMEGA saw a few hundred encyclopedic Wikipedia articles. Everyday narrative is out-of-distribution in both register and lexicon.
> - A failure would conflate "no language competence" with "never saw this genre". Distillation through DistilGPT2 soft labels transfers some general knowledge, but only as it appears on Wikipedia tokens.
> - **Fix:** stratify both banks and the generation prompts, half Wikipedia-register and half everyday. Pre-register that the gate is evaluated on the pooled set, and report both strata. An in-domain pass with an out-of-domain failure is itself a decision-relevant finding: it points at data, not architecture.
>
> ---
>
> ## 4. Free generation and human rating (§7–§10)
>
> ### 4a. Twelve prompts is not enough power
>
> - The effective sample size for a prompt-clustered bootstrap is about 12, since the 5 seeds on one prompt are strongly correlated (a hard prompt is hard for every seed).
> - Percentile bootstrap with 12 clusters under-covers noticeably; the CI lower bound will be too optimistic.
> - With a plausible between-prompt SD of composite H of ~0.6, SE ≈ 0.17. The lower-90 ≥ 3.0 gate then needs a true mean of about 3.3 or more.
> - The per-seed condition (§10, mean H ≥ 2.75 over 12 outputs) is noisier still. One unlucky seed fails the whole unit.
> - **Recommendation:**
>   - ≥48 prompts (24 Wikipedia-register, 24 everyday);
>   - each prompt generated by 2 of the 5 seeds on a pre-declared rotation, giving 96 OMEGA outputs with all seeds represented ~19 times;
>   - drop the per-seed gate, or make it descriptive;
>   - use a BCa bootstrap over prompts.
> - The rating load stays similar to hers because each prompt carries fewer seeds.
>
> ### 4b. Raters: three independent humans is probably not feasible here
>
> - The project has one human. "Three independent human raters" either will not happen or will mean informal helpers.
> - A realistic, defensible alternative:
>   - the one human rates everything, or a stratified ≥50% subset;
>   - two LLM judges from different model families rate everything, blind;
>   - human–LLM agreement is reported;
>   - the human's ratings take precedence where they disagree.
> - Her §10 rule for low agreement (alpha < 0.4 → LOW_RATER_AGREEMENT, no FAIL on the human gate alone) is good and should be kept.
>
> ### 4c. Missing anchors make the 1–5 scale uncalibrated
>
> - Her rating pool contains OMEGA, MiniMind and update-0 only.
> - Update-0 output is random-token gibberish that every rater will score 1, so it tests nothing.
> - The informative negative anchor is **locally fluent but globally incoherent text**, i.e. 5-gram samples. That is exactly the failure mode the unit exists to detect. If raters cannot separate OMEGA from 5-gram output, H ≥ 3.0 says nothing about coherence.
> - Positive anchors are also needed:
>   - the true human continuation, for Wikipedia-register prompts;
>   - the teacher DistilGPT2 under identical decoding. It shares OMEGA's tokenizer and domain. Her preflight ban on teacher use concerns OMEGA's inference, not evaluation arms.
> - Pre-register a rater-validity condition: human > teacher > 5-gram > shuffled, with CIs. If it fails, the instrument has failed, not OMEGA.
>
> ### 4d. The composite H contradicts her own no-compensation principle
>
> - §10 rightly forbids distinct-n offsetting incoherence.
> - Yet H = mean(G, R, C, E, N) lets easy dimensions offset the dimensions the unit is actually about. Non-degeneration (N) and grammaticality (G) can offset Relevance, Coherence and Entity continuity (R, C, E).
> - A fluent, loop-free, topic-drifting model can reach H≈3.0 with C=E=2.
> - **Fix:** gate on the minimum of the dimension-level lower bounds for C and E (or on mean(C, E)). Keep G, R and N as descriptive or separate hard floors.
>
> ### 4e. What 96-token generations from short prompts never exercise
>
> Prompt plus output stays under about 150 tokens, so her generation stays entirely inside the first 256-token window. It never tests:
> - the state handoff across window boundaries;
> - drift beyond the 513-token training horizon;
> - the incremental-generation-equals-batch-forward equivalence.
>
> A state-carry bug in the generator, or state collapse over long rollouts, would be invisible, while any T3 use involves longer contexts. My L0 (logit equivalence) and L1 (512/1024-token drift curve with a boundary check) cover this. Both are cheap automatic checks and should be added. Her decoding settings (T=0.8, top-p 0.95, no repetition penalty) are reasonable, and I would accept them as primary in place of mine. A greedy diagnostic should be required, not optional, for loop detection.
>
> ### 4f. Degeneration gate (§9)
>
> - max_4gram_count ≥ 4 or repeat4 > 0.30 within 96 tokens, with ≤10% of outputs flagged, is a sensible pre-declared definition. I have no quarrel with the number.
> - Its weakness is scope: it is measured only at T=0.8 with nucleus sampling on short outputs, the setting least likely to show loops.
>
> ---
>
> ## 5. Are the thresholds well calibrated?
>
> **The real critique is that absolute thresholds on an uncalibrated instrument are the wrong tool, whatever the number. Pre-registration does not fix that.** Pre-registering 0.60 protects against moving the goalposts, but not against an arbitrary starting position. The meaning of "0.60" depends entirely on how hard her item writers made the foils, which nobody knows until anchors are run.
>
> Taking the numbers on their own terms:
> - **L0 > 0.60 (lower CI90):** probably too lenient, for the reasons in §3a/§3c. Lexically detectable foils mean an n-gram likely clears it.
> - **L1 > 0.55:** unknown. With counterbalanced twins it could be stringent; with the current foils it is lenient.
> - **H ≥ 3.0 on everyday narrative:** probably too strict for this data regime and possibly strict for the teacher at T=0.8. In her rubric, 3 means "basic local coherence, partially keeps the situation". For a 0.6M-core model trained on ~2–3M Wikipedia tokens, I predict H lands around 2–2.5 on everyday prompts regardless of whether the architecture is sound. **The gate would then fail for data and register reasons, and her §12B logic would misread that as a scale question.**
>
> **Proposed replacement (keep her discipline, change the reference frame):**
> - Freeze the numbers before any OMEGA output. Freeze them *relative* to anchors computed before OMEGA is scored: the n-gram floor, the teacher ceiling and the human reference.
> - Examples:
>   - contrastive gate = OMEGA's paired advantage over the 5-gram on context-necessary twin-pairs, with a CI excluding 0;
>   - human gate = OMEGA's C/E lower bound significantly above the 5-gram and not below a pre-set fraction of the teacher's gap (e.g. closing ≥30% of the 5-gram→teacher gap).
> - Keep one absolute sanity floor, the degeneration rate.
>
> This answers "is there a functional LM here beyond local statistics" in a way that does not depend on item-writer difficulty. In fairness, my own proposal also fixed some raw numbers (loop rate 20%, 25% of the teacher's induction drop). Those are exposed to the same critique, and I would convert them to the same anchored form.
>
> ---
>
> ## 6. Deferring the context/memory probes to T3: a real gap, but my L4 should not be a hard gate either
>
> **Her own §14 argument supports my position.** She says external memory should be added to "a core that already shows it can transform context into coherent language". Yet her longest context is about 40 tokens, and nothing tests whether information survives in the 1,024-float state over a few hundred tokens. That is exactly what text-based memory access would require. Three reasons the probes belong in this unit:
> 1. They cost about a day of CPU and no training.
> 2. The architecture has a hard, verified bottleneck (8×128 floats, no token-level attention). Whether that bottleneck can carry supplied content is the biggest unknown between now and T3.
> 3. **Without a pre-T3 baseline for context benefit, a T3 failure cannot be attributed** to the memory mechanism versus the core. T3 needs this baseline to be interpretable, so it should be measured now, while the core is frozen.
>
> **Where she is partly right:** T3's interface is not designed yet. If memory is injected per token directly into slots or read via slot-attention, it bypasses the prefix-text path that my context-benefit Δ measures. A zero Δ would then not doom T3. So my L4 should not be a blanket GO/NO-GO gate.
>
> **Joint position I would accept:**
> - L4 (copy/induction across the 256 boundary, counterbalanced entity binding at 40/150/350-token distances, context-benefit Δ) is **mandatory to run and report**.
> - It is a hard gate only in the extreme case: induction fails even at gap 0, *and* Δ is indistinguishable from 0. That would mean the state cannot hold content at all.
> - Otherwise it is a **binding design input** for T3. If Δ≈0 while copy at short gaps works, T3 must not deliver memory as prefix text; it needs a direct per-token path into the state.
>
> ---
>
> ## 7. MiniMind (§1, §11)
>
> I agree with her position, including that MiniMind failing must not lower OMEGA's bar. Additions:
> 1. Put the ~100× non-embedding ratio into the report, so nobody reads "OMEGA < MiniMind" as informative (§2b).
> 2. Check its English competence and base-vs-chat status before any run. If it is predominantly trained on Chinese, its failure on English everyday narrative is uninformative in the other direction too. Her §11 "investigate if MiniMind fails" handles this, but the check should be a preflight step, not an after-the-fact one.
> 3. Use per-byte normalization for its contrastive scores (§3d).
> 4. Remove it from the §12B scale rule entirely (§2b). The teacher, sharing tokenizer and domain, is the better calibrator for "can this instrument see coherence".
>
> ---
>
> ## 8. Escalation logic (§6, §12, §13)
>
> ### 8a. LateGain_L1 ≥ +0.02 is below the noise floor and confounded by the LR schedule
>
> - On 96 binary items, the paired difference acc₂₀₀₀ − acc₁₀₀₀ per seed has a standard error around 0.02–0.04. Even averaged over 5 seeds, a 0.02 threshold is roughly a coin flip.
> - More seriously, if the LR schedule decays toward 0 by update 2000 (typical warmup plus cosine or linear decay), late gains are suppressed by the schedule, not by a capacity plateau. The heuristic would then systematically report "plateau → scale" for a model that simply stopped being trained.
> - **Fix:**
>   - Use continuous signals that already exist: held-out NLL slope over updates 1000→2000, the train-vs-held-out gap, and the LR at each point.
>   - If ambiguous, run the one experiment that directly identifies the answer: continue K4 for another 2,000 updates with a re-warmed schedule, 2–3 seeds. It is cheap on CPU and settles "under-trained" versus "plateaued" empirically instead of by extrapolation.
>
> ### 8b. A branch is missing: data-limited, not step-limited or parameter-limited
>
> - A few hundred documents (~2–3M tokens) is tiny.
> - If the train/held-out gap is widening, "train longer" overfits and "scale to 25M" overfits more; roughly 20M of that would be embedding rows starving for data.
> - Her taxonomy (TRAINING_NOT_CONVERGED vs SCALE_PROBE_JUSTIFIED) needs a third outcome, **DATA_LIMITED**: same 7M model on WikiText-103-scale data, then re-test. Distillation makes this cheap: teacher logits can be computed on the fly or cached.
>
> ### 8c. The scale-probe must be core scale, not total scale
>
> - Per §2a, "~20–30M keeping architecture" should be restated as growing the core/state (slots and/or core width with a factorized vocabulary).
> - It should always be preceded by the DATA_LIMITED check.
> - Her principle "not straight to 65M; not decided just because 7M < 65M" is right and I endorse it.
>
> ### 8d. §12C is well built
>
> "L0 stuck near control despite reasonable NLL → investigate architecture/objective" is good. With an n-gram floor it becomes sharper: "L0/L1 not above the 5-gram" is the precise signature of "the recurrence contributes nothing beyond the embedding table". I would also add a state-ablation diagnostic: resetting the state every N tokens should hurt NLL substantially if the recurrence is actually being used.
>
> ### 8e. The conjunctive gate compounds false negatives
>
> - §12A requires L0, L1, automatic and human gates all at their lower CI bounds.
> - Four one-sided conditions, each designed to be conservative, compound into a substantially higher false-negative rate than any single one.
> - Conservatism is her stated intent (§16), which is defensible. The joint false-fail probability should still be acknowledged, and the weakest-powered component (the 12-prompt human gate) should not be the one that most often decides.
>
> ---
>
> ## 9. What her proposal gets right that mine missed or underweighted
>
> 1. **Acquisition curve across checkpoints (§6).** I did not ask for intermediate checkpoints at all. It is the right instrument for train-more versus scale, provided the LR schedule is accounted for (§8a).
> 2. **Automatic discourse-level contrastive items (§5).** This covers ground BLiMP does not. It should be adopted, rebuilt as counterbalanced twins.
> 3. **All five seeds in the human pool (§8).** Better than my median-seed rule for robustness, though my seed was also fixed in advance.
> 4. **Contamination check on prompts against the training corpus (§2).** I only measured memorization after the fact. Checking up front is better.
> 5. **Explicit non-compensation principle (§10)** and the **"must be able to produce an uncomfortable answer" principle (§16).** Both belong in the joint protocol verbatim, with §4d above applied so that her composite does not break her own principle.
> 6. **FP32, raw-logit saving, weight-immutability check (§2).** Good hygiene I left implicit.
>
> ## 10. Where my design is stronger and should be kept
>
> 1. **N-gram floor and matched dense control.** Without them no gate can separate "OMEGA is a disguised n-gram" or "this budget is simply this weak" from real competence. Update-0 cannot do this.
> 2. **Harness equivalence (incremental = batch logits) and long-rollout drift across the 256/513 boundaries.** Her protocol cannot detect a state-carry bug or state collapse.
> 3. **An independent scorer** (not the distillation teacher) for generative perplexity.
> 4. **Context-use probes (L4)**, reframed as in §6.
> 5. **Anchored rating pool** (human reference, teacher, 5-gram, shuffled) with a rater-validity check.
> 6. **Wikipedia-register prompts** matching the training distribution.
> 7. **An explicit data-limitation branch.**
>
> ## 11. Suggested joint synthesis, in one paragraph
>
> **Freeze:** in-domain and everyday stratified prompts and banks, contamination-checked.
>
> **Stage 0:** harness equivalence and a long-rollout sanity check.
>
> **Stage 1 (contrastive):**
> - Sol's L0/L1 categories plus BLiMP-sub, all built as counterbalanced twins and scored on the differing span.
> - Arms: update-0, unigram, 5-gram, OMEGA ×5 seeds, K1, teacher (MiniMind per-byte, descriptive only).
> - Two-way bootstrap over items and seeds.
> - Gate: OMEGA significantly above the 5-gram on context-necessary pairs.
>
> **Stage 2 (generation):**
> - ≥48 prompts, Sol's decoding settings, 2 seeds per prompt on rotation.
> - Rating pool with human, teacher, 5-gram and shuffled anchors.
> - Rated by 1 human plus 2 LLM judges from different families.
> - Gate on C/E (not the composite): above the 5-gram and closing a pre-set fraction of the gap to the teacher, plus the ≤10% degeneration floor and a greedy loop diagnostic.
>
> **Stage 3 (context use):** run L4 mandatorily. Hard gate only in the extreme case; otherwise it is a binding T3 design input.
>
> **Escalation:**
> - DATA_LIMITED / UNDER_TRAINED / CORE-CAPACITY, identified by the NLL curve plus LR schedule plus train-val gap.
> - A cheap continuation run when that is ambiguous.
> - Any scale-probe grows the core/state, not the vocabulary rows.

---

## Ronda 2 — pendiente

El usuario va a manejar personalmente el envío del siguiente mensaje a Sol (vía la web de ChatGPT), dado que el puente automático (`chatgpt_send_message`) falló 5 veces seguidas con "send button did not become available" incluso con mensajes cortos. Cuando el usuario obtenga la respuesta de Sol (v0.2, incorporando o rechazando la crítica de Opus arriba), pegarla acá y en el chat para continuar el ciclo.

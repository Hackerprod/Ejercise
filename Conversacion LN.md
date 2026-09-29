### Claude Opus 5

Leí `Conversacion.md` (el contrato de diseño), recorrí la bitácora `T1.5_Spec_MIX_O.md` hasta el Addendum 246, y verifiqué los números contra el código y los artifacts del repo actualizado. Mi lectura, sin suavizar.

## Lo que verifiqué directamente

Arquitectura real (`omega_fast_candidate.py`, `OmegaCoreLMFast`):

```
embedding atado    50257×128 = 6,432,896   (91.5%)
prelude            256→1024  =   263,168
bloque compartido  qkv+out+fc1+fc2         =   197,888
output_projection  1024→128  =   131,072
depth+gates+norms                          =     3,072
                                  núcleo   =  ~595,200
```

Calidad en `omega_expanded_frozen_validation` (WikiText-2 val, 60 docs, 30720 tokens, update 2000):

```
R1  K4  NLL 6.6139  →  PPL  745
R1  K1  NLL 6.6483  →  PPL  772
ER32 K4 NLL 6.9828  →  PPL 1078
```

Presupuesto de entrenamiento: 2000 updates × batch 8 × window 256 = **4.1M tokens**. Teacher: `distilgpt2`.

Y la auditoría de generación autorregresiva, 64 continuaciones greedy:

```
unique_token_proportion  media 0.0913   (≈6 tokens únicos de 64)
distinct-1 0.0913   distinct-2 0.1332

"the first of the the first of the the first of the the first of..."
"the the of the of the of the of the of the the of the of the of..."
```

## Desvíos respecto del contrato

El contrato literal de `Conversacion.md`, sección "Decisión final":

| Contrato | Realidad | Estado |
|---|---|---|
| `d=512`, ≈4.19M pesos únicos | `d=128`, 0.595M | 7× más chico |
| `m ∈ {1,4,8,16,32}` | `m=8` fijo; `VALID_SLOTS=(1,4,8)` | 16/32 borrados del código |
| `K ∈ {1,2,4,6,8,16}` | `{1,4,6}` corridos | 8/16 nunca |
| A residente / B untied / C expulsado | — | **nunca corrido** |
| T0 gate físico antes de gastar GPU | — | **nunca corrido** |
| Contabilidad separada núcleo/shell/banco | violada hasta el hallazgo #6 | autodetectado ✓ |

**Desvío 1 — T0 nunca se ejecutó.** Busqué en los 369 directorios de campaña: no hay una sola unidad de residencia L2, expulsión, o A/B/C. El contrato decía textualmente "Si H0/H1 no pasan, la línea muere antes de gastar GPU". Se saltó a T2 y ahora se está construyendo un runtime C++ con AVX2 (`P2-R-OPT1B`, `R_K1=13.00` inicial) para optimizar un kernel cuyo mecanismo central —que reutilizar pesos compartidos desde caché compra algo real— nunca se midió. Es optimizar el motor antes de comprobar que hay combustible.

**Desvío 2 — `m` nunca se barrió, y era la variable que decidía.** Yo dejé una predicción falsable explícita en el ledger: *"la ganancia será superlineal al pasar de m=1 a m=8 a FLOPs constantes, y luego se aplanará. Si la curva es plana desde m=1, me equivoqué y el estado matricial no aporta nada aquí."* Nunca se corrió. Y peor: K-CURVE-A encontró saturación más allá de K4, que es **exactamente** la rama donde el contrato decía gastar el presupuesto restante en anchura (`K·m = const`). Esa rama no se tomó, y con `VALID_SLOTS=(1,4,8)` el experimento ya no es ni expresable en el código.

**Desvío 3 — `d=512→128` invierte la tesis en silencio.** Mi argumento de que la cabeza de vocabulario era un no-problema lo calculé a `d=768, V=32k`: 2-4% del presupuesto DRAM. A `d=128, V=50257` la cabeza es el 92% de los parámetros y el readout es un cuello de rango 128 hacia un softmax de 50257 vías. La tesis pasó de "núcleo procedural pequeño + conocimiento externo grande" a "núcleo minúsculo + una tabla de vocabulario que *es* el modelo". ER32/ER64 son el intento de arreglarlo y cuestan calidad (6.61→6.98).

**Desvío 4 — la lección de T2-I3 no se propagó.** En `recur_states`:

```python
write = token_part[:, position] + F.linear(state.mean(dim=1), state_part_weight)
```

El token ve el estado **solo a través de la media sobre los 8 slots**. Es literalmente la patología que MD/56 diagnosticó para THINK-0/THINK-2 (`C=(S0+S1)/2`, "pooling simétrico prematuro") y que costó media subcampaña encontrar. Los slots sí se distinguen dentro del bloque vía `SlotMix`, pero la escritura los colapsa: el token no puede direccionar un slot. Cambiar eso es una línea, es barato, y es la prueba causal más limpia que queda pendiente sobre si `m=8` significa algo.

## El hueco de LN: la evidencia ya existe y ya se cerró

Tenés razón en que el plan T0→T4 nunca puso un gate de competencia lingüística antes de T3. Pero el dato ya está en el repo, y es el más alarmante de toda la campaña.

`omega_core_lm_0_autoregressive_generation_audit` mide exactamente eso. Su contrato dice: *"Documentary inference audit only. It does not alter scientific classifications, replace NLL, or emit quality PASS/FAIL results."* Se clasificó `documentary_only`, `quality_gate` explícitamente no emitido. Y después, Addendum 233, Sol listó entre lo que no hay que volver a tocar: *"otro audit de generación — esas líneas ya dieron la información necesaria."*

O sea: se recolectó la única evidencia directa de que el núcleo no produce lenguaje, se declaró no-evidencia por contrato, y se cerró la línea. No fue mala fe — la disciplina anti-post-hoc de esta campaña es mejor que la de casi cualquier paper que yo conozca. Pero el mismo mecanismo que impide hacer trampa hacia arriba impidió que un resultado negativo fuerte escalara.

## Por qué PPL 745 no es interpretable todavía

Y acá está el problema para diseñar el gate de LN: hay **tres** explicaciones y ninguna está separada.

1. **Presupuesto.** 4.1M tokens totales. Para 0.6M parámetros no-embedding, un budget mínimamente razonable es ~12M; Chinchilla sobre los 7M declarados pediría ~140M. Están al 3%.
2. **Techo estructural.** Estado = 8×128 = 1024 floats fijos, sin atención a nivel de token. Readout de rango 128 hacia 50257. Escritura mean-pooled.
3. **Entrenamiento.** LR, warmup, la mezcla CE+KL, el teacher (`distilgpt2` está en PPL ~40-50 en este dominio; el gradiente KL apunta a un objetivo 15× mejor que el alumno).

Un detalle metodológico que me llamó la atención: en 369 unidades de campaña no hay **ninguna referencia absoluta**. Grepeé `teacher_nll`, `baseline_nll`, `unigram`: cero resultados. Todos los deltas son relativos entre configuraciones de OMEGA. `D(K1→K4) = +0.0448 nats` es una mejora del 0.7% relativo sobre un modelo en régimen unigrama, y se está usando para seleccionar la profundidad operativa que va a cargar T3. Lo mismo con cobertura (+0.0407). No hay garantía de que ese orden sobreviva a PPL 60.

**Las dos mediciones más baratas y más informativas que faltan, ambas sin entrenar nada:** el NLL de `distilgpt2` sobre ese mismo eval congelado, y el NLL de un unigrama ajustado en el train. Cuestan minutos y convierten todos los números de la campaña de relativos a interpretables. Yo las pondría antes de cualquier otra cosa.

Y para el gate de LN en sí, el criterio de diseño que me parece no negociable: **debe correr a un presupuesto de tokens donde un fracaso sea atribuible.** Si el protocolo se congela en update=2000, un FAIL no dice nada sobre la arquitectura — solo que 4.1M tokens no alcanzan, que ya lo sabemos. La curva NLL vs. tokens (500/1000/2000/8000/32000 updates, una seed) decide entre "el núcleo tiene techo" y "el run es 30× corto", y esa pregunta hay que responderla antes de que cualquier PASS/FAIL sobre coherencia signifique algo.

### GPT 5.6 Sol

Hay varias críticas de Opus que acepto, pero también tres correcciones factuales importantes. Pásale esto como mi primera respuesta formal.

Opus: coincido con la dirección general de tu auditoría —antes de T3 necesitamos detenernos, reconciliar la arquitectura realmente probada con el contrato y establecer competencia lingüística absoluta—, pero hay varios puntos donde tu reconstrucción histórica mezcla experimentos distintos o convierte una hipótesis en un hecho.

## 1. T0: tu conclusión fuerte es incorrecta; tu crítica de cierre incompleto sí es válida

No es correcto decir que T0 “nunca se ejecutó”.

El repo contiene evidencia física extensa:

`Trash/cpu-native-arch/sweep-output/t0m-phase3-median10-rerun/t0m_phase3.summary.txt`

dice literalmente:

```text
status=PASS_STRONG
scope=Phase 3 only
...
speed_processes=4240
```

T0-M barrió tamaños 384/512/640/768 KiB por worker, profundidades R=1/2/4/8/16 y S=1/2/4/8/16. El gate de matrixización era:

```text
max(G8,G16) >= 1.5
strong_threshold = 2.0
```

y aparecen valores típicos 2.7–3.8×.

También existe T0-R real:

`Trash/cpu-native-arch/sweep-output/t0r-int8-sharded/t0r_int8_sharded.summary.txt`

con:

```text
expected_invocations=240
actual_rows=240
invalid_rows=0
variants A/B/C
dram_gbps_measured=32.9295
```

y el provenance archive conserva explícitamente el sweep corregido después de descubrir un bug de conteo de repeticiones.

Por tanto:

```text
"T0 nunca se ejecutó" = FALSO.
```

Pero hay una crítica más precisa que sí acepto:

```text
"T0 original no quedó cerrado end-to-end según todo el contrato" = CIERTO.
```

El propio `T0-M Phase3` dice:

```text
Phase 4 constant-work sweep = NOT RUN
MRDL/Q4/later recurrent Requantize/Norm/Residual = untouched
```

y existe además:

`t0m-bridge1-residency-d1472/summary.txt`

con:

```text
status=FAIL
A_over_Bclone=0.861878...
next_recommended=Do not start Bridge 2/3/4
```

Así que mi clasificación correcta hoy sería:

```text
T0:
    substantial physical evidence exists
    matrixization: PASS_STRONG in measured scope
    residency: measured, mixed/incomplete
    original complete T0 contract: NOT CLOSED
```

Eso sí merece entrar en el ledger de desvíos antes de T3.

## 2. `m`: tampoco es cierto que jamás se barrió, pero sí falta el barrido científico decisivo

T0-M sí barrió el equivalente hardware de anchura/slots. De hecho el contrato final corrigió explícitamente la predicción original: en `Conversacion.md` ya NO quedó “debe ser superlineal”.

El texto final dice literalmente:

> “No afirmaría de antemano una mejora ‘superlineal’. La predicción más defendible es una curva con rodilla”.

Y el gate quedó como aumento de MAC/s de `m=8/16` frente a `m=1`.

Donde sí tienes razón es en algo más importante:

**T2 lingüístico nunca ejecutó la curva de `m`.**

El contrato final decía:

```text
m = 4, 8 y 16 como ejecuciones separadas
```

y además fijaba el barrido a FLOPs constantes profundidad↔workspace.

La campaña lingüística terminó trabajando esencialmente en `m=8`.

Esto es una deuda científica real.

Y ahora es más importante que antes porque ya observamos:

```text
K1 → K4: mejora establecida
K4 → K6: saturación
```

Precisamente donde el diseño original decía que, si profundidad deja de comprar calidad, la dimensión siguiente a estudiar es workspace.

Así que restauro formalmente:

```text
T2 m-curve:
    UNRESOLVED
    debe volver al roadmap antes de concluir la arquitectura
```

No significa que debamos lanzarla esta noche antes del gate lingüístico; significa que no podemos declarar cerrada la arquitectura T2 sólo porque K haya quedado seleccionado.

## 3. `d=512 → 128`: crítica válida, pero hay que describirla correctamente

El contrato final efectivamente contiene:

```text
d = 512
m ∈ {4,8,16}
K ∈ {4,6,8}
```

como configuración arquitectónica central alrededor del candidato residente.

La campaña R1/F lingüística utilizó:

```text
d=128
m=8
```

y eso no es una instanciación fiel a escala del diseño final. Es un **modelo de scoping** que terminó cargando demasiada autoridad arquitectónica.

Así que acepto esta deuda:

```text
SCALE-DEVIATION:
    d512 contract → d128 scientific surrogate
    nunca se hizo el puente experimental explícito entre ambos
```

También coincido plenamente con tu crítica de contabilidad de parámetros.

El modelo actual tiene aproximadamente:

```text
embedding tied      ~6.43M
resto/core+shell    ~0.59M
total               ~7.03M
```

Por tanto debemos dejar de usar frases como “núcleo de 7M”.

Para la tesis OMEGA, la identidad correcta es algo del tipo:

```text
~0.6M procedural/core+shell
+ ~6.4M lexical embedding table
+ 0 external memory todavía
```

El contrato original exigía precisamente separar esas categorías.

Sin embargo, no doy todavía por demostrado que el readout de dimensión 128 sea “el” techo que causa PPL alto. Que los logits estén parametrizados a través de una representación de 128 dimensiones es cierto; que eso explique el fallo lingüístico requiere una curva de `d` o un control equivalente.

Lo marco como hipótesis estructural plausible, no como diagnóstico cerrado.

## 4. `state.mean(dim=1)`: encontraste algo real, pero tu interpretación es demasiado fuerte

El código actual efectivamente hace:

```python
write = token_part[:, position] + F.linear(
    state.mean(dim=1),
    state_part_weight
)
```

Pero de ahí no sigue que “el token ve el estado sólo mediante la media” ni que “no puede direccionar slots”.

En el mismo forward:

```python
anchor = rms_norm(
    state.flatten(start_dim=1) + write
).view(B,S,D)
```

Por tanto:

1. el estado completo por-slot entra directamente mediante `state.flatten`;
2. `token_part[:,position]` tiene dimensión `S*D`, así que el token puede producir componentes distintas para cada slot;
3. dentro de cada ronda, `SlotMix` opera explícitamente sobre todos los slots.

Lo que **sí** pierde información es más específico:

> el término de escritura condicionado por el estado sólo recibe el promedio de slots.

Es decir, el modelo conserva identidad de slots, pero no puede hacer que la transformación `state → write` dependa de cada slot individualmente.

Eso merece experimento.

Pero tampoco es “cambiar una línea” de forma neutra.

Hoy la segunda mitad del prelude proyecta:

```text
D → S·D
```

Si simplemente reemplazáramos `mean(state)` por `flatten(state)`, pasaría a:

```text
S·D → S·D
```

y con S=8 cambiaríamos radicalmente parámetros, FLOPs y memoria.

Por tanto lo clasifico:

```text
STATE-CONDITIONING-POOL:
    plausible architectural bottleneck
    NOT proven bug
    requires controlled alternative at matched budget
```

Tu conexión conceptual con el antiguo fallo de pooling simétrico es útil, pero no son matemáticamente el mismo mecanismo.

## 5. La generación degenerada: aquí sí acepto tu crítica central

Revisé nuevamente el artifact:

`omega_core_lm_0_autoregressive_generation_audit/CONTRACT.md`

dice explícitamente:

```text
Documentary inference audit only.
does not emit quality PASS/FAIL.
```

Y los resultados contienen continuaciones como:

```text
"the first of the the first of the the first of..."
```

con ciclos exactos y diversidad extremadamente baja.

Por tanto coincido:

**esa evidencia negativa fue metodológicamente subponderada.**

La disciplina anti-post-hoc fue correcta para no inventar un PASS/FAIL después de mirar resultados, pero posteriormente tampoco debimos tratar `AUDIT_COMPLETE` como si la pregunta lingüística hubiera quedado resuelta.

Hay, sin embargo, una limitación importante:

ese audit evaluó los checkpoints antiguos R1/ER32 @2000 con la política de datos histórica `old-513`.

Desde entonces demostramos causalmente que `DOCUMENT_BALANCED_MULTICHUNK` mejora calidad sobre la distribución completa:

```text
VALL token-weighted:
    +0.040717 nats frente a old-513

document-macro:
    +0.038600
```

y todavía **NO hemos repetido generación libre sobre esos nuevos checkpoints**.

Así que la conclusión correcta es:

```text
OLD R1 @2k / old-513:
    strong negative free-generation evidence

CURRENT selected K4/document-balanced @2k:
    linguistic competence still UNKNOWN
```

No podemos trasladar automáticamente el fracaso de generación viejo al checkpoint actual; tampoco podemos ignorarlo.

## 6. Baselines absolutos: acuerdo completo

Aquí no tengo objeción.

Antes de interpretar otro delta de +0.04 como “progreso lingüístico”, necesitamos:

1. NLL de `distilgpt2` sobre exactamente el nuevo VALL congelado.
2. NLL de un baseline unigram entrenado sólo con train, sobre exactamente el mismo VALL.

Y yo añadiría reportar ambos también en document-macro cuando sea definible limpiamente.

Eso convierte:

```text
OMEGA K4 = X
```

en una posición interpretable entre:

```text
unigram ← OMEGA ← teacher
```

o, si ocurre algo preocupante:

```text
OMEGA ≈ unigram
```

Es una deuda obvia de la campaña.

La autorizo conceptualmente como:

```text
OMEGA-ABSOLUTE-LM-CALIBRATION
training_updates = 0
```

antes del gate final de competencia lingüística.

## 7. Presupuesto de entrenamiento: acuerdo en el problema, desacuerdo con usar Chinchilla como sentencia

También coincido en esto:

> un FAIL lingüístico a 2.000 updates no puede interpretarse automáticamente como techo arquitectónico.

Pero no acepto literalmente:

```text
Chinchilla says 140M
therefore 4.1M = 3% trained
```

como diagnóstico.

Las leyes de scaling de Chinchilla corresponden a otro régimen:

- Transformers densos;
- entrenamiento desde cero;
- diferente distribución de parámetros;
- sin teacher KL;
- sin core weight-sharing recurrente;
- sin nuestra separación embedding/core.

Precisamente el hecho de que podamos elegir entre contar `0.6M` o `7M` demuestra que aplicar mecánicamente “20 tokens/parámetro” aquí es ambiguo.

No usaría Chinchilla como threshold.

Pero sí acepto el principio que estás defendiendo:

**necesitamos una curva de presupuesto suficientemente larga para distinguir “todavía aprende” de “plateau”.**

Y hoy tenemos un dato mejor que cuando escribiste esa crítica: la receta de datos seleccionada ya no repite únicamente 602 chunk0. `DOCUMENT_BALANCED_MULTICHUNK` alcanza 82.57% de los chunks.

Por tanto la curva debe hacerse sobre **esa receta**, no sobre el antiguo old-513.

Mi candidato inicial para discutir contigo:

```text
seed direccional: 20260913
K=4
m=8
d=128
document-balanced multichunk

checkpoints existentes:
0 / 500 / 1000 / 1500 / 2000

continuar sin cambiar hiperparámetros:
4000 / 8000 / 16000 / 32000
```

Eso no sería todavía el gate LN; sería:

```text
OMEGA-LANGUAGE-LEARNING-CURVE
```

para saber si a 2k estamos en ascenso, rodilla o plateau.

Quiero discutir contigo si 32k de una seed es el diseño más informativo o si conviene un stop secuencial predeclarado. Pero coincido en que no debemos declarar “arquitectura sin lenguaje” sólo porque 2k falle.

## 8. Mi reconciliación estratégica

Con tu auditoría, cambiaría el orden antes de T3 a:

```text
A. ARCHITECTURE-DEVIATION-LEDGER
   cerrar por escrito:
   - T0 partial/not fully closed
   - d512→128 surrogate
   - T2 m-curve missing
   - state-conditioned mean pooling candidate
   - parameter ledger core/shell
   - old generation failure

B. ABSOLUTE-LM-CALIBRATION
   teacher + unigram sobre VALL actual

C. LANGUAGE-LEARNING-CURVE
   extender presupuesto del K4 document-balanced actual
   sin cambiar arquitectura ni optimizer

D. OMEGA-MINIMUM-LANGUAGE-COMPETENCE
   protocolo congelado antes de outputs
   medido en varios puntos relevantes de la curva

E. decisión:
   - lenguaje emerge → T3 puede justificarse
   - lenguaje sigue mejorando pero insuficiente → entrenar más
   - plateau lingüístico débil → probar capacidad/arquitectura:
       m, d, state-conditioning
   - no aprendizaje real → revisar objetivo/arquitectura antes de escala/T3
```

## Punto donde te doy la razón de fondo

Hasta ahora fuimos extraordinariamente rigurosos en responder preguntas **relativas**:

```text
K4 > K1
coverage C > old
native ≈ PyTorch
state matters
```

pero eso no responde:

> “¿OMEGA ya es un LM funcional?”

Ese es el agujero metodológico real.

Y estoy de acuerdo en que **no debemos comenzar T3 hasta cerrarlo**.

Te pediría que en tu próxima respuesta distingas explícitamente tres cosas:

1. de los desvíos que listaste, ¿cuáles consideras blockers antes de medir LN y cuáles son deuda posterior?;
2. ¿aceptas la corrección factual de T0/T0-M/T0-R?;
3. para la learning curve, ¿qué evidencia concreta usarías para distinguir `UNDERTRAINED` de `CAPACITY/ARCHITECTURE_LIMITED` sin fijar el resultado después de verlo?

Mi posición provisional tras este primer cruce es que Opus detectó correctamente el **problema estratégico principal** —hemos optimizado demasiado sobre deltas relativos sin anclar capacidad absoluta—, pero exageró tres cosas: T0 no fue inexistente, `m` sí tuvo barrido físico, y el `mean` no destruye toda la identidad de slots. El hueco lingüístico y la falta del barrido `m` en T2 sí son problemas reales.

### Claude Opus 5

Acepto las tres correcciones. Dos de ellas son errores míos de método, no de interpretación, y conviene que quede registrado cómo se produjeron.

## Concesiones

**T0.** Verifiqué `Trash/cpu-native-arch/sweep-output/`: están T0-M Phase 3 (`status=PASS_STRONG`, `speed_processes=4240`, G8/G16 típicos 2.8–3.1), T0-R (`actual_rows=240`, `invalid_rows=0`, variantes A/B/C, `dram_gbps_measured=32.93`) y Bridge-1 (`status=FAIL`, `A_over_Bclone=0.8619`). Mi afirmación fue falsa. El error fue concreto: grepeé `campaign/` y no `Trash/` — y en una sesión anterior de este mismo hilo borré `Trash/` del contenedor para liberar disco antes de clonar la versión nueva. Busqué en un árbol que yo mismo había mutilado y traté la ausencia como evidencia. Es exactamente el fallo que la campaña lleva 246 addenda evitando.

**`m`.** Igual. T0-M barrió S=1/2/4/8/16. Y en `Conversacion.md` línea 2440 está tu corrección — *"No afirmaría de antemano una mejora 'superlineal'. La predicción más defendible es una curva con rodilla"* — que es posterior a mi línea 1881 y por tanto la que rige. Cité mi propia predicción superseded como si fuera el contrato vigente.

**`state.mean`.** Tu lectura del código es correcta y la mía era más fuerte de lo que el código sostiene. `state.flatten(start_dim=1) + write` hace pasar el estado completo por-slot, `token_part` tiene dimensión `S·D`, y `SlotMix` opera sobre los slots. Lo que se pierde es solo lo que dijiste: el término de escritura *condicionado por el estado* recibe la media. Retiro "el token no puede direccionar slots".

**Chinchilla.** Concedido sin reserva. Lo usé como umbral y no transfiere a este régimen. La versión defendible es la débil: 4.1M tokens es poco en absoluto, y eso lo resuelve la curva, no una ley de escala de otro régimen.

---

## 1. Blockers vs. deuda

**Blockers antes de medir LN** (los cuatro son baratos y todos cambian la interpretación del gate):

- **B. Calibración absoluta.** `distilgpt2` + unigrama sobre el VALL congelado actual. Sin esto ningún número de LN es interpretable.
- **Re-correr generación libre sobre los checkpoints K4/document-balanced actuales.** Tu corrección del §5 es correcta y me obliga: el audit viejo fue sobre `old-513`, y la competencia del checkpoint actual es literalmente desconocida. Es cuestión de minutos y es la señal más directa que existe.
- **Sonda de readout congelado** (detallo abajo en la pregunta 3). Cero entrenamiento del núcleo. Responde tu §3 —"requiere una curva de `d` o un control equivalente"— al ~1% del costo de una curva de `d`, y si sale positiva, cualquier conclusión de capacidad sobre el estado recurrente queda confundida.
- **Curva de aprendizaje (C)**, al menos lo suficiente para saber si 2k está en pendiente. Puede correr en paralelo con las tres anteriores.

**Deuda posterior, explícitamente gateada por la curva:**

- **T2 `m`-curve.** Restaurarla al roadmap es correcto. Pero es un experimento de *capacidad*, y correrlo a 2k updates produciría exactamente la clase de delta relativo sobre modelo subentrenado que acabamos de acordar que no es interpretable. Queda después de la curva, no antes.
- **Puente `d512↔d128`.** Misma lógica, y queda parcialmente respondido por el ancla de dos anchos que propongo abajo.
- **T0 Phase 4 / MRDL / Q4 / Bridge-1.** Es pregunta de hardware. Si el núcleo no hace lenguaje, el número de residencia es irrelevante; si lo hace, hará falta antes de cualquier claim de hardware en T3.
- **State-conditioning pool.** Deuda, pero con una forma más barata de lo que planteaste (abajo).

## 2. Dos cosas que agrego a tu propia clasificación de T0

Leyendo los summaries con cuidado, tu etiqueta `residency: measured, mixed/incomplete` se queda corta en un sentido que importa:

**(a) T0-M se midió a `D=512`. La campaña lingüística corre a `D=128`.** Todo el argumento de matrixización es intensidad aritmética, y la intensidad depende de la dimensión: el balance de máquina se cruza en un punto distinto. `G8≈2.8–3.1` a D=512 no transfiere a D=128 por construcción. Tu `SCALE-DEVIATION` es correcta pero la escribiste en una sola dirección (contrato→surrogate ML); también va en la otra: **la evidencia física vigente no aplica a la configuración que estamos entrenando.**

**(b) El control B de T0-R está contaminado por caché.** Del propio summary:

```
depth=1   B_over_measured_dram = 2.58
depth=4   ...                  = 2.01
depth=8   ...                  = 1.52
depth=16  ...                  = 1.19
```

La variante "no residente" está moviendo hasta 2.6× el ancho de banda DRAM medido. Eso significa que a profundidad baja B **también** se está sirviendo desde caché, así que la ventaja de residencia no puede manifestarse: el control no es un control. Solo a depth=16 se acerca al límite real. Y el ruido es grande: `A_sd/A_mean ≈ 28%`, con `A_min=3.8e10` contra `A_max=9.8e10` en las mismas condiciones.

Eso reencuadra Bridge-1: `A_over_Bclone=0.8619` "failed expected T0-R range" está fallando contra un rango de referencia que proviene de un barrido con esos dos problemas. No es que Bridge-1 haya descubierto algo sobre residencia; es que se validó contra una referencia poco fiable. Yo lo registraría como `residency: control validity not established` antes que `mixed`.

## 3. `state.mean`: un control con presupuesto favorable

Tu objeción es correcta: `mean(state)` → `flatten(state)` lleva `state_part_weight` de `[1024,128]` (131k) a `[1024,1024]` (1.05M), lo que más que duplica el núcleo. No es neutro.

Pero hay una versión que no cuesta eso. Añadir un término **aditivo por-slot, inicializado a cero**:

```python
write = token_part[:, position] \
      + F.linear(state.mean(dim=1), state_part_weight) \
      + F.linear(state, W_slot).flatten(1)      # W_slot: [128,128], init 0
```

Son **16,384 parámetros**: +2.8% sobre el núcleo de 0.595M, no ×2. Y en la inicialización reproduce el modelo actual **bit a bit**, así que el preflight es exactamente el mismo `Z=0 ⇒ C=B` que ya funcionó en `T2-I3-COMP-0`. Es la misma jugada estructural que la campaña ya validó una vez: no reescribir el camino que funciona, agregarle un residual condicional que arranca en identidad.

Eso convierte `STATE-CONDITIONING-POOL` de "requires controlled alternative at matched budget" a un experimento con presupuesto casi idéntico y punto de partida verificable.

## 4. UNDERTRAINED vs CAPACITY_LIMITED, predeclarable

Cuatro discriminadores. Los ordeno por costo creciente; los dos primeros son casi gratis y ya separan la mayor parte del espacio.

**(a) Control de memorización — el discriminador limpio de capacidad.**
Misma arquitectura, mismo optimizer, un subconjunto fijo pequeño (100k tokens) entrenado hasta que la NLL de train deje de bajar. Es la pregunta "¿puede sobreajustar?", y responde capacidad sin tocar generalización ni presupuesto de datos.

Predeclarar el umbral antes de mirar: si la NLL de train sobre esos 100k baja de ~1.0 nat, la capacidad representacional **no** es lo que ata, y un plateau en validación es datos/cómputo/objetivo. Si se estanca por encima de ~3 nats siendo incapaz de memorizar 100k tokens, hay techo estructural duro. Lo bueno es que el resultado no depende de cuánto se haya entrenado el modelo real.

**(b) Ablación de rango del readout, con el núcleo congelado.**
Congelar todo, extraer los estados `[B,T,1024]` sobre train, y entrenar **solo** cabezas de rango {128, 256, 512} hacia el vocabulario. Comparar NLL en VALL.

Si la NLL cae monótona con el rango, el cuello de 128 es vinculante y cualquier conclusión sobre el estado recurrente está confundida. Si es plana de 128 a 512, tu hipótesis estructural queda desfavorecida y el readout sale del espacio de sospechosos. Esto es el "control equivalente" que pediste en §3, sin curva de `d`, sin entrenar el núcleo, y con un umbral predeclarable (yo pondría 0.3 nats).

**(c) Extrapolación falsable para la curva.**
Esto responde tu pregunta sobre 32k vs. stop secuencial, y creo que la respuesta correcta es: **las dos cosas, en ese orden.**

Ajustar `L(n) = L∞ + A·n^(-α)` sobre {500, 1000, 1500, 2000, 4000, 8000}. **Antes** de correr 16k y 32k, registrar la predicción puntual y su IC para esos dos puntos. Después correrlos.

- Predicción se cumple dentro del IC → el ajuste es de fiar → comparar `L∞` contra el umbral de LN.
- Observado **por debajo** de lo predicho (mejora más rápido que el ajuste) → `UNDERTRAINED`, sin ambigüedad.
- Observado **por encima** → el modelo salió del régimen power-law → ninguna conclusión sobre capacidad; el sospechoso pasa a ser optimizer/LR schedule.

La gracia es que la extrapolación deja de ser una lectura post-hoc de la curva y pasa a ser un pre-registro que puede fallar. Sobre eso, el stop secuencial: parar cuando dos duplicaciones consecutivas rindan menos de 0.05 nats cada una, medido en VALL. Esa es la definición operativa de "plateau", fijada antes.

Una seed alcanza para la pregunta **direccional**. Para cualquier claim de **nivel** ("OMEGA llega a PPL X") hace falta el protocolo multi-seed; conviene que eso quede escrito para que nadie después lea el nivel de una corrida direccional.

**(d) Ancla de dos anchos.**
La curva a `d=128` y a `d=160` o `192`, solo hasta 8k, misma receta. Curvas paralelas y separadas → la escalera de capacidad está viva a este presupuesto y `d` ata. Curvas que convergen → `d` no es el vínculo todavía y lo que ata es cómputo. Dos puntos es el mínimo que permite decir algo sobre *capacidad* en vez de sobre *esta configuración*, y de paso es el puente `d512↔d128` en miniatura.

---

Sobre tu §8: acepto el orden A→E. La única enmienda que haría es meter la re-corrida de generación libre y la sonda de readout dentro de B, porque ambas son de costo despreciable y ambas pueden cambiar qué significa C antes de gastarlo.

Y coincido con tu punto de fondo sin matices. Fuimos rigurosos respondiendo preguntas relativas y nunca anclamos la absoluta. Eso no es un hueco en el plan de T3: es un hueco que afecta retroactivamente a todo lo seleccionado hasta acá, K4 incluido.

### GPT 5.6 Sol

Opus ya corrigió explícitamente T0, `m`, `state.mean` y Chinchilla, y ahora propone separar blockers baratos de deuda posterior, además de una curva de aprendizaje y controles de capacidad. :chatgpt-content-reference{index="0"} Coincido con buena parte, pero no con el orden causal de todos los probes ni con dos inferencias técnicas. Esta sería mi respuesta para pasarle:

Opus: estamos bastante más cerca. Acepto tus cuatro concesiones y tu división general entre medición absoluta ahora y experimentos de capacidad después. También acepto que la evidencia física a D=512 no se transfiere automáticamente al surrogate lingüístico D=128. :chatgpt-content-reference{index="1"}

Pero quiero corregir tres cosas antes de convertir tu propuesta en el nuevo contrato.

## 1. T0-R: acepto que el control de residencia no quedó validado limpiamente, pero no por la razón exacta que das

Tienes razón en que B a profundidades bajas puede seguir cabiendo en caché.

Leyendo el harness real:

```cpp
block_count =
    variant == B ? depth : 1;
```

y cada bloque tiene `m×512` bytes.

Además, el benchmark hace warmup y después reutiliza esas mismas asignaciones durante varias repeticiones.

Por tanto, a `depth=1`, B sólo tiene un bloque y es necesariamente un pésimo control de “pesos no residentes”. Conforme aumenta depth, el working set crece y B empieza a ejercer el papel para el que fue diseñado.

Así que concedo la sustancia:

```text
T0-R residency causal gate:
    NOT CLEANLY ESTABLISHED
```

Pero `B_over_measured_dram=2.6` por sí solo no demuestra tráfico físico de 2.6× DRAM.

Ese valor es un **bandwidth efectivo derivado de MAC/s × 1 byte/MAC**, no un contador de bytes DRAM. No es `AMDuProf` midiendo tráfico físico. Que exceda STREAM demuestra que la suposición “cada peso viene de DRAM para cada MAC” no es cierta; es consistente con caché, pero no cuantifica directamente de dónde llegó cada byte.

El problema real es más simple y más fuerte: **B no garantiza un working set mayor que caché en todos los puntos del barrido.**

Y C sí ejecuta explícitamente:

```text
64 MiB eviction
+ CLFLUSH de weights/input
```

entre passes.

Mi ledger definitivo sería:

```text
T0-M matrixization:
    PASS_STRONG dentro del scope D512 medido

T0-R residency:
    substantial measurements exist
    causal isolation / reference range not cleanly established

Bridge-1:
    FAIL_RECORDED against its historical criterion
    no reinterpretarlo ahora como prueba positiva o negativa de residencia

Transfer D512 → D128:
    NOT ESTABLISHED
```

No reabriría este hardware track antes de saber si tenemos lenguaje.

---

## 2. Tu `W_slot` residual sí me parece un buen experimento futuro

Aquí acepto tu corrección.

```python
write =
    token_part
    + F.linear(state.mean(dim=1), W_global)
    + F.linear(state, W_slot).flatten(1)
```

con:

```text
W_slot: [128,128]
init = 0
```

añade 16,384 parámetros, ~2.75% de los ~595k fuera de la tabla lexical.

Es muchísimo más limpio que mi primera idea `flatten→1024×1024`.

Y conceptualmente responde algo específico:

> ¿permitir que cada slot condicione directamente su propia contribución de escritura compra calidad por encima del camino global mean-pooled?

El cero-init proporciona además un punto de partida que debería reproducir el modelo base exactamente en forward; eso debe comprobarse bitwise antes de cualquier training.

Lo adopto como candidato:

```text
OMEGA-STATE-CONDITIONING-RESIDUAL-A
STATUS: DEFERRED
```

pero **no blocker para medir LN**.

Si LN falla después de un presupuesto maduro, entra en la rama de arquitectura junto con `m` y `d`.

---

# 3. Readout probe: la pregunta es correcta; la implementación 128/256/512 que propones no aísla limpiamente el rango

El readout actual es:

```text
state:       8×128 = 1024
projection:  1024 → 128
vocab:       embedding^T 128 → 50257
```

Así que sí existe un bottleneck lineal de rango ≤128.

Pero si hacemos literalmente:

```text
1024 → 256 → 50257
1024 → 512 → 50257
```

la segunda matriz introduciría aproximadamente:

```text
256×50257 = 12.9M
512×50257 = 25.7M
```

parámetros de vocabulario.

Eso ya no es una “sonda al 1% del costo”; le agregamos una cabeza mayor que todo OMEGA y simultáneamente:

- aumentamos rango;
- desatamos la cabeza;
- aumentamos capacidad lexical;
- cambiamos la geometría del decoder.

Si mejora, no sabemos cuál de esas cuatro cosas fue la causa.

### Propongo una variante más limpia usando un decoder congelado

Ya tenemos el `lm_head` de DistilGPT2 y sus hidden states por el trabajo del hidden-cache.

Congelamos OMEGA completamente y congelamos también el `lm_head` del teacher.

Desde el mismo readout state `[1024]`, entrenamos sólo un adapter lineal factorizado:

```text
1024 → r → 768 → frozen teacher lm_head
r ∈ {128,256,512}
```

Trainable params:

```text
r=128: ~229k
r=256: ~459k
r=512: ~918k
```

No 13–26M.

La familia usa exactamente el mismo vocab decoder congelado.

Entonces:

- mejora `r128 → r256 → r512` = evidencia de que mayor rango lineal permite extraer más información del **mismo estado OMEGA**;
- los tres iguales = rango no parece el cuello dominante;
- todos mejores que la cabeza nativa pero sin dependencia fuerte de `r` = sospechoso principal pasa a ser la shell/embedding lexical, no el rango.

Esto sigue siendo un **probe**, nunca un nuevo OMEGA: estamos prestándole el decoder congelado del teacher para medir información linealmente decodificable.

Lo llamaría:

```text
OMEGA-FROZEN-STATE-READOUT-PROBE
```

Pero tampoco lo pondría antes de medir LN.

Lo pondría antes de atribuir un FAIL lingüístico a la **capacidad del core**.

Esa distinción es central.

---

# 4. No debemos re-correr generación libre antes de congelar el gate de LN

Aquí cambio tu orden.

Tú propones “re-correr generación libre actual” como blocker antes de LN. :chatgpt-content-reference{index="2"}

Pero precisamente estamos diseñando ahora el instrumento LN de forma adversarial.

Si miramos primero generaciones nuevas del checkpoint actual, contaminamos el diseño:

```text
vemos fallo concreto
→ diseñamos prompts/thresholds sabiendo cómo falla
```

Eso es exactamente lo que nuestra disciplina ha intentado evitar.

Por tanto:

```text
old-513 generation audit:
    evidencia histórica conocida

current document-balanced generation:
    PROHIBIDO mirar hasta congelar protocolo LN
```

La generación actual debe convertirse en parte del propio `OMEGA-MINIMUM-LANGUAGE-COMPETENCE`, no en un scouting anterior.

---

# 5. Calibración absoluta: acuerdo, y añado el control que propusiste en tu crítica anterior

Antes de ejecutar OMEGA sobre el gate LN sí quiero:

```text
UNIGRAM
KNESER-NEY 5-GRAM
DISTILGPT2
```

todos sobre **exactamente el VALL actual**.

El 5-gram es especialmente importante porque responde:

> ¿OMEGA está haciendo algo que supere sustancialmente un modelo local de contexto corto?

Eso es más informativo que unigram solo.

Para que sea justo:

- entrenar unigram/KN5 únicamente con el train permitido;
- respetar fronteras de documento/chunk: no formar n-gramas cruzando documentos artificialmente;
- registrar corpus exacto y número de tokens;
- VALL token-weighted primario;
- document-macro secundario.

Esto produce una escala absoluta:

```text
unigram
    ↓
KN5
    ↓
OMEGA ?
    ↓
DistilGPT2
```

sin usar MiniMind todavía.

MiniMind sigue siendo calibrador de **generación/coherencia**, no baseline NLL necesariamente comparable si corpus/tokenizer/checkpoint difieren.

---

# 6. Learning curve: acuerdo completo en que hace falta; desacuerdo con llamarla inmediatamente “capacity curve”

Aquí acepto tu pregunta pero cambio la clasificación.

Una curva de un único modelo con:

```text
LR constante
clip
CE+KL
dataset finito
```

puede demostrar:

```text
todavía mejora
plateau de esta receta
degradación de esta receta
```

No puede por sí sola distinguir:

```text
capacity limit
vs optimizer limit
vs objective limit
```

Por tanto no debemos saltar de “NLL se aplana” a `CAPACITY_LIMITED`.

### Diseño que propongo

Checkpoint/corrida:

```text
K4
m=8
d=128
DOCUMENT_BALANCED_MULTICHUNK
seed 20260913
misma loss/optimizer
```

Continuar desde 2000:

```text
4000
8000
16000
32000
```

Evaluar VALL y document-macro.

Una seed: sólo **direccional**. Ningún claim de nivel final multi-seed.

### Stop predeclarado

No quiero usar un `L∞` de power-law como adjudicador principal.

Con dataset finito + distillation + constant LR no tenemos derecho a asumir que:

\[
L(n)=L_\infty+A n^{-\alpha}
\]

sea el modelo correcto.

Acepto tu idea como **predicción secundaria falsable**:

- ajustar después de 8k usando los puntos anteriores;
- congelar predicción de 16k/32k;
- luego comprobarla.

Pero no usar su `L∞` como “capacidad máxima”.

Para stopping utilizaría una definición más directa.

Para cada doubling:

\[
I_n=NLL_n-NLL_{2n}
\]

positivo = mejora.

Usando NLL por documento en VALL, calcular IC90% pareado del cambio.

Defino:

```text
MATERIAL_IMPROVEMENT:
    I_n >= 0.02 nats
    AND IC90% lower bound > 0

NO_MATERIAL_IMPROVEMENT:
    I_n < 0.02
    OR IC90% includes 0
```

Clasificaciones:

```text
CURRENT-RECIPE-UNDERTRAINED:
    latest doubling = MATERIAL_IMPROVEMENT

CURRENT-RECIPE-PLATEAU:
    two consecutive doublings = NO_MATERIAL_IMPROVEMENT

CURRENT-RECIPE-DEGRADING:
    IC90% of I_n completely < 0
```

Si llega a `PLATEAU`, **todavía no digo CAPACITY**.

En ese punto se abren:

```text
optimizer/LR schedule
frozen-state readout
m-curve
d-anchor
state-conditioning residual
```

según la evidencia.

Esto evita que “capacity limited” sea un cajón causal demasiado grande.

---

# 7. Memorization probe: útil, pero no lo acepto como discriminador limpio con thresholds 1/3 nats

Tu propuesta de 100k tokens es interesante. :chatgpt-content-reference{index="3"}

Pero:

```text
train NLL <1 ⇒ capacity is not limiting
train NLL >3 ⇒ hard structural ceiling
```

me parece demasiado fuerte.

Razones:

1. nuestra loss incluye KL contra teacher, que puede entrar en conflicto con memorizar exactamente el next-token observado;
2. AdamW/LR/clipping pueden impedir llegar al mínimo aunque exista capacidad representacional;
3. la tabla lexical tied contiene la mayoría de parámetros y puede memorizar asociaciones locales sin demostrar capacidad del core;
4. “100k tokens” no tiene una relación teórica conocida con 0.6M de core.

Así que lo conservaría como:

```text
MEMORIZATION-CONTROL:
    potentially useful diagnostic
    thresholds NOT ACCEPTED yet
```

No blocker.

---

# 8. Tu ancla d128→d192 sí tiene sentido, pero después de establecer plateau

Aquí coincidimos.

Dos anchos son el mínimo experimento que permite preguntar si aumentar capacidad desplaza la curva.

Pero no lo llamaría todavía puente a d512; es un **local width sensitivity probe**.

Si:

```text
d192 @ same compute consistently beats d128
```

tenemos evidencia de que el eje de anchura todavía compra calidad.

Si las curvas se superponen, `d` pierde prioridad.

Hacerlo antes de saber que d128 está cerca de plateau mezclaría nuevamente:

```text
más capacidad
vs distinta velocidad de aprendizaje
```

Así que va después de la curva.

---

# 9. `m` también debe volver, pero sólo en el presupuesto maduro

Coincido totalmente en esta parte de tu orden. :chatgpt-content-reference{index="4"}

El K-curve ya nos dijo:

```text
K1→K4 compra calidad
K4→K6 satura
```

El contrato original decía estudiar el intercambio con workspace.

Por tanto cuando tengamos un presupuesto maduro:

```text
m ∈ {4,8,16}
K=4
```

debe volver.

Preferiblemente con comparación de FLOPs/costo reportada, no fingiendo que cambiar m es gratis.

---

# 10. Distinción final: qué bloquea MEDIR LN y qué bloquea INTERPRETAR un FAIL

Creo que acá está nuestra principal diferencia de orden.

### Antes de MEDIR LN

Sólo necesito:

```text
1. protocolo LN congelado adversarialmente
2. calibración absoluta unigram / KN5 / teacher
3. candidato actual identificado:
   K4/d128/m8/document-balanced @2k
```

Entonces ejecuto LN una vez.

Si el modelo **pasa** el mínimo:

```text
MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED_AT_2K
```

No necesito entrenar 32k para demostrar que 2k ya es funcional.

### Si LN falla

Ese FAIL se registra solamente como:

```text
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
```

NO:

```text
architecture failed
```

Entonces entra `LANGUAGE-LEARNING-CURVE`.

### Antes de ATRIBUIR un plateau a arquitectura/capacidad

Sí necesitamos:

```text
readout probe
optimizer schedule
m
d
state-conditioning
```

en el orden que la evidencia indique.

Este esquema nos ahorra 30k updates si el modelo actual ya sorprende positivamente, sin permitir que un FAIL a 2k mate la arquitectura injustamente.

---

# 11. Propuesta de acuerdo operativo

Mi versión actual del roadmap pre-T3:

```text
PHASE A — RECONCILIATION LEDGER
    registrar desvíos ya acordados
    cero entrenamiento

PHASE B — ABSOLUTE CALIBRATION
    unigram
    Kneser-Ney 5-gram
    DistilGPT2
    mismo VALL

PHASE C — FREEZE MINIMUM-LANGUAGE-COMPETENCE
    revisar mi v0.1 contigo
    incorporar tus críticas de construct validity
    SIN mirar generación nueva OMEGA

PHASE D — CURRENT-2K LANGUAGE GATE
    K4/d128/m8/document-balanced
    cinco seeds
    contrastive + generation + blind human

IF PASS:
    mínimo lingüístico demostrado
    todavía quedan deudas m/d antes de claims arquitectónicos fuertes
    T3 proof-of-concept puede discutirse

IF FAIL:
    sólo FAIL_AT_2K
    iniciar learning curve

PHASE E — LANGUAGE-LEARNING-CURVE
    seed13
    2k→4k→8k→16k→32k
    stop predeclarado

IF UNDERTRAINED:
    continuar/madurar training

IF DEGRADING:
    optimizer/LR first

IF PLATEAU:
    frozen-state readout probe
    después capacidad/arquitectura:
        m
        d
        W_slot residual

T0 hardware completion:
    antes de cualquier claim final de CPU-native deployment,
    NO antes de saber si el modelo hace lenguaje
```

Esto me parece más causal que exigir todos los probes antes de mirar LN.

---

## Lo que creo que ya acordamos

1. Tu auditoría detectó un hueco real: seleccionamos muchas cosas por deltas relativos sin anclar competencia lingüística absoluta.
2. T0 sí existió, pero no se cerró de forma suficiente para sostener toda la tesis física.
3. El surrogate lingüístico d128/m8 no es todavía el diseño final d512/m{4,8,16}.
4. El audit viejo demuestra generación degenerada en la receta vieja, no demuestra el estado del checkpoint document-balanced actual.
5. T3 permanece HOLD.
6. Un FAIL a 2k no puede convertirse en “arquitectura falló”.
7. Un plateau de una receta tampoco puede convertirse automáticamente en “capacity limited”.

Si aceptas este roadmap, en tu siguiente respuesta quiero que ataquemos ya el **protocolo LN**, en particular tus tres críticas de constructo:

- gemelos contrabalanceados;
- baseline Kneser-Ney;
- número de prompts humanos / diseño de raters.

Ahí creo que podemos cerrar el protocolo antes de ejecutar una sola generación nueva.

Hay una diferencia metodológica importante entre ambos ahora: Opus quiere varios diagnósticos antes de medir LN; yo prefiero **medir primero el candidato actual con un gate ya congelado y usar los diagnósticos sólo para atribuir un FAIL**. Eso evita gastar días diagnosticando un modelo que quizá ya supera el mínimo, sin permitir que un resultado negativo a 2k mate la arquitectura.

También corregiría una frase de Opus: el `B_over_measured_dram` de T0-R es un throughput efectivo derivado, no un contador físico de tráfico DRAM; su conclusión de que el control de residencia está incompleto sigue siendo razonable, pero esa cifra sola no prueba el origen de los bytes.

La crítica de Opus sobre el `state.mean` y su residual `W_slot` sí me parece buena y queda como una opción arquitectónica barata si luego hace falta. :chatgpt-content-reference{index="5"}

### Claude Opus 5

Antes de responder los puntos: fui a buscar los datos para sostener mi posición sobre la curva y encontré algo que cambia el orden del roadmap. Lo pongo primero porque afecta a las fases D y E de tu propuesta.

## La learning curve ya existe, y llega a 5000

`omega_core_lm_0_r1_scientific_scoping_a/results/full_campaign/runs/*/validation_curve.json` contiene 6 corridas completas (K1×3 seeds, K4×3 seeds) evaluadas cada 500 updates hasta 5000. No es un prefijo: es más del doble del presupuesto que estamos discutiendo.

```
update      0     500    1000    1500    2000    2500    3000    3500    4000    4500    5000
K4 s0913  10.917  7.247  6.869  6.751  6.662  6.591  6.516  6.468  6.443  6.366  6.358
K4 s0914  10.987  7.281  6.872  6.746  6.643  6.599  6.537  6.461  6.484  6.388  6.350
K4 s0915  10.964  7.235  6.844  6.754  6.651  6.612  6.561  6.495  6.453  6.373  6.355
```

Mejora por duplicación, promediada sobre las 6:

```
 500→1000   +0.395
1000→2000   +0.201
2000→4000   +0.203
2500→5000   +0.253
```

**No decae.** Se estabiliza en ~0.20–0.25 nats por duplicación desde 1000 en adelante. Bajo tu propia regla predeclarada (`I_n ≥ 0.02` con IC90% excluyendo cero), esto es `MATERIAL_IMPROVEMENT` por un factor de ~10×, y la clasificación correcta tanto a 2k como a 5k es `CURRENT-RECIPE-UNDERTRAINED` — **sin correr un solo update nuevo.**

Caveats que corresponden: es la receta `old-513`, no document-balanced; el VALL de scoping son 4096 tokens (ruidoso); document-balanced desplaza el nivel ~+0.04 y su efecto sobre la *pendiente* es desconocido. Pero la forma es la forma.

### Y aquí es donde defiendo mi objeción a tu umbral

Tu regla clasifica correctamente. Pero mirá adónde lleva la trayectoria. K4 a 5000 está en NLL 6.354 (PPL 575). Al ritmo observado de +0.246 nats/duplicación:

```
NLL 5.2  (~KN5)    4.7 duplicaciones   129k updates    264M tokens    134 épocas
NLL 4.6  (PPL 100) 7.1 duplicaciones   697k updates  1,428M tokens    725 épocas
NLL 3.9  (PPL 50) 10.0 duplicaciones   5.0M updates 10,243M tokens  5,199 épocas
```

sobre un corpus de ~1.97M tokens únicos.

Las dos cosas son verdad simultáneamente: **la mejora es estadísticamente material y la trayectoria no termina en ningún lado útil.** Una regla que solo pregunta "¿`I_n` > 0.02 con IC que excluye cero?" dice "seguí entrenando" indefinidamente mientras el modelo pasa de ver el corpus 5 veces a verlo 134.

Por eso insisto en que la materialidad tiene que anclarse a la **distancia restante**, no a una constante absoluta. La forma que propongo:

```
D = NLL_actual − NLL_referencia        (referencia = KN5, o el umbral de LN)
n = D / I_n                            duplicaciones necesarias al ritmo actual

VIABLE_TRAJECTORY:      n ≤ 3
MARGINAL:               3 < n ≤ 6
NON-VIABLE_AT_THIS_RATE: n > 6
```

Eso convierte un test de significancia en una decisión. Y con los números de arriba, la referencia KN5 ya cae en `NON-VIABLE` sobre este corpus.

### Falta una categoría en tu taxonomía

Tenés `UNDERTRAINED / PLATEAU / DEGRADING`, y después `capacity / optimizer / objective`. Ninguna cubre esto. El diagnóstico más probable dados estos números no es capacidad ni optimizador: es **`CORPUS_LIMITED`**. 1.97M tokens únicos no alcanzan para llevar ninguna arquitectura a competencia lingüística, y a 134 épocas estás midiendo memorización, no lenguaje. Si eso es lo que pasa, ni la `m`-curve, ni la `d`-ancla, ni el readout probe van a mover nada — y los tres estaban en tu rama post-plateau.

### Un segundo hallazgo en los mismos archivos, que pide verificación

```
update 2000:   K1 media 6.704    K4 media 6.652    Δ = 0.052
update 5000:   K1 media 6.379    K4 media 6.354    Δ = 0.025
```

La ventaja de K4 **se reduce a la mitad** entre 2000 y 5000. A 5000, la mejor seed K1 (6.360) está prácticamente en la media de K4 (6.354).

`K-CURVE-A` seleccionó K4 con `D(K1→K4)=+0.0448, IC90% [+0.0217,+0.0678]` medido exactamente a 2000 updates — el presupuesto donde el efecto es mayor. No estoy diciendo que K4 esté refutado: son corridas de scoping, `old-513`, VALL de 4096 tokens, y K-CURVE-A es una campaña distinta con 5 seeds. Lo que digo es que **la selección arquitectónica central de la campaña puede ser un artefacto de presupuesto, y hay datos propios que lo sugieren.** Eso merece una verificación antes de que K4 siga cargando autoridad.

---

## Concesiones

**`B_over_measured_dram`.** Tenés razón sobre el instrumento: es MAC/s × 1 byte/MAC, un ancho de banda efectivo derivado, no un contador físico. No es uProf. Retiro "contaminado por caché" como si fuera una medición de origen de bytes.

Pero defiendo la inferencia como **cota, no como hipótesis**. Si `dram_gbps_measured=32.93` es un techo válido para esta máquina, entonces cuando B mueve 85.01 GB/s de bytes de peso, como máximo 32.93 pudieron venir de DRAM: **≥61% no vino de DRAM**, por aritmética. Y como una calibración tipo STREAM es un techo *optimista* para accesos no streaming, la cota subestima la fracción de caché. No dice de dónde vino cada byte; sí acota cuántos no pudieron venir de ahí.

Dicho eso, tu diagnóstico es mejor que mi número, porque es mecanístico y accionable: `block_count = variant==B ? depth : 1` explica *por qué*, y dice que el arreglo es dimensionar el working set de B explícitamente, no agregar contadores. Y agrego un dato que refuerza tu conclusión: incluso a `depth=16`, el punto donde B por fin ejerce su papel, `B_over_measured_dram=1.19` — sigue 19% por encima del techo. El control no queda limpio en ningún punto del barrido. Acepto tu ledger de T0 tal como lo escribiste.

**Readout probe.** Tenés razón: `1024→256→50257` confunde rango, atado/desatado, capacidad lexical y geometría del decoder en un solo cambio. Mi diseño era malo.

**Generación antes de congelar LN.** Acepto el orden. El argumento de contaminación de constructo es correcto y es la disciplina de la casa.

**Umbrales del memorization probe.** Concedidos. Eran arbitrarios y tus cuatro objeciones son válidas.

---

## Donde defiendo

### 1. Tu readout probe tiene un confound propio, y peor porque es invisible

`estado[1024] → r → 768 → lm_head congelado de DistilGPT2` resuelve el problema de parámetros, pero introduce otro: el `lm_head` del teacher está atado a su embedding y definido sobre **la geometría de representación de DistilGPT2**. La pregunta que responde es "¿el estado de OMEGA es linealmente alineable con el espacio del teacher?", que no es "¿el estado de OMEGA determina linealmente el siguiente token?". Un estado puede contener la información completa en una geometría no alineable con la del teacher y dar FAIL por alineación, no por contenido. Y además cambia dos cosas a la vez igual que la mía: rango *y* decoder — solo que con menos parámetros.

La separación limpia es reconocer que hay **dos preguntas distintas**:

**Q1 — ¿la proyección pierde información?** Congelar todo, incluido el embedding atado, y reentrenar **únicamente `output_projection` [1024→128]**. 131k parámetros. Cero cambio de rango, cero cambio de decoder, cero cambio de capacidad lexical. Si la NLL baja de forma material, la proyección actual está mal ajustada. Si no baja, ya extrae todo lo que este decoder puede usar. Es más barato y más apretado que tu versión y que la mía, y debería ir primero.

**Q2 — ¿rango 128 es techo?** Esta no se puede responder sin cambiar la dimensión del decoder, porque el decoder *es* el rango. Pero **ya tenés datos**:

```
VALL 60-doc @2000:  rango 128 = 6.6139    rango 32 = 6.9828     Δ = 0.369
VALL 8-doc  @2000:  rango  64 ≈ 6.854     rango 32 ≈ 7.005      Δ = 0.151
```

ER32 y ER64 se diseñaron como reducción de costo y se leyeron como "la factorización cuesta calidad". Leídos como barrido de rango dicen otra cosa: **cada halving de rango cuesta 0.15–0.22 nats, o sea la derivada respecto al rango no es cero en 128.** Eso no prueba que 256 compraría algo, pero es evidencia de que el eje no está saturado, y ya está pagada. Es el dato más barato del roadmap: cero corridas nuevas.

Tu probe con decoder del teacher sigue siendo útil, pero como medición distinta: "cuánto de la distribución del teacher es linealmente recuperable del estado de OMEGA" — fidelidad de destilación, que es relevante dado el término KL. No lo llamaría readout-rank probe.

### 2. Memorization probe: concedo umbrales, defiendo el instrumento con tres arreglos

Tus cuatro objeciones tienen arreglo de diseño, no requieren descartar la sonda:

1. *KL en conflicto con memorizar* → correrla **CE-only**. El control de memorización no tiene que compartir el objetivo del entrenamiento real.
2. *Optimizer puede impedir el mínimo* → **permitir tuning de LR dentro del control**. No hay claim de generalización; cuando la pregunta es "¿esta clase de funciones puede ajustar estos datos?", tunear está permitido.
3. *La tabla lexical atada puede memorizar sin demostrar capacidad del core* → esta es tu objeción más fuerte y acepto que invalidaba mi versión. Arreglo: **congelar el embedding** en sus valores entrenados y entrenar solo los 0.6M del core. El éxito pasa a ser atribuible al core.
4. *100k tokens no tiene relación teórica con 0.6M* → correcto, y por eso el umbral absoluto sobraba. Reemplazo por uno **relativo**: correr el mismo control a ancho `d` y `2d`. Si duplicar el ancho no mueve el piso de memorización, no es capacidad. Si lo mueve, sí.

Con esos cuatro cambios sigue siendo el discriminador más limpio de capacidad que tenemos, y sigue sin ser blocker.

### 3. La bifurcación en generación tiene una síntesis que no cuesta contaminación

Acepto no mirar. Pero tu Fase D gasta cinco seeds y raters humanos sobre un checkpoint de 2k cuya NLL está en 6.6 — territorio donde las generaciones viejas eran `distinct-1 = 0.091`.

La salida no es mirar antes: es meter un **early-exit predeclarado dentro del protocolo congelado**, decidido ahora:

```
Stage 0 (automático, sin raters):
    generar bajo la política de decode congelada
    si distinct-1 < X en todas las seeds:
        registrar DEGENERATE_OUTPUT
        no ejecutar la fase humana
```

Y `X` se deriva **del corpus, no de outputs del modelo**: calcular distinct-1 sobre ventanas de 64 tokens de texto real del mismo VALL. Eso da un número de referencia sin que ningún modelo participe, y sin contaminación posible. Un piso en la mitad de ese valor es defendible y queda predeclarado.

Así no se mira nada antes de congelar, y la rama FAIL —que dados estos números es la probable— es barata. Tu argumento de eficiencia ("no gastemos días diagnosticando un modelo que quizá ya pasa") es correcto; el mismo argumento aplica en la otra dirección.

### 4. Calibración absoluta: acepto KN5 y agrego un cuarto ancla

KN5 es la adición correcta y mejor que mi propuesta original. Dos precisiones: entrenarlo sobre **la misma tokenización BPE de 50257**, no word-level, o las NLL no son comparables; y registrar el conteo exacto de tokens de train.

Agrego un ancla que cuesta cero y es la más diagnóstica de las cuatro: **OMEGA con el estado recurrente anulado**, evaluado sobre el mismo VALL. Ya demostraron causalmente que resetear/mezclar el estado empeora la NLL, pero no dónde cae "sin estado" en la escala absoluta. Si resulta:

```
unigram  →  OMEGA-sin-estado  ≈  OMEGA-con-estado  →  KN5  →  DistilGPT2
```

entonces la recurrencia —el corazón de la tesis— está comprando menos que un contador de 5-gramas, y eso se sabe antes de gastar un rater.

---

## Predicción falsable, para el ledger

Antes de que corra la Fase B: predigo que KN5 sobre el mismo VALL BPE cae **por debajo de NLL 5.5**, y que OMEGA K4 @2k (6.61–6.66) quedará **más de 1.1 nats por encima de un 5-gram**. Si me equivoco y OMEGA supera a KN5, retiro buena parte de lo que dije arriba y la lectura de la campaña cambia sustancialmente.

---

## Roadmap: acepto el tuyo con tres enmiendas

Acepto A→E y acepto tu distinción entre *bloquea medir LN* y *bloquea interpretar un FAIL*. Es más limpia que mi ordenamiento. Enmiendas:

1. **La Fase E ya tiene datos.** Antes de programar 2k→32k, extraer y clasificar formalmente las 6 curvas existentes hasta 5000 bajo tu propia regla. Eso es Fase A (ledger, cero entrenamiento), no Fase E.
2. **`CORPUS_LIMITED` entra a la taxonomía**, y la regla de parada se ancla a la distancia restante (`n = D / I_n`), no a una constante de 0.02.
3. **El early-exit por degeneración entra dentro del protocolo LN congelado**, con `X` derivado del corpus.

Y una que no es enmienda sino bandera: **la convergencia K1↔K4 entre 2000 y 5000 necesita verificación antes de que K4 siga siendo la profundidad operativa seleccionada.** Si se confirma, no invalida la campaña, pero sí significa que la selección se hizo en el punto de máximo efecto y que hay que rehacerla en presupuesto maduro — junto con `m`, no después de `m`.

Sobre tu pedido de atacar ya el protocolo LN: de acuerdo, y creo que ahora tenemos mejor base para fijar los umbrales, porque KN5 y las anclas absolutas dan una escala donde "mínimo" deja de ser una intuición. Los gemelos contrabalanceados, el diseño de raters y el número de prompts los ataco en la próxima si vos mandás tu v0.1 revisada.

### GPT 5.6 Sol

Verifiqué las seis curvas históricas hasta 5000: existen y sostienen que la receta **old-513 seguía mejorando claramente**; también confirmé que la diferencia K1−K4 cae de ~0.052 en 2000 a ~0.025 en 5000, pero **no de forma monótona** —en 4500 vuelve a ~0.054—, así que todavía no llamaría a eso convergencia K1↔K4. La observación de Opus sobre el presupuesto es importante, pero debe vincularse a la receta histórica, no trasladarse automáticamente al document-balanced actual. :chatgpt-content-reference{index="0"}

Ésta es la siguiente respuesta que le pasaría:

Opus: ya estamos de acuerdo en la estructura general. Revisé directamente las seis `validation_curve.json`; tu hallazgo de la curva hasta 5000 es real. Pero de ahí saco una conclusión ligeramente distinta.

## 1. Learning curve histórica: acepto `UNDERTRAINED_THROUGH_5K`, pero sólo para old-513

Los valores crudos confirman:

```text
500→1000  ≈ +0.395 nats
1000→2000 ≈ +0.201
2000→4000 ≈ +0.203
2500→5000 ≈ +0.253
```

No hay plateau de la receta histórica.

Así que incorporo al ledger:

```text
OLD513_R1:
    MATERIAL_IMPROVEMENT_THROUGH_5K
    UNDERTRAINED_THROUGH_5K
```

Pero no:

```text
CURRENT document-balanced:
    UNDERTRAINED
```

porque son recetas de datos distintas.

Hay una diferencia cuantitativa crítica: old-513 exponía sólo ~308k targets únicos; Coverage-C alcanza ~82.6% del corpus multichunk. El experimento que acabamos de hacer demostró que ampliar cobertura cambia materialmente la calidad ya a igual compute.

Por tanto la forma de la curva old-513 es evidencia de que **2k nunca fue un presupuesto maduro para esa receta**, pero no predice de forma válida la pendiente del modelo actual.

Si LN@2k falla, la learning curve que necesitamos es la del **document-balanced actual**, no prolongar inferencias desde old-513.

---

## 2. No acepto todavía `CORPUS_LIMITED` como diagnóstico

Acepto que falta esa categoría causal en nuestro árbol.

Pero todavía no está establecida.

Tu cálculo de cientos/miles de épocas extrapola una mejora aproximadamente constante durante muchas duplicaciones. Eso es útil como alarma de coste, no como pronóstico.

Lo registraría como:

```text
PROJECTED_CURRENT-RECIPE-COST:
    distancia a referencia / mejora reciente
```

y si exige >6 duplicaciones:

```text
CURRENT_RECIPE_NOT_PRACTICALLY_VIABLE_AT_OBSERVED_RATE
```

Eso significa:

> no sigamos simplemente duplicando updates bajo la misma receta.

No significa todavía:

```text
CORPUS_LIMITED
```

porque el culpable podría ser:

- corpus;
- LR constante;
- clipping;
- distillation objective;
- capacidad;
- readout;
- combinación de ellos.

Además, Coverage-C ya produjo una observación causal contraria a una lectura simple de “sólo repetir más”: **más diversidad de texto mejora a igual número de updates**.

Así que añado `CORPUS_LIMITED` al árbol, pero requiere evidencia propia, por ejemplo una comparación donde aumentar corpus manteniendo razonablemente presupuesto/exposición desplace la frontera de generalización.

---

## 3. K1↔K4: la interacción con presupuesto es real; “convergencia” todavía no

Recalculé las tres seeds históricas:

```text
update   mean(K1-K4)
2000     +0.05218
2500     +0.03861
3000     +0.03861
3500     +0.04475
4000     +0.03040
4500     +0.05356
5000     +0.02515
```

Eso no es una reducción monótona.

Con sólo tres seeds y validation de 4096 tokens, el valor de 5000 es demasiado débil para afirmar convergencia; de hecho en 4500 el efecto vuelve prácticamente al nivel de 2000.

Por tanto cambio el ledger de:

```text
K4 selected
```

a algo más exacto:

```text
K4_SELECTED_AT_2K_DOCUMENT_BALANCED:
    ESTABLISHED

K_EFFECT_VS_TRAINING_BUDGET:
    UNRESOLVED
```

Estoy de acuerdo en que **si terminamos necesitando un presupuesto maduro >2k**, entonces K1/K4 debe volver a compararse en ese presupuesto antes de tratar K4 como operating depth definitivo.

Pero no bloquearía LN@2k por esta razón: LN evalúa el candidato que realmente seleccionamos a ese presupuesto.

---

## 4. Readout: acepto tu Q1; rechazo la interpretación fuerte de ER32/ER64 para Q2

Tu Q1 es buena:

```text
freeze core/state/embedding
retrain only output_projection [1024→128]
```

Eso pregunta:

> ¿la proyección actual extrae mal la información que ya está presente?

Es un diagnóstico limpio y barato.

Lo dejo:

```text
READOUT-Q1:
    APPROVED IF LN FAILS/PLATEAUS
```

Pero no acepto:

> “ER32/ER64 demuestran que la derivada con respecto al rango no es cero en 128”.

ER32 y ER64 no son una ablación congelada del mismo estado. Cambiaron la parametrización y se entrenaron trayectorias nuevas completas.

Demuestran:

```text
comprimir por debajo de la representación R1
puede costar calidad bajo esas recetas
```

No demuestran:

```text
128→256 compraría calidad
```

ni una derivada local en 128.

Eso sigue abierto.

Tu crítica a mi decoder DistilGPT2 también es correcta: mediría alineabilidad lineal al espacio del teacher, no sólo contenido next-token. Lo renombro:

```text
TEACHER-SPACE-LINEAR-PROBE
```

y deja de ser mi propuesta primaria de readout.

---

## 5. Memorization control: tus arreglos lo vuelven defendible, pero sigue siendo post-FAIL

Acepto esta versión conceptual:

```text
CE-only
embedding congelado
core entrenable
LR permitido como variable del control
comparación de capacidad entre dos anchos
```

Eso ya pregunta algo útil sobre la clase de funciones del core.

No acepto aún thresholds absolutos.

Y no lo considero blocker para ejecutar LN.

---

## 6. Acepto un early-exit automático, pero no basado sólo en Distinct-1

La idea es buena: congelamos la regla antes de generar, por lo que no contamina el diseño.

Pero `distinct-1` aislado no mide coherencia. Un texto incoherente puede ser muy diverso y uno válido puede repetir vocabulario.

Propongo derivar del **VALL real**, antes de ejecutar OMEGA, distribuciones para ventanas de 96 tokens:

```text
distinct-1
distinct-2
distinct-4
max 4-gram recurrence
exact short cycles (1..8)
```

Definimos límites `q05/q95` exclusivamente desde texto humano VALL.

Un output es `CORPUS-EXTREME-DEGENERATE` si:

```text
viola >=2 límites corpus-derived
OR
presenta un ciclo exacto de longitud <=8 repetido >=4 veces
```

Early exit humano sólo si:

```text
>=80% de los outputs OMEGA son CORPUS-EXTREME-DEGENERATE
AND
cada una de las 5 seeds tiene >=60% de outputs degenerados
```

Entonces:

```text
DEGENERATE_OUTPUT
human phase = SKIPPED_BY_PREREGISTERED_RULE
```

No es un PASS/FAIL inventado tras ver ejemplos.

Si no se activa, pasan a evaluación humana.

---

## 7. Calibración absoluta: acuerdo con tres anclas; no acepto todavía “OMEGA sin estado” como cuarta

Antes de ejecutar OMEGA en LN:

```text
UNIGRAM
MODIFIED KNESER-NEY 5-GRAM
DISTILGPT2
```

sobre exactamente el VALL seleccionado.

KN5 debe ser BPE-level con vocabulario GPT-2 y train permitido únicamente.

Tu predicción:

```text
KN5 NLL < 5.5
```

queda registrada como predicción tuya, no como threshold del experimento.

No añado todavía `OMEGA-sin-estado` porque esa frase no define una intervención única.

Nuestro `anchor_reset` existente elimina sólo el carry W0→W1; no elimina:

- estado dentro de una ventana;
- recurrencia dentro de un token;
- SlotMix.

Y resetear estado en cada token sería una intervención nueva, muy fuera de distribución.

Podemos reportar junto a la calibración los **deltas causales de estado ya establecidos**, pero no llamar a eso “NLL de OMEGA sin estado”.

---

# 8. LN v0.2: acepto tu crítica de validez de constructo

Cambio sustancialmente mi v0.1.

## 8.1 Contrastivos: gemelos contrabalanceados

Los ítems contextuales se construirán en pares espejo.

Ejemplo:

```text
A:
Mara owns the red bicycle.
Theo owns the blue bicycle.
Mara rode to the park.
Her bicycle was ...

correct = red
foil    = blue

B:
Mara owns the blue bicycle.
Theo owns the red bicycle.
Mara rode to the park.
Her bicycle was ...

correct = blue
foil    = red
```

Las continuaciones tienen exactamente las mismas palabras intercambiadas.

Por tanto una preferencia lexical fija no puede acertar el par completo.

Defino dos métricas:

```text
ITEM_ACCURACY
PAIR_SUCCESS = ambas mitades correctas
```

`PAIR_SUCCESS` será primaria para tareas de contexto.

---

## 8.2 Dos niveles automáticos

### L0 — local language competence

64 minimal pairs de:

- acuerdo/sintaxis;
- selección semántica;
- completitud local;
- plausibilidad inmediata.

Su función es comprobar que existe modelado lingüístico básico.

No todo L0 necesita contexto largo; precisamente por eso KN5 sirve como calibración.

### L1 — context-dependent short coherence

64 **pares gemelos**, 128 ítems, repartidos en:

```text
16 entity/attribute binding
16 temporal ordering
16 simple causal consequence
16 discourse/situation consistency
```

La información decisiva debe quedar >5 tokens antes de la continuación, de manera que un KN5 puro no pueda resolverla por construcción salvo correlaciones accidentales.

Score de cada candidato:

\[
S(c|p)=\frac1{|c|}\sum_i\log P(c_i|p,c_{<i})
\]

para evitar ventaja por longitud.

---

## 8.3 Gates automáticos provisionales

Quiero que los discutamos antes de congelar.

Mi propuesta actual:

### L0

```text
lower_CI90(seed accuracy) > 0.60
AND
CI90(trained - init) entirely > +0.10
```

### L1

No usaría ya mi viejo `accuracy >0.55`.

Usaría:

```text
lower_CI90(PAIR_SUCCESS) > 0.40
AND
lower_CI90(PAIR_SUCCESS_OMEGA - PAIR_SUCCESS_KN5) > +0.10
AND
CI90(item_accuracy trained-init) > 0
```

¿Por qué 0.40?

Random independent pair success = 0.25.

0.40 exige señal contextual sustancial sin pedir a 7M comportamiento de teacher.

Pero estoy dispuesto a que lo ataquemos: todavía no está congelado.

DistilGPT2 debe rendir claramente por encima de ese instrumento; si no, `INSTRUMENT_INVALID`.

---

# 9. Generación humana: acepto 48 prompts y diseño incompleto balanceado de seeds

Tu crítica de n efectivo=12 era correcta.

Cambio a:

```text
48 prompts independientes
2 de 5 OMEGA seeds por prompt
96 outputs OMEGA entrenado
```

Asignación seed↔prompt precomputada para que:

```text
cada seed aparezca 19 o 20 veces
cada par de seeds aparezca aproximadamente igual
```

Esto da un balanced incomplete block design.

Controles humanos:

```text
48 MiniMind
48 OMEGA update0 (una seed predeclarada)
```

Total:

```text
192 outputs
```

Todos anonimizados y mezclados.

3 raters independientes.

---

## 9.1 Prompts

No serán instrucciones.

Serán 48 comienzos de prosa, 12 de cada clase:

```text
entity persistence
simple causal event
temporal sequence
ordinary narrative continuation
```

20–60 tokens de contexto aproximadamente.

Nada factual-heavy.

El banco completo se congela y se comprueba contra training antes de generación.

---

## 9.2 Decode

Mantengo:

```text
max_new_tokens = 96
temperature = 0.8
top_p = 0.95
top_k = disabled
repetition_penalty = 1.0
beam = false
```

Sin mecanismos anti-loop artificiales.

Greedy se guarda como diagnóstico secundario, no como condición humana primaria.

---

## 9.3 Evaluación humana

Dimensiones 1–5:

```text
G grammaticality
R prompt relevance
C internal coherence
E entity/causal continuity
N non-degeneration
```

Mediana entre raters por output.

\[
H=(G+R+C+E+N)/5
\]

Agrupación/bootstrapping por **prompt**, no por output.

Gate que mantengo por ahora:

```text
lower_bootstrap90(H) >= 3.0
each OMEGA seed mean H >= 2.75
catastrophic human failure <=10%
```

Con 48 prompts, la varianza del gate ya es bastante más razonable que con 12.

`Krippendorff alpha ordinal <0.4`:

```text
LOW_RATER_AGREEMENT
human result cannot alone cause FAIL
```

---

# 10. Clasificación final de LN

### `MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED_AT_2K`

requiere:

```text
L0 PASS
L1 PASS
automatic non-degeneration PASS
human blind PASS
```

No exige igualar MiniMind.

### `LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K`

si no.

Y esto **NO** significa fallo arquitectónico.

Después entra la curva del candidato actual.

---

# 11. Learning curve actual sólo si LN falla

Si falla:

```text
K4 / d128 / m8
document-balanced
seed13
```

continuar:

```text
2k → 4k → 8k → 16k → 32k
```

Pero con dos salidas distintas:

### A. señal estadística

\[
I_n=NLL_n-NLL_{2n}
\]

### B. viabilidad práctica

\[
D=NLL_{\text{actual}}-NLL_{\text{referencia}}
\]

\[
N_{\text{doublings,projected}}=D/I_n
\]

Acepto usar tu métrica como **proyección operativa**, no como inferencia causal.

Si:

```text
projected doublings > 6
```

no seguimos doblando ciegamente.

Clasificación:

```text
CURRENT_RECIPE_IMPROVING_BUT_NOT_PRACTICALLY_VIABLE
```

y abrimos diagnóstico de:

```text
data/corpus
optimizer
objective
capacity
readout
```

No declaramos automáticamente `CORPUS_LIMITED`.

---

# 12. Roadmap que considero casi acordado

```text
A. RECONCILIATION LEDGER
   + old-513 learning curve through 5k
   + K-budget interaction unresolved

B. ABSOLUTE CALIBRATION
   unigram / KN5 / DistilGPT2
   zero OMEGA generation inspected

C. FREEZE LN v0.2
   twins
   48 human prompts
   preregistered degeneration early-exit

D. EXECUTE CURRENT @2K LN
   five document-balanced K4 seeds

IF PASS:
   minimum language established at 2k
   T3 proof-of-concept may be discussed
   m/d/T0 debts remain before broad architectural claims

IF FAIL:
   LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
   run current-recipe learning curve

IF CURRENT RECIPE IMPROVES PRACTICALLY:
   evaluate LN again only at predeclared checkpoint

IF IMPROVING BUT IMPRACTICAL / PLATEAU / DEGRADING:
   diagnose in evidence-driven order:
       LR/optimizer
       readout-Q1
       m
       d
       W_slot residual
       corpus expansion if indicated
```

---

## Lo que todavía necesito que critiques antes de que declaremos acuerdo

Quedan, a mi juicio, sólo cuatro cuestiones realmente abiertas:

1. **L0 threshold:** ¿0.60 + trained-init >0.10 es defendible?
2. **L1:** ¿`PAIR_SUCCESS >0.40` y ventaja ≥0.10 sobre KN5 es un mínimo correcto, o debemos definirlo de otra manera?
3. **Human gate:** ¿H≥3.0 / seed≥2.75 / catastrophic≤10% sigue siendo razonable con 48 prompts?
4. **Early-exit:** ¿aceptas el criterio corpus-derived multimétrico en lugar de Distinct-1 aislado?

Si cerramos esos cuatro puntos, creo que ya no necesitamos seguir discutiendo arquitectura antes de ejecutar la calibración y congelar LN.

Aún **sí hace falta otra respuesta de Opus**: ya estamos cerca del acuerdo, pero todavía no cerramos los cuatro thresholds/gates del protocolo LN. Cuando esos puntos queden resueltos y no sobreviva ninguna objeción material, te diré explícitamente que **ya no es necesario enviarle más mensajes a Opus**.

### Claude Opus 5

Respondo los cuatro puntos, y después tres cosas donde mantengo posición.

Antes: acepto tu corrección sobre K. Recalculé y tenés razón, no es monótono —2000: +0.052, 4500: +0.054, 5000: +0.025—. "Convergencia" fue overclaim y lo retiro. Pero el reemplazo no es "unresolved" sin más. La serie es `0.052, 0.039, 0.039, 0.045, 0.030, 0.054, 0.025`: media 0.041, **SD 0.011, rango 0.029**. K-CURVE-A reportó `IC90% [+0.0217, +0.0678]`, ancho 0.046. O sea: **la varianza del efecto según en qué checkpoint lo midas es comparable al ancho entero del IC que lo certificó, y no está dentro de ese IC.** El IC cubre varianza entre seeds; no cubre varianza de selección de checkpoint. Eso no refuta K4 — dice que un efecto medido en un único update tiene una fuente de error no contabilizada del mismo tamaño que el efecto. Lo pondría así en el ledger, no como "unresolved".

---

## 1. Umbral L0: no es defendible como está, por dos razones

**(a) El umbral nominal y el operativo difieren en 10 puntos.** Con n=64 y p=0.60, `SE = 0.061`. Para que `lower_CI90 > 0.60` hace falta observar **≈0.70**. Si el mínimo que querés exigir es 0.60, el gate está pidiendo 0.70; si querés 0.70, decilo. Igual pasa en L1: `lower_CI90(PAIR_SUCCESS) > 0.40` exige observar **≈0.50**. Los cuatro thresholds deberían escribirse en términos de lo que hay que observar, porque es lo que va a discutirse cuando salga el número.

**(b) La segunda condición es la primera.** En init, con scoring longitud-normalizada sobre minimal pairs de igual longitud, el modelo está en ~0.50 por simetría. Entonces `CI90(trained − init) > +0.10` es aritméticamente casi idéntico a `accuracy > 0.60`. Son dos condiciones que son una.

**(c) Y el problema de fondo: un KN5 pasa L0.** Concordancia, sintaxis local, selección semántica y plausibilidad inmediata son precisamente lo que un 5-grama modela bien. L0 con umbral absoluto no demuestra "modelado lingüístico"; demuestra estadística local. Vos aplicaste el comparador KN5 a L1 y no a L0, y es al revés de donde hace falta: en L1 lo diseñaste para que KN5 no pueda, en L0 es donde KN5 sí puede.

Dos salidas coherentes, elegí una: o L0 lleva `lower_CI90(OMEGA − KN5) > 0` y pasa a ser competencia, o L0 se renombra **sanity check** y se declara explícitamente que **no puede contribuir a PASS**, solo a FAIL. Lo que no sostengo es L0 como una de las cuatro condiciones de `MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED`.

## 2. L1: el comparador correcto no es KN5, es OMEGA con contexto truncado

El diseño de gemelos contrabalanceados es bueno y la métrica `PAIR_SUCCESS` resuelve limpiamente el problema de preferencia lexical. Sin objeción ahí.

Pero el comparador KN5 no hace el trabajo que parece. Diseñaste los ítems para que la información decisiva quede >5 tokens atrás, "de manera que un KN5 puro no pueda resolverla por construcción". Si eso funciona, KN5 queda en ~0.25, y entonces `lower_CI90(OMEGA − KN5) > 0.10` exige OMEGA ≈ 0.48 — casi exactamente lo mismo que el gate absoluto de 0.50. **Dos condiciones que codifican la misma exigencia.** Y si no funciona, KN5 sube, el gate se mueve, y no sabés dónde queda hasta la Fase B.

El comparador que sí muerde, y que además mide la tesis de OMEGA directamente:

```
PAIR_SUCCESS_full      prompt completo, estado inicial cero
PAIR_SUCCESS_trunc5    mismos pesos, mismo scoring, prompt truncado
                       a los últimos 5 tokens, estado inicial cero

condición: lower_CI90(full − trunc5) > 0
```

Mismos pesos, misma arquitectura, misma política de decode, **una sola intervención bien definida**. Esto responde tu objeción del §7 a "OMEGA sin estado": truncar contexto no es resetear estado en cada token ni es fuera de distribución — es evaluar con un prompt más corto, algo que el modelo hace de todos modos al principio de cada documento.

Y es la pregunta de la tesis en un número: *¿el estado recurrente transporta información a través de una distancia que un modelo local no cubre?* Si `full ≈ trunc5`, OMEGA no está usando su estado en esta tarea y `PAIR_SUCCESS = 0.55` no significaría lo que queremos que signifique.

Lo pondría como condición primaria de L1, con KN5 como referencia descriptiva.

Un detalle menor: 16 ítems por categoría (32 con gemelos) es demasiado ruidoso para gatear por categoría. Predeclarar que solo el pooled gatea.

## 3. Gate humano: la estructura está bien, faltan dos anclajes y una simetría

48 prompts con BIBD y clustering por prompt corrige mi objeción de n=12. Bien.

**(a) Falta el control techo.** Tenés MiniMind y OMEGA-update0 (piso). No tenés texto humano real. Agregá 48 ventanas del VALL, misma longitud, mismo anonimato, mezcladas en el pool. Dos razones: es el `INSTRUMENT_INVALID` que sí exigiste para L1 (DistilGPT2 debe superar el instrumento) y omitiste acá; y sin él, "H ≥ 3.0" es un punto arbitrario de una escala sin anclar. Con texto real en ~4.3 y update0 en ~1.1, el 3.0 pasa a ser una posición interpretable. Cuestan 48 ítems más en el mismo pool.

**(b) La regla de Krippendorff es asimétrica y eso es un agujero.** Escribiste: `α < 0.4 → human result cannot alone cause FAIL`. Si raters ruidosos no pueden causar FAIL pero sí pueden causar PASS, el ruido es un pase libre. Tiene que ser simétrico: `α < 0.4 → el resultado humano no puede por sí solo causar FAIL **ni** PASS`, y LN queda `HUMAN_INCONCLUSIVE`. Además, α con 3 raters tiene su propio IC ancho; predeclarar si el corte de 0.4 usa el estimador puntual o el límite inferior, o vas a tener la misma discusión un nivel más abajo.

**(c) "catastrophic human failure ≤ 10%" necesita definición predeclarada.** Hoy es un juicio post-hoc. Sugerencia: cualquier output donde la mediana de raters sea 1 en ≥2 de las 5 dimensiones.

Con esas tres cosas, H ≥ 3.0 / seed ≥ 2.75 me parece razonable para 48 prompts. Sin el ancla de texto real, no puedo defender el 3.0 ni atacarlo — no significa nada todavía.

## 4. Early-exit: acepto tu versión, con una validación del instrumento

Tu criterio multimétrico corpus-derived es mejor que mi distinct-1 aislado. Concedido. Dos agregados:

**(a) Medí la tasa de falsos positivos de la regla sobre texto humano.** Con límites `q05/q95` y 5 métricas, una ventana humana normal viola al menos un límite con probabilidad no despreciable; bajo independencia, P(≥2 violaciones) ≈ 8%. Están correlacionadas, así que será menos, pero hay que **medirlo, no asumirlo**: correr la regla sobre ventanas humanas held-out del VALL y reportar qué fracción sale `CORPUS-EXTREME-DEGENERATE`. Si supera ~20%, la regla es demasiado laxa y el `≥80%` deja de ser informativo. Es el mismo tipo de control de instrumento que ya hacés en todo lo demás.

**(b) El early-exit tiene que estar conectado a la clasificación, no solo al ahorro de raters.** `DEGENERATE_OUTPUT` implica que `automatic non-degeneration` falla, y por tanto LN es FAIL. Tu estructura lo implica pero conviene escribirlo, para que nadie lea "human phase SKIPPED" como resultado neutro.

**(c)** Las distribuciones de referencia deben contarse a nivel BPE con el tokenizer GPT-2, no a nivel palabra.

---

## Un hueco de orden en tu roadmap

La Fase B como la escribiste produce **NLL** de unigram/KN5/DistilGPT2. Pero la Fase C congela un gate L1 con un threshold relativo a KN5, y L0 —si aceptás mi punto 1— también. **Para congelar C hace falta el score de KN5 y DistilGPT2 *a través de los instrumentos L0 y L1*, no su NLL.** Son cosas distintas y la Fase B actual no las produce.

Concretamente, la Fase B debe emitir:

```
NLL:            unigram, KN5, DistilGPT2
L0 accuracy:    KN5, DistilGPT2
L1 PAIR_SUCCESS: KN5, DistilGPT2
degeneration:   distribuciones q05/q95 sobre texto humano VALL
                + tasa de falsos positivos de la regla
```

Nada de eso toca OMEGA, así que no hay contaminación. Y sin eso, C no se puede congelar honestamente.

---

## Donde mantengo posición

**1. La curva old-513 apunta más fuerte de lo que le concedés.** Usás la diferencia de receta para bloquear la inferencia hacia document-balanced. Pero document-balanced multiplica los targets únicos por ~6.4× (308k → 1.97M). Más datos únicos con el mismo número de updates debería, si algo, mantener la pendiente **más tiempo**, no menos — menos repetición, saturación más tardía. La curva old-513 sin plateau a 5k es entonces una **cota inferior conservadora** sobre cuánto sigue mejorando document-balanced, no una observación de otra receta que no aplica. Sigo sosteniendo que `UNDERTRAINED` es la lectura más probable del candidato actual, aunque acepte que formalmente falta medirlo.

**2. Sobre `CORPUS_LIMITED` coincidimos más de lo que decís.** Acepto `PROJECTED_COST` y retiro el diagnóstico. Pero tu contraargumento —"Coverage-C demostró que más diversidad mejora a igual compute"— es evidencia *a favor* de que los datos son una restricción activa, no en contra. Si agregar tokens únicos compra calidad sin agregar cómputo, el corpus está atando. Lo que refutás es "repetir más alcanza", que es exactamente lo que yo estaba diciendo. Lo dejo como hipótesis en el árbol, con la evidencia que pedís, pero no la trataría como la menos probable de las seis.

**3. ER32/ER64: concedido.** Tenés razón, son reparametrizaciones con trayectorias de entrenamiento nuevas, no una ablación congelada del mismo estado. No establecen una derivada local en 128. Retiro esa lectura. Q2 queda abierta y el único camino limpio es el que vos rechazaste por costo y yo por confound, así que por ahora Q1 es todo lo que tenemos y está bien que así sea.

---

Si incorporás: L0 con comparador o renombrado a sanity, L1 con la condición `full − trunc5`, el control de texto humano y la simetría de α en el gate humano, la validación de falsos positivos del early-exit, y la ampliación de la Fase B a los instrumentos — entonces no me queda objeción material al protocolo y estoy de acuerdo en congelar.

### GPT 5.6 Sol

Opus ya no tiene objeciones materiales al esquema general; sus últimos puntos son principalmente de **definición exacta de los gates** y dos interpretaciones causales que conviene corregir. También acepta ampliar la calibración previa para que KN5/teacher pasen por los propios instrumentos L0/L1, y propone anclar la fase humana con texto real. :chatgpt-content-reference{index="0"}

Pásale esta respuesta:

Estamos prácticamente en acuerdo. Incorporo casi todas tus correcciones. Quiero cerrar dos cuestiones semánticas antes de congelar el protocolo, y después te doy los gates exactos que propongo como versión final.

# 1. K vs presupuesto: acuerdo en el fenómeno, corrección en cómo describirlo

Acepto plenamente esto:

> el efecto K1→K4 depende del presupuesto de entrenamiento y esa dependencia no fue caracterizada por K-CURVE-A.

Pero no lo describiría como una “fuente de error no contabilizada” del IC de K-CURVE-A.

Ese IC respondía a un estimando predeclarado y perfectamente específico:

```text
efecto K1→K4
en update=2000
bajo document-balanced
```

La variación del efecto a update 2500/3000/... es **effect modification por training budget**, no error muestral omitido en el estimando de 2000.

Comparar directamente:

```text
rango temporal del efecto
vs
ancho del IC entre seeds
```

mezcla dos dimensiones diferentes de incertidumbre.

El ledger que propongo es:

```text
K4 > K1 AT_2K:
    ESTABLISHED

K_EFFECT_STABLE_ACROSS_TRAINING_BUDGET:
    NOT ESTABLISHED

K4_AS_MATURE_OPERATING_DEPTH:
    NOT ESTABLISHED
```

Eso conserva el resultado válido y evita darle una autoridad que no tiene.

Si LN obliga a madurar el presupuesto más allá de 2k, entonces K debe revalidarse en ese presupuesto junto con `m`.

---

# 2. Old-513 no es una cota inferior formal de la pendiente document-balanced

Entiendo tu intuición:

```text
más datos únicos
→ menos repetición
→ potencialmente saturación más tardía
```

y la considero una hipótesis plausible.

Pero no es una cota matemática.

Cambiar old-513→document-balanced altera:

- distribución de ejemplos;
- frecuencia de revisita;
- diversidad del gradiente;
- clipping;
- dificultad media de los chunks;
- correlación entre updates.

La pendiente puede subir, bajar o cambiar de forma.

Así que registro:

```text
CURRENT_DOCUMENT_BALANCED_UNDERTRAINED_AT_2K:
    PLAUSIBLE / HIGH PRIOR PROBABILITY
    NOT YET ESTABLISHED
```

y old-513 aporta evidencia histórica que hace esa hipótesis razonable.

No la trato como prueba.

---

# 3. Distinción de datos: ya demostramos DATA-COVERAGE-LIMITED, no CORPUS-SIZE-LIMITED

Aquí creo que podemos afinar terminología.

Coverage-C sí estableció causalmente:

```text
OLD TRAINING POLICY:
    DATA-COVERAGE-LIMITED
```

porque mostrar más regiones del mismo corpus, a igual compute y controlando document weighting, mejoró VALL.

Eso es distinto de:

```text
CORPUS-SIZE-LIMITED:
    1.97M tokens únicos son insuficientes para competencia lingüística
```

Esto último sigue siendo hipótesis.

Así evitamos discutir usando “corpus limited” para dos cosas distintas.

---

# 4. L0: acepto tu segunda opción — pasa a ser SANITY, no evidencia positiva principal

Renombro:

```text
L0_LOCAL_LANGUAGE_SANITY
```

Su función:

> si ni siquiera existe estadística lingüística local aprendida, LN no puede pasar.

Pero **pasar L0 no contribuye por sí mismo a afirmar competencia lingüística**.

El gate queda escrito en términos observables:

```text
mean_accuracy_across_5_seeds >= 0.60

AND

CI90(
    accuracy_trained_seed
    -
    accuracy_init_same_seed
) entirely > 0
```

No vuelvo a escribir `lower CI >0.60`, porque correctamente señalaste que eso operativamente exigía ~0.70.

KN5 y DistilGPT2 se reportan en L0 como calibradores.

Si KN5 supera ampliamente a OMEGA en L0, eso no hace fallar automáticamente LN; deja claro que la parte local de OMEGA no supera un baseline corto.

L0 sólo puede:

```text
PASS_SANITY
FAIL_SANITY
```

Nunca producir `LANGUAGE_COMPETENCE_ESTABLISHED` por sí mismo.

---

# 5. L1: acepto `full − trunc5` como comparador causal primario

Ésta es una mejora clara respecto a KN5 como gate.

Banco:

```text
64 twin pairs
128 items
```

Primaria:

```text
PAIR_SUCCESS_full
```

Control causal:

```text
PAIR_SUCCESS_trunc5
```

donde trunc5 usa:

- mismos pesos;
- mismo checkpoint;
- misma continuación candidata;
- sólo los últimos 5 tokens del prompt;
- estado inicial cero.

La información decisiva de cada twin debe estar fuera de esos últimos cinco tokens.

## Gate L1 final

Quiero escribirlo también en cantidades observables.

### Condición A — competencia contextual absoluta

```text
mean(PAIR_SUCCESS_full across 5 seeds) >= 0.50
```

Random pair success independiente es 0.25.

Exigir 0.50 significa que el modelo resuelve correctamente **ambas mitades del twin en al menos aproximadamente la mitad de los pares**, en promedio.

### Condición B — uso causal del contexto distante

Por seed:

```text
Δcontext_s =
    PAIR_SUCCESS_full_s
    -
    PAIR_SUCCESS_trunc5_s
```

Gate:

```text
CI90(mean Δcontext) entirely > 0
```

### Condición C — aprendizaje

```text
CI90(
    item_accuracy_trained
    -
    item_accuracy_init
) entirely > 0
```

KN5 deja de ser threshold primario.

Se conserva como calibración descriptiva.

DistilGPT2 debe demostrar que el instrumento no está roto.

No gateamos categorías individuales; sólo pooled.

---

# 6. Instrument validity de L1 antes de tocar OMEGA

Acepto tu corrección del orden.

PHASE B debe ejecutar el banco congelado sobre:

```text
KN5
DistilGPT2
```

antes de OMEGA.

Predeclaro:

```text
INSTRUMENT_INVALID_L1
```

si ocurre cualquiera:

```text
DistilGPT2 PAIR_SUCCESS_full < 0.60

OR

DistilGPT2 full - trunc5 <= 0

OR

KN5 PAIR_SUCCESS_full >= 0.50
```

Interpretación:

- teacher incapaz → banco demasiado difícil/mal construido;
- teacher no usa contexto → twins no están midiendo lo pretendido;
- KN5 alcanza 0.50 → demasiadas pistas locales.

En ese caso se corrige el instrumento **antes de generar cualquier resultado OMEGA** y se vuelve a sellar.

No se cambian gates después de ver OMEGA.

---

# 7. Human gate: acepto tus tres correcciones

## Pool

```text
96 OMEGA trained
48 MiniMind
48 OMEGA update0
48 REAL-HUMAN continuation
```

Total:

```text
240 outputs
```

Los 48 `REAL-HUMAN` son continuaciones reales de VALL:

```text
prompt real
→ siguientes 96 tokens reales
```

misma presentación y anonimización que los modelos.

No son “respuestas escritas por evaluadores”; son texto corpus real.

---

## Instrument anchoring

Antes de juzgar OMEGA:

```text
H_human > H_init
```

debe quedar claramente establecido.

Predeclaro:

```text
lower_bootstrap90(
    H_human - H_init
) > 0.75
```

Si no:

```text
HUMAN_INSTRUMENT_INVALID
```

La escala no está separando texto real de un modelo sin entrenar y no sirve como gate.

MiniMind permanece calibrador, no threshold.

---

## Agreement

Usamos:

```text
Krippendorff alpha ordinal
```

sobre las cinco dimensiones.

El gate usa el **estimador puntual**, no el límite inferior del IC:

```text
alpha >= 0.40
```

El IC bootstrap se reporta, pero no crea otro gate.

Si:

```text
alpha < 0.40
```

entonces:

```text
HUMAN_INCONCLUSIVE
```

y el resultado humano:

```text
NO puede causar PASS
NO puede causar FAIL
```

Simétrico.

Por ser co-primario, si queda HUMAN_INCONCLUSIVE:

```text
MINIMUM_LANGUAGE_COMPETENCE
    = INCONCLUSIVE
```

No READY_FOR_T3.

---

# 8. Catastrophic human failure: definición congelada

Acepto tu definición:

Un output es `HUMAN_CATASTROPHIC` si:

```text
median across raters == 1
en >=2 de las 5 dimensiones
```

Gate:

```text
catastrophic_rate <= 10%
```

sobre los 96 outputs OMEGA entrenados.

---

# 9. Human PASS final

Si `alpha>=0.40` y el instrumento está válido:

```text
lower_bootstrap90(H_OMEGA) >= 3.0

AND

each OMEGA training seed mean_H >= 2.75

AND

HUMAN_CATASTROPHIC_RATE <= 10%

AND

lower_bootstrap90(
    H_OMEGA - H_init
) >= 0.50
```

Bootstrap clusterizado por prompt.

El último término evita que un rater sistemáticamente generoso haga pasar tanto init como trained.

MiniMind y texto humano se reportan como calibración.

OMEGA no tiene que igualarlos.

---

# 10. Early-exit automático: acepto tu validación de falsos positivos

Reference construction:

```text
GPT-2 BPE
96-token windows
VALL real
```

Metrics:

```text
distinct-1
distinct-2
distinct-4
max 4-gram recurrence
exact cycles length 1..8
```

Usaremos dos subconjuntos disjuntos de ventanas reales:

```text
REFERENCE_SET:
    deriva q05/q95

CONTROL_SET:
    mide false-positive rate
```

Ninguno puede solaparse con los 48 controles humanos.

Antes de generar OMEGA:

```text
FP_rate_CORPUS_EXTREME_DEGENERATE <= 20%
```

Si no:

```text
DEGENERATION_INSTRUMENT_INVALID
```

y se corrige la regla sin haber visto OMEGA.

---

## Clasificación de un output

`CORPUS_EXTREME_DEGENERATE` si:

```text
viola >=2 intervalos corpus q05/q95

OR

contiene ciclo exacto de longitud <=8
repetido >=4 veces
```

## Early exit OMEGA

Si:

```text
>=80% de los 96 outputs OMEGA
son CORPUS_EXTREME_DEGENERATE

AND

cada una de las 5 seeds
tiene >=60% degenerados
```

entonces:

```text
AUTOMATIC_NONDEGENERATION = FAIL
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
human phase = SKIPPED_BY_PREREGISTERED_RULE
```

Esto es FAIL, no neutral.

---

# 11. PHASE B definitiva

Antes de cualquier scoring/generation de OMEGA trained:

## NLL absoluto

Sobre exactamente VALL seleccionado:

```text
UNIGRAM
MODIFIED_KNESER_NEY_5GRAM
DISTILGPT2
```

token-weighted primario.

Document-macro secundario.

## Instrument calibration

```text
L0:
    KN5
    DistilGPT2

L1:
    KN5 full/trunc5
    DistilGPT2 full/trunc5

Generation-reference:
    corpus degeneration distributions
    false-positive control
```

No MiniMind todavía.

No outputs OMEGA trained todavía.

No output humano rating todavía.

Eso permite congelar el instrumento sin contaminación.

---

# 12. Sobre la predicción KN5

Tu predicción:

```text
KN5 NLL < 5.5
OMEGA@2k > KN5 by >1.1 nats
```

queda preservada en el ledger como:

```text
OPUS_PREDICTION
```

No se usa como gate.

Si OMEGA supera KN5, actualizamos interpretación, no thresholds.

---

# 13. Clasificación LN final

Para:

```text
MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED_AT_2K
```

requiere:

```text
L0_LOCAL_LANGUAGE_SANITY = PASS

L1_CONTEXTUAL_COMPETENCE = PASS

AUTOMATIC_NONDEGENERATION = PASS

HUMAN_BLIND = PASS
```

Pero conceptualmente:

```text
L0 = prerequisite sanity
L1 + generation/human = positive evidence
```

Si L0 falla:

```text
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
```

Si L1 falla:

```text
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
```

Si automatic degeneration early-exit activa:

```text
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
```

Si human queda inconclusive por acuerdo:

```text
MINIMUM_LANGUAGE_COMPETENCE = INCONCLUSIVE
```

Si human válido falla:

```text
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
```

Ninguno significa `ARCHITECTURE_FAILED`.

---

# 14. Después de un FAIL@2k

Entonces sí medimos la curva **actual** document-balanced.

Primero:

```text
K4/d128/m8
seed13
2k→4k→8k
```

A 8k:

- congelamos ajuste secundario de power-law sólo como predictor;
- predecimos 16k/32k antes de ejecutarlos.

Después:

```text
16k
32k
```

si la regla de viabilidad no indica que continuar sea inútil.

Dos ejes de lectura:

### evidencia empírica

```text
I_n = NLL_n - NLL_2n
```

### coste proyectado

```text
D = NLL_current - NLL_reference
projected_doublings = D / I_n
```

Referencia primaria para esa proyección:

```text
KN5 NLL
```

y secundaria:

```text
threshold/level asociado al LN
```

Si:

```text
projected_doublings > 6
```

clasifico:

```text
CURRENT_RECIPE_IMPROVING_BUT_NOT_PRACTICALLY_VIABLE
```

y no sigo doblando por inercia.

No lo llamo corpus-limited.

---

# 15. Deudas arquitectónicas después

Si hace falta diagnosticar:

```text
LR / optimizer
READOUT-Q1
m={4,8,16}
local d sensitivity
W_slot residual
larger corpus
```

en ese orden sólo si la evidencia previa lo justifica.

T0 físico se completa antes de claims de deployment, no antes del gate lingüístico.

---

# 16. Mi propuesta de acuerdo

Con esto considero cerrados tus cuatro puntos:

1. L0 ya no pretende probar competencia contextual.
2. L1 usa `full−trunc5`, no KN5, como intervención causal.
3. Human tiene techo real, piso init y acuerdo simétrico.
4. Early-exit tiene control de falsos positivos corpus-derived.

También queda incorporada tu corrección de PHASE B.

Si no ves un defecto material en estas definiciones, considero que **el protocolo metodológico pre-T3 ya está acordado**.

Lo único que quedaría sería implementación/verificación mecánica de los instrumentos, no más debate arquitectónico previo a ejecutarlos.

Hay dos correcciones de fondo que conviene preservar aunque Opus esté de acuerdo con el resto: su afirmación de que old-513 es una “cota inferior conservadora” de la pendiente document-balanced no está demostrada, y la variación de K con update es una **dependencia del efecto respecto al presupuesto**, no un error que invalide el IC fijado en update 2000. La propia respuesta de Opus ya reconoce que la receta old-513 es distinta y que la pendiente actual aún no se midió. :chatgpt-content-reference{index="1"}

Todavía necesito **una última confirmación de Opus** de que no ve un defecto material en esta versión cerrada. Si la acepta —aunque sugiera detalles puramente de implementación— te diré explícitamente que **ya no es necesario enviarle más mensajes a Opus** y retomamos la orquestación normal.

### Claude Opus 5

Acepto las dos correcciones de fondo. "Cota inferior conservadora" era un argumento de plausibilidad, no una cota — lo retiro. Y tenés razón en el estimando: el IC de K-CURVE-A responde correctamente a "efecto en update=2000"; la variación con presupuesto es *effect modification*, no error omitido. Mi framing mezclaba dos dimensiones de incertidumbre. Tu ledger de tres líneas es la descripción correcta. También acepto la distinción `DATA-COVERAGE-LIMITED` vs `CORPUS-SIZE-LIMITED`.

Sobre el protocolo: hice la revisión final buscando defectos, no confirmando. Encontré cuatro. Ninguno es arquitectónico; los cuatro son huecos de definición en el documento congelado, y tres se arreglan con una línea. El cuarto exige una decisión ahora.

## 1. `AUTOMATIC_NONDEGENERATION` no tiene criterio de PASS

§13 lo exige como una de las cuatro condiciones co-primarias. §10 define solamente el disparo del early-exit: `≥80% global AND ≥60% por seed → FAIL`. En ningún punto se define qué es PASS.

Por defecto queda: "no disparó el early-exit". Es decir, **79% de outputs `CORPUS_EXTREME_DEGENERATE` cuenta como PASS de no-degeneración.** Ese es el hueco más grande del documento: una de las cuatro condiciones de `MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED` tiene el umbral más permisivo posible, por omisión.

Fix, anclado al corpus como todo lo demás:

```
AUTOMATIC_NONDEGENERATION = PASS  si:
    degenerate_rate(96 OMEGA) <= 2 × FP_rate(CONTROL_SET)

entre ese valor y el disparo del early-exit:
    AUTOMATIC_NONDEGENERATION = FAIL
    pero la fase humana SÍ se ejecuta (informativa)
```

Eso usa el número de falsos positivos que ya vas a medir, y convierte el early-exit en lo que debe ser —un ahorro de raters— en vez de en la única definición del criterio.

## 2. El umbral que invalida el instrumento es idéntico al que declara competencia

§5: `mean(PAIR_SUCCESS_full) >= 0.50` → OMEGA pasa L1.
§6: `KN5 PAIR_SUCCESS_full >= 0.50` → `INSTRUMENT_INVALID_L1`.

El protocolo dice simultáneamente que 0.50 significa "el banco tiene demasiadas pistas locales" y "hay competencia contextual". Si KN5 saca 0.49, el instrumento es válido y OMEGA necesita superarlo por 0.01.

Fix: separarlos. `KN5 >= 0.40 → INVALID`, y dejar el PASS de OMEGA en 0.50. Eso deja un margen real entre "un 5-grama puede" y "OMEGA demostró".

## 3. Falta un check de validez: DistilGPT2 en trunc5

Tus tres condiciones de `INSTRUMENT_INVALID_L1` cubren "teacher incapaz", "teacher no usa contexto" y "KN5 resuelve por pistas locales". Falta la cuarta, que es la que valida directamente la intervención causal que ahora es primaria:

```
DistilGPT2 PAIR_SUCCESS_trunc5 >= 0.40  →  INSTRUMENT_INVALID_L1
```

Si el teacher, con solo cinco tokens, ya resuelve los twins, la información decisiva no está donde el diseño dice que está, y `full − trunc5` deja de medir uso de contexto distante. KN5 no cubre este caso: es otro mecanismo de explotación de pistas locales.

Relacionado, y de transparencia más que de defecto: tus condiciones A y B de L1 son casi la misma. Si trunc5 queda en ~0.25 (chance), entonces `A: full ≥ 0.50` implica `B: full − trunc5 > 0` casi automáticamente. Es la misma redundancia que criticamos en la versión con KN5, reaparecida. No lo hace inválido —el gate no queda más débil— pero conviene escribir que B es confirmatoria de A, no independiente, para que nadie la cuente como segunda evidencia.

## 4. Dos modos de fallo del instrumento ocurren *después* de ver OMEGA, y no tienen rama

Esta es la que necesita decisión, no edición.

`INSTRUMENT_INVALID_L1` y `DEGENERATION_INSTRUMENT_INVALID` ocurren en Fase B. Tu regla —"se corrige el instrumento antes de generar cualquier resultado OMEGA y se vuelve a sellar"— es correcta y segura.

`HUMAN_INSTRUMENT_INVALID` (`lower_bootstrap90(H_human − H_init) ≤ 0.75`) y `HUMAN_INCONCLUSIVE` (`α < 0.40`) ocurren en Fase D, **después** de que los 96 outputs de OMEGA entrenado ya fueron generados y puntuados. Ahí no podés "corregir y resellar": cualquier ajuste posterior se hace con los resultados a la vista. Y §14 solo define la rama FAIL@2k; INCONCLUSIVE no tiene continuación definida.

No es hipotético. α ordinal ≥ 0.40 con 3 raters sobre cinco dimensiones subjetivas de calidad textual no está garantizado — es un régimen donde 0.3–0.5 es común. Esta rama tiene probabilidad sustancial y hoy desemboca en una decisión post-hoc, que es exactamente lo que el protocolo entero existe para impedir.

Dos cosas, ambas antes de congelar:

**(a) Predeclarar la consecuencia.** Mi propuesta: `INCONCLUSIVE` o `HUMAN_INSTRUMENT_INVALID` → no se procede a T3, y se ejecuta **una** re-corrida humana con un pool de raters ampliado y predeclarado (5 en vez de 3), sobre los mismos outputs sellados, sin tocar ítems ni dimensiones. Una sola vez. Si vuelve a quedar inconclusivo, LN se clasifica `INCONCLUSIVE` de forma terminal y la decisión de T3 se apoya solo en L1 + no-degeneración, con esa limitación escrita.

**(b) Validar el instrumento humano antes**, con un piloto. Los 48 `REAL-HUMAN` y los 48 `OMEGA update0` no requieren el modelo entrenado. Un piloto de α sobre un subconjunto de ítems del mismo tipo —distintos de los 240 finales, mismos raters, misma rúbrica— mide el acuerdo antes de gastar la sesión principal. Si α piloto < 0.40, arreglás la rúbrica o el entrenamiento de raters sin que exista un solo output de OMEGA entrenado. Es práctica estándar y elimina el modo de fallo más caro.

---

## Detalles de implementación, no objeciones

- En L0 condición 2 y L1 condición C: especificar si el CI90 es sobre las 5 seeds (n=5) o sobre los ítems. Cambia mucho el ancho y hoy es ambiguo.
- El `AND` del early-exit (≥80% global **y** ≥60% en cada seed) es conservador: con 80% global y una seed en 55% no dispara, y se gastan raters sobre un pool 80% degenerado. Con el fix del punto 1 eso deja de importar, porque esa configuración ya sería FAIL de no-degeneración aunque la fase humana corra.
- `H_init` aparece en el anclaje (`H_human − H_init`) y en el PASS (`H_OMEGA − H_init`) con una sola seed predeclarada de update0. Con n=48 outputs de una sola seed, es un piso razonable, pero conviene registrar que es un piso de una seed, no de la familia.

---

## Verdicto

Los puntos 1, 2 y 3 son ediciones de una línea cada uno. El punto 4 requiere que decidas la rama y, si aceptás, agregar un piloto de α a la Fase B.

**Con esos cuatro incorporados, no tengo objeción material al protocolo y estoy de acuerdo en congelarlo.** El resto de lo que quedaría sería verificación mecánica de los instrumentos, no debate.

Una nota final sobre el conjunto, no sobre el protocolo: lo que se congela acá es un instrumento honesto para una pregunta que la campaña llevaba 246 addenda sin hacer. Aunque el resultado sea FAIL —y mi predicción registrada dice que lo será— el valor de haberlo construido antes de mirar no cambia. Eso es lo que hace que el FAIL, si llega, signifique algo.

### GPT 5.6 Sol

Sí. Con esta última respuesta de Opus, **considero que llegamos a acuerdo material**. Él condicionó explícitamente su aceptación a incorporar cuatro correcciones; las acepto, con una precisión estadística en el punto 1 para evitar que un `FP_rate=0` accidental vuelva imposible el PASS.

**Ya no es necesario que le envíes más mensajes a Opus.** A partir de aquí retomo yo la orquestación. La discusión Sol↔Opus puede cerrarse y archivarse en MD.

Opus también deja correctamente asentado que las curvas históricas old-513 muestran mejora material hasta 5k, pero que eso no establece formalmente la pendiente del régimen document-balanced actual. :chatgpt-content-reference{index="0"}

# CONSENSO FINAL SOL–OPUS
## Reconciliación arquitectónica y protocolo pre-T3

### Estado

```text
ADVERSARIAL_REVIEW:
    CLOSED / AGREEMENT_REACHED

T3:
    HOLD

Next:
    absolute calibration
    instrument verification
    freeze + execute OMEGA-MINIMUM-LANGUAGE-COMPETENCE
```

No quedan objeciones arquitectónicas o metodológicas que requieran otra ronda Sol↔Opus antes de ejecutar el programa acordado.

---

# 1. Ledger arquitectónico reconciliado

## T0

```text
T0-M matrixization:
    PASS_STRONG en el scope D512 medido

T0-R residency:
    substantial physical evidence exists
    causal isolation / clean reference range NOT ESTABLISHED

Bridge-1:
    FAIL_RECORDED bajo su criterio histórico

Complete original T0 program:
    NOT CLOSED

Transfer D512 hardware evidence → D128 linguistic surrogate:
    NOT ESTABLISHED
```

No se reabre T0 antes del gate lingüístico.

Se completará antes de cualquier claim amplio de deployment CPU-native.

---

## Escala

```text
original architectural target:
    d≈512
    m∈{4,8,16}

linguistic surrogate actually studied:
    d=128
    m=8
```

El surrogate actual no constituye todavía validación de la escala arquitectónica final.

Contabilidad obligatoria:

```text
lexical embedding ≈ 6.43M
procedural/core+shell ≈ 0.60M
external memory = 0 actualmente
```

No volver a describir el sistema actual simplemente como “núcleo de 7M”.

---

## Profundidad K

```text
K4 > K1 AT UPDATE 2000:
    ESTABLISHED

K4 > K6 additional benefit:
    NOT ESTABLISHED

K_EFFECT_STABLE_ACROSS_TRAINING_BUDGET:
    NOT ESTABLISHED

K4_AS_MATURE_OPERATING_DEPTH:
    NOT ESTABLISHED
```

La variación del efecto con update es `effect modification`, no error omitido en el IC de K-CURVE-A.

Si el presupuesto lingüístico debe madurar >2k, K debe volver a comprobarse en ese presupuesto.

---

## Workspace m

```text
hardware m/S sweep:
    ejecutado en T0-M

linguistic T2 m-curve:
    NOT EXECUTED
```

Debe volver al roadmap:

```text
m ∈ {4,8,16}
K≈4
```

pero sólo cuando exista un presupuesto lingüístico suficientemente maduro.

---

## Datos

Coverage-C estableció:

```text
DATA-COVERAGE-LIMITED:
    ESTABLISHED para la receta old-513
```

porque ampliar regiones observadas del mismo corpus mejoró calidad a igual compute controlando weighting documental.

No confundir con:

```text
CORPUS-SIZE-LIMITED:
    NOT ESTABLISHED
```

La hipótesis de que ~1.97M tokens únicos sean insuficientes sigue abierta.

---

## State conditioning

El modelo preserva identidad por slot.

La pérdida específica es sólo:

```text
state-conditioned write
    recibe state.mean(slots)
```

Candidato futuro acordado:

```python
write =
    token_part
    + global_mean_path
    + per_slot_residual
```

con:

```text
W_slot [128,128]
zero-init
+16,384 parámetros
```

Estado:

```text
OMEGA-STATE-CONDITIONING-RESIDUAL-A:
    DEFERRED
```

No blocker para LN.

---

# 2. Evidencia lingüística histórica

El audit antiguo mostró generación severamente degenerada bajo:

```text
R1 old-513 @2k
```

pero no debe extrapolarse al candidato actual.

Estado:

```text
OLD old-513 free generation:
    strong negative evidence

CURRENT document-balanced K4 @2k:
    language competence UNKNOWN
```

No se observarán generaciones nuevas del candidato actual antes de congelar el protocolo LN.

---

# 3. Fase B — ABSOLUTE-LM-CALIBRATION

Ejecutar antes de cualquier resultado de OMEGA entrenado.

## NLL absoluto

Mismo VALL seleccionado:

```text
UNIGRAM
MODIFIED KNESER-NEY 5-GRAM
DISTILGPT2
```

Condiciones:

```text
GPT-2 BPE vocab
training data permitido únicamente
no cruzar artificialmente fronteras documentales
VALL token-weighted = primario
document-macro = secundario
```

Predicción de Opus, preservada pero NO gate:

```text
KN5 NLL < 5.5
OMEGA@2k quedará >1.1 nats peor que KN5
```

---

## Instrument calibration

Antes de OMEGA:

```text
L0:
    KN5
    DistilGPT2

L1:
    KN5 full/trunc5
    DistilGPT2 full/trunc5

Degeneration detector:
    corpus REFERENCE_SET
    corpus CONTROL_SET
```

No ejecutar todavía outputs del OMEGA entrenado.

---

# 4. L0 — LOCAL-LANGUAGE-SANITY

64 minimal pairs.

Función:

> detectar si existe aprendizaje lingüístico local básico.

No constituye evidencia positiva suficiente de competencia contextual.

Score:

\[
S(c|p)=\frac{1}{|c|}\sum_i\log P(c_i|p,c_{<i})
\]

Tie:

```text
item accuracy = 0.5
```

Gate:

```text
mean_accuracy_across_5_trained_seeds >= 0.60
```

y, pareado por seed:

```text
CI90(
    accuracy_trained_seed
    -
    accuracy_init_same_seed
) entirely > 0
```

El CI se calcula sobre las **5 seeds**, no sobre los ítems.

Salida:

```text
PASS_SANITY
FAIL_SANITY
```

L0 es prerequisite; no puede por sí solo establecer competencia lingüística.

KN5/teacher se reportan como calibradores.

---

# 5. L1 — CONTEXT-DEPENDENT SHORT COHERENCE

64 twin pairs = 128 ítems.

Categorías:

```text
16 entity/attribute binding
16 temporal ordering
16 simple causal consequence
16 discourse/situation consistency
```

Cada twin invierte qué continuación es correcta manteniendo vocabulario comparable.

La información decisiva debe encontrarse fuera de los últimos cinco tokens.

## Scoring

Por cada mitad:

```text
preferred score > foil score → correct
```

Tie:

```text
item accuracy = 0.5
pair success = 0
```

`PAIR_SUCCESS` requiere ambas mitades estrictamente correctas.

---

## Full vs trunc5

```text
FULL:
    prompt completo
    initial state zero

TRUNC5:
    últimos 5 tokens del mismo prompt
    initial state zero
```

Mismos pesos y mismas continuaciones.

---

## Validación del instrumento antes de OMEGA

`INSTRUMENT_INVALID_L1` si:

```text
DistilGPT2 PAIR_SUCCESS_full < 0.60

OR

DistilGPT2 PAIR_SUCCESS_trunc5 >= 0.40

OR

DistilGPT2 full − trunc5 <= 0

OR

KN5 PAIR_SUCCESS_full >= 0.40
```

Si ocurre:

```text
corregir banco
revalidar
reseal
```

todo antes de observar OMEGA entrenado.

---

## Gate L1 OMEGA

### A — competencia contextual absoluta

```text
mean PAIR_SUCCESS_full across 5 seeds >= 0.50
```

### B — efecto causal de contexto distante

Para cada seed:

\[
\Delta context_s =
PAIR\_SUCCESS_{full,s}
-
PAIR\_SUCCESS_{trunc5,s}
\]

Gate:

```text
CI90(mean Δcontext) entirely > 0
```

El CI se calcula sobre las 5 seeds.

B es confirmatoria/mecanicista de A; no se cuenta como evidencia estadísticamente independiente.

### C — aprendizaje

```text
CI90(
    item_accuracy_trained_seed
    -
    item_accuracy_init_same_seed
) entirely > 0
```

sobre las 5 seeds.

KN5 es referencia descriptiva, no threshold de PASS de OMEGA.

---

# 6. Automatic degeneration detector

Métricas GPT-2 BPE en ventanas reales de 96 tokens:

```text
distinct-1
distinct-2
distinct-4
max 4-gram recurrence
exact cycles length 1..8
```

## Reference/control split

```text
REFERENCE_SET:
    deriva q05/q95

CONTROL_SET:
    totalmente disjunto
    estima false-positive rate
```

También disjuntos de los 48 textos reales usados por raters.

---

## Validez

Sea:

```text
FP = observed false-positive rate on CONTROL_SET
U_FP = one-sided 90% binomial upper confidence bound
```

`DEGENERATION_INSTRUMENT_INVALID` si:

```text
FP > 0.20
```

El `U_FP` evita que un control con 0 falsos positivos produzca artificialmente threshold=0.

---

## Output degenerado

`CORPUS_EXTREME_DEGENERATE` si:

```text
viola >=2 límites q05/q95

OR

tiene ciclo exacto de longitud <=8
repetido >=4 veces
```

---

## PASS de no-degeneración

```text
threshold_auto =
    max(0.10, 2 × U_FP)
```

El piso `0.10` conserva el umbral automático que existía desde el primer protocolo pre-output; no se introduce tras ver resultados.

```text
AUTOMATIC_NONDEGENERATION = PASS
iff
OMEGA degenerate_rate <= threshold_auto
```

En caso contrario:

```text
AUTOMATIC_NONDEGENERATION = FAIL
```

---

## Early exit humano

Si:

```text
global OMEGA degenerate_rate >= 0.80

AND

cada una de las 5 seeds >= 0.60
```

entonces:

```text
AUTOMATIC_NONDEGENERATION = FAIL
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
human main phase = SKIPPED_BY_PREREGISTERED_RULE
```

Si automatic falla pero no alcanza early-exit:

```text
human phase todavía se ejecuta
pero no puede rescatar el FAIL automático
```

---

# 7. Human instrument pilot — ANTES de OMEGA entrenado

Éste queda incorporado como requisito.

Usar prompts/outputs PILOT completamente distintos del pool final:

```text
REAL-HUMAN pilot continuations
OMEGA update0 pilot continuations
```

Mismos raters y misma rúbrica final.

No usar ningún output OMEGA entrenado.

## Rúbrica

Cinco dimensiones 1–5:

```text
G grammaticality
R relevance
C internal coherence
E entity/causal continuity
N non-degeneration
```

Per-rater composite:

\[
H_r=(G+R+C+E+N)/5
\]

---

## Agreement

Krippendorff:

```text
alpha_ordinal
```

sobre unidades:

```text
(output, dimension)
```

y ratings 1–5.

Pilot válido si:

```text
alpha >= 0.40
```

y además:

```text
lower_bootstrap90(
    H_human_pilot - H_init_pilot
) > 0.75
```

Si falla, se permite corregir:

```text
rubric wording
rater training
instructions
```

porque todavía no existe resultado OMEGA entrenado.

Después se repite con un conjunto piloto disjunto y se vuelve a sellar.

La versión final de la rúbrica debe quedar congelada **antes** de generar OMEGA trained.

---

# 8. Human main evaluation

Pool final:

```text
96 OMEGA trained
48 MiniMind
48 OMEGA update0
48 REAL-HUMAN
```

Total:

```text
240 outputs
```

48 prompts OMEGA:

```text
2 de 5 seeds por prompt
balanced incomplete assignment
```

Cada seed aparece 19–20 veces.

---

## Decode OMEGA

```text
max_new_tokens = 96
temperature = 0.8
top_p = 0.95
top_k = disabled
repetition_penalty = 1.0
beam = false
```

No anti-loop hacks.

---

## Agreement principal

```text
alpha >= 0.40
```

con la misma definición del piloto.

Reportar IC de alpha, pero el gate usa estimador puntual.

---

## Human instrument validity

```text
lower_bootstrap90(
    H_human - H_init
) > 0.75
```

Si se cumple y `alpha>=0.40`, instrumento válido.

---

## Human catastrophic output

Output `HUMAN_CATASTROPHIC` si:

```text
median rating across raters == 1
en >=2 de las 5 dimensiones
```

---

## Human PASS

Requiere:

```text
lower_bootstrap90(H_OMEGA) >= 3.0

AND

each OMEGA trained seed mean_H >= 2.75

AND

HUMAN_CATASTROPHIC_RATE <= 0.10

AND

lower_bootstrap90(
    H_OMEGA - H_init
) >= 0.50
```

Bootstrap clusterizado por prompt.

`H_init` procede de **una seed update0 predeclarada**; se documenta expresamente que es un piso de una seed, no de toda la familia.

MiniMind y REAL-HUMAN son calibradores; OMEGA no tiene que igualarlos.

---

# 9. Rama humana INCONCLUSIVE — predeclarada

Si en la main phase:

```text
alpha < 0.40
```

o:

```text
lower_bootstrap90(H_human - H_init) <= 0.75
```

no se cambia:

```text
rúbrica
prompts
outputs
dimensiones
thresholds
```

Se permite exactamente **un rescue predeclarado**:

```text
añadir 2 raters
total raters = 5
```

sobre los **mismos outputs sellados** y misma rúbrica.

Se recalculan agreement y gates.

---

## Si el rescue pasa

Usar resultado de 5 raters.

---

## Si vuelve a fallar

Clasificación terminal:

```text
HUMAN_INCONCLUSIVE_TERMINAL
MINIMUM_LANGUAGE_COMPETENCE = INCONCLUSIVE
T3 = HOLD
```

No se permite:

```text
hacer PASS sólo con L1
hacer FAIL sólo con automático
redefinir rúbrica después de outputs
```

Una futura evaluación humana requeriría una unidad nueva y preregistrada.

---

# 10. Clasificación LN final

## PASS

```text
MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED_AT_2K
```

iff:

```text
L0_LOCAL_LANGUAGE_SANITY = PASS
AND
L1_CONTEXTUAL_COMPETENCE = PASS
AND
AUTOMATIC_NONDEGENERATION = PASS
AND
HUMAN_BLIND = PASS
```

Esto significa:

> el actual OMEGA pequeño demuestra lenguaje funcional y coherencia corta suficientes para justificar discutir un T3 proof-of-concept.

No demuestra razonamiento general.

---

## FAIL

```text
LANGUAGE_COMPETENCE_NOT_ESTABLISHED_AT_2K
```

si falla de manera válida cualquiera de:

```text
L0
L1
automatic nondegeneration
human blind
```

No significa:

```text
ARCHITECTURE_FAILED
```

---

## INCONCLUSIVE

```text
MINIMUM_LANGUAGE_COMPETENCE = INCONCLUSIVE
```

si el instrumento humano permanece inválido/inconcluso después del único rescue permitido.

T3 continúa HOLD.

---

# 11. Qué ocurre después de FAIL@2k

Sólo entonces se ejecuta la curva **del candidato actual**:

```text
K4
d128
m8
DOCUMENT_BALANCED_MULTICHUNK
seed 20260913
```

Continuar:

```text
2k
4k
8k
16k
32k
```

Primero 4k/8k.

A 8k congelar una predicción secundaria para 16k/32k.

---

## Métrica de mejora

\[
I_n=NLL_n-NLL_{2n}
\]

## Proyección práctica

\[
D=NLL_{current}-NLL_{reference}
\]

\[
projected\_doublings=D/I_n
\]

Referencia primaria:

```text
KN5 NLL
```

Secundaria:

```text
nivel asociado al gate LN
```

Si:

```text
projected_doublings > 6
```

clasificación:

```text
CURRENT_RECIPE_IMPROVING_BUT_NOT_PRACTICALLY_VIABLE
```

No seguir duplicando por inercia.

---

# 12. Diagnóstico después de curva insuficiente

Orden guiado por evidencia:

```text
1. LR / optimizer
2. READOUT-Q1
3. m ∈ {4,8,16}
4. local d sensitivity
5. W_slot residual
6. corpus expansion
```

`READOUT-Q1`:

```text
core/state/embedding congelados
retrain only output_projection [1024→128]
```

Pregunta:

> ¿la proyección actual está dejando información ya presente en el estado?

No interpretar ER32/ER64 como derivada local de rango en 128.

---

# 13. T3

T3 permanece:

```text
HOLD
```

hasta que:

```text
MINIMUM_LANGUAGE_COMPETENCE_ESTABLISHED
```

o hasta una futura unidad explícita que cambie ese criterio.

Incluso con PASS:

```text
m-curve
scale bridge
T0 completion
```

siguen siendo deudas antes de claims arquitectónicos/deployment amplios.

---

# 14. Fin de revisión adversarial

```text
Sol:
    ACCEPT

Opus:
    ACCEPT conditional on 4 fixes

All 4 fixes:
    INCORPORATED

Material objections remaining:
    NONE
```

El trabajo siguiente es implementación/verificación mecánica, no renegociación del protocolo.

**Ya no es necesario enviarle más mensajes a Opus.**

A partir de aquí vuelvo a asumir la orquestación principal. El siguiente paso operativo es `OMEGA-ABSOLUTE-LM-CALIBRATION + instrument preflight`, sin generar todavía outputs del OMEGA entrenado.

# MRDL: Modelo Relacional Disperso de Lenguaje

**Estado:** especificación consolidada + núcleo de lenguaje real implementado, escalado a 10.000 historias, con hallazgo decisivo sobre composición relacional y rediseño del operador (TFA) en evaluación — ver Parte 10.4–10.9
**Versión:** 4.0 (documento vivo — actualizar tras cada ronda de revisión, no confiar en historial de chat)
**Objetivo del documento:** registrar con precisión toda la arquitectura acordada hasta ahora, incluidas las correcciones nacidas de revisión adversarial y de benchmarks reales ejecutados, para que cualquier agente (LLM o humano) pueda retomarla sin reconstruir contexto perdido.

> **Restricción de diseño original, sigue vigente:** no asumir que la solución debe converger hacia Transformers, attention densa o backpropagation global tradicional. El núcleo del conocimiento son relaciones explícitas y dispersas entre tokens/conceptos, no matrices densas de parámetros.

---

# PARTE 0 — Qué se ha hecho realmente (para no perder el hilo)

1. **Diseño en papel**, iterado con un Validator adversarial (rondas v0.2 → v2.3): representación, propagación, binding, composición, memoria y auditoría quedaron especificadas con matemática concreta, no metáforas.
2. **Implementación en VPS** (`/root/mrdl`, agente `opencode`), Stages 1–3: álgebra de evidencia (M0/M1/M2, no-lavado), trazas de ejecución/sombra, y suites de invariantes — **verificadas de forma independiente** por el Validator (no solo por el reporte del implementador; se encontraron y corrigieron 2 bugs reales en el camino).
3. **Tres benchmarks de capacidad** ejecutados y verificados con números reales (no simulados en papel): Δ-Mix, frontera no lineal (XOR/product), folding profundo.
4. **Hallazgo decisivo**: el benchmark de folding profundo demostró que el árbol de evidencia simbólico (Leaf/Serial/Alternative/Joint) explota combinatoriamente con la profundidad, independientemente de que el lado de la transformación esté acotado por beam. Esto **descarta esa representación de evidencia para composición profunda**.
5. **Rediseño completado y verificado**: "ejecución epistémica estratificada" (carriles FULL/CLEAN) reemplaza el árbol de evidencia. Los 3 puntos abiertos se cerraron, y las 5 pruebas de verificación (A–E) pasaron con re-ejecución independiente del Validator, incluidos 2 bugs reales encontrados y corregidos en el camino y un hallazgo de robustez real (colapso de CLEAN con ramificación/beam insuficiente a alta densidad M1 — ver 9.1). **Núcleo de lenguaje real (embeddings + relaciones + entrenamiento sobre corpus) queda habilitado para implementarse a partir de este punto.**

6. **Núcleo real implementado, fase E (escalado de corpus) en curso**: se encontraron y corrigieron 4 bugs reales en la maquinaria de aprendizaje del controller (presupuesto sin escalar, regla de actualización unidireccional, contaminación por nodo waypoint, falta de decaimiento de tasa de aprendizaje — ver Parte 10.2). Con los fixes, MRDL le gana a trigram en corpus chico (300 historias) pero la brecha se agranda monótonamente al escalar (137→425→627 de accuracy en sentido inverso al de trigram, ver Parte 10.3) — **patrón de fondo real, no problema de entrenamiento inestable**. También se confirmó que `a_e` (escala del operador monomial) es coseno de embeddings congelados, no una cantidad aprendida del corpus (Parte 10.1).

7. **Dos bugs más encontrados y corregidos al escalar a 10.000 historias** (contaminación de candidatos VSA "fantasma", contaminación del entrenamiento del controller por objetivo ausente del pool — Parte 10.4), llevando accuracy full-scale a 0.3483 (random_frozen) / 0.2596 (pretrained_frozen) en 4.025 historias — el primer estado del pipeline verificado como correcto de punta a punta.

8. **Hallazgo decisivo, con el pipeline ya corregido**: la composición relacional (2+ hops) no aporta nada medible. A 4.025 y 10.000 historias, con metodología correctamente controlada (Parte 10.5), las predicciones compuestas aciertan 0/29 y 0/43 veces respectivamente, usándose en <0.5% de las decisiones — escalar el corpus 2,5x hizo bajar el uso de composición, no subirlo. AOC (celdas de polaridad contextual) y la hipótesis de "crowding" detrás de RCR/REN se cerraron con evidencia negativa limpia (Parte 10.6–10.7) tras que el Validator rechazara aceptarlas sin verificación independiente.

9. **Causa aislada y corregida parcialmente**: el embudo causal G0-G5 (Parte 10.8) localizó el cuello de botella exacto en `Fold_B` (`P(G4|G3)=0.00676`, colapso de 148x). Se encontró que la implementación real de `Fold_B` no fusionaba ramas por misma permutación antes del corte top-K, contra lo que exige 4.4 — bug real, corregido y verificado con evidencia concreta (no solo reporte del implementador). Con el fix, el colapso mejora a ~105x, pero la accuracy de composición en corpus completo (10k) sigue en 1,44% contra 31,14% de predicción directa — **22x peor incluso cuando la ruta compuesta sobrevive el beam y gana top-1**. Esto descarta a `Fold_B` como causa restante y apunta a la calidad semántica del operador monomial mismo.

10. **Rediseño del operador en evaluación — TFA (Transporte Fásico Anclado)**, propuesto para reemplazar `a_e=coseno` (Parte 10.9): descompone el estado en ancla de nodo (identidad léxica) + residuo contextual transportado por una isometría por pares (preserva norma exactamente, sin decaimiento con la profundidad). Matemática verificada (P2.0: cierre bajo composición, conservación de norma, cuantización de fase exacta, todo confirmado con artifact hasheado, no solo reporte). Evaluación en sombra sobre el checkpoint real (P2.1, con rehidratación correcta del estado VSA, dos rondas de resultados rechazadas por bugs de ejecución antes de una tercera válida): ninguna variante (U=identidad, fase sola, fase+innovación+prototipos) se distingue estadísticamente del baseline — resultado es ruido, no señal, según el propio umbral que Sol pre-registró. Pendiente: definir si el problema es que el residuo contextual VSA no tiene suficiente estructura recuperable, o si la familia de operadores isométricos es insuficiente para explotarla — decisión de replanteo en curso con Sol, no resuelta a la fecha de este documento.

Si sos un agente nuevo leyendo esto: la arquitectura de representación de conocimiento (Partes 1–4) **no cambió** en ninguna ronda. Lo que cambió fue exclusivamente la maquinaria de aprendizaje/auditoría/memoria (Partes 5–7, y ahora la implementación real del núcleo en Parte 10) en respuesta a problemas concretos encontrados por revisión y benchmarks — no impresiones ni intuición.

---

# PARTE 1 — Representación fundamental (sin cambios desde v0.2)

## 1.1 Hipótesis central

Conocimiento operativo almacenado no en matrices densas de pesos, sino en una **red dispersa de relaciones aprendidas entre tokens, conceptos y patrones**.

Cinco ideas base:

1. Cada token/concepto tiene un **embedding base**, inicialmente congelado, que aporta geometría semántica general.
2. El entrenamiento crea y modifica **conexiones explícitas** entre elementos relacionados en el corpus.
3. Cada conexión contiene un **vector relacional multidimensional**, no un peso escalar.
4. El contexto es un **estado activo y disperso**, no una ventana fija de tokens anteriores.
5. La predicción surge de **competencia entre rutas relacionales compatibles con el contexto**, no de atención densa.

La intención: evaluar si esta dinámica puede ser la arquitectura PRINCIPAL de un modelo de lenguaje, no una memoria auxiliar pegada a un Transformer.

## 1.2 Nodo

```text
Node {
    id
    embedding_base        # x_i ∈ R^d_e, congelado durante entrenamiento principal
    activation
    frequency
    surface_or_concept_type
    optional_canonical_link
    active_ports[]         # ver Parte 3
}
```

Dos configuraciones experimentales necesarias para distinguir qué aporta la arquitectura de qué aporta el embedding importado:
- **Embeddings preentrenados y congelados**: semántica importada desde el día uno.
- **Embeddings aleatorios y congelados**: mide cuánto conocimiento construye la arquitectura relacional por sí sola.

## 1.3 Arista / relación

```text
e_ij = (i, j, r_ij, s_ij, q_ij, m_ij)
```
- `r_ij ∈ R^d_r` (o `Z^d_r` cuantizado, int8/int16): vector relacional.
- `s_ij`: soporte (evidencia acumulada).
- `q_ij`: confianza.
- `m_ij`: metadatos (última actualización, frecuencia, estado de consolidación).

Canales funcionales del vector relacional (regiones de un vector compacto, no necesariamente separados físicamente):

```text
RelationVector {
    semantic_delta
    role_signature
    temporal_signature
    composition_signature
    continuation_signal
    closure_signal
    confidence_state
}
```

Puede existir más de un prototipo por par de nodos (`r_ij^(1)...r_ij^(K)`, K pequeño, 2–4) para no promediar sentidos incompatibles (ej. "banco" dinero/parque/sentarse).

---

# PARTE 2 — Contexto y razonamiento (sin cambios desde v2.1)

## 2.1 Contexto como estado activo

```text
C_t = (A_t, R_t, O_t, H_t)
```
- `A_t`: nodos activos dispersos.
- `R_t`: rutas recientemente activadas.
- `O_t`: expectativas abiertas (aún sin resolver).
- `H_t`: traza temporal comprimida.

## 2.2 Cápsula de ruta (unidad activa real, no el nodo)

> **La unidad activa no es el nodo ni una activación escalar. Es una cápsula que conserva contexto, roles, procedencia y responsabilidad.** Dos cápsulas pueden coexistir sobre el mismo nodo (ej. "de") representando situaciones distintas, sin fusionarse.

```text
RouteCapsule {
    current_node
    contextual_state
    role_bindings
    open_expectations
    energy
    route_signature
    parent_references
    local_contributions
}
```

## 2.3 Sustituto de atención: propagación competitiva

No se calcula matriz de atención densa. Se activan nodos del contexto reciente, cada uno consulta un subconjunto pequeño de conexiones (Top-K disperso), la activación se propaga unas pocas rondas, las rutas incompatibles pierden energía.

```text
a_j^(k+1) = φ( Σ_{i∈A_t} a_i^(k) · g(r_ij, C_t) − I_j )
```
- `g(r_ij, C_t)`: compatibilidad relación-contexto.
- `I_j`: inhibición/repetición/conflicto/saturación.
- `φ`: integración y filtrado local.

La suma se restringe a un conjunto recuperado por índices dispersos + límite Top-K — nunca a todas las relaciones del modelo.

## 2.4 Cierre y repetición (restaurado v0.2 §22-23 — se había perdido en la consolidación v3.0, drift real detectado y corregido)

### Repeticiones y bucles

Generación tipo "de de de de..." ocurre cuando una transición o ciclo se vuelve autosostenido. Causas en esta arquitectura: autoarista fuerte, dos nodos reforzándose mutuamente, el contexto no cambia, no hay inhibición temporal, la relación más frecuente domina siempre, no se detecta que la ruta ya fue recorrida.

Mecanismos necesarios:
1. **Período refractario**: reducir temporalmente la activación de tokens recién emitidos.
2. **Inhibición de ciclos**: penalizar rutas repetidas dentro de una ventana.
3. **Saturación**: una relación pierde fuerza temporal tras usarse repetidamente.
4. **Cobertura**: favorecer relaciones que añadan información pendiente (expectativas abiertas sin resolver).
5. **Cambio de estado real**: cada token generado debe modificar de forma real el contexto — no repetir el mismo estado.
6. **Detección de bucles**: reconocer secuencias periódicas cortas explícitamente.
7. **Competencia con cierre**: si no aparece información nueva, `<EOS>` gana fuerza relativa.

La repetición no debe resolverse solo con penalización externa de decodificación — el estado interno también debe reflejar que una ruta ya fue consumida.

### Cómo termina una idea

- **Nodo de fin**: `<EOS>` compite como cualquier otro candidato en la puntuación, no es un caso especial fuera del mecanismo normal.
- **Estado de expectativas abiertas** (`O_t`): entidad mencionada sin predicado, relación iniciada pero incompleta, estructura abierta, pregunta sin respuesta, enumeración aún no cerrada.
- **Puntuación de cierre**: aumenta la probabilidad de cierre cuando no quedan expectativas fuertes, la energía de propagación baja, el patrón activo suele finalizar en la experiencia previa, la puntuación de continuar es débil, o un signo de puntuación/`<EOS>` es coherente con el estado.
- **Cierre aprendido, no programado por gramática**: no se escriben reglas tipo "después de un adjetivo termina la oración" — el sistema aprende qué estados relacionales tienden a cerrarse.

## 2.5 Profundidad dinámica

Rondas de propagación reemplazan capas fijas de Transformer: ronda 1 = relaciones directas, ronda 2 = segundo orden, rondas posteriores = composición/abstracción. Un contexto simple usa 1–2 rondas; uno ambiguo, más. Termina cuando la distribución de candidatos se estabiliza, la energía cae, o se alcanza un límite.

---

# PARTE 3 — Puertos contextuales y túneles (v2.1, corregido para no repetir el error de Capsule Networks)

## 3.1 Por qué existen

Un nodo hub (palabras frecuentes: de, la, el, que) recibe demasiada información distinta y la mezcla en un único estado — over-squashing/over-smoothing documentado en literatura de GNN (Alon & Yahav 2021; Li et al. 2018). La solución: **puertos**, no un único vector por nodo.

```text
ContextPort {
    frozen_key
    capacity
    current_load
    utility
}
```

## 3.2 Regla dura: routing de un solo paso, NUNCA agreement iterativo

Riesgo detectado y cerrado explícitamente: el mecanismo de puertos se parece a Capsule Networks (Sabour et al. 2017), cuyo routing-by-agreement iterativo es la razón documentada por la que esa arquitectura nunca escaló a producción. **Decisión de diseño: prohibido routing iterativo.**

Asignación (equivalente a lookup de codebook, estilo VQ-VAE — van den Oord et al. 2017, técnica que sí escala en producción):

```text
k_h = HashQuantized(role_h, expectations_h, path_h, bindings_h)

p* = argmax_p sim(k_h, k_p)

si similarity >= threshold y hay capacidad: asignar a p*
si no hay coincidencia y Pressure(v) es alto: abrir puerto nuevo
si se alcanzó max_ports: tránsito independiente / puerto de desbordamiento / descarte por utilidad
```

Las claves de puerto quedan **congeladas durante la ronda actual**; se actualizan por media móvil solo después. No hay reasignación de cápsulas ya procesadas.

Ambigüedad controlada sin negociación: una cápsula puede duplicarse en máximo 2 puertos, dividiendo energía (`E_{h→p1} + E_{h→p2} = E_h`) — nunca ramificación ilimitada.

```text
Pressure(v) = N_capsules · (1 + λ·H(k_h)) / (1 + N_ports)
```
Responde "¿este nodo canaliza demasiados estados distintos por pocos puertos?", no "¿cuál es la partición óptima?" — evita reintroducir routing iterativo por la puerta de atrás.

## 3.3 Túneles efímeros (graph rewiring)

Bypass compuesto para evitar que toda ruta atraviese un hub:

```text
r_{casa⇒verde} = Compose(r_{casa→es}, r_{es→verde})

EphemeralTunnel {
    source, destination, composed_relation,
    provenance_path, context_key, ttl
}
```

Existe solo en el contexto actual; si se reutiliza en muchos contextos independientes, se consolida como relación permanente. Dos niveles: túnel efímero (atajo temporal) vs. relación consolidada (patrón general aprendido).

---

# PARTE 4 — Binding de roles y composición algebraica (v2.1)

## 4.1 El binding no emerge de similitud (error de v0.2, corregido)

Objeción resuelta: similitud entre embeddings no resuelve quién es agente/objeto/recipiente. Fundamento: Fodor & Pylyshyn (1988), problema de binding de variables en conexionismo clásico.

**Roles anónimos autoinducidos** — no se definen a mano (`AGENTE`, `OBJETO`); se descubren por invariancia de sustitución:

```text
VariableScore(k) = IdentityEntropy(k) / StructuralEntropy(k)
```
Alta variación de identidad + baja variación estructural alrededor de una posición → candidata a rol.

## 4.2 Binding vectorial invertible (Vector Symbolic Architecture / Holographic Reduced Representations, Plate 1995)

```text
Bind(role, entity) = P_role(entity)      # P_role: permutación, rotación estructurada, etc.
Frame = Bind(ρ0, perro) + Bind(ρ1, pelota)
Unbind(ρ0, Frame) ≈ perro
```

Diferencia clave con VSA clásico: los roles no los define el diseñador, se crean cuando el sistema descubre posiciones estructuralmente invariantes. Roles viven **temporalmente en la cápsula contextual**, no como nodo permanente por combinación (evita explosión combinatoria: no existe nodo "perro-como-agente-de-perseguir-pelota").

## 4.3 Compose — Álgebra Relacional Monomial (v2.1, reemplaza el `Compose` sin definir de v0.2)

Cada modo de relación es un operador estructurado:

```text
R_e = (P_e, a_e, b_e, k_e^in, k_e^out, A_e, K_e)
T_e(z) = a_e ⊙ P_e(z) + b_e
```
- `P_e`: permutación con signo.
- `a_e`: escala diagonal.
- `b_e`: desplazamiento.
- Almacenamiento/aplicación/composición: O(d).

**Cerrado bajo composición** (grupo de matrices monomiales / hiperoctaédrico — verificado matemáticamente, no solo afirmado):
```text
T_{2∘1}(z) = T_2(T_1(z)) = a_{21} ⊙ P_{21}(z) + b_{21}
P_{21} = P_2 P_1
a_{21} = a_2 ⊙ Q_2(a_1)          # Q_2 = permutación de P_2 sin signos
b_{21} = a_2 ⊙ P_2(b_1) + b_2
```

Composición de expectativas abiertas:
```text
O' = (O \ K_e) ∪ A_e
K_{21} = K_1 ∪ K_2
A_{21} = (A_1 \ K_2) ∪ A_2
```

**Límite conocido, confirmado por benchmark (ver Parte 8):** suma de k ramas monomiales sigue siendo una transformación LINEAL — no puede representar interacciones no lineales (XOR, producto de canales). Expansión mínima si un benchmark demuestra que hace falta: átomos de fusión pequeños que mezclen exactamente 2 canales (`φ(z_i,z_j)`), nunca matriz densa ni MLP grande — **solo agregar si el benchmark lo exige, no antes**.

## 4.4 Explosión de ramas y Fold_B

Componer L relaciones con k ramas cada una da k^L combinaciones sin control. Solución: beam acotado + composición perezosa.

```text
1. Expandir ramas de la siguiente relación.
2. Aplicar al estado contextual actual.
3. Puntuar resultados.
4. Fusionar duplicados.
5. Conservar presupuesto B de ramas diversas.
```
```text
Compose_B(R2, R1) = Fold_B({M_{2,j}M_{1,i}}_{i,j})
```
`Fold_B` fusiona ramas con la misma permutación, suma/combina escalas, elimina ramas causalmente irrelevantes, limita a B ramas. Es beam search estándar (probado en producción desde hace décadas en traducción/parsing) — no una técnica nueva sin precedente.

## 4.5 CompositionGate — separar validez de utilidad, tres estados

```text
V(r1,r2,C) = probabilidad de que la composición sea válida
U(r1,r2,C) = utilidad computacional/predictiva del túnel

COMPOSE si V alto y U positivo
REJECT  si V bajo
DEFER   si V incierto   # no crear túnel, conservar ruta original — reduce daño de decisión incierta
```
```text
g_12 = σ(α·cos(k1_out, k2_in) + β·B_12 − ρ·X_12 − δ)
```
Pocos parámetros compartidos, no un MLP grande.

**Prueba adversarial obligatoria** (el ejemplo AMA/PRODUCIDA_POR construido a mano NO cuenta como validación — es confirmación de diseño, no generalización). Metodología inspirada en HANS (McCoy, Pavlick & Linzen, 2019 — dataset estándar para cazar heurísticas superficiales en NLI):
- **Separación por bloques**: excluir el producto cartesiano completo de familias de relaciones, no solo triples puntuales.
- **Gemelos contrafactuales**: mismo patrón estadístico local, composición inválida (`a→b, b'→c` en vez de `a→b, b→c`).
- **Alteraciones adversariales**: inversión de dirección, intercambio de roles, sustitución de nodo intermedio, negación, cambio de cuantificador, alias léxicos, embeddings similares con reglas composicionales distintas.
- **Minería de contraejemplos**: buscar falsos positivos de alta confianza, generar variantes alrededor del fallo, reevaluar sin reentrenar antes.
- **Túnel en modo sombra**: correr en paralelo sin afectar la predicción hasta comprobar que sobrevive a sustitución de entidades (si falla bajo sustitución pequeña, memorizó coincidencia, no operación general).

Métrica principal, no precisión promedio (un falso túnel puede propagar error a muchos contextos):
```text
HarmFPR = Σ(túneles inválidos aceptados) ΔLoss / N_decisiones
```

## 4.6 Confianza de ruta — sin decaimiento multiplicativo

Error de v0.2 (`q_path = Π q_i`, decaimiento exponencial disfrazado), corregido dos veces:
```text
Q_path = logit(min_i q_i) + ε·(1/ℓ)·Σ_i logit(q_i) − λ·log(1+ℓ)      # 0 < ε ≪ 1
```
El mínimo domina (eslabón más débil), el promedio solo desempata, el costo por longitud es logarítmico (sublineal), no exponencial.

---

# PARTE 5 — Memoria en tres niveles (v2.2)

## 5.1 Por qué existen tres niveles

Contradicción que había en v2.1 y se resolvió: no se puede sostener a la vez "escritura one-shot barata" Y "auditoría causal completa antes de cada escritura". Separación correcta: *recordar algo inmediatamente* ≠ *convertirlo en conocimiento relacional generalizable*.

```text
M0: estado contextual — dura la sesión/secuencia activa. Cápsulas, bindings, expectativas abiertas, túneles efímeros.

M1: memoria en custodia — escritura inmediata SIN auditoría. Costo ≈ forward + O(|trace|). Influencia acotada (confidence_cap bajo), asociada a fuente/contexto, con caducidad (ttl), nunca sobrescribe M2, no se usa para generalización lejana.

M2: memoria relacional consolidada — aristas permanentes, prototipos, abstracciones. Solo se llega vía Promote() explícito, tras auditoría contrafactual.
```

```text
EscrowRecord {
    contextual_key, observed_content, bound_frame,
    active_trace, source, timestamp, support,
    confidence_cap, ttl, contradiction_state
}
```

## 5.2 Qué propiedad one-shot se conserva y cuál no

Afirmación correcta (no la de v0.2): **"Puede almacenar y recuperar inmediatamente una observación, pero necesita evidencia o auditoría para transformarla en una relación generalizable de alta confianza."**

Pierde: atribución causal precisa, generalización segura, resistencia a información falsa, capacidad de modificar relaciones establecidas, certeza de que el contexto no fue accidental.
Conserva: memorización/recuperación inmediata, trazabilidad, reversibilidad, costo bajo.

## 5.3 Recuperación desde M1

```text
S_escrow = sim(k(C_t), k_e) · q_source · q_binding · decay(age)
```
Dos claves posibles por episodio: `surface_key` (segura, generaliza poco) y `structural_key` (generaliza más, contribución limitada mientras no esté auditada). Familia de técnica real y probada: retrieval-augmented generation (kNN-LM, Khandelwal et al. 2020; RETRO, DeepMind 2021).

## 5.4 Cuándo se paga la auditoría

NO en cada escritura — solo cuando un registro intenta ganar impacto:
```text
E[C] = C_forward + p_promotion · C_audit
```
Activadores: reutilización repetida, aparición en contextos distintos, contradicción con memoria consolidada, influencia creciente sobre predicciones, propuesta explícita de consolidación, fuente de baja confianza, error potencial alto.

## 5.5 Regla de no-lavado a través de parámetros (v2.3 → v3.0, crítico)

Riesgo identificado: separar activaciones no alcanza si un registro M1 actualiza directamente los **parámetros del controlador**, que luego se usan también para conocimiento consolidado — sería lavado por la puerta de los pesos, no de la evidencia.

> **Regla dura: una escritura M1 puede modificar memoria provisional, pero nunca los parámetros ni estados persistentes usados por niveles superiores** (controlador, claves de puertos, operadores de composición, estadísticas de normalización, cachés compartidas, criterios de beam, codebooks). La actualización compartida del controlador ocurre solo después de promoción.

**Consecuencia a tener presente, no defecto**: el aprendizaje "de escala lenta" (procedimientos generales del controlador) queda completamente gateado detrás de promoción — el sistema no mejora sus procedimientos de razonamiento a partir de experiencia fresca no auditada. Es un trade-off deliberado, consistente con el resto del diseño, pero limita la velocidad de adaptación general del sistema. Pendiente de confirmar si es aceptable a largo plazo.

---

# PARTE 6 — Auditoría, trazas y evidencia (v2.2 → v2.3 → **rediseño v3.0 en curso**)

## 6.1 Aprendizaje en dos modos (v2.1, sigue vigente)

**Modo A** (todo diferenciable): EdgeMemory + Controller en el grafo diferenciable, backprop por el subgrafo activo, sin fórmula de crédito aparte.

**Modo B** (memoria externa, preferido para conservar one-shot real):
```text
Controller θ: entrenado lentamente por gradiente.
EdgeMemory M: mutable, actualización inmediata, M_e = stopgrad(M_e) durante backprop.

M_e ← Project(M_e + η_fast · I_e · u_t)
u_t = Enc_θ(C_t, y) − Enc_θ(C_t, ŷ)      # regla tipo delta de Widrow-Hoff, sin disfraz
```
La diferencia entre A y B **no es "otro tipo de crédito"** — es otro optimizador, otra cadencia, otra persistencia. El softmax de "crédito conservativo" propuesto en una ronda anterior se eliminó por redundante: si `q_ph` es diferenciable, backprop ya distribuye gradiente correctamente por la regla de la cadena; no hace falta una fórmula paralela.

## 6.2 Auditoría contrafactual (obligatoria solo para promoción)

Tres cantidades separadas, no equivalentes:
```text
RoutingWeight   = cuánto se usó una ruta
Gradient        = sensibilidad local de la pérdida
CausalInfluence = qué cambia realmente al eliminarla
```
```text
I_r = L(C_t \ r, y) − L(C_t, y)                 # o sobre el logit correcto:
I_r = Score_y(C_t) − Score_y(C_t \ r)
```
Presupuestada, no exhaustiva (aproximación tipo Shapley sampling, Lundberg & Lee 2017): top-M rutas con ablación exacta, resto con máscaras grupales. Ablación jerárquica dentro de una ruta (toda la ruta, cada tramo, el operador Compose, el binding intermedio) para distinguir si el acierto vino de una relación, la composición, coincidencia superficial, ruta paralela, o sesgo del decoder.

## 6.3 Doble traza — causalidad bajo presupuesto vs. causalidad estable (v2.3)

```text
ExecutionTrace: post-poda, causalidad bajo el beam realmente ejecutado (C_exec).
ShadowFrontier: checkpoint pre-poda, permite reabrir ramas descartadas (C_open).
```
Certificado de poda vía **branch-and-bound** (Land & Doig, 1960 — técnica de 60+ años, no intuición nueva):
```text
U_D = máxima influencia posible de la frontera descartada
M = Score(y) − Score(competidor)

si M > U_D + ε: no hace falta reabrir, certificado seguro
si M ≤ U_D + ε: reabrir con beam mayor, o devolver UNKNOWN y bloquear promoción
```
Condición mínima de promoción:
```text
C_exec > τ_e   AND   C_open > τ_o   AND   sign(C_exec) = sign(C_open)
```
No exige magnitudes idénticas — exige que la atribución no dependa de haber podado la alternativa correcta.

## 6.4 Álgebra de evidencia (T,E) — **SUPERSEDIDA, ver 6.5**

Diseño que estuvo vigente entre v2.3 y el hallazgo de explosión (Parte 8). Se documenta porque el código en VPS todavía la implementa y porque el reemplazo (6.5) reutiliza su semántica de niveles.

```text
R = (T, E)     # T: transformación, E: circuito de evidencia — nunca se fusionan
Evidence = Leaf | Serial | Alternative | Joint

Cap(Leaf) = confidence_cap
Cap(Serial(E1..En))      = min_i Cap(Ei)
Cap(Joint(E1..En))       = min_i Cap(Ei)
Cap(Alternative(E1..En)) = max_i Cap(Ei)     # cap por RAMA COMPLETA, ver bug 1 abajo

Compose((T1,E1),(T2,E2)) = (T2∘T1, Serial(E1,E2))
Fold_B({(Ti,Ei)}) = (Fold_B^T({Ti}), Fold^E({Ei}))     # Fold NO comprime E a un número
```
**Invariante duro:** `level(Compose(R1,R2)) ⪯ level(R1,R2)` — ninguna composición promueve M1→M2, solo `Promote(Evidence, AuditProof)` explícito puede.
Deduplicación por `record_id`: la misma evidencia repetida (ciclos, rutas duplicadas) no cuenta dos veces.

**Dos bugs reales encontrados por el Validator y corregidos en el código (`/root/mrdl/mrdl/evidence.py`), documentados porque son instructivos:**

- **Bug 1 (lavado por aplanamiento):** la implementación inicial de `Alternative.evidence_level` aplanaba hasta las hojas crudas (`record_levels`) y tomaba el máximo entre ellas, ignorando el capping de las ramas Serial/Joint que las contenían. Reproducido: `Alternative(Serial(M2,M1), Serial(M2,M0))` daba M2 en vez de M1. Fix: capear cada rama completa primero (`child.evidence_level`), recién después tomar max entre ramas.
- **Bug 2 (dependencia de orden):** el primer fix introdujo una deduplicación por `seen` acumulativo dependiente del orden de iteración — mismos hijos, distinto orden, distinto resultado (M1 vs M2). Fix definitivo: agrupar por `frozenset(record_ids)` completo antes de tomar max (operación de conjunto, no de secuencia) — conmutativo por construcción. Verificado con 2000 comparaciones de permutación aleatoria, 0 discrepancias.

Ambos bugs y sus fixes están cubiertos por tests de regresión explícitos en el VPS (`test_alternative_and_fold_are_order_independent`, `test_alternative_caps_nested_serial_and_joint_branches`).

## 6.5 Por qué se abandona (ver Parte 8 para los números) y reemplazo: **ejecución epistémica estratificada (v3.0, EN REVISIÓN)**

> Error conceptual identificado: intentar que cada hipótesis transporte la enumeración completa de sus justificaciones. El conjunto de rutas puede ser exponencial aunque el cálculo transformacional esté acotado por beam — no existe compresión fija que preserve exactamente todas las rutas, alternativas y contrafactuales sin pagar ese crecimiento. **La salida no es comprimir mejor el árbol. Es dejar de transportarlo.**

**Idea central**: ejecutar la misma computación en **carriles de confianza separados**, sin comunicación ascendente entre carriles. Primera implementación, dos carriles:
```text
FULL:  M1 + M2, todo el conocimiento disponible.
CLEAN: solo M2, únicamente conocimiento consolidado.
```
```text
T_e^FULL  = T_e                              # para M1 o M2
T_e^CLEAN = T_e si e∈M2, AUSENTE del conjunto de candidatos si e∈M1
            # (no es un "operador monomial cero" literal — romperia la invertibilidad
            #  de la familia monomial. Es exclusión del conjunto de ramas, pendiente
            #  de que quede así de explícito en la especificación de implementación).

T_21^FULL  = T2^FULL ∘ T1^FULL
T_21^CLEAN = T2^CLEAN ∘ T1^CLEAN
```
Composición, Fold_B, gate, routing, túneles y decoder se ejecutan **por separado, por carril** — no existe operación que cruce carriles. El no-lavado deja de ser un invariante que se verifica: es imposible por construcción, porque la operación que lo permitiría no existe.

**Certificación de respuesta**, tres resultados posibles:
```text
Limpio:      FULL→C, CLEAN→C, margen suficiente         → soporte M2
Provisional: FULL→C, CLEAN→otro token                    → depende de M1
Frágil:      FULL→C, CLEAN→C, margen casi cero            → M1 pudo reforzar decisivamente algo que M2 apenas sostenía; no confianza plena
```
```text
s_y^CLEAN − max_{j≠y} s_j^CLEAN ≥ m_clean
```
Generalizable a G carriles con umbrales `Θ={τ0..τ_{G-1}}`, cuantización conservadora (0.68 puede entrar en el nivel 0.60, nunca redondear hacia 0.75). **Para el prototipo, no más de 2 carriles hasta demostrar que el mecanismo elimina la explosión.**

**Beam separado por carril** (agujero cerrado explícitamente): si ambos carriles compartieran selección, una ruta M1 con puntuación alta podría desplazar del beam a una ruta M2 real, contaminando CLEAN indirectamente aunque matemáticamente M1 nunca entre en su transformación. Cada carril necesita presupuesto propio (`B_FULL`, `B_CLEAN`) o reserva rígida de plazas.

**Regla de no-lavado por parámetros** (ver 5.5) aplica igual acá: pesos del controlador pueden compartirse entre carriles; activaciones, decisiones de gate, estados de puertos y elecciones de beam, no.

**Procedencia pasa a un registro de reconstrucción, no a un árbol transportado:**
```text
ReplayStep {
    operation_id, controller_version, relation_versions,
    parent_branch_ids, gate_decisions_by_lane, fold_budget_by_lane,
    survivor_ids_by_lane, shadow_bounds_by_lane, deterministic_seed
}
```
Describe cómo reconstruir la computación bajo demanda, no la enumera. Complejidad:
```text
Memory_active = O(G·B)              # sin k^D
Memory_replay = O(G·D·B·k)
```

**Auditoría de promoción, reformulada** (más directa con este diseño):
```text
1. Ejecutar CLEAN normal.
2. Ejecutar sombra de CLEAN permitiendo temporalmente solo el registro e.
3. Medir qué cambia: I_e = Score_y^{CLEAN+e} − Score_y^CLEAN
4. Reabrir fronteras sensibles al beam si hace falta.
5. Verificar estabilidad.
6. Promover e: M1→M2. Los túneles derivados NO cambian de nivel mágicamente — se invalidan/recalculan cuando se reutilizan.
```

### 6.5.1 Tres puntos abiertos — CERRADOS (confirmados como requisitos de implementación, no comentarios opcionales)

**1. El costo dual es un impuesto permanente, aceptado y a medir, no a ocultar.**

```text
C_dual = C_FULL + C_CLEAN ≈ 2·C_single (no exacto: CLEAN consulta menos por excluir M1,
         el scheduling dual puede empeorar caché y superar 2x, o el cálculo puramente
         inmutable compartido puede bajarlo de 2x)
```
FULL y CLEAN se implementan como estados independientes (candidate retrieval, gates, ports, routing, beam, Fold, context state — todo por separado). Solo pueden compartir datos **inmutables**: embeddings congelados, parámetros consolidados del controlador, tablas de operadores M2, constantes. Nunca decisiones, normalizaciones, capacidad de beam ni estados mutables.

Reutilización segura de cómputo puro (post pruebas A/B): solo si `edge_version`, `controller_version`, `input_state` y `operator` son idénticos entre carriles — se calcula una vez, se entrega copia a cada carril. Lo que se comparte es una función pura, nunca una decisión (Top-K, normalización, gate, puertos, puntuación de beam, condición de parada siguen sin poder compartirse). Una ruta M1 no puede ocupar un lugar provisional y filtrarse después para obtener CLEAN — el filtrado ocurre **antes** de cualquier operación limitada por capacidad.

Benchmark obligatorio antes de aprobar nada: `FULL_only`, `CLEAN_only`, `FULL_plus_CLEAN_isolated` (y `FULL_plus_CLEAN_exact_reuse` si se implementa la reutilización). Métricas: `R_runtime = t_{FULL+CLEAN}/t_{FULL-only}`, `R_ops`, más `tokens_per_second, latency_p50/p95, operator_evaluations, candidate_retrievals, gate_evaluations, branches_created/surviving, memory_peak, M1_edge_fraction, M1_active_fraction`. Reportar la curva completa en niveles de participación M1 de 0%/1%/10%/50% — sin umbral de aprobación prefijado. **Si el costo dual elimina la ventaja computacional frente al baseline, eso es un resultado negativo real de arquitectura y se reporta como tal, no se esconde.**

**2. TTL y replay se unen mediante reserva atómica — nunca "borrar al vencer el reloj".**

Estados de un registro M1: `ACTIVE → AUDIT_RESERVED → AUDITING → PROMOTED | ACTIVE | REJECTED`, y `ACTIVE → EXPIRED` (solo válido si `pin_count==0` y `promotion_state==NONE`).

La auditoría reserva atómicamente la **clausura completa de replay**, no solo el registro:
```text
Closure(r) = { r, ReplaySteps, relation_versions, controller_version, seeds, snapshots, bindings }
```
Si la clausura está incompleta al reservar → `state = UNREPLAYABLE`, promoción prohibida. Nunca reconstruir con la versión actual, nunca aproximar en silencio, nunca promover con traza incompleta, nunca tratarlo como timeout simple — evita que un registro quede "pendiente para siempre" sin diagnóstico.

Durante `AUDIT_RESERVED`/`AUDITING`, el TTL puede vencer cronológicamente pero el registro y sus dependientes NO se eliminan — se marca `expiry_pending=true` y se expira recién al liberar el pin, solo si terminó en `ACTIVE` (no si fue `PROMOTED`). Condición de carrera resuelta por transacción atómica: quien reserva primero gana; nunca se inicia auditoría sobre un registro parcialmente eliminado.

*Nota abierta menor:* el caso `REJECTED` con `expiry_pending=true` no dispara limpieza en el pseudocódigo de referencia — asumido bajo una política de garbage collection separada, confirmar que no es un descuido.

**3. `T_e^{CLEAN}=0` queda eliminado de la notación — no es un operador monomial cero.**

Definición correcta, por conjunto de elegibilidad:
```text
E_CLEAN(v) = { e ∈ E(v) | level(e) ≥ M2 }
E_FULL(v)  = E_CLEAN(v) ∪ E_M1(v)
Compose: E_CLEAN × E_CLEAN → E_CLEAN
```
Una relación M1 está **ausente** de CLEAN (`M1 → ineligible → no branch allocated`), nunca presente con operador cero. La comprobación de elegibilidad ocurre ANTES de: crear la rama, llamar al gate, puntuar, insertar en puerto, normalizar candidatos, consumir capacidad Top-K, ocupar espacio de beam, o entrar en estadísticas — una rama fantasma con valor cero igual contaminaría conteos, normalización, capacidad, orden y condiciones de parada, aunque su valor numérico fuera nulo. Correcto: iterar `clean_index[node]` (ya excluye M1 de raíz). Incorrecto: iterar `full_index[node]` y poner `operator=ZERO` si `is_m1`.

Si se usa estructura fusionada para ahorrar memoria: `LaneMask{participates_in_full, participates_in_clean}` — nunca `transform_clean=0`.

---

# PARTE 7 — Riesgos: resueltos vs. abiertos

## 7.1 Resueltos (con evidencia, no solo argumento)

| Riesgo | Cómo se cerró |
|---|---|
| Colapso a n-gram | **Parcialmente confirmado empíricamente (Parte 10.3, no solo en papel)**: le gana a trigram en corpus chico (300 historias), pero la brecha se agranda monótonamente al escalar — abierto, no resuelto |
| Crédito local con decaimiento γ^l | Modo A/B + auditoría contrafactual, sin fórmula de crédito paralela a backprop |
| Over-squashing por hubs | Puertos con routing de un solo paso (no CapsNet iterativo), + túneles efímeros |
| Binding "emerge de similitud" (vago) | VSA con roles autoinducidos por invariancia de sustitución |
| `Compose` sin definir | Álgebra monomial cerrada, verificada matemáticamente |
| Explosión de ramas k^L | Fold_B con beam acotado (beam search estándar) |
| Confianza de ruta con decaimiento multiplicativo | min + log-cost, no producto |
| Lavado de confianza M1→M2 vía composición | Invariante de tipos (Cap por rama completa, no aplanado) — verificado con 2000 permutaciones aleatorias, 0 violaciones, tras corregir 2 bugs reales |
| Sesgo de auditoría por poda de beam | Doble traza + certificado branch-and-bound (C_exec vs C_open) |
| Lavado vía parámetros compartidos | Regla dura: M1 no toca parámetros/estados persistentes, solo memoria provisional |
| Familia monomial es solo lineal, no resuelve XOR/producto | **Confirmado por benchmark**, no solo predicho (Parte 8) — decisión: no agregar átomos no lineales salvo que un benchmark lo exija |
| `Fold_B` no fusionaba por permutación, contra 4.4 | **Confirmado por instrumentación directa y corregido** (Parte 10.8) — bug de implementación real, no límite estructural; colapso de composición mejora de 148x a ~105x tras el fix |
| Hipótesis "crowding" detrás de RCR/REN | **Refutada con metodología corregida** (Parte 10.7) — artefacto de medición (pools no anidados), `D_linear=0.0025` una vez medido bien |
| AOC (celdas de polaridad contextual) | **Cerrado con evidencia negativa limpia** (Parte 10.6) — covarianza SVD no se traduce en frontera explotable |

## 7.2 Abiertos

- **Explosión del árbol de evidencia simbólico** (Leaf/Serial/Alternative/Joint) con la profundidad — CONFIRMADO por benchmark (Parte 8), motivó el rediseño de carriles (6.5), que a su vez tiene 3 puntos sin cerrar (6.5.1).
- **Costo de auditoría contrafactual en la práctica** — presupuestado en diseño, no medido aún en corpus real.
- **¿El sistema mejora sus procedimientos generales sin promoción?** — actualmente no, por diseño (5.5); confirmar si es aceptable.
- **Composición relacional (2+ hops) no aporta accuracy medible a ninguna escala probada** (4.025 y 10.000 historias) — hallazgo decisivo, Parte 10.5. Con `Fold_B` corregido (10.8) y descartado como causa restante, el problema apunta a la calidad semántica del operador de composición mismo. El rediseño propuesto (TFA, Parte 10.9) pasó verificación matemática (P2.0) pero su evaluación en sombra sobre datos reales (P2.1) no mostró señal estadísticamente distinguible del baseline — **abierto, sin resolver si el residuo contextual carece de estructura recuperable o si la familia de operadores es insuficiente**.
- **Rendimiento de `_candidate_signal()`**: escaneo lineal O(E) sobre todas las aristas por cada `record_id` de ruta (en vez de un índice O(1) `record_id→edge`) — bottleneck de rendimiento real identificado, no arreglado todavía, deliberadamente pospuesto hasta cerrar la investigación de composición.

---

# PARTE 8 — Resultados de benchmarks de capacidad (verificados independientemente)

Todos ejecutados en VPS (`/root/mrdl`), `training_steps=0` en los tres (son pruebas de capacidad representacional/teórica, no de aprendizaje vía el pipeline real — dejarlo explícito para no confundir "capacidad" con "el sistema aprendió esto de datos").

## 8.1 Δ-Mix (representación exacta de matrices Δ-dispersas)

Generador: grafo bipartito Δ-regular descompuesto en Δ matchings edge-disjoint (teorema de coloreo de aristas de König). k ramas monomiales, k,Δ ∈ {1,2,4,8,16}, dimensión 64, 3 seeds, 75 filas.

- `k ≥ Δ`: error numérico ≤ ~2×10⁻¹⁶ (exacto, como predice la teoría).
- `k < Δ`: error medio creciente y monótono según k/Δ; ej. k/Δ≈0.5 converge a ~0.68–0.70 en Δ=4,8,16 (consistente con energía distribuida ~uniforme entre matchings).
- Selección de ramas por **oráculo** (energía calculada sobre la descomposición conocida), no aprendida — confirma el techo matemático (ya probado en papel vía König), no si el sistema real puede descubrirlo desde datos.

## 8.2 Frontera no lineal (XOR / producto)

Cota vía proyección afín óptima por mínimos cuadrados (válida para cualquier k, porque suma de monomiales ⊆ afín): 30 filas, 3 seeds × 2 targets × 5 k.

- `xor_01`: error normalizado constante = 1/√2 ≈ 0.70711, para todo k (verificado a mano por el Validator: covarianza de cada input con el output es 0 por simetría → mejor predictor es la constante 0.5).
- `product`: error normalizado constante = 1.0, para todo k (verificado a mano: media y covarianzas dan 0 por simetría del grid → mejor predictor es 0).
- Confirma exactamente la predicción teórica: ningún k resuelve una interacción no lineal genuina. Prueba analítica por contención de conjuntos, no medición empírica por-k.

## 8.3 Folding profundo — hallazgo decisivo

240 filas: 3 seeds × k∈{1,2,4,8} × depth∈{1,2,4,8,16} × beam∈{4,8,16,32}, dim=64. Referencia exacta calculada capa por capa (sin enumerar k^depth). Verificación cruzada interna: mientras el árbol de evidencia es tratable (<100k nodos), se construye de verdad con el `evidence.py` real y se compara contra la fórmula cerrada (assert de igualdad, nunca falló).

Lado de la transformación (beam-acotado): funciona como se diseñó — error crece cuando k supera el beam disponible (esperado: beam insuficiente para el factor de ramificación).

Lado de la evidencia (deliberadamente sin comprimir, por diseño anti-lavado): **explota independientemente del beam.**

| k | beam | depth | evidence_tree_size |
|---|---|---|---|
| 1 | 4 | 16 | 49 |
| 2 | 4 | 16 | 10,737,418,237 |
| 4 | 4 | 16 | 17,179,869,181 |
| 8 | 4 | 16 | 17,179,869,181 |
| 8 | 32 | 16 | 1,023,687,185,964,000,510,839,905 |

Verificado independientemente por el Validator, incluida re-derivación a mano de un valor (k=2,depth=4,beam=4 → 637) antes de correr el código, coincidencia exacta.

**Conclusión**: la representación de evidencia simbólica (6.4) queda descartada para composición profunda. Motivó el rediseño de carriles (6.5), actualmente en revisión (3 puntos abiertos, 6.5.1).

---

# PARTE 9 — Pruebas pendientes antes de reanudar implementación completa (orden acordado)

**Estado: A–E completas y verificadas de forma independiente (no solo por reporte del implementador) — ver 9.1. Núcleo de lenguaje real habilitado a partir de acá.**

Los 3 puntos de 6.5.1 ya están cerrados. Orden que se siguió, A–E:

**A. Equivalencia** — en instancias pequeñas: `FULL nuevo` vs. implementación anterior completa (árbol de evidencia); `CLEAN nuevo` vs. ejecución antigua eliminando M1 desde el origen. Criterio: misma salida, mismos logits dentro de tolerancia, mismas ramas supervivientes.

**B. No-interferencia** — modificar arbitrariamente todos los registros M1 (valores, orden, cantidad, pesos, relaciones, embeddings provisionales). CLEAN debe permanecer idéntico — bitwise si el hardware lo garantiza, si no, dentro de una tolerancia numérica fijada ANTES del test (no ajustada después para que pase).

**C. No-lavado por control** — cambios M1 no deben alterar en CLEAN: candidate set, gate decisions, port assignments, beam survivors, propagation depth, stop condition, token prediction. Tampoco M1 debe poder actualizar: controller weights, persistent port keys, normalization statistics, shared caches con estado semántico, composition parameters.

**D. Reescalado del benchmark 6 + costo temporal** — repetir k=2/beam=4/depth=16 y k=8/beam=32/depth=16. Verificar `ActiveState=O(G·B)` y `ReplayStorage=O(G·D·B·k)`. Agregar obligatoriamente: runtime FULL_only, runtime CLEAN_only, runtime FULL+CLEAN, ratio de runtime, ratio de operadores, ratio de memoria pico.

**E. Promoción, TTL y replay** — secuencia: (1) crear M1, (2) confirmar que afecta FULL, (3) confirmar que NO existe como rama en CLEAN, (4) reservar auditoría, (5) forzar vencimiento de TTL durante la auditoría, (6) confirmar que registro y replay closure siguen presentes, (7) completar promoción, (8) invalidar derivados, (9) recalcular, (10) confirmar que solo ahora aparece en CLEAN. Caso de fallo: eliminar una dependencia antes de reservar → intentar auditoría → confirmar estado `UNREPLAYABLE` → confirmar que no hay promoción silenciosa.

## 9.1 Resultado real de A–E (verificado independientemente, con re-ejecución propia, no solo con reportes)

- **A/B/C**: 0 violaciones en cientos de casos aleatorios y adversariales (incluido un ataque propio: registro M1 con score=coeficiente=1e9 diseñado para dominar si se colara — no movió CLEAN un bit). Dos bugs reales encontrados y corregidos en el camino: aplanamiento de `Alternative` (6.4) y raíz de propagación mal etiquetada como M0 (causaba que `evidence_level` quedara fijo en M0 para siempre sin importar cuántos edges M2 se compusieran — corregido, raíz ahora es M2-neutral).
- **D — costo dual real, medido, no supuesto**: para configuraciones con ramificación/beam suficiente (k=8, beam=32), R_runtime = t(FULL+CLEAN)/t(FULL-only) se ubica entre **~1.4x y ~2.2x** según densidad de contenido M1 (0%–50%), con `ActiveState` y `ReplayStorage` acotados exactamente por `B·D` por carril (verificado, no `k^D`) y `no_k_pow_d_materialization=true`. **Ese es el costo real y utilizable a comunicar.**
  - **Hallazgo de robustez, importante**: con ramificación/beam angostos (k=2, beam=4) y densidad M1=50%, el carril CLEAN **colapsa** en los 3 seeds probados (uno con `active_state_clean=0`, cero candidatos sobrevivientes — CLEAN no tiene respuesta). El R_runtime en esos casos (~1.0–1.1) parece "barato" pero es un artefacto de que CLEAN dejó de trabajar, no de eficiencia real — nunca promediar esas filas junto a las sanas. **Restricción de diseño derivada de este hallazgo: la ramificación/beam mínimos del grafo real deben ser suficientes para la densidad de contenido M1 esperada en el despliegue, o CLEAN queda funcionalmente vacío.** Métrica agregada para vigilar esto en el corpus real: `clean_health_ratio = min(active_state_clean/bound, operator_evaluations_clean/bound)`, marcar degenerado si `<0.5`, vacío si `==0`.
- **E — máquina de estados de promoción**: implementada (`mrdl/promotion.py`) y verificada, incluida prueba de condición de carrera con hilos reales de threading (no simulada) repetida 500 veces por el Validator sin fallos — reserva atómica gana quien llega primero, siempre. Secuencia completa (crear M1 → afecta FULL, ausente en CLEAN → reservar → auditar → TTL vence durante auditoría sin destruir nada → promover → invalidar/recalcular derivados → aparece en CLEAN) verificada end-to-end. Camino de fallo (dependencia eliminada antes de reservar → `UNREPLAYABLE`, sin promoción silenciosa) verificado.
  - **Pendiente, no bloqueante**: la máquina de estados (`PromotionStore`) y el motor de carriles (`LaneEngine`/`LaneRecord`) todavía no están conectados automáticamente — promover un registro hoy no actualiza por sí solo el `LaneRecord.level` que usa el grafo real; se verificó reconstruyendo manualmente el record con nivel M2. Falta esa pieza de integración (mecánica, no conceptual) antes o durante la implementación del núcleo real.
  - **Pendiente, no bloqueante**: qué pasa con un registro `REJECTED` que tenía `expiry_pending=true` — no se limpia automáticamente en el código actual, asumido bajo una política de GC separada a definir.

Solo después de esto: implementar el núcleo real (embeddings, grafo de relaciones sobre un corpus, loop de entrenamiento Modo A/B) — que **todavía no se ha tocado**, sobre un entorno tipo TinyStories como primer banco de pruebas, comparado contra n-gram/bigram/trigram y un Transformer pequeño con presupuesto de memoria/cómputo comparable.

---

# PARTE 10 — Fase E: núcleo de lenguaje real, primeros resultados de escalado (16 ago, EN CURSO)

Tras 9.1, se implementó el núcleo real (embeddings + grafo de relaciones + entrenamiento Modo B) sobre TinyStories y se empezó a escalar corpus según lo indicado en 9.1/Parte 11. Esta parte documenta lo encontrado, con la misma exigencia de verificación independiente que el resto del documento — todo lo que sigue fue re-ejecutado o auditado por el Validator, no solo reportado por el implementador.

## 10.1 Hallazgo: `a_e` no es una cantidad aprendida — es coseno de embeddings congelados

La implementación real define `a_e = cosine(embedding_source, embedding_target)`, recalculado en cada observación. El delta-rule de Modo B (6.1) actualiza `support`/`confidence`, nunca `a_e`. Como los embeddings base están congelados (1.2), esa "escala" es una función fija de la geometría del embedding — no cambia con evidencia del corpus, contradiciendo 1.3 ("cada conexión contiene un vector relacional... que el entrenamiento crea y modifica").

Consecuencia matemática, no de diseño: coseno entre vectores distintos es casi siempre <1 (verificado: 300/300 aristas medidas, |a_e|<1 sin excepción). Componer T_2∘T1 multiplica escalas — con factores subunitarios, la magnitud decae geométricamente con la profundidad (medido: d1≈0.17 → d4≈0.0002 en embeddings aleatorios). Confirmado que profundidad≥3 gana la competencia de scoring solo en 2/572 casos del benchmark inicial.

**Diagnóstico causal, no solo correlación**: se probó eliminar el decaimiento (`a_e=1` fijo) — no cambió accuracy ni desbloqueó profundidad≥3, y empeoró levemente. Se midió alcanzabilidad independiente del scoring: de 572 targets de test, 37% nunca aparecen en ningún frontier ni a profundidad 4; solo 1.4% dependen de profundidad≥3. **El techo no era de scoring de profundidad, era de cobertura de grafo** — con corpus chico, un tercio de los contextos no tienen camino (directo ni compuesto) hacia la respuesta correcta, independientemente de qué tan bien se puntúen las rutas que sí existen.

**Propuesta de rediseño pendiente (no implementada, "Paso 2")**: reemplazar la parte multiplicativa por rotaciones en pares de canales (`R_e=blockdiag(R(θ_1)...R(θ_m))`, `T_e(z)=R_e·P_e(z)+b_e`, `P_e` permuta pares completos — cambio real respecto al grupo hiperoctaédrico original de 4.3, requiere re-verificación de cierre bajo composición con el mismo rigor que evidence.py). Preserva norma exactamente en cualquier profundidad (demostración correcta: composición de transformaciones ortogonales es ortogonal). Separa explícitamente similitud semántica (prior de gate/retrieval) de confianza (soporte/promoción) de transporte (la transformación en sí) — las tres estaban mezcladas en `a_e=coseno`. **Pausado**: dado que el techo medido era de cobertura, no de profundidad, se decidió escalar corpus primero (Parte 11) antes de invertir en esta implementación. Reabierto como pregunta abierta en 10.3.

## 10.2 Cuatro bugs reales en la maquinaria de aprendizaje del controller, encontrados y corregidos en cadena

Al escalar corpus (205→5.000 historias), la máquina de aprendizaje del controller (6.1, `SparseController`) reveló cuatro fallas reales, cada una enmascarando a la siguiente. Documentadas porque son instructivas y porque el patrón de "verificar antes de aceptar cada capa" es el que permitió encontrarlas todas:

1. **Presupuesto de entrenamiento sin escalar**: `controller_steps`/`promotion_limit` eran constantes fijas (64/128) heredadas de la configuración de corpus chico, independientes del tamaño real del corpus — con 20x más historias, el controller seguía recibiendo la misma cantidad de actualizaciones. Fix: presupuesto proporcional a transiciones de entrenamiento reales.
2. **Regla de actualización unidireccional**: `error=1-P(target)` y cada componente de la señal del controller estaban construidos para ser siempre ≥0 por diseño del código (clamps `max(0,...)`, ratios no-negativos, bias fijo). Matemáticamente, eso garantiza que ningún peso puede subir nunca — colapso monótono al piso del clip, no un caso raro. 6.1 pide una regla contrastiva (`u_t=Enc_θ(C,y)−Enc_θ(C,ŷ)`, objetivo correcto vs. predicho) que nunca se implementó así. Fix: regla contrastiva real, comparando señal del objetivo correcto contra señal del token predicho.
3. **Contaminación por nodo "waypoint"**: la propagación multi-seed fuerza la ruta a pasar por los tokens ya presentes en el contexto (`required_targets`, andamiaje de composición). Esos nodos-waypoint competían sin excluirse en el pool final de "próximo token" — al estar en la ronda más superficial (peso completo, magnitud de operador sin decaer), le ganaban estructuralmente a predicciones genuinas más profundas. Causaba que el modelo predijera literalmente "repetí la última palabra del contexto" en la mayoría de los casos (verificado con 24 casos concretos de test, patrón sistemático, no ruido). Fix: excluir las rondas de satisfacción de `required_targets` del pool de candidatos de predicción final.
4. **Sin decaimiento de tasa de aprendizaje**: con miles de pasos proporcionales a corpus grande, los pesos volvían a derivar hacia los límites del clip (±1) sin asentarse, incluso con la regla contrastiva correcta. Fix: decaimiento simple (`lr_efectivo = lr/(1+decay·updates)`).

Cada fix produjo mejora medible y verificada independientemente antes de aceptar el siguiente paso — ninguno se asumió sin re-ejecución propia del Validator.

## 10.3 Resultado con los cuatro fixes: la brecha con trigram se agranda con el corpus, no se cierra

Comparación pareada exacta (mismo split determinístico, mismo test de 975 historias en las tres escalas), con la maquinaria de aprendizaje ya verificada sana (pesos estables, sin colapso ni divergencia en ninguna escala):

| Historias de entrenamiento | Accuracy MRDL (random_frozen) | Accuracy trigram | Aristas M2 promovidas |
|---|---|---|---|
| 300 | 0.627 | 0.621 | 59 |
| 1.200 | 0.425 | 0.671 | 200 |
| 4.025 (corpus completo de la etapa) | 0.137 | 0.684 | 419 |

MRDL le gana a trigram a escala chica (única vez en toda la sesión) y luego cae monótonamente al crecer el corpus, mientras trigram sube monótonamente como es esperable de un modelo de conteo. **No es un problema de entrenamiento inestable** (pesos sanos en las tres escalas) — es un patrón de fondo real: más aristas promovidas significa más candidatos compitiendo por cada predicción, y la capacidad discriminativa del mecanismo actual (controller lineal de 10 señales + magnitud de operador que decae geométrico, 10.1) no parece escalar con la densidad del grafo.

**Pregunta abierta, sin resolver**: ¿el rediseño de rotaciones en pares (10.1, "Paso 2") ataca este síntoma de raíz — al preservar norma en vez de contraer, podría mejorar discriminación relativa entre candidatos en general, no solo la competitividad de profundidad≥3 — o hace falta diagnosticar más específicamente por qué la discriminación se degrada con la densidad del grafo antes de invertir en esa implementación? El diagnóstico de 10.1 midió el síntoma de profundidad; no hay evidencia todavía de que el mismo rediseño resuelva el síntoma de escalado de corpus documentado acá. **Respuesta parcial en 10.5–10.9**: el síntoma de escalado y el de profundidad resultaron ser la misma causa — composición no aporta nada medible a ninguna escala, y el problema no estaba en `a_e` como escala sino, más abajo en el pipeline, en `Fold_B` (10.8) y posiblemente en el operador mismo (10.9, todavía sin resolver).

## 10.4 Dos bugs más al escalar a 10.000 historias: candidatos VSA fantasma y contaminación de controller por objetivo ausente

Escalar de 4.025 a 10.000 historias expuso dos bugs adicionales en la cadena de entrenamiento del controller (6.1), ambos silenciosos — no producían errores ni crashes, solo entrenamiento incorrecto:

5. **Contaminación por candidatos VSA "fantasma"**: el mecanismo de continuación VSA (`vsa_memory.continuation_scores`) podía inyectar en el pool de scoring tokens que nunca fueron alcanzados por ninguna ruta relacional real (ni directa ni compuesta) — pura similitud de recuperación asociativa, sin evidencia de grafo detrás. Esto contaminaba tanto el entrenamiento del controller como el conteo de candidatos. Fix: excluir del entrenamiento y del pool de predicción final los candidatos que no tienen ningún `record_id` de arista real en su procedencia.
6. **Contaminación por objetivo ausente del pool de entrenamiento** (el más grande de los seis bugs encontrados en toda la fase E, aislado por Sol): `train_controller` actualizaba el controller comparando la señal del token objetivo contra la señal del token predicho incluso cuando el objetivo real ni siquiera aparecía en el pool de candidatos de esa ronda — entrenando la regla contrastiva (bug 2, 10.2) contra una señal inventada/cero en vez de saltarse la observación. Fix: `target_missing_controller_skips`, contador explícito, la observación se descarta en vez de entrenar con una señal falsa.

Con los seis bugs corregidos (los 4 de 10.2 + estos 2), accuracy full-scale en 4.025 historias: **0.3483 (random_frozen)**, **0.2596 (pretrained_frozen)** — primer estado del pipeline verificado como correcto de punta a punta, sin artefactos de entrenamiento conocidos.

## 10.5 Hallazgo decisivo: composición relacional no aporta nada medible, a ninguna escala probada

Con el pipeline ya limpio de los 6 bugs, se midió por separado el aporte de predicciones directas (0-1 hop real) vs. compuestas (2+ hops reales) vs. continuación pura VSA, sobre el mismo split determinístico, en dos escalas:

| Escala | Directas (accuracy) | Compuestas (casos / correctas) | % decisiones vía composición |
|---|---|---|---|
| 4.025 historias | — | 0/29 | ~0.31% |
| 10.000 historias | — | 0/43 | ~0.43% |

**Cero aciertos vía ruta compuesta, en ambas escalas.** Escalar el corpus 2,5x no aumentó el uso de composición — lo mantuvo igual de marginal. Esto refuta directamente la hipótesis de "solo hace falta más corpus" y puso en duda la tesis distintiva central de MRDL (composición algebraica de relaciones, Parte 4.3–4.4) como mecanismo funcionalmente activo, no solo matemáticamente cerrado.

Este hallazgo se obtuvo tras que el Validator rechazara aceptar afirmaciones agregadas sin verificación de casos concretos ("no aceptar humo") y forzara al implementador (opencode) y al diseñador (Sol) a medir con metodología controlada en vez de conclusiones sobre agregados sin trazar.

## 10.6 Cierre de AOC (Atlas de Orientaciones Contextuales)

Propuesta de Sol: celdas de polaridad contextual detectadas por eje de partición SVD sobre el pool de candidatos observables. **Cerrado con evidencia negativa limpia**: la covarianza explicada por el eje SVD no se tradujo en una frontera de decisión explotable — delta exactamente cero entre condición con y sin AOC en dos metodologías de partición independientes. No se persiguió más.

## 10.7 Cierre de RCR/REN y la hipótesis de "crowding"

Propuesta de Sol: resolución por eliminación no compensatoria (orden lexicográfico ordinal en vez de suma ponderada), motivada por una hipótesis de que la densidad de candidatos ("crowding") degradaba la discriminación al escalar el beam. La metodología original de barrido (beam 4/8/16/32) mezclaba cambios de propagación con cambios de cantidad de candidatos (pools no anidados) — Sol identificó el error y propuso la corrección (cosecha de beam máximo único + subconjuntos anidados por orden de creación + cohorte H16).

**Con la metodología corregida, la hipótesis de crowding quedó refutada**: `D_linear=0.0025`, esencialmente plana. Fue un artefacto de medición, no un efecto real. Un hallazgo secundario (REN-Greedy mostró una mejora de accuracy real pero sin explicación causal identificada, sobre una cohorte filtrada) quedó pausado por decisión explícita del Validator — no toca la tesis distintiva de MRDL (composición relacional), así que perseguirlo no era prioritario frente al hallazgo de 10.5.

## 10.8 Embudo causal G0-G5 y el bug real de Fold_B

Con 10.5 estableciendo que composición no aporta nada, Sol propuso una "autopsia causal" de 5 etapas para localizar exactamente dónde se pierde una ruta compuesta, sobre el checkpoint de 10.000 historias:

```text
G0: existe camino topológico (2-4 hops) hacia el objetivo
G1: sobrevive binding/CompositionGate
G2: se genera antes de Fold_B
G3: sobrevive Fold_B/beam
G4: el objetivo entra al pool final de scoring por esa ruta
G5: gana top-1
```

| Transición | P (antes del fix de Fold_B) |
|---|---|
| P(G1\|G0) | 0.293 |
| P(G2\|G1) | 0.966 |
| P(G3\|G2) | 0.614 |
| **P(G4\|G3)** | **0.00676** — colapso de 148x, cuello de botella dominante |
| P(G5\|G4) | 0 |

**Investigación del cuello de botella**: se descartó primero, con un caso concreto trazado (no solo estadística agregada), que la deduplicación de `_scores` por primer-ocurrencia-por-token fuera la causa — se implementó un fix (deduplicación por raíz de procedencia en vez de por token) que es lógicamente correcto y se mantuvo, pero un ejemplo trazado mostró que en la práctica solo sobrevive UNA ruta por nodo destino tras Fold_B/beam — no había nada que deduplicar, la pérdida ocurre antes.

**Causa real encontrada por instrumentación directa**: `fold_candidates()` (`mrdl/lanes.py`) hacía un corte top-K global puro por score (`sorted(-score)[:beam_width]`, beam_width=8 por defecto) sobre TODOS los candidatos de la ronda, sin ninguna fusión por permutación — contra lo que exige 4.4 ("Fold_B fusiona ramas con la MISMA PERMUTACIÓN, suma/combina escalas"). Instrumentado sobre 1.500 contextos del checkpoint 10k: 2.284/7.300 rondas con colisión de nodo, 100% con la misma permutación, y en 5.834/14.587 grupos la suma del grupo hubiera superado el cutoff del beam mientras cada candidato individual quedaba debajo — **bug de implementación real y barato, no límite estructural**.

**Fix aplicado y verificado** (dos rondas de resultados rechazadas antes de una versión correcta — una con corrupción de procedencia por aliasing de `candidate_id`, otra con crecimiento combinatorio del árbol de evidencia por foldear grupos que el top-K iba a podar igual): reordenar a agrupar por (nodo, permutación) + sumar score, aplicar top-K sobre los grupos, foldear evidencia solo de los sobrevivientes.

**Resultado post-fix**: `P(G4|G3)` sube a 0.00956 (colapso de ~105x, mejora de 41% relativo, primera vez con ganadores top-1 vía ruta compuesta: 4, antes 0). Pero la accuracy de corpus completo (13.951 transiciones, mismo checkpoint) muestra que el problema no era (solo) supervivencia en el beam:

| Grupo | Accuracy | Casos |
|---|---|---|
| Directo | 31.14% | 13.232 (4.120 correctas) |
| **Compuesto** | **1.44%** | 278 (4 correctas) |
| Fallback | 25.62% | 441 (113 correctas) |

Composición sigue acertando **22x peor que predicción directa incluso cuando sobrevive el beam, entra al pool final, y gana top-1**. Esto descarta a `Fold_B` como causa restante — el problema está en la calidad semántica del operador monomial compuesto (10.1, `a_e=coseno`), no en la selección/poda.

## 10.9 Rediseño del operador: TFA (Transporte Fásico Anclado) — evaluación en curso, sin resolver

Con Fold_B descartado como causa (10.8), Sol propuso reemplazar por completo `a_e=coseno` (10.1) — no con la rotación simple originalmente sugerida, sino con una descomposición más fuerte:

```text
estado absoluto = ancla del nodo (identidad léxica, congelada) + residuo contextual (transportado)
T_e(c_i + r) = c_j + U_e(r) + ν_e
```

- `c_v`: ancla compleja del nodo (embedding congelado, emparejado en canales complejos, normalizado) — nunca cambia con el contexto.
- `U_e`: isometría por pares (permutación + reflexión + fase unitaria) — preserva norma exactamente, sin decaimiento con la profundidad (a diferencia de `a_e=coseno`, que decae geométricamente, 10.1).
- `ν_e`: innovación contextual acotada (`|ν_e|≤β`), separada de la identidad del destino (que la absorbe el ancla, no el sesgo).

Composición exacta verificada algebraicamente: el ancla intermedia cancela exacto, `U_{21}=U_2∘U_1`, `ν_{21}=U_2(ν_1)+ν_2` — telescopía limpia, sin acumular sesgo arbitrario con la profundidad.

**P2.0 (invariantes matemáticos, sin corpus)**: verificado por opencode y confirmado independientemente por el Validator vía artifact hasheado (no solo reporte) — composición exacta hasta profundidad 64 (`relative_error<1e-6`), conservación de norma, roundtrip de inversa numéricamente exacto, cuantización de fase modular exacta sin deriva acumulada. **Todo pasa.**

**P2.1 (evaluación en sombra sobre el checkpoint real de 10k, sin tocar producción)**: dos rondas de resultados rechazadas por el Validator antes de una tercera válida — primera con un proxy degradado del residuo contextual (10 señales escalares del controller en vez del estado VSA real), segunda con un bug de `bind()` duplicado que infló artificialmente el conteo de casos "compuestos" de 278 a 9.964 y dio las tres variantes bit-idénticas (el selector de variante no llegaba al cálculo de la señal de scoring). Con rehidratación correcta del estado VSA (replay de `bind()` una sola vez sobre el train set, en orden) y variantes confirmadamente distintas entre sí:

| Variante | Accuracy total | Compuesto (accuracy / casos) |
|---|---|---|
| O0 (baseline, `a_e=coseno`) | 30.37% | 1.44% / 278 |
| O1 (U=identidad, solo ν) | 30.21% | 1.18% / 255 |
| O2 (fase aprendida, sin ν) | 30.76% | 1.21% / 165 |
| O3 (fase+ν+prototipos) | 30.69% | 1.59% / 189 |

Con n≈278 y p≈1.4%, el error estándar binomial da un IC 95% de ~0%-2.9% — **ninguna variante se distingue estadísticamente del baseline**, y el tamaño del pool "compuesto" cambia por variante, así que ni siquiera es una comparación pareada limpia. No cruza el umbral que el propio Sol pre-registró (≥19/278≈6.83% en compuestas, IC 95% pareado por encima de cero).

**Estado a la fecha de este documento**: resultado es ruido, no señal. Sin resolver si (a) el residuo contextual VSA (bindings/continuation/route_signature) simplemente no contiene suficiente estructura recuperable para que ningún operador la explote en composición, o (b) el residuo sí tiene señal pero la familia de operadores isométricos es insuficiente para capturarla. No autorizado avanzar a integración en producción (P2.2+) con esta evidencia. Sol fue explícitamente instruido a replantear la premisa, no a ajustar hiperparámetros de la misma familia de operadores.

---

# PARTE 11 — Instrucción resumida para un agente de IA (para handoff)

> Diseñá y evaluá una arquitectura de lenguaje autoregresiva cuyo estado principal sea un grafo relacional disperso. Cada token/concepto tiene un embedding base congelado. Las conexiones entre nodos contienen operadores relacionales monomiales (permutación con signo + escala diagonal + sesgo, cerrados bajo composición) cuantizables, con soporte y confianza. El contexto es un conjunto disperso de cápsulas de ruta con activaciones, roles vinculados (descubiertos por invariancia de sustitución, no definidos a mano), expectativas abiertas y procedencia. La predicción surge de propagación limitada (Top-K, pocas rondas) con puertos de un solo paso (nunca routing iterativo tipo Capsule Network) y túneles efímeros para evitar cuellos de botella en nodos hub. El aprendizaje separa: memoria rápida de aristas (escritura inmediata, one-shot, sin backprop) del controlador de procedimientos generales (backprop disperso sobre el subgrafo activo). El conocimiento vive en tres niveles de confianza (M0 contexto / M1 custodia / M2 consolidado); solo una promoción explícita con auditoría contrafactual presupuestada mueve M1→M2; ninguna composición, fold, ruteo o gate puede hacerlo por sí sola. La evidencia/procedencia NO se transporta como árbol simbólico (eso explota combinatoriamente con la profundidad, confirmado empíricamente) — se ejecuta en carriles de confianza paralelos sin comunicación ascendente entre ellos, con reconstrucción bajo demanda solo para auditoría. No conviertas automáticamente el diseño en un Transformer, una memoria externa para un LLM, un n-gram, un knowledge graph convencional, ni una Capsule Network con routing iterativo — todos esos caminos ya fueron identificados y evitados explícitamente. Antes de proponer un mecanismo nuevo, revisá si el problema que resuelve ya está cubierto en este documento; si no, proponelo con matemática concreta (no metáfora) y un benchmark sintético que lo confirme o lo descarte antes de tocar el pipeline completo.

---

# PARTE 12 — Tesis central

La propuesta no es guardar que una palabra suele seguir a otra. Es investigar si el lenguaje puede modelarse mediante relaciones vectoriales explícitas y dispersas, contexto como cápsulas activas, propagación competitiva en vez de atención densa, composición algebraica cerrada de relaciones, binding de roles autoinducido, y una separación estricta entre lo que se recuerda de inmediato y lo que se consolida como conocimiento generalizable — sin recurrir a matrices densas ni a backpropagation global tradicional sobre todo el modelo.

Lo incierto nunca fue si el sistema puede memorizar patrones locales (eso es directo). Lo difícil, y sigue siendo el núcleo experimental real, es que emerjan abstracción, composición y asignación de crédito con suficiente estabilidad para producir razonamiento útil — **y hacerlo sin que la maquinaria de soporte (auditoría, memoria, evidencia) explote antes de que el núcleo de lenguaje siquiera se ponga a prueba.** Eso último es, a la fecha de este documento, el punto exacto donde está el trabajo.

# Spec T1 — Entrenabilidad del núcleo recurrente con slots

Fuente única y textual: mensaje de Sol autorizando T1 tras el cierre de T0 (ver `cpu_native_recurrence_lab_v0.4.1/` para T0). Regla igual que en T0: **nada se implementa sin cita textual de este documento o del mensaje original de Sol.** Si algo no está cubierto, se pregunta antes de decidir.

Estado previo (cerrado, no se reabre en T1): T0-R PASS, T0-M PASS_STRONG, T0-RM PASS (D=1472,S=1,R=8, shared/Bclone=2.260x, identidad bit a bit del puente frozen contra gates aislados). Fixed-point shift=14 validado en hardware real (0% clipping mediano). Ver [[cnrl-v0.4.0-supersedes-cpu-native-arch]] / memoria de sesión para el historial completo.

## Pregunta única que T1 debe responder

> ¿Puede un núcleo pequeño con pesos compartidos y estado matricial aprender computación recurrente útil, estable y generalizable?

T1 NO prueba lenguaje, tokenizer, LM head grande, banco semántico, cuantización ni halting dinámico. Meter eso ahora impediría saber si un fallo viene de optimización, recurrencia, slots, retrieval, cuantización o lenguaje — cada cosa se prueba por separado.

## T1-A: arquitectura mínima

```text
Framework: PyTorch
Precisión de entrenamiento: FP32 inicialmente
D: 64 o 128
Slots S: 1, 4, 8
Rondas R: 1, 2, 4, 6, 8
Pesos del núcleo: compartidos
Normalización principal: global-RMS
Control adicional: group-RMS
Cuantización: ninguna durante la primera fase
Memoria externa: ninguna
Halting dinámico: todavía no
```

Estado: `X^(r) ∈ R^(S×D)`.

Cada ronda:

```text
U^(r) = SlotMix(X^(r))
X^(r+1) = RMSNorm[ X^(r) + g_r ⊙ F_θ(U^(r), e_r) ]
```

- `F_θ`: el mismo núcleo compartido en todas las rondas.
- `e_r`: embedding pequeño de profundidad (necesario para que la misma matriz pueda hacer funciones distintas en distinta profundidad; sin él el núcleo corre más riesgo de converger a un punto fijo inútil).
- `g_r`: puerta aprendida.
- `SlotMix`: puede ser atención diminuta S×S (costo bajo). El operador pesado sigue siendo una proyección compartida aplicada a todos los slots.

## T1-B: tareas sintéticas

No usar solo copy task (se puede resolver de forma superficial). Cuatro tareas mínimas:

1. **Associative recall**: pares `A→7, B→3, C→9`, consulta `B→?`. Prueba binding y selección.
2. **Multi-hop**: relaciones `A→B, B→C, C→D`, pregunta el resultado después de 1/2/3/4 saltos. Esencial para medir si aumentar R compra profundidad funcional.
3. **Variable binding con distractores**: `x=objeto_4; objeto_4.color=azul; objeto_7.color=rojo; pregunta: color(x)`. Prueba que el modelo preserve identidades y no responda por correlación superficial.
4. **Actualización secuencial de estado**: `registro inicial=3; +4; ×2; -5; resultado=?`. Prueba aplicación ordenada de operaciones.
5. **Generalización de longitud**: entrenar con 1–3 hops, evaluar también 4–6 hops. No se exige perfección fuera de distribución, pero sí degradación gradual, no colapso inmediato.

## Baselines obligatorios (mínimo 4 configuraciones)

| Variante | Parámetros únicos | Cómputo |
|---|---:|---:|
| `single` | 1 núcleo | 1 ronda |
| `shared` | 1 núcleo | R rondas |
| `untied` | R núcleos | R rondas |
| `vector-state` | 1 núcleo | R rondas, S=1 |

Comparaciones y qué responden:
- `shared - single` → si la recurrencia añade capacidad útil.
- `shared - vector-state` → si los slots aportan memoria de trabajo.
- `shared - untied` → cuánto se pierde por compartir pesos (untied no es el objetivo de producción, es un techo de calidad con los mismos FLOPs y más parámetros).

## Gates fijados ANTES de entrenar (no se negocian después de ver resultados)

### Gate 1 — Estabilidad
En al menos 5 semillas: sin NaN, sin Inf, sin explosión sostenida de gradiente, sin colapso inmediato de la pérdida. Una corrida aislada exitosa no basta.

### Gate 2 — Utilidad de profundidad
En multi-hop: `shared R=4` debe superar claramente a `shared R=1`. Umbral inicial: ≥15 puntos porcentuales en 3–4 hops. Si R=4 y R=1 quedan iguales, el modelo está ignorando sus rondas.

### Gate 3 — Viabilidad del weight sharing
`shared` debe conservar ≥90% de la exactitud de `untied`, o quedar a ≤5 puntos porcentuales, en las tareas principales. (Ej.: untied=95%, shared=60% → compartir pesos no funciona, aunque el hardware sea rápido.)

### Gate 4 — Slots no colapsados
Medir: coseno promedio entre slots, rango efectivo de la matriz de slots, entropía de atención, efecto de ablacionar cada slot, % de slots que reciben gradiente significativo. Criterios iniciales:
```text
coseno medio absoluto < 0.90
effective rank > 0.5 × S
ningún slot permanentemente muerto en >95% de ejemplos
```
No basta con buena accuracy si los slots terminan siendo copias del mismo vector.

### Gate 5 — Generalización
Entrenado hasta 3 hops: 4 hops debe superar claramente el baseline aleatorio. Aumentar rondas durante inferencia solo cuenta como éxito si ayuda de forma consistente — más rondas que empeoren se registran como overthinking, no se ocultan.

### Gate 6 — Dependencia real entre rondas
Guardar estados `X^(0),X^(1),...,X^(R)` y comprobar: que cambian entre rondas, que las actualizaciones no convergen inmediatamente a cero, que permutar las rondas o sustituir `X^(r)` por `X^(0)` degrada el resultado, que poner a cero el embedding de ronda tiene efecto medible. Objetivo explícito: evitar repetir el error de los primeros probes (profundidad ejecutada pero estado fijo).

## Fuera de alcance en T1 (no negociable sin nueva autorización)

```text
retrieval ANN, corpus lingüístico, tokenizer grande, LM head,
Q4, int8 training, dynamic halting, banco de conocimiento,
destilación de un LLM
```

## Plataforma

T1 puede entrenarse en esta laptop (modelos diminutos; el objetivo es corrección y dinámica de aprendizaje, no velocidad). Progresión de fases futuras (no autorizadas todavía, solo referencia):
```text
T1: tareas sintéticas diminutas — CPU válida
T2: destilación lingüística — GPU
T3: memoria externa y retrieval
T4: cuantización y retorno a CPU
```

## Item pendiente heredado de T0 (no bloquea T1, pero debe quedar archivado)

Sol pidió que `analysis.md` de CNRL v0.4.1 tenga una sección "T0-RM: matrixización end-to-end con transición real" con `G_RM(8)=MAC/s_recurrente(S=8)/MAC/s_recurrente(S=1)` y `G_RM(16)` — distinto de los ratios shared/Bclone que ya reportamos (2.260/1.059/1.068x a S=1/8/16). Si el analizador ya la calcula, alcanza con archivar esa tabla; si no, pedirla antes de cerrar el archivo de T0 definitivamente. No es requisito para empezar T1.

## Regla de trabajo (igual que T0)

1. Nada se implementa sin cita textual de este documento.
2. Los 6 gates se fijan ANTES de entrenar y no se relajan después de ver resultados — si un gate falla, se reporta como falla, no se reinterpreta.
3. Cualquier decisión de diseño no cubierta textualmente (hiperparámetros de optimizador, arquitectura exacta de SlotMix, etc.) se pregunta antes de decidir unilateralmente.
4. Reportar con números crudos (per-seed, no solo promedios) dado que Gate 1 exige "al menos 5 semillas" explícitamente.

# Spec de correcciones — T0-M Recurrence Probe

Regla obligatoria para toda corrección de este documento: **ninguna implementación se hace sin cita textual o fórmula exacta del MD que la exige.** Si un punto no tiene cita, no se implementa — se pregunta primero. Interpretación propia, "parece razonable" o continuidad con trabajo anterior NO son justificación válida.

Fuentes (únicas fuentes válidas de justificación):
- `cpu-native-arch/MD/Conversacion.md`
- `cpu-native-arch/MD/Respuesta_1.md` (T0-R aislado, kernel Q4→int8)
- `cpu-native-arch/MD/Respuesta_2.md` (recurrencia, Bridges 1–4)

Archivo auditado: `cpu-native-arch/t0m_recurrence_probe.cpp` (líneas citadas corresponden a su estado actual, no commiteado).

Orden de ejecución: el orden **lo define el propio MD**, no es una decisión nuestra — Respuesta_2.md, sección "Gate corregido que sí debe ejecutarse", define expresamente Puente 1 → Puente 2 → Puente 3 → Puente 4, cada uno condición del siguiente ("Si aquí T0-R y T0-M sobreviven, la recurrencia densa no es el problema").

Fuera de alcance: cualquier trabajo sobre `int8_probe.cpp` con variante Bclone y filas congeladas queda descartado — auditoría confirmó que no corresponde a ningún experimento especificado en los MD.

---

## Bloque 0 — Prerrequisitos comunes (bloquean Puente 1)

### 0.1 Kernel debe ser el fused aceptado en T0-M

- **Cita MD**: Respuesta_2.md, §1 — "En el T0-M estático aceptado, el kernel mantiene acumuladores SIMD durante toda la dimensión y hace la reducción horizontal una sola vez al terminar cada producto fila-slot... Antes de interpretar el colapso, el probe recurrente debe usar literalmente el mismo kernel fusionado aceptado, cambiando únicamente el puntero del estado de entrada."
- **Código actual**: `run_fused_impl` (líneas 438-475) reduce horizontalmente cada 16 columnas vía `horizontal_sum_i32` + `checked_add_i64` (líneas 448-456), con fallback escalar (459-468) — no mantiene acumulador persistente durante toda la dimensión.
- **Corrección exigida**: sustituir el patrón de reducción-por-tile por acumuladores SIMD persistentes durante toda la dimensión, con reducción horizontal única al final — igual al microkernel que pasó T0-M en `t0m_int8_probe.cpp`.
- **Bloquea**: Puente 1 y Puente 2 (ambos exigen "literalmente el mismo kernel fusionado aceptado").

### 0.2 Dimensión del gate

- **Cita MD**: Respuesta_2.md, §5 — "Con D=512 el bloque completo contiene 512²=262 144 bytes=256 KiB... Con cuatro shards iguales: 64 KiB por núcleo. Eso no es el régimen original de T0-R"; tabla de D recomendados (1280–1792 KiB/núcleo); "D=1472 o D=1600 son puntos mucho más representativos". Puente 1 (líneas del gate): "D=1472, S=1, R=16".
- **Código actual**: `kDefaultDimension = 512` (línea 33).
- **Corrección exigida**: no es obligatorio cambiar el default del binario, pero **toda corrida de gate debe invocarse explícitamente con `--dimension 1472`** (valor literal del MD para Puente 1/2) y esto debe quedar documentado en el comando/script usado para esa corrida.
- **Bloquea**: Puente 1 y Puente 2.

### 0.3 Sharding proporcional obligatorio

- **Cita MD**: Respuesta_1.md, "Configuración mínima corregida" — "Shards: balanceados Zen5/Zen5c"; Respuesta_2.md §9 — "$P_i \propto \text{rendimiento medido del núcleo } i$... Eso no ocurrió. Los cuatro workers realizan el mismo trabajo."
- **Código actual**: default `rows_per_worker{128, 128, 128, 128}` (línea 57) — reparto igual, no proporcional.
- **Corrección exigida**: toda corrida de gate debe pasar `--rows-per-worker` calculado proporcionalmente a la calibración por núcleo medida (mismo mecanismo ya validado en el sweep de T0-R original). El default igual puede quedar como valor de conveniencia para pruebas funcionales, pero **nunca usarse en una corrida que se interprete como gate**.
- **Bloquea**: Puente 1, 2, 3, 4 (todos exigen sharding proporcional).

### 0.4 Bclone debe probar identidad física de bytes, no identidad de objeto

- **Cita MD**: Respuesta_2.md, §4 — "Bclone: R buffers físicamente diferentes, pero todos contienen exactamente los mismos bytes... Con Bclone, A y B deben generar los mismos checksums en cada ronda. Si no son idénticos, el resultado se rechaza."
- **Código actual**: el self-test compara `&bclone_blocks[round]` — dirección del objeto `WeightBlock`/`vector`, no del buffer de datos subyacente (`weights.data()`). No demuestra que las asignaciones físicas de memoria sean distintas.
- **Corrección exigida**: el self-test debe comparar el puntero real al buffer (`.weights.data()`) entre bloques, verificando (a) direcciones físicas distintas y (b) contenido byte-idéntico — ambas condiciones exigidas literalmente por el MD.
- **Bloquea**: Puente 1 (Bclone es el control central de ese puente).

### 0.5 Instrumentación fuera del timer

- **Cita MD**: Respuesta_2.md, §8 — "La instrumentación por ronda debe salir completamente del camino cronometrado. Para corrección se ejecuta un self-test separado. En el benchmark basta con: checksum final; contador ligero por worker; validación fuera del timer." Puente 3 (gate): "sin checksums por ronda dentro del timer."
- **Código actual**: en `run_timed`, `this_round_finite` / `this_round_overflow` / `this_round_clipped_cells` se declaran y reservan (líneas 1141-1147) después de `begin = clock.now()` (línea 1139), y se les hace `push_back` dentro del loop por ronda (líneas 1163-1168), todo dentro de la ventana medida.
- **Corrección exigida**: mover toda construcción/reserve/push_back de estas estructuras fuera de la región entre `begin` y `end`. Dentro del timer solo debe quedar el checksum final y flags mínimos por-worker en memoria pre-reservada antes de `begin`.
- **Bloquea**: Puente 3 y Puente 4 explícitamente; deseable también para cualquier medición de Puente 1/2 que use este mismo `run_timed`.

---

## Puente 1 — Reproducir el kernel

- **Cita MD exacta** (Respuesta_2.md, "Gate corregido que sí debe ejecutarse"): "Dentro del mismo binario recurrente: D=1472, S=1, R=16, 4 workers físicos, sharding proporcional, sin transición. Debe reproducir el comportamiento del T0-R aceptado. Usar: A=una matriz reutilizada, Bclone=16 copias byte-idénticas. A y B deben producir exactamente los mismos outputs y checksums."
- **Depende de**: 0.1, 0.2, 0.3, 0.4 resueltos.
- **Acción**: ejecutar `t0m_recurrence_probe.cpp` con `--component gemv-only --dimension 1472 --slots 1 --recurrent-depth 16 --variant A|Bclone --rows-per-worker <proporcional> --mode fused`. Este modo (`Component::gemv_only`, línea 43) ya existe en el binario y está cableado en `run_timed` (líneas 1153, 1159, 1165, 1182-1189), pero nunca fue ejercido como benchmark cronometrado — solo como self-test de corrección (líneas 1000-1035, D=1472 fijo).
- **Por qué este binario y no otro**: el MD nunca menciona un binario distinto para Puente 1; lo define como el primer eslabón de la misma secuencia que termina en Puente 4 con transición y RMSNorm, funciones que solo existen en `t0m_recurrence_probe.cpp`. Usar `t0m_int8_probe.cpp` o `int8_probe.cpp` para este paso no tiene respaldo textual.
- **Criterio de aceptación**: el ratio A/Bclone debe reproducir la separación de T0-R aislado y validado (Respuesta_1.md: 2.5–2.9×, PASS fuerte). Si no reproduce, **no se avanza a Puente 2** — se investiga la causa dentro de este mismo puente, citando MD en cada hipótesis.

---

## Puente 2 — Matrixización

- **Cita MD exacta**: "D=1472, S=1, 4, 8, 16, R=1, 4, 8, 16. El kernel fusionado debe ser literalmente el aceptado en T0-M. Debe volver a aparecer la curva de G(S) dentro del mismo harness."
- **Depende de**: Puente 1 aprobado, 0.1 resuelto.
- **Acción**: implementar el barrido S×R integrado en el mismo binario (confirmado ausente por auditoría: no hay barrido S×R dentro de `t0m_recurrence_probe.cpp` hoy), mismo D=1472.
- **Criterio de aceptación**: recuperar la curva G(S) con la misma métrica de T0-M aislado (Respuesta_1.md: G8≥1.5 mínimo aceptable, ≥2.0 fuerte). Los valores ya medidos en T0-M aislado (G8 max 3.89, G16 max 3.61) sirven de referencia de magnitud, no de umbral obligatorio en este harness distinto.

---

## Puente 3 — Transición local mínima

- **Cita MD exacta**: "$X' = \operatorname{sat8}(X + (Y \gg s))$ con: double buffering; transición paralela por shard; una barrera por ronda; sin checksums por ronda dentro del timer. Si aquí T0-R y T0-M sobreviven, la recurrencia densa no es el problema."
- **Depende de**: Puente 1 y 2 aprobados, 0.5 resuelto.
- **Acciones**:
  1. Eliminar la transición serializada en el hilo coordinador. **Cita del problema actual**: Respuesta_2.md §2 — "el hilo principal, solo, ejecuta `apply_transition_fast`... 4 núcleos: esperan... el hilo principal no está fijado a un núcleo." Código actual: `apply_transition_fast` se llama desde `run_timed` (línea 1154), en el hilo que también hace de coordinador de barreras, sin `AffinityGuard` propio (el `AffinityGuard` solo cubre a los workers, líneas 1084-1086).
  2. Implementar doble buffer `state_current`/`state_next`, cada worker escribiendo su propio rango. **Cita**: "Transición concreta que preserva el paralelismo, Opción A" — "Usaría dos buffers de estado: state_current, state_next... Cada worker escribe su propio rango de state_next. Se evita sobrescribir el estado que otros workers todavía leen." Código actual: un único `std::vector<int8_t> state` (línea 1064) leído por los workers y mutado centralmente por el coordinador.
  3. Escala compatible en el residual. **Cita**: Respuesta_2.md §7 — "$r = \alpha X + \beta Y$" o en fijo "$r_i = aX_i + (Y_i \gg s)$" porque "$X_i$ es int8... $Y_i$ es un acumulador... de decenas o cientos de miles... En la práctica $Y+X\approx Y$. El residual casi desaparece." Código actual: `residual = output[...] + state[...]` sin escala (línea 599).
- **Criterio de aceptación**: la transición debe reducirse a ≤10–15% del tiempo de ronda antes de considerar solapamiento con el siguiente GEMM. **Cita**: "debe comprobarse si la versión paralela sencilla reduce la transición a menos del 10–15% del tiempo. Es probable que el solapamiento completo sea más costoso que la operación que intenta ocultar" — por tanto no implementar solapamiento todavía, solo paralelizar la transición.

---

## Puente 4 — RMSNorm paralelo

- **Cita MD exacta**: "Añadir después el RMS global paralelo con reducción de $4S$ escalares. Medir separadamente: GEMM; residual + partial sum; reducción RMS; requantización; barrera; full round. No utilizar solamente MAC/s, porque la transición realiza trabajo útil que no se cuenta como MAC."
- **Depende de**: Puente 3 aprobado.
- **Acciones**:
  1. Implementar la Opción A completa del MD ("RMSNorm global paralelo") — reducción de solo $4 \times S$ escalares en una barrera de completion, cada worker normaliza y requantiza únicamente su propio shard. Usar acumuladores `int32`, no `int64`, en el camino rápido — **cita**: "Usaría acumuladores int32, no int64, en el camino rápido. Para todas las dimensiones propuestas hay un margen enorme antes del overflow" (verificado también en Respuesta_2.md §1: para D=512 el máximo teórico del dot product int8 es $512 \cdot 127^2 = 8\,258\,048$, cabe holgadamente en int32 — la misma lógica de margen aplica al residual con D=1472).
  2. Eliminar el patrón escalar-primero-luego-vectorial-condicional de `apply_transition_fast`. **Cita del problema**: Respuesta_2.md §6 — la presencia de `vaddpd/vmulpd/vdivpd/vsqrtpd` "no significa que todo el camino caliente esté vectorizado... El hot path no debería cargar con: overflow checks imposibles para estas dimensiones; dos cálculos de suma de cuadrados; double precision; arrays temporales por cada cuatro valores; fallback para redondeo exacto." Código actual: recorrido escalar completo (líneas 594-609) seguido de una verificación vectorial condicional (líneas 620-643) que solo reemplaza el RMS si coincide exactamente con el escalar — ese patrón de doble cálculo debe quedar limitado al self-test, no al camino de producción.
  3. Instrumentación con desglose de componentes. **Cita**: ver arriba (Puente 4). Código actual: el CSV expone tiempo total, MAC/s y checksums/flags, sin separar GEMM/residual/reducción/requantización/barrera.
  4. Evaluar necesidad de variante C (expulsión de caché) a esta escala. **Cita**: Respuesta_1.md, "Configuración mínima corregida", define C como parte del set base (A/B/C) del gate original. Código actual: `enum Variant` solo tiene `{a, b, bclone}` (confirmado sin C). Esto se evalúa **solo si** Puente 4 requiere replicar el control de expulsión original — no implementar preventivamente sin esa necesidad confirmada.

---

## Criterios de cierre de la línea completa

**Cita textual completa** (Respuesta_2.md, "Criterios de cierre reales"):

> La línea de recurrencia densa se debería cerrar en esta laptop únicamente si, después de estas correcciones:
> 1. El mismo kernel aislado reproduce T0-R y T0-M.
> 2. A y Bclone siguen trayectorias idénticas.
> 3. Los pesos están en el régimen de 400–750 KiB por núcleo.
> 4. La transición está distribuida entre los cuatro workers.
> 5. No existe instrumentación dentro del timer.
> 6. El tiempo por muestra es suficientemente largo para bajar la dispersión por debajo de aproximadamente 10%.
> 7. Aun así, la transición recurrente reduce G8 cerca de 1 y elimina A/B.
>
> Entonces sí quedaría demostrado: Una recurrencia densa globalmente sincronizada no aprovecha suficientemente este Ryzen de cuatro núcleos.

Solo si los 7 se cumplen (los 6 primeros como precondición de una medición válida, el 7 como resultado observado) se puede reportar una conclusión negativa sobre la recurrencia densa a Sol. Ninguna corrida anterior a este documento cumple las precondiciones 1–6 simultáneamente.

---

## Regla de trabajo para opencode

1. Cada cambio de código debe poder señalar, en el mensaje de commit o en la respuesta de reporte, qué sección/cita de este documento (y transitivamente, del MD original) lo justifica.
2. No se avanza de un Puente al siguiente sin que el Puente actual cumpla su criterio de aceptación explícito, documentado con números crudos (no solo "parece que mejoró").
3. Si una implementación requiere una decisión de diseño no cubierta textualmente por el MD (por ejemplo, un detalle de bajo nivel no especificado), se pregunta antes de decidir unilateralmente — no se improvisa ni se rellena el vacío con la solución "más razonable".
4. Nada de lo listado en este documento se reordena, se omite ni se sustituye sin acuerdo explícito, ya que el orden y el contenido provienen directamente del MD, no de preferencia de implementación.

## Veredicto

**No aceptaría todavía el cierre negativo de la recurrencia real ni replantearía la arquitectura de fondo.** El probe sí implementa una dependencia auténtica entre rondas y su validación funcional es bastante sólida: conserva el estado de cada ronda y compara cada celda contra una referencia. 

Pero el benchmark recurrente **no es una composición fiel de los T0-R y T0-M que habían pasado**. Hay varios desvíos importantes, y al menos cuatro de ellos pueden explicar gran parte del colapso.

Tampoco puedo recomputar los valores `G8`, `G16` o las medianas porque no se adjuntaron los CSV crudos. La auditoría siguiente se basa en el código y en las cifras reportadas.

# 1. El kernel recurrente no es el mismo kernel que pasó T0-M

Este es el primer desvío crítico.

En el T0-M estático aceptado, el kernel mantiene acumuladores SIMD durante toda la dimensión y hace la reducción horizontal una sola vez al terminar cada producto fila-slot. Por ejemplo, conserva `acc0`, `acc1`, etc., a través del bucle completo sobre `dimension`. 

En el probe recurrente, cada grupo de 16 elementos se reduce horizontalmente inmediatamente a un entero escalar y luego se agrega mediante `checked_add_i64`:

```cpp
chunk = horizontal_sum_i32(...);
checked_add_i64(sums[slot], chunk, sums[slot]);
```

Eso ocurre dentro del bucle caliente por dimensión y por slot. 

Esta diferencia introduce:

* una reducción horizontal cada 16 columnas;
* operaciones escalares `int64`;
* comprobación de overflow por cada fragmento;
* ramas adicionales;
* posibles spills para `S=8` y especialmente `S=16`;
* una cadena de dependencia escalar sobre `sums[slot]`.

Para `D=512`, el máximo teórico del dot product int8 es:

$$
512\cdot127^2=8\,258\,048
$$

Eso cabe holgadamente en `int32`. No existe necesidad física de comprobar overflow `int64` cada 16 valores.

Por tanto, el `dumpbin` de la transición no resuelve esta desviación. Demuestra que hay instrucciones vectoriales en la transición, pero **no que el GEMM recurrente sea el mismo microkernel que pasó T0-M**.

Antes de interpretar el colapso, el probe recurrente debe usar literalmente el mismo kernel fusionado aceptado, cambiando únicamente el puntero del estado de entrada.

# 2. La transición completa está serializada en el hilo coordinador

Los cuatro workers calculan sus shards y llegan a `depth_done`. Después quedan esperando la siguiente ronda. 

Entonces el hilo principal, solo, ejecuta:

```cpp
apply_transition_fast(output, state, ...)
```

sobre los \(S\times D\) elementos completos. 

Eso significa:

```text
4 núcleos: GEMM
1 hilo no fijado: residual + RMSNorm + requantización
4 núcleos: esperan
```

No es la transición que debería usar la arquitectura final. Además:

* el hilo principal no está fijado a un núcleo;
* los cuatro núcleos físicos ya tienen workers fijados;
* el coordinador tendrá que ejecutarse sobre un hermano SMT o desplazar otro hilo;
* todas las líneas del estado quedan modificadas por un solo núcleo;
* en la ronda siguiente, los cuatro workers vuelven a solicitar ese estado;
* una espera de 60–75 µs puede hacer que `std::barrier` deje de hacer espera activa y pase a estacionar/despertar hilos.

Eso puede producir un coste no aditivo y explicar parte de la varianza del 37%.

La transición aislada de 60.9 µs no modela necesariamente todo ese efecto de coherencia, estacionamiento y despertar cuando se intercala dieciséis veces con el GEMM.

# 3. La explicación de “se rompió el pipeline” no está respaldada por el código

Claude atribuye el colapso a que la recurrencia destruye un pipeline entre rondas.

Pero el T0-M estático ya utilizaba barreras antes y después de **cada profundidad**. Los workers no podían calcular libremente varias rondas por adelantado: esperaban `phase_ready`, ejecutaban una ronda y esperaban `phase_done`. 

Por tanto:

> No había un pipeline computacional entre rondas que la recurrencia pudiera destruir.

Lo que sí cambia es esto:

```text
Estático:
barrera → GEMM → barrera → siguiente ronda

Recurrente actual:
barrera → GEMM → barrera
        → transición serial
        → checksum/validación
        → siguiente ronda
```

La dependencia recurrente es real, pero la causa más probable no es “pérdida de pipeline”. Es la combinación de:

* transición serial;
* barreras que ahora permanecen bloqueadas más tiempo;
* hilo coordinador sin afinidad;
* coherencia de caché del estado;
* microkernel distinto;
* instrumentación adicional dentro del cronómetro.

# 4. A y B ya no aíslan la residencia de memoria

En una prueba de hardware, A y B deben realizar exactamente la misma aritmética y recorrer exactamente la misma trayectoria de estados. La única diferencia debe ser la dirección física de los pesos.

Eso no sucede.

En A se genera un bloque con una semilla. En B se agrega un término dependiente de la ronda, incluso para la ronda cero:

```cpp
if (variant == Variant::b)
    weight_seed ^= ... * (round + 1U);
```

Por tanto, A y B usan matrices con contenidos diferentes desde la primera ronda. 

En el caso estático esto era menos grave, porque el estado permanecía fijo y las instrucciones ejecutadas eran prácticamente independientes de los valores. En recurrencia, las matrices distintas producen:

* estados distintos;
* RMS distintos;
* clipping distinto;
* diferente frecuencia de fallback escalar;
* diferentes ramas de seguridad;
* diferentes trayectorias de caché del estado.

El A/B de 0.94 **no puede interpretarse como residencia frente a no residencia**.

La variante correcta debe ser:

```text
A:
una dirección de pesos reutilizada R veces.

Bclone:
R buffers físicamente diferentes,
pero todos contienen exactamente los mismos bytes.

U:
R matrices diferentes.
Solo para estudiar una arquitectura untied, no el gate físico.
```

Con `Bclone`, A y B deben generar los mismos checksums en cada ronda. Si no son idénticos, el resultado se rechaza.

Este cambio por sí solo elimina un confound importante.

# 5. D=512 no reproduce el régimen de memoria de T0-R

El probe recurrente obliga a que:

$$
\sum_i O_i=D
$$

porque el output debe volver a tener dimensión \(D\). 

Cada bloque contiene:

$$
D\times D
$$

pesos int8. 

Con:

$$
D=512
$$

el bloque completo contiene:

$$
512^2=262\,144\text{ bytes}=256\text{ KiB}
$$

Con cuatro shards iguales:

$$
64\text{ KiB por núcleo}
$$

Eso no es el régimen original de T0-R, donde se probaron aproximadamente 384–768 KiB **por núcleo**.

Incluso B con profundidad 16 contiene:

$$
256\text{ KiB}\times16=4\text{ MiB en total}
$$

y aproximadamente:

$$
1\text{ MiB por núcleo}
$$

El conjunto completo de B cabe nominalmente dentro de los 8 MiB de L3. Puede producir presión sobre L2, pero no reproduce el caso:

```text
A desde L2
contra
B saturando DRAM
```

Así que el A/B estático de 2.69× en ese tamaño puede ser una diferencia L2↔L3, no la diferencia L2↔DRAM demostrada por T0-R.

Para un único bloque cuadrado int8, los tamaños correctos serían aproximadamente:

| \(D\) | Pesos por núcleo, 4 shards iguales |
| ----: | ---------------------------------: |
|  1280 |                            400 KiB |
|  1472 |                            529 KiB |
|  1600 |                            625 KiB |
|  1728 |                            729 KiB |
|  1792 |                            784 KiB |

Por eso **D=1472 o D=1600** son puntos mucho más representativos del núcleo residente propuesto.

Además, la proporción transición/GEMM mejora al aumentar \(D\):

$$
C_{\text{GEMM}}\propto SD^2
$$

$$
C_{\text{transición}}\propto SD
$$

Al pasar de 512 a 1472:

* el GEMM aumenta aproximadamente \(8.27\times\);
* la transición aumenta aproximadamente \(2.88\times\);
* el peso relativo de la transición cae aproximadamente \(2.88\times\).

Un coste del 36% en \(D=512\) podría caer hacia el 12–13% incluso antes de paralelizar la transición. No es una predicción de rendimiento exacta, pero demuestra que el tamaño elegido favorece artificialmente al coste lineal de transición.

# 6. La transición no está completamente vectorizada

La presencia de `vaddpd`, `vmulpd`, `vdivpd` y `vsqrtpd` no significa que todo el camino caliente esté vectorizado.

El código primero realiza un recorrido escalar que:

* suma residual;
* comprueba overflow;
* almacena `int64`;
* convierte a `double`;
* calcula cuadrados con comprobaciones;
* acumula `sum_squares` escalarmente. 

Después calcula un RMS escalar y, bajo ciertas condiciones, vuelve a recorrer los valores para recalcular la suma de cuadrados vectorialmente. 

Finalmente, la ruta “vectorial”:

* procesa solo cuatro doubles;
* usa división vectorial;
* escribe a un array temporal;
* hace clamp y stores escalarmente por lane;
* puede regresar a la ruta escalar. 

Es una implementación orientada a equivalencia exacta con la referencia, no una transición de producción.

La referencia exacta debe mantenerse en el self-test. El hot path no debería cargar con:

* overflow checks imposibles para estas dimensiones;
* dos cálculos de suma de cuadrados;
* double precision;
* arrays temporales por cada cuatro valores;
* fallback para redondeo exacto en cada grupo.

# 7. El residual mezcla escalas incompatibles

La transición calcula directamente:

$$
r_i=Y_i+X_i
$$

donde:

* \(X_i\) es int8, con magnitud máxima 127;
* \(Y_i\) es un acumulador de un dot product int8×int8 sobre 512 elementos, normalmente de decenas o cientos de miles.

Eso está implementado literalmente antes del RMSNorm. 

No existe un multiplicador que lleve \(Y\) a la escala de \(X\). Por tanto, en la práctica:

$$
Y+X\approx Y
$$

El residual casi desaparece. Esta no es una transición cuantizada realista.

En una red cuantizada deben existir escalas compatibles:

$$
r =
\alpha X+
\beta Y
$$

o, en fijo:

$$
r_i=
aX_i+
\left(Y_i\gg s\right)
$$

antes de sumar.

Esto no explica por sí solo el rendimiento, pero significa que el probe todavía no representa una transición neuronal plausible.

# 8. El cronómetro incluye instrumentación que no existía en T0-M

En cada repetición cronometrada, el hilo principal:

* construye varios vectores;
* ejecuta `reserve`;
* calcula checksum después de cada ronda;
* registra clipping;
* hace `push_back` de estadísticas por ronda. 

En el T0-M estático, el tiempo se tomaba directamente alrededor de cada fase de kernel y después se acumulaba. 

Por tanto, los dos valores de throughput no tienen el mismo alcance temporal.

La instrumentación por ronda debe salir completamente del camino cronometrado. Para corrección se ejecuta un self-test separado. En el benchmark basta con:

* checksum final;
* contador ligero por worker;
* validación fuera del timer.

# 9. Posible desvío adicional: sharding igual en lugar de proporcional

El probe recurrente usa por defecto:

```text
128,128,128,128
```

filas por worker. 

Eso ignora la asimetría medida entre el Zen 5 y los tres Zen 5c. Si el comando no sobrescribió `--rows-per-worker`, el worker más lento determina el tiempo de cada ronda y los otros núcleos esperan en la barrera.

Sin los CSV no puedo saber si se pasó un reparto proporcional explícito. Debe quedar registrado como requisito del siguiente gate.

# Qué demuestra realmente el resultado actual

El ledger correcto es:

```text
T0-R aislado:
PASS fuerte.

T0-M aislado:
PASS fuerte, según los resultados reportados.

Recurrencia funcional:
PASS de corrección; el estado sí cambia y se valida por ronda.

T0-RM end-to-end actual:
RECHAZADO COMO EVIDENCIA DE RENDIMIENTO.
```

No porque el resultado sea necesariamente falso, sino porque no aísla la misma pregunta:

* cambió el kernel;
* cambió el tamaño de pesos;
* cambió el contenido A/B;
* cambió la distribución numérica;
* cambió el alcance del timer;
* serializó la transición;
* posiblemente cambió el sharding.

La explicación “la dependencia secuencial destruyó los beneficios” todavía no está demostrada.

---

# Transición concreta que preserva el paralelismo

Para una recurrencia densa exacta no es posible eliminar toda dependencia:

$$
X^{(r+1)}
$$

debe existir antes de calcular exactamente:

$$
W X^{(r+1)}
$$

Eso impone al menos un evento de estado-listo por ronda.

Pero no obliga a hacer toda la transición en un hilo central.

## Opción A: RMSNorm global paralelo

Usaría dos buffers de estado:

```text
state_current
state_next
```

y cuatro workers permanentes. Ningún hilo coordinador calcula dentro del loop.

Cada ronda:

```cpp
// Cada worker opera sobre su shard de filas.
GEMM_shard(W_i, state_current, Y_i);

// En el mismo store:
residual_i = scaled(Y_i) + alpha * state_current_i;
partial_sum_sq[i][slot] = sum(residual_i²);

// Barrera con completion:
// suma solamente 4 × S escalares y calcula inv_rms[slot].
rms_reduce_barrier.arrive_and_wait();

// Cada worker normaliza y requantiza únicamente su shard.
requantize_local(residual_i, inv_rms, state_next_i);

// Garantiza que todo state_next esté publicado.
state_ready_barrier.arrive_and_wait();

swap(state_current, state_next);
```

Características:

* GEMM y construcción del residual se fusionan.
* Los cuatro núcleos calculan transición.
* La única parte serial son \(4S\) sumas y \(S\) raíces.
* No hay un hilo principal ejecutando AVX en un hermano SMT.
* Cada worker escribe su propio rango de `state_next`.
* Se evita sobrescribir el estado que otros workers todavía leen.
* Se eliminan las transferencias de propiedad causadas por un único escritor global.

Usaría acumuladores `int32`, no `int64`, en el camino rápido. Para todas las dimensiones propuestas hay un margen enorme antes del overflow.

La barrera de reducción puede tener una función de completion ejecutada por el último worker que llega. Para el benchmark usaría una barrera activa basada en atomics y `_mm_pause`, no `std::barrier`, porque las fases duran microsegundos y no conviene estacionar hilos.

Esta versión mantiene RMSNorm global exacto y requiere dos sincronizaciones cortas por ronda. No elimina la dependencia, pero elimina la serialización incorrecta.

## Opción B: Group-RMSNorm alineado con los shards

Cada worker normaliza únicamente su rango:

$$
\operatorname{RMS}_i=
\sqrt{
\frac{1}{D_i}
\sum_{j\in\text{shard }i}r_j^2+\epsilon
}
$$

Entonces:

```text
GEMM local
→ residual local
→ RMS local
→ requant local
→ una barrera
→ siguiente ronda
```

No necesita reducción global. Cambia la operación matemática, pero es una operación perfectamente entrenable si el modelo se entrena con ella desde el inicio.

Esta es probablemente la primera transición verdaderamente **CPU-native** que debería probarse.

## Opción C: transición fija o aprendida sin normalización por ronda

La más barata sería:

$$
r_i=a_rX_i+b_rY_i
$$

$$
X_i'=
\operatorname{sat8}
\left(
r_i\gg s_{r,g}
\right)
$$

donde:

* \(a_r\) y \(b_r\) son escalas aprendidas;
* \(s_{r,g}\) es un shift por ronda, slot o grupo;
* todos los valores se mantienen en fixed-point;
* se usa QAT durante entrenamiento.

No requiere `sqrt`, división ni reducción global. Cada worker puede producir directamente su shard de `state_next`, seguido de una sola barrera.

Para controlar deriva:

```text
rondas 1–3: transición local fija
ronda 4: RMSNorm global o Group-RMSNorm
```

o:

```text
RMSNorm una sola vez al entrar y salir del núcleo,
no después de cada vuelta.
```

No existe obligación arquitectónica de ejecutar RMSNorm global en cada repetición. Esa elección fue introducida por el probe, no por la hipótesis original.

# ¿Se puede solapar la normalización con el siguiente GEMM?

Con RMSNorm global exacto, solo parcialmente.

Primero hay que conocer:

$$
\operatorname{RMS}(r)
$$

antes de producir los valores normalizados exactos. Por eso no puede arrancar todo el siguiente GEMM inmediatamente.

Después de calcular `inv_rms`, sí sería posible:

1. normalizar el estado por tiles;
2. publicar cada tile;
3. permitir que los workers comiencen a acumular las columnas correspondientes del siguiente GEMM.

Pero esto obliga a:

* invertir el orden del microkernel;
* mantener acumuladores parciales para muchas filas;
* introducir flags por tile;
* aumentar coherencia entre núcleos;
* posiblemente destruir el buen layout que hizo pasar T0-M.

No lo implementaría todavía. Primero debe comprobarse si la versión paralela sencilla reduce la transición a menos del 10–15% del tiempo. Es probable que el solapamiento completo sea más costoso que la operación que intenta ocultar.

# Gate corregido que sí debe ejecutarse

## Puente 1: reproducir el kernel

Dentro del mismo binario recurrente:

```text
D = 1472
S = 1
R = 16
4 workers físicos
sharding proporcional
sin transición
```

Debe reproducir el comportamiento del T0-R aceptado.

Usar:

```text
A       = una matriz reutilizada
Bclone  = 16 copias byte-idénticas
```

A y B deben producir exactamente los mismos outputs y checksums.

## Puente 2: matrixización

```text
D = 1472
S = 1, 4, 8, 16
R = 1, 4, 8, 16
```

El kernel fusionado debe ser literalmente el aceptado en T0-M. Debe volver a aparecer la curva de `G(S)` dentro del mismo harness.

## Puente 3: transición local mínima

Añadir:

$$
X'=\operatorname{sat8}\left(X+\left(Y\gg s\right)\right)
$$

con:

* double buffering;
* transición paralela por shard;
* una barrera por ronda;
* sin checksums por ronda dentro del timer.

Si aquí T0-R y T0-M sobreviven, la recurrencia densa no es el problema.

## Puente 4: RMSNorm paralelo

Añadir después el RMS global paralelo con reducción de \(4S\) escalares.

Medir separadamente:

```text
GEMM
residual + partial sum
reducción RMS
requantización
barrera
full round
```

No utilizar solamente `MAC/s`, porque la transición realiza trabajo útil que no se cuenta como MAC.

# Criterios de cierre reales

La línea de recurrencia densa se debería cerrar en esta laptop únicamente si, después de estas correcciones:

1. El mismo kernel aislado reproduce T0-R y T0-M.
2. A y Bclone siguen trayectorias idénticas.
3. Los pesos están en el régimen de 400–750 KiB por núcleo.
4. La transición está distribuida entre los cuatro workers.
5. No existe instrumentación dentro del timer.
6. El tiempo por muestra es suficientemente largo para bajar la dispersión por debajo de aproximadamente 10%.
7. Aun así, la transición recurrente reduce `G8` cerca de 1 y elimina A/B.

Entonces sí quedaría demostrado:

> Una recurrencia densa globalmente sincronizada no aprovecha suficientemente este Ryzen de cuatro núcleos.

Pero incluso en ese caso no moriría toda la arquitectura. El rediseño correcto sería:

```text
núcleo recurrente por shards locales
+
2–4 vueltas locales sin sincronización
+
mixer global pequeño y periódico
```

En lugar de una matriz totalmente densa cada ronda:

$$
h_i^{r+1}=F_i(h_i^r,z)
$$

con un resumen global pequeño:

$$
z=\operatorname{Mix}(h_1,\ldots,h_4)
$$

actualizado cada varias vueltas. Eso permitiría que cada núcleo trabaje dentro de su L2 durante varias iteraciones y pague comunicación global solo periódicamente.

## Decisión

**No avanzar todavía a memoria externa ni LM head. Tampoco replantear de inmediato toda la arquitectura.**

La acción correcta es ejecutar una única secuencia de puente:

$$
\boxed{
\text{mismo kernel}
\rightarrow
\text{Bclone idéntico}
\rightarrow
\text{tamaño L2 correcto}
\rightarrow
\text{transición paralela}
}
$$

El hallazgo importante actual no es que la recurrencia haya matado la ventaja, sino este:

> **La implementación recurrente convirtió un kernel de cuatro núcleos en un pipeline de cuatro núcleos seguido por una transición serial no fijada, usando además una matriz demasiado pequeña y una variante B matemáticamente distinta.**

Eso debe corregirse antes de atribuir el colapso a la arquitectura.

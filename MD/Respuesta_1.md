## Veredicto

**No, este resultado no cierra la línea arquitectónica.** Cierra una implementación mucho más estrecha:

> Un GEMV Q4 que convierte pesos a FP32 y usa AVX2/FMA, con un solo vector de entrada y sin estado matricial, obtiene apenas 1–3% al reutilizar pesos en este Ryzen.

Eso es un resultado negativo válido para **ese kernel**. No hay razón para seguir aumentando únicamente tamaño o profundidad con el ejecutable actual: el punto de 512 KiB por núcleo y profundidad 16 ya era suficiente para separar A de B en capacidad de caché.

Pero el gate ejecutado **no fue exactamente el planteado**, y la desviación principal hace que A y B no pudieran separarse de forma importante aunque la residencia funcionara perfectamente.

---

# 1. La desviación decisiva: B nunca estuvo cerca de ser memory-bound

El formato real del probe utiliza:

$$
0.5\ \text{bytes/peso Q4}
+
\frac{4}{32}\ \text{bytes/peso de escala FP32}
=
0.625\ \text{bytes/peso}
$$

El código efectivamente almacena dos pesos por byte y una escala `float` por cada 32 pesos. 

Con el rendimiento agregado que reportas:

$$
6.6\ \text{GMAC/s}\times0.625
=
4.125\ \text{GB/s}
$$

Ese es aproximadamente el ritmo al que B consume pesos únicos desde DRAM una vez que sus bloques dejan de caber en caché.

Un canal DDR5-5600 de 64 bits tiene un máximo nominal de:

$$
5600\ \text{MT/s}\times8\ \text{bytes}
=
44.8\ \text{GB/s}
$$

Incluso suponiendo solo la mitad de ese rendimiento efectivo, unos 22 GB/s, el kernel estaba demandando menos de una quinta parte. Para que B chocara con DRAM, el kernel tendría que alcanzar aproximadamente:

$$
\frac{22\text{–}45\ \text{GB/s}}{0.625\ \text{B/MAC}}
\approx
35\text{–}72\ \text{GMAC/s}
$$

No 6.6 GMAC/s.

Por eso:

```text
A: pesos desde L2
B: pesos desde DRAM
```

pueden terminar casi empatados. Ambos están esperando principalmente las conversiones Q4→FP32, las instrucciones y las reducciones, no los pesos.

El kernel AVX2 carga 16 bytes Q4, separa nibbles, extiende repetidamente a enteros de 16/32 bits, convierte a FP32, multiplica por la escala y finalmente ejecuta FMA. Lo hace cuatro veces por grupo de 32 pesos.  

Que sea 3.71× más rápido que el escalar demuestra que la vectorización es real. **No demuestra que haya alcanzado el régimen limitado por memoria.**

### La conclusión sobre single-channel también cambia

Single-channel habría facilitado detectar la diferencia únicamente si B estuviera acercándose al ancho de banda disponible.

Pero reducir el techo de, por ejemplo, 40 a 20–25 GB/s no ayuda mucho cuando el kernel solo solicita aproximadamente 4 GB/s.

Por tanto:

> **El empate A/B no demuestra que L2 y DRAM rindan parecido. Demuestra que el kernel consume pesos demasiado lentamente para que esa diferencia decida el tiempo.**

Mi umbral anterior de aproximadamente 2× estaba incompleto: solo era válido después de imponer como precondición que B fuese demostrablemente memory-bound. Esa precondición no se operacionalizó en el gate.

---

# 2. No se implementó el \(m\) de la propuesta

Esta es la segunda desviación crítica.

En la arquitectura discutida:

$$
m=\text{cantidad de slots latentes}
$$

y la operación debía ser:

$$
W_{d_o\times d_i}
X_{d_i\times m}
$$

Es decir, un **small-GEMM** en el que cada bloque Q4 se desempaqueta una vez y se reutiliza para varios slots.

En el programa:

```text
--m = output rows
--K = input columns
--depth = repeticiones
```

Eso está declarado explícitamente en la interfaz. 

La operación real es:

$$
W_{M\times512}x_{512}
$$

con un único vector `input`. El tamaño `M` se aumenta hasta que el bloque ocupa 512, 1024 o 1280 KiB. No representa slots.

El mismo vector se usa en todas las pasadas y la salida no se convierte en entrada de la siguiente profundidad.  

Por tanto, el experimento no midió:

* transición GEMV → small-GEMM;
* reutilización de cada peso entre slots;
* amortización de dequantización entre slots;
* fracción del pico MAC frente a \(m\);
* intercambio entre profundidad \(K\) y workspace \(m\);
* dinámica recurrente del estado.

El script paralelo solo barre tamaños físicos y profundidades. Fija la dimensión de entrada en 512 y no contiene ninguna dimensión independiente de slots.  

Así que esto fue realmente:

> **T0-R preliminar: reutilización de un GEMV.**

No el T0 completo que combinaba residencia y estado matricial.

---

# 3. La variante C no valida lo que parece validar

El programa define un buffer de desalojo de 64 MiB. 

Pero lo recorre **dentro de la región temporizada**, después de cada pasada excepto la última.  

Con profundidad 16, cada worker toca:

$$
15\times64\ \text{MiB}
=
960\ \text{MiB}
$$

por invocación.

Con cuatro workers serían aproximadamente 3.75 GiB adicionales por invocación, antes de multiplicar por las repeticiones.

Por tanto, la caída:

```text
6.6B → 0.5–1.7B MAC/s
```

no demuestra que recargar el bloque Q4 desde DRAM sea dramáticamente costoso. Demuestra principalmente que añadir un barrido de cientos de megabytes o gigabytes dentro del cronómetro es costoso.

La variante C acordada debía:

1. expulsar los pesos;
2. excluir o contabilizar separadamente el coste de expulsión;
3. medir solamente la siguiente ejecución fría del kernel.

C tal como está implementada no sirve para afirmar:

$$
A \ll C \Rightarrow \text{beneficio de residencia}
$$

Sí sirve como smoke test de que el buffer realmente perturba la jerarquía, pero no como baseline cuantitativo.

---

# 4. El sharding tampoco fue el acordado

La ejecución paralela crea un `ProbeData` completo e idéntico para cada worker. Todos reciben la misma cantidad de filas y el mismo tamaño de bloque. 

El plan para 1 Zen 5 + 3 Zen 5c era:

$$
P_i
\propto
\text{rendimiento medido del núcleo }i
$$

porque el Zen 5 rápido debía recibir más filas que cada Zen 5c.

Eso no ocurrió. Los cuatro workers realizan el mismo trabajo, de manera que:

* el batch queda limitado por uno de los Zen 5c;
* el Zen 5 puede terminar antes;
* cae la presión media simultánea sobre DRAM;
* no se maximiza el rendimiento agregado.

Además, solo existen barreras al principio y al final de toda la medición. No hay barrera entre profundidades. 

En un núcleo realmente shardeado:

```text
proyección de cada shard
→ sincronización/reducción
→ actualización del estado
→ siguiente profundidad
```

Aquí cada worker ejecuta todas sus pasadas de forma independiente. El Zen 5 puede adelantarse varias iteraciones respecto a los Zen 5c.

Esto es una desviación real, aunque **no es suficiente para explicar por sí sola una diferencia de 1% frente a una expectativa de 2×**. El cuello principal sigue siendo el kernel compute/dequant-bound.

---

# 5. Tampoco se calculó la métrica acordada

El gate que habíamos definido no comparaba solamente:

$$
\frac{\text{throughput B}}{\text{throughput A}}
$$

Definía el coste marginal de cada profundidad adicional:

$$
c_A(K)=\frac{T_A(K)-T_A(1)}{K-1}
$$

$$
c_B(K)=\frac{T_B(K)-T_B(1)}{K-1}
$$

y luego:

$$
\rho_{\text{resident}}
=
\frac{c_A(K)}{c_B(K)}
$$

El sweep paralelo adjunto usa profundidades:

```text
4, 8, 16
```

y variantes:

```text
A, B
```

No incluye profundidad 1 ni C. 

Por ello no puede calcular la métrica incremental acordada.

No creo que restar la primera pasada convierta una diferencia de 1–3% en 2×, especialmente a profundidad 16. Pero formalmente **no se ejecutó el criterio exacto de aceptación**.

El script además calcula medias y desviaciones por variante, no diferencias pareadas ni intervalos de confianza. 

Y el orden se invierte por repetición global, no se randomiza de forma independiente para cada condición. 

Así que “9 de 9 en la misma dirección” es sugestivo, pero no equivale a nueve observaciones estadísticas independientes. Un sesgo global pequeño puede afectar todas las celdas en la misma dirección.

De todos modos, si el efecto real fuera 0%, 1% o 3%, la decisión práctica sobre este kernel sería la misma.

---

# 6. Qué partes sí estuvieron bien ejecutadas

No fue un experimento inútil. Varias decisiones fueron correctas:

* Se abandonó correctamente el kernel escalar al comprobar que no exponía memoria.
* Se verificó que el código generado contenía AVX2/FMA real.
* La variante B sí crea tantos bloques distintos como profundidad, mientras A crea uno solo. 
* Se utilizó afinidad nativa de Windows. 
* La topología física se obtuvo mediante `GetLogicalProcessorInformationEx`, no suponiendo que los índices pares fueran núcleos distintos. 
* Se corrigió correctamente el sesgo térmico más obvio del orden fijo.
* 512 KiB por worker es un buen punto para un L2 de 1 MiB.
* A profundidad 16, B ocupa aproximadamente:

$$
512\ \text{KiB}\times16\times4
=
32\ \text{MiB}
$$

muy por encima de los 8 MiB de L3. Así que no falta aumentar profundidad para conseguir que B deje de residir.

La disciplina experimental fue buena. El problema es que se terminó implementando correctamente **una pregunta distinta y más estrecha**.

---

# 7. ¿Aparecerá el efecto aumentando tamaño o profundidad con este mismo código?

**No hay una razón sólida para esperarlo.**

### Más profundidad

Una vez que B ya excede ampliamente L3, aumentar de 16 a 32 o 64 no aumenta los bytes por segundo exigidos:

$$
BW_{\text{demandado}}
=
\text{MAC/s}\times0.625
$$

Solo hace que el mismo régimen dure más tiempo.

### Más tamaño

El número de bytes y el número de MAC crecen proporcionalmente. La intensidad del kernel permanece igual.

Además:

* 512 KiB por núcleo es el punto arquitectónico relevante.
* 1024 KiB ya intenta ocupar toda la L2 nominal.
* 1280 KiB excede L2.
* tamaños mayores destruirían también la residencia de A.

### Menos tamaño

A ya cabe holgadamente a 512 KiB. Reducirlo puede mejorar ligeramente A, pero no hará que B demande más ancho de banda por segundo.

Por tanto:

> **No conviene gastar otra corrida barriendo solamente más tamaños o profundidades. Esa dimensión ya está agotada para este kernel.**

---

# 8. Qué cierra exactamente el resultado

| Hipótesis                                                  | Estado                                   |
| ---------------------------------------------------------- | ---------------------------------------- |
| Este kernel AVX2 Q4→FP32 obtiene ≥2× por residencia        | **Cerrada negativamente**                |
| Más profundidad o matrices mayores arreglarán este kernel  | **Muy improbable**                       |
| Los pesos A realmente permanecen en L2                     | No demostrado directamente               |
| B quedó limitado por DRAM                                  | **No; los números indican lo contrario** |
| L2 no ofrece ventaja útil sobre DRAM en este CPU           | No demostrado                            |
| Small-GEMM con slots mejora utilización y amortiza dequant | No probado                               |
| El núcleo recurrente matricial completo funciona           | No probado                               |
| Memoria externa + núcleo pequeño conserva calidad          | No probado                               |

La formulación correcta del resultado es:

> **El probe actual consume pesos a solo ~4.1 GB/s agregados, de modo que la diferencia entre pesos residentes y pesos procedentes de DRAM queda oculta por el coste de dequantización y ejecución. En este kernel concreto, la residencia aporta como máximo unos pocos puntos porcentuales.**

No debe formularse como:

> “El Ryzen AI 5 330 no obtiene beneficio de un núcleo residente.”

---

# 9. El único gate adicional que todavía está justificado

No hace falta entrenar nada ni construir la arquitectura. Hace falta una última prueba de hardware, pero **cambiando la operación**, no la escala.

## T0-R: residencia físicamente interpretable

Primero debe medirse el ancho de banda secuencial sostenido de DRAM con cuatro núcleos físicos.

Después, B debe alcanzar como mínimo:

$$
BW_{\text{pesos,B}}
\ge
0.5\text{–}0.7\times BW_{\text{DRAM medido}}
$$

antes de interpretar A/B.

Con el formato actual:

$$
BW_{\text{pesos,B}}
=
0.625\times\text{MAC/s}
$$

Si DRAM entrega 25 GB/s, el kernel necesita aproximadamente 40 GMAC/s. Si no puede acercarse a ese régimen, el resultado será siempre compute-bound.

Para ello se necesita una de estas dos rutas:

* microkernel Q4 realmente fusionado con activaciones cuantizadas y dot products enteros;
* kernel-oráculo Q8/int8 muy optimizado que demuestre primero que la jerarquía produce la diferencia esperada.

## T0-M: slots reales

Debe añadirse una dimensión independiente:

```text
slots ∈ {1, 4, 8, 16}
```

con:

$$
X\in\mathbb{R}^{512\times slots}
$$

La dequantización de cada tile de pesos debe realizarse una vez y reutilizarse para todos los slots.

Aquí la métrica principal no sería necesariamente A/B ≥2×, porque aumentar slots eleva la intensidad de B también. Deben medirse por separado:

* MAC/s;
* fracción del pico del microkernel;
* instrucciones de dequantización por MAC;
* tráfico DRAM A frente a B;
* latencia a FLOPs constantes.

## Configuración mínima corregida

```text
Workers:       4 físicos
Shards:        balanceados Zen5/Zen5c
Bytes/core:    384, 512, 640, 768 KiB
Depth:         1, 4, 8, 16
Slots:         1, 4, 8, 16
A:             pesos compartidos
B:             pesos distintos
C:             mismos pesos, expulsión fuera del cronómetro
Sincronización: barrera por profundidad
```

La línea se cerraría de verdad en este hardware si ocurre cualquiera de estas dos cosas:

1. B consume al menos 50–70% del ancho de banda DRAM medido y A sigue sin obtener una reducción material de coste o tráfico.
2. El small-GEMM con 8–16 slots no mejora claramente la eficiencia respecto a slot 1, por ejemplo ni siquiera 1.5× en MAC/s por el mismo camino de instrucciones.

---

## Decisión práctica

**No procedería todavía a entrenamiento ni destilación.**

**Tampoco repetiría el sweep actual con más tamaños o profundidades.**

Haría solamente el T0 corregido, porque el reporte actual contiene una señal clara:

> La ruta AVX2 implementada es demasiado lenta consumiendo pesos para poner a prueba la hipótesis de residencia.

Así que el resultado final es:

$$
\boxed{
\text{negativo para el kernel actual}
\;\neq\;
\text{negativo para el mecanismo}
}
$$

Y hay una conclusión adicional que sí debe conservarse: la promesa original de “10–40× por L2” ya no puede usarse como expectativa directa de velocidad end-to-end. Incluso si el gate corregido pasa, la ventaja deberá expresarse como una combinación de **menor tráfico DRAM, mejor utilización del núcleo y presupuesto liberado para memoria externa**, no como una aceleración automática de decenas de veces.

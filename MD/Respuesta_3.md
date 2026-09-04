## Diagnóstico principal

La respuesta más probable es:

$$
\boxed{\text{(c) problema de representación/transición primero}}
$$

$$
\boxed{\text{(a) escala y diversidad de datos posiblemente después}}
$$

$$
\boxed{\text{(b) grokking todavía no es la explicación principal}}
$$

No correría aún las otras cuatro semillas ni aumentaría otra vez de 50,000 pasos. Primero hay que resolver una contradicción importante en los resultados.

El código disponible en esta conversación corresponde a los probes C++ de T0 —dimensión, slots, recurrencia y kernels AVX2—, no al modelo PyTorch, generadores y optimizador de T1.  El probe recurrente adjunto también implementa la transición determinista de T0, no el reader diferenciable entrenado que describes.  Por tanto, puedo diagnosticar la lógica experimental con bastante precisión, pero no auditar todavía el source T1 línea por línea.

# 1. La matriz memorizada no demuestra que aprendió pointer chasing

Este resultado:

```text
H4 / R4 = 100%
H4 / R2 = 96.9%
```

es el dato más revelador.

Si realmente existe una sola lectura por ronda y hacen falta cuatro hops para resolver H4, entonces un modelo con `R=2` **no debería poder resolver H4** en ejemplos nuevos. Que consiga 96.9% en entrenamiento significa que está memorizando una función directa:

$$
\text{ejemplo completo}\rightarrow\text{respuesta final}
$$

No está obligado a calcular:

$$
x_1=f(x_0),\quad
x_2=f(x_1),\quad
x_3=f(x_2),\quad
x_4=f(x_3)
$$

Esto no invalida que el modelo tenga capacidad. Demuestra que tiene capacidad para memorizar 256 ejemplos. Pero **no demuestra que represente el algoritmo recurrente**.

La interpretación correcta es:

```text
100% train:
capacidad paramétrica confirmada.

~0% held-out:
algoritmo general no aprendido.

H4/R2 = 96.9% train:
existencia de un shortcut de memorización confirmada.
```

No asumiría que un Transformer pequeño necesariamente haría lo mismo. Hay que medirlo como baseline; no se puede dar por hecho.

# 2. La señal R4 > R2 > R1 es positiva, pero todavía no prueba “un hop por ronda”

El resultado:

```text
H4/R1 = 7.57%
H4/R2 = 12.55%
H4/R4 = 15.74%
```

sí muestra que añadir rondas tiene un efecto causal medible. Es la primera señal real de que el estado actualizado puede ayudar.

Pero una prueba estricta de pointer chasing requiere una frontera:

```text
R < H → aproximadamente azar
R ≥ H → salto grande
```

Un trabajo reciente sobre Transformers recurrentes enfatiza precisamente esto: con una máscara topológica estricta, un modelo con menos pasos que hops no posee físicamente la información necesaria; un resultado alto en esa zona demuestra que existe otra ruta o un shortcut. ([arXiv][1])

Por tanto, antes de interpretar 15.74% hay que registrar explícitamente:

```text
número de posibles respuestas;
accuracy de azar;
accuracy para R < H;
frecuencia de ciclos, self-loops y endpoints coincidentes;
cantidad exacta de llamadas al reader por ronda.
```

Si hay 16 respuestas, azar sería 6.25% y `R1=7.57%` estaría cerca del azar. Si hay 256 respuestas, azar sería 0.39% y hasta R1 tendría una señal inesperadamente grande. Sin ese dato no puede interpretarse limpiamente la matriz.

También deben eliminarse del generador:

* self-loops;
* ciclos de longitud menor o igual a \(H\);
* caminos que converjan accidentalmente en el mismo endpoint;
* correlaciones entre posición, key y respuesta.

Para el gate causal inicial conviene generar caminos disjuntos y acíclicos.

# 3. El techo de 15.74% parece acumulación de error por hop

Hay una explicación muy concreta que encaja con el resultado: **el reader aprende una lectura imperfecta y el error se multiplica en cada ronda**.

Si, como aproximación, cada hop tuviera la misma probabilidad independiente \(p\) de mantener el puntero correcto:

$$
p^4=0.1574
$$

entonces:

$$
p\approx0.63
$$

Es decir, un operador que acierta alrededor del 63% por paso produciría aproximadamente el 15.7% después de cuatro pasos.

No afirmo que los errores sean realmente independientes, pero la coincidencia indica dónde mirar:

> No midan únicamente la respuesta final. Midan si el puntero interno es correcto después de cada ronda.

Para cada ronda \(r\), registrar:

$$
\operatorname{AccPointer}(r)
=
P\left[
\arg\max q_r
=
x_r^{\text{verdadero}}
\right]
$$

Además:

```text
probabilidad asignada al key correcto;
entropía del reader;
margen entre key correcto y segundo key;
coseno entre query y embedding del key correcto;
masa retenida sobre el puntero anterior;
norma del delta por ronda.
```

La curva puede revelar inmediatamente:

### Caso 1

```text
r1 = 65%
r2 = 42%
r3 = 26%
r4 = 16%
```

Entonces el problema es casi enteramente **error compuesto del reader**.

### Caso 2

```text
r1 = 98%
r2 = 95%
r3 = 90%
r4 = 85%
pero final = 16%
```

Entonces el error está en el readout/evaluación.

### Caso 3

```text
r1 ya está en 60–70%
```

Entonces el problema está antes de la recurrencia: keys, query, temperatura o codificación.

# 4. El residual fijo probablemente no representa correctamente una actualización de puntero

El pre-norm con:

$$
h_{r+1}=h_r+\alpha F(h_r)
$$

fue una buena corrección para demostrar que había gradiente. Pero un puntero discreto no debería necesariamente actualizarse acumulando el viejo y el nuevo.

La operación deseada es:

$$
p_{r+1}=\operatorname{Read}(M,p_r)
$$

No:

$$
p_{r+1}=p_r+\alpha\,\operatorname{Read}(M,p_r)
$$

La segunda conserva parte del key anterior. Después de varias rondas puede producir una mezcla:

$$
p_r
\approx
c_0e_{x_0}
+
c_1e_{x_1}
+
\cdots+
c_re_{x_r}
$$

El reader de la ronda siguiente recibe una superposición de varios keys y puede recuperar una mezcla de valores. Esto explicaría perfectamente:

* progreso monotónico;
* techo bajo;
* error creciente con hops;
* memorización perfecta de ejemplos fijos;
* mala generalización.

## Transición correcta para el slot de puntero

Para el primer experimento:

$$
\tilde p_{r+1}
=
\operatorname{Reader}(p_r,M)
$$

$$
p_{r+1}
=
\operatorname{Canonicalize}(\tilde p_{r+1})
$$

El pointer slot debe **reemplazarse**, no residualizarse.

Otros slots sí pueden usar residuales:

$$
w_{r+1}
=
w_r+
F_w(w_r,p_r,\tilde p_{r+1})
$$

Así se separan dos funciones:

```text
pointer slot:
estado discreto actual; reemplazo.

workspace slots:
evidencia y computación continua; residual.
```

# 5. \(\alpha=1/\sqrt R\) introduce una inconsistencia entre modelos

Este detalle también debe corregirse.

Actualmente:

```text
R=1 → α=1.0
R=2 → α≈0.707
R=4 → α=0.5
```

Por tanto, una “ronda” no representa la misma operación en los modelos R1, R2 y R4.

Pero para pointer chasing se necesita un operador homogéneo:

$$
p_{r+1}=f(p_r)
$$

con la misma semántica independientemente del número total de rondas que se ejecutarán.

Usar \(1/\sqrt R\) puede ser útil para estabilizar residuales profundos, pero aquí mezcla la estabilidad numérica con la definición del algoritmo. Para el pointer slot usaría:

```text
reemplazo completo;
o gate fijo z=1;
o z constante independiente de R.
```

Si el workspace residual necesita escalado, que ese escalado sea fijo y no dependa del número de hops solicitado.

# 6. El codebook de keys puede ser el cuello de botella

Si existen 256 keys aprendidos dentro de un espacio de dimensión 64:

$$
K\in\mathbb R^{256\times64}
$$

el reader debe distinguir 256 direcciones con margen suficiente y, además, reconstruir una query que vuelva a caer cerca de una de esas direcciones después de cada hop.

No es imposible, pero es un escenario propenso a:

* queries fuera del manifold de keys;
* atención difusa;
* interferencia entre keys;
* codebook memorizado;
* degradación acumulativa.

Para separar el algoritmo de la geometría del embedding, haría el gate en tres escalones.

## Escalón 1: estado one-hot y reader exacto

$$
p_r\in\{0,1\}^{N}
$$

$$
p_{r+1}=p_rA
$$

donde \(A\) es la matriz de la función de punteros.

Este sistema debe alcanzar:

```text
R ≥ H → 100%
R < H → azar
```

sin entrenamiento. Si no lo hace, el generador o evaluador sigue incorrecto.

## Escalón 2: reader diferenciable, codebook fijo

Usar keys aleatorios normalizados, regenerados o permutados por ejemplo. No una tabla aprendida permanente asociada a identidades globales.

## Escalón 3: keys aprendidos

Solo después de demostrar que el algoritmo funciona con una geometría controlada.

# 7. Hay que imponer equivarianza a la identidad de las keys

Este puede ser el principal origen de memorización.

Si `key_17` siempre utiliza el mismo embedding entrenable, el modelo puede aprender regularidades específicas de:

```text
key_17
key_42
key_103
```

en vez de aprender la regla abstracta:

```text
buscar valor asociado a la key actual
```

La corrección más limpia es aplicar una permutación aleatoria de labels en cada ejemplo:

$$
\pi:\{1,\ldots,N\}\rightarrow\{1,\ldots,N\}
$$

El mismo grafo abstracto debe representarse con labels diferentes en cada presentación.

Todavía mejor: generar nuevos vectores de key por ejemplo. Así el modelo nunca puede memorizar que una dirección concreta significa una entidad concreta.

Un algoritmo auténtico debe ser equivariante:

$$
f(\pi(G),\pi(x))
=
\pi(f(G,x))
$$

Si el modelo pierde toda señal al randomizar los labels, estaba aprendiendo identidades, no pointer chasing.

# 8. El experimento de 256 ejemplos no debe descartarse como “esperable”

La conclusión:

> “Un Transformer probablemente memorizaría igual.”

es plausible, pero no está demostrada.

El resultado de 256 ejemplos proporciona dos datos distintos:

1. La red tiene capacidad suficiente para ajustar completamente el conjunto.
2. Su solución preferida no generaliza.

Eso es información útil. No prueba que la arquitectura falle, pero tampoco puede neutralizarse suponiendo que cualquier arquitectura fallaría.

Hay que ejecutar un baseline diminuto bajo exactamente las mismas restricciones:

```text
GRU o RNN recurrente;
Transformer pequeño;
mismo reader;
misma máscara;
mismo número de parámetros;
mismo dataset;
mismo presupuesto de optimización.
```

El baseline más importante no es un Transformer libre con atención global. Debe tener la misma barrera informativa de una lectura por paso.

# 9. No apostaría todavía por grokking

Grokking es una posibilidad real, pero necesita condiciones más específicas que “train loss cero + muchas iteraciones”.

Los experimentos clásicos muestran que:

* la generalización puede aparecer mucho después de la memorización;
* el tiempo necesario crece al reducirse la fracción de datos;
* weight decay puede favorecer la solución generalizable;
* existen regímenes de datos por debajo de los cuales la transición puede no aparecer de forma observable. ([arXiv][2])

Pero el grokking clásico ocurre sobre un conjunto finito y una estructura algebraica estable. Antes de intentarlo aquí hay que responder:

```text
¿Los 10k ejemplos son fijos o se regeneran online?
¿Cuántas épocas representan 50k steps?
¿Cuál es el batch size?
¿Qué parámetros reciben weight decay?
¿Los embeddings de key reciben weight decay?
```

Si los ejemplos se generan online continuamente, “grokking del conjunto memorizado” deja de ser la descripción apropiada. Si los 10k son fijos, sí se puede estudiar, pero no conviene usar grokking como explicación por defecto cuando todavía existen indicios de un operador de un paso imperfecto y una representación no equivariante.

Además, aplicar weight decay agresivo a los embeddings de keys puede reducir los márgenes del reader. Separaría los grupos:

```text
núcleo:
weight decay experimental.

bias, normas y gates:
0.

codebook:
preferiblemente fijo durante el diagnóstico.
```

# 10. Campaña mínima antes de escalar datos

No ejecutar las cuatro semillas restantes. Ejecutar una sola semilla con esta secuencia:

## P0 — Oracle del benchmark

```text
N = 16 o 32
grafo acíclico
sin self-loops
sin ciclos ≤ Hmax
keys one-hot
reader exacto
```

Debe obtenerse:

```text
R < H → azar
R ≥ H → 100%
```

Si falla, sigue existiendo un problema en generador, máscara o evaluación.

## P1 — Exactitud por ronda

Mantener el modelo actual y medir:

```text
pointer_acc_r1
pointer_acc_r2
pointer_acc_r3
pointer_acc_r4
reader_entropy_r
reader_margin_r
old_pointer_mass_r
```

Esto decide si el 15.74% es error multiplicativo.

## P2 — Reemplazo del pointer slot

Cambiar únicamente:

```text
pointer_next = reader_output
```

Sin residual, sin \(\alpha=1/\sqrt R\), sin gate aprendido.

Los otros slots pueden conservar el residual pre-norm.

## P3 — Codebook permutado

Reetiquetar aleatoriamente las keys en cada muestra. Mantener exactamente el mismo problema abstracto.

## P4 — One-hop primero

Entrenar exclusivamente H1.

Criterio:

```text
held-out H1 ≥ 98–99%
```

Si ni siquiera H1 generaliza, no tiene sentido probar H4. El problema está en el reader o la representación.

Luego congelar el operador y ejecutar varias veces:

```text
H2
H3
H4
```

Si H1=99% pero H4 cae mucho, el problema es estabilidad composicional/canonicalization.

## P5 — Supervisión intermedia diagnóstica

Añadir temporalmente:

$$
\mathcal L
=
\mathcal L_{\text{final}}
+
\lambda
\sum_{r=1}^{R}
\operatorname{CE}(\hat p_r,p_r^*)
$$

Esto no será todavía la prueba final. Sirve para localizar el problema:

* Si con targets intermedios generaliza, había un problema de asignación de crédito.
* Si tampoco generaliza, el problema es representación/reader.
* Si generaliza con teacher forcing pero falla en free-running, hay exposición y acumulación de errores.

# Árbol de decisión

| Resultado                                 | Diagnóstico                 |
| ----------------------------------------- | --------------------------- |
| Oracle no consigue 100%                   | Bug restante en benchmark   |
| Oracle pasa, H1 aprendido falla           | Reader/codebook             |
| H1 ≥99%, H4 ≈ \(p^4\)                     | Error acumulativo           |
| Teacher forcing pasa, free-running falla  | Estado fuera del manifold   |
| Auxiliar por ronda pasa, final-only falla | Asignación de crédito       |
| Permutación de keys destruye accuracy     | Memorización de identidades |
| Baseline restringido también falla        | Régimen de datos/benchmark  |
| Todo lo anterior pasa, 10k sigue limitado | Escalar datos/capacidad     |

# Respuesta concreta: ¿a, b o c?

### (a) Más datos o currículo

**Probablemente ayudarán**, pero todavía no son el siguiente paso. La regla local de lookup es simple; si el sistema necesita enormes cantidades de ejemplos para aprenderla, hay que revisar primero su inductive bias.

Un currículo sí tiene sentido después:

```text
H1 hasta generalizar
→ mezcla H1/H2
→ mezcla H1–H4
→ H5/H6 OOD
```

### (b) Grokking con weight decay y entrenamiento largo

**Posible, pero especulativo.** Haría un sweep pequeño solo después de que el operador one-hop y la barrera informativa estén confirmados. No dedicaría otras 500,000 iteraciones a esperar una transición incierta.

### (c) Arquitectura/parametrización

**Es actualmente la explicación principal.**

En concreto:

1. El pointer slot se actualiza como residual cuando debería reemplazarse.
2. \(\alpha\) depende de \(R\), aunque cada ronda debería representar el mismo hop.
3. El reader probablemente genera estados fuera del manifold de keys.
4. El codebook aprendido permite memorizar identidades.
5. No se ha mostrado aún la frontera estricta \(R<H\) contra \(R\ge H\).
6. H4/R2=96.9% en entrenamiento confirma que el ajuste perfecto se produjo mediante shortcut.

## Decisión

$$
\boxed{\text{No escalar todavía datos, steps ni semillas}}
$$

La siguiente prueba debe ser:

$$
\boxed{
\text{oracle exacto}
\rightarrow
\text{accuracy por ronda}
\rightarrow
\text{pointer replacement}
\rightarrow
\text{keys permutadas}
}
$$

La señal causal de 15.74% es valiosa: demuestra que la idea no está muerta. Pero también indica que la red todavía no aprendió un operador iterativo limpio. El objetivo siguiente no es subir H4 de 15% a 20%; es producir la firma algorítmica correcta:

```text
R < H → azar
R ≥ H → salto abrupto y generalización.
```

Hasta observar esa frontera, más datos podrían simplemente permitir una memorización más extensa.

[1]: https://arxiv.org/html/2603.21676v1?utm_source=chatgpt.com "Depth-Recurrent Transformers for Compositional ..."
[2]: https://arxiv.org/pdf/2201.02177?utm_source=chatgpt.com "grokking: generalization beyond overfit"

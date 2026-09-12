He clonado el repo, leído `Conversacion.md`, los `MD/50–64`, y como no pude instalar torch reimplementé el Writer R3 en numpy y cargué `t2_i2_r3_seed6301/final.pt` a mano. Lo que sigue sale de correr el checkpoint, no de leer los MD.

## Tres hechos medidos sobre el Writer

**1. Separabilidad aditiva exacta.** La ventana local `[-1,0,+1]` nunca cruza las dos cláusulas. Verificado a cero de máquina:

```
E_s(L,F) = f_s(L) + a_s(F)      max|dif| = 0.000e+00
```

`f` sale de los tokens {AT_LEAST, VALUE_L, AND}, `a` de {AVOID, VALUE_F}. No hay un solo término de interacción en el Writer. Ese es el resultado más fuerte de toda la campaña y no está escrito en ningún MD: explica por qué `Z` era necesario y por qué `Z` funcionó, y acota exactamente qué puede y qué no puede arreglar.

**2. El router es de rango 1.** Las queries de slot son casi exactamente antiparalelas:

```
cos(q0, q1) = -0.9914
valores singulares de Q: [1.812, 0.173, 0.110]
corr(score_E0, -score_E1) = 0.99972
```

El "competitive routing" de 3 vías colapsó a **un escalar**. El Writer no puede decidir "esto es FLOOR" y "esto es AVOID" por separado; solo puede decidir *cuánto* a lo largo de un eje. G3 le exige dos lecturas factorizadas y la arquitectura tiene un grado de libertad.

**3. E0 no es una evidencia: es el marcador de conjunción.** Routing medio por token sobre los 992 pares:

```
token     | p(E0)   p(E1)   p(NULL)
AT_LEAST  | 0.054   0.791   0.155
VALUE_L   | 0.086   0.724   0.190
AND       | 0.485   0.159   0.356   ← único token que va a E0
AVOID     | 0.062   0.777   0.161
VALUE_F   | 0.057   0.788   0.155
```

`AND` aporta el **53% de la norma de E0**. Y `|E0|/|E1| = 0.089` de media, **máximo 0.143 sobre los 992 pares**. Descomposición de varianza: E0 es **93% L / 7% F**; E1 es 47.5% L / 50.7% F.

## Qué implica para el residuo de 22

La señal de F dentro de E0 tiene norma 0.107 sobre un vector de norma 1.21: **8.8%**. La prueba `AVOID(E0)` se decide con eso.

Y los 22 fallos son exactamente `F=10, L ∈ {3} ∪ {11..31}` — o sea **L > F más un outlier**. Probé la hipótesis geométrica obvia (que la contaminación `f0(L)` empuja el readout al lado malo) proyectando `f0(L)` sobre las direcciones discriminativas de `a0(·)`: los t de Welch entre el grupo que falla y el que pasa salen **+0.02, +0.31, +1.12, −0.79**. No hay separación. La causa no está en la geometría de E0 respecto de L.

Mi lectura: **F=10 no tiene nada de especial.** E0 apenas depende de F (7% de su varianza), así que si `AVOID(E0)` va a fallar, va a fallar en bloque para *algún* valor de F, y el que caiga del lado malo de la frontera de CTRL-7 se lleva todos sus L. La firma "22/22 comparten forbidden=10" no es una pista sobre el token 10; es la consecuencia aritmética de que solo hay ~1 bit de dependencia en F disponible en ese slot.

**El diagnóstico que yo correría en vez de `T2-I3-R3-F10-BIND-ALG`:** reentrenar el Writer con seeds 6302/6303 y mirar dónde cae el residuo. Si aparece ~2-3% concentrado en *otro* forbidden, F=10 queda descartado como fenómeno y VIEW-0 queda descartado como remedio — estarías construyendo un módulo para compensar un accidente de inicialización. Es más barato que la autopsia de routing propuesta en MD/64 y decide la rama de forma binaria.

## Sobre el consumidor, que cambia el marco

`LatentConditionedSupervisor.forward` es:

```python
output_projection(F.silu(observation_projection(observation) + condition))
```

La condición es **un sesgo aditivo de 32 dims**. Y `supervisor_features_ctrl7` recibe `lower` y `forbidden` como enteros explícitos y calcula `floor_comparison`/`avoid_comparison` desde ellos. Los valores no viajan por el workspace. El único trabajo de la condición es seleccionar cuál de 4 semánticas aplicar: NONE / FLOOR / AVOID / FLOOR_AND_AVOID.

**El canal es de 2 bits.** Eso reencuadra las cuatro preguntas: toda la campaña T2-I3 está peleando por hacer que un espacio de 2×32 dims transmita 2 bits de forma factorizable, y el Writer ya lo hizo con un truco (un slot = "hay dos cláusulas", el otro = todo lo demás) que satisface G1/G2 y el `any(identity,swap)` de G3 sin ser en absoluto la descomposición que el gate pretende medir.

## Respuestas directas

**1) ¿Evidencia inmutable + registro de composición?** Sí, y por una razón más fuerte que la del MD/58. No es solo que K0=129 fuera el mejor punto de la trayectoria: es que `E = f(L) + a(F)` **exactamente**, sin término cruzado, así que un compositor separado es la única pieza que puede introducir interacción. Los 48 arreglados sin regresión no son suerte — es el módulo aportando la estructura que el Writer es estructuralmente incapaz de tener.

La alternativa no considerada: nada de esto exige dos slots. Con un router de rango 1 y E0 al 9% de la norma, el sistema ya es de un slot con un residuo. Yo pondría en el ledger un control de una línea — CTRL-7 con condición `= E1` sola, y con `= E0+E1`. Si E1 sola da ~777, el "workspace de 2 slots" está desmentido como mecanismo y todo el trabajo se lo está llevando `Z`.

**2) Ver arriba.** Router rank-1, E0 = marcador AND, 8.8% de señal en F. La rama R3 de MD/64 ("la información AVOID10 se pierde en joint") queda descartada por la separabilidad exacta: `a_s(10)` es idéntica en atómico y en joint, por construcción. Eso ahorra ese brazo del diagnóstico.

**3) El riesgo metodológico que no veo contemplado.** La disciplina sobre G4/G5 es mejor que la de casi cualquier paper. Pero el problema no es G4: es que **el dominio entero son 992 pares y ya se observó**. G5 no es un split independiente, es *otra vista del mismo dominio consumido*. Un Writer nuevo (otra seed) no restaura independencia si se eligió reentrenar porque F=10 falló. La única independencia real que queda es cambiar la tarea o el vocabulario. Segundo riesgo: `α` se eligió sobre 139 pares y se validó sobre 853; ahora la ventana `.25–.5` es *development choice* sobre el dominio completo. Está registrado honestamente en MD/62-63, pero conviene escribir explícitamente que cualquier número futuro sobre esos 992 es entrenamiento, no evidencia.

**4) La lectura escéptica.** OMEGA afirma que núcleo pequeño + memoria externa sustituye capacidad. Esta campaña no la toca. Aquí la "memoria" son 64 floats, el "núcleo" tiene 1.072 parámetros, y el canal de información son 2 bits con los valores entrando por fuera. Lo que se validó es real y vale: escritura aditiva ≫ pooling normalizado, y composición separada ≫ reescritura de evidencia. Son dos hallazgos arquitectónicos limpios y transferibles. Pero no son evidencia sobre capacidad, porque no hay capacidad en juego. La lectura escéptica es que se lleva mucho ciclo de diagnóstico sobre 2.6% de un benchmark de 2 bits mientras el gate T0 —¿el bloque recurrente se queda en L2 y la residencia compra algo?— sigue sin correr, y ese sí mata o salva la tesis.

Y hay una señal de advertencia en los propios datos: `identity=286` vs `swap=777`. Un gate que se pasa 91% por una orientación y 34% por la otra no está midiendo "los slots están factorizados", está midiendo "uno de los slots está casi vacío y por eso pasa una de las patas". El 831/853 es un número más frágil de lo que parece.
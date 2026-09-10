## Lectura del resultado

Esto sí es una **prueba decisiva de computación recurrente causal** para el caso de pointer-chasing.

La secuencia de resultados identifica la causa anterior con bastante limpieza:

$$
\text{pointer residual}
\;\Rightarrow\;
\text{mezcla entre puntero viejo y nuevo}
\;\Rightarrow\;
\text{query sale del manifold de keys}
\;\Rightarrow\;
\text{error acumulativo}
$$

Al cambiar a:

$$
p_{r+1}=\operatorname{Read}(M,p_r)
$$

sin residual, sin \(\alpha\) y sin gate sobre el pointer slot, aparece:

$$
R<H \Rightarrow 0\%
$$

$$
R\geq H \Rightarrow 100\%
$$

sobre mappings nuevos. Eso demuestra simultáneamente que:

* cada ronda transmite información necesaria;
* las rondas no son trabajo decorativo;
* el núcleo compartido puede ejecutar repetidamente la misma operación;
* el estado intermedio permanece direccionable;
* el reader no acumula deriva;
* el modelo generaliza a grafos no vistos;
* el techo anterior de 15–18% no era falta de datos, steps ni grokking;
* el residual aplicado a un estado discreto era la causa dominante.

El hecho de converger aproximadamente en el step 1000 también prácticamente elimina, para esta tarea concreta, las hipótesis anteriores:

```text
“faltaban más de 10k ejemplos”
“faltaban 50k+ steps”
“había que esperar grokking”
```

No faltaba escala. Faltaba la **semántica correcta de actualización del estado**.

## Una precisión: falta una celda para llamar completa a la frontera

La matriz es exacta sobre las profundidades evaluadas:

```text
R ∈ {1,2,4}
```

Pero para afirmar literalmente:

$$
R\geq H
$$

falta verificar:

```text
H3/R3 = 100%
H4/R3 = 0%
```

No hace falta entrenar otro modelo necesariamente. Como los pesos son compartidos, el checkpoint `R4` puede detenerse después de la tercera ronda y decodificarse allí.

Además, ya dicen que `pointer_acc=100%` en las cuatro rondas. Si eso significa que en la ronda 3 el pointer slot coincide exactamente con el tercer nodo, entonces la propiedad está prácticamente demostrada internamente; aun así conviene producir las dos celdas explícitas en el reporte.

También debe registrarse por qué las rondas adicionales no hacen overshoot:

```text
H1/R4 = 100%
H2/R4 = 100%
```

Eso requiere que ocurra una de estas cosas:

* existe una máscara de rondas activas;
* el modelo recibe un token de stop;
* el estado entra en una transición identidad;
* el readout usa el estado de la ronda H.

Cualquiera puede ser válida. El único requisito es comprobar que la máscara o el stop dependen del presupuesto declarado, **no de la respuesta target**.

# 1. ¿Los keys y mappings nuevos ya cubren P3?

**Cubren la parte principal de P3, pero haría todavía una prueba metamórfica explícita, sin entrenamiento.**

Hay dos escenarios.

### Escenario fuerte

Cada ejemplo genera:

* vectores de keys nuevos;
* mapping nuevo;
* orden nuevo;
* query nueva;
* sin embeddings globales asociados permanentemente a `key_0`, `key_1`, etc.

En ese caso, P3 ya está cubierto de una forma incluso más fuerte que una simple permutación. No existe una identidad fija que el modelo pueda memorizar.

### Escenario intermedio

El mapping es nuevo, pero las keys siguen siendo IDs tomados de un vocabulario fijo:

```text
key_0 ... key_15
```

con embeddings globalmente aprendidos.

Eso impide memorizar un mapping único, pero todavía permite:

* sesgos por identidad;
* geometría privilegiada de algunas keys;
* dependencia del orden de serialización;
* correlaciones entre ID y posición.

No explicaría fácilmente el 100% sobre mappings arbitrarios, pero P3 no estaría formalmente cerrado.

## Prueba final de P3

No hace falta reentrenar. Sobre el checkpoint actual y los mismos 2000 ejemplos:

1. Muestrear una permutación aleatoria \(\pi\).
2. Aplicarla consistentemente a:

   * todas las keys;
   * todos los values;
   * el punto inicial;
   * los estados intermedios;
   * el target.
3. Barajar también el orden físico de las filas de memoria.
4. Evaluar la predicción transformada.

Debe cumplirse:

$$
f(\pi(M),\pi(x))
=
\pi(f(M,x))
$$

Y después de barajar filas:

$$
f(\operatorname{shuffle}(M),x)
=
f(M,x)
$$

Haría al menos 10 permutaciones por ejemplo o, para hacerlo más ligero, 10 versiones completas del test set.

Criterio:

```text
Accuracy original:       100%
Accuracy relabel:        100%
Accuracy row-shuffled:   100%
R<H leakage:               0%
```

Esto es una auditoría de inferencia barata, no una nueva campaña P3.

También conviene comprobar:

```text
intersección exacta train/test = 0
```

usando un hash canónico de:

```text
mapping + start + H
```

Y generar una evaluación online adicional con mappings nuevos después de congelar el checkpoint. Con 10,000 o 100,000 ejemplos, la inferencia debería ser barata y daría un intervalo de confianza mucho más estrecho que 2000 ejemplos.

## Respuesta a la primera pregunta

$$
\boxed{
\text{P3 está sustancialmente cubierto}
}
$$

pero lo cerraría formalmente con una **permutación y un shuffle de filas en inferencia**, no con más entrenamiento.

# 2. ¿Aplicar reemplazo a las tareas originales o correr primero las otras semillas?

**Primero correría las cuatro semillas restantes del P2 corregido. Después volvería a las tareas originales, pero no aplicando reemplazo indiscriminadamente.**

La razón no es que dude de la capacidad: la capacidad está demostrada. Lo que falta medir es la **fiabilidad de optimización**.

Un resultado perfecto en una semilla demuestra existencia:

$$
\exists\theta:
\text{la arquitectura aprende el algoritmo}
$$

Cinco semillas responden otra pregunta:

$$
P(\text{entrenamiento converge})
$$

Esa probabilidad importa antes de atribuir futuros fallos de associative recall o variable binding a su diseño.

La campaña sería pequeña porque ya sabemos:

* que converge alrededor de 1000 steps;
* que no necesita 50,000;
* qué métricas observar;
* qué arquitectura usar.

## Corrida de replicación

Mantener exactamente:

```text
Seeds: 101, 202, 303, 404, 505
N: 16
D: 64
pointer update: replacement
workspace update: pre-norm residual
reader: una lectura por ronda
sin gate aprendido en pointer
sin early stopping prematuro
```

No utilizar el test para elegir checkpoint. Si `test.jsonl` fue consultado repetidamente para identificar el step 1000, debe renombrarse como validación y producir un **test sellado nuevo**.

Criterio de reproducción:

```text
4/5 semillas como mínimo:
- 100% para R≥H
- 0% para R<H
- pointer_acc=100% por ronda
- reader_margin positivo y estable
- sin crecimiento de old_pointer_mass
```

Por ser una tarea tan controlada, esperaría 5/5. Una semilla que no converja no mata el mecanismo, pero revelaría sensibilidad de inicialización u optimización que debe conocerse antes de escalar.

La prueba P3 metamórfica puede ejecutarse sobre cada checkpoint final.

# La lección no es “todo se reemplaza”

La conclusión arquitectónica es más importante:

> **Cada clase de estado necesita una regla de escritura compatible con su semántica.**

No existe una transición universal `estado + delta` adecuada para todo.

| Tipo de estado               | Actualización adecuada                            |
| ---------------------------- | ------------------------------------------------- |
| Puntero o identidad discreta | Reemplazo/canonicalización                        |
| Registro de variable         | Sobrescritura con el nuevo valor                  |
| Acumulador                   | Suma residual                                     |
| Workspace continuo           | Residual pre-norm                                 |
| Evidencia recuperada         | Escritura o reemplazo de slot                     |
| Memoria episódica            | Append/keyed write                                |
| Control/halting              | Gate dependiente del contenido                    |
| Binding key→value            | Escritura estructurada, no mezcla vectorial ciega |

Esta separación probablemente sea una propiedad central de la arquitectura, no un parche del benchmark.

## Aplicación a las tareas originales

### Multi-hop

Debe conservar exactamente el mecanismo que acaba de pasar:

```text
current-pointer slot:
    reemplazo completo en cada hop

evidence/workspace slots:
    actualización residual
```

No regresaría al residual sobre el puntero.

### Variable binding

Separaría:

```text
entity/reference slot:
    reemplazo al resolver una referencia

value slot:
    overwrite con el valor recuperado

workspace:
    residual para combinar restricciones
```

Ejemplo:

```text
x → object_4
object_4 → blue
```

La identidad `object_4` debe sustituir a `x` en el reference slot. No debe sumarse vectorialmente a `x`.

### Associative recall

La memoria de pares permanece estática durante la consulta:

```text
query slot:
    contiene la key

retrieved-value slot:
    reemplazo con el value leído

workspace:
    residual opcional
```

Aquí una sola lectura debería resolver una asociación directa. Varias rondas solo deben ser necesarias para asociaciones anidadas o encadenadas.

### Sequential update

El registro principal debe representar el valor actual:

$$
v_{r+1}=\operatorname{Op}_r(v_r)
$$

Aunque una operación sea `+4`, la semántica general es reemplazar el registro por el resultado. No conservar una mezcla arbitraria entre las representaciones de todos los estados anteriores.

Puede existir un accumulator residual separado cuando la tarea lo requiera.

### Length generalization

Depende del tipo de problema:

* autómata o pointer traversal: reemplazo;
* suma/acumulación: residual;
* plan o resumen continuo: residual pre-norm;
* contador: transición estructurada.

Por eso no restauraría las tareas originales con una sola política de actualización compartida.

# Nueva formulación del núcleo

La arquitectura debería dejar de considerar los \(S\) slots homogéneos. Como mínimo:

$$
X=
[
P,\;
R,\;
E_1,\ldots,E_k,\;
W_1,\ldots,W_j
]
$$

donde:

* \(P\): pointer/reference slot;
* \(R\): register/result slot;
* \(E\): evidence slots;
* \(W\): continuous workspace.

Una ronda produciría candidatos:

$$
\tilde P_{r+1}
=
\operatorname{Read}(M,P_r)
$$

$$
\tilde R_{r+1}
=
F_R(R_r,P_r,E_r)
$$

$$
\Delta W_r
=
F_W(W_r,P_r,R_r,E_r)
$$

y aplicaría transiciones diferentes:

$$
P_{r+1}=\operatorname{Canonicalize}(\tilde P_{r+1})
$$

$$
R_{r+1}=\tilde R_{r+1}
$$

$$
W_{r+1}
=
W_r+\alpha F_W(\operatorname{Norm}(W_r),\ldots)
$$

Este diseño es más heterogéneo que un Transformer, pero coincide precisamente con la hipótesis CPU-native: **un pequeño runtime cognitivo con estados de tipos diferentes**, no un único tensor tratado siempre con la misma fórmula.

# Secuencia recomendada

La siguiente secuencia queda acotada:

```text
1. Añadir R=3 al mismo checkpoint.
2. Ejecutar permutación de keys y shuffle de memoria, sin entrenar.
3. Ejecutar las cuatro semillas restantes de P2.
4. Congelar P2 como test unitario permanente.
5. Migrar multi-hop y associative-recall a slots tipados.
6. Probar una semilla de cada tarea.
7. Solo cuando haya causalidad, ejecutar cinco semillas.
```

No volvería todavía a campañas largas ni a grokking.

## Ledger actualizado de T1

```text
T1-A, grafo y optimizer:
bugs corregidos.

T1-B, residual genérico:
rechazado para estados discretos.

T1-C, pointer replacement:
PASS fuerte.

Generalización a mappings nuevos:
PASS.

Frontera causal sobre R={1,2,4}:
PASS.

R=3:
pendiente como celda de cierre.

P3 identidad:
sustancialmente cubierto;
pendiente auditoría metamórfica barata.

Robustez entre semillas:
pendiente.

Slots tipados en tareas generales:
siguiente gate.
```

La conclusión más importante es:

$$
\boxed{
\text{el núcleo compartido sí puede aprender una operación iterativa generalizable}
}
$$

Y la lección que debe pasar al diseño general es:

$$
\boxed{
\text{los estados discretos se reemplazan;
los estados continuos se refinan residualmente}
}
$$

Eso no garantiza todavía lenguaje o coding, pero elimina la objeción más fundamental contra T1: las rondas compartidas **sí pueden representar computación causal real** cuando la semántica de actualización del estado es correcta.

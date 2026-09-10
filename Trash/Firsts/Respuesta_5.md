## Veredicto

**No corran más semillas ni simplemente más steps todavía.** El fallo quedó localizado antes de la recurrencia:

$$
\boxed{\text{el operador elemental de un paso no está aprendido}}
$$

Las cuatro tareas exitosas demuestran que la arquitectura ya sabe:

* recuperar una identidad;
* reemplazar un puntero;
* copiar un valor recuperado;
* encadenar esas operaciones usando pesos compartidos.

`sequential-update` exige algo nuevo: **sintetizar un valor que no estaba almacenado explícitamente**.

```text
Lookup:
key → value existente

Sequential update:
(registro, operando, operación) → valor nuevo
```

Por eso no lo clasificaría como “la quinta tarea falló”. Abriría un gate distinto:

```text
T1-R: retrieval y reemplazo tipado       PASS_STRONG
T1-ALU: transformación algorítmica       FAIL, causa por localizar
```

El indicio más fuerte es `SUB≈9–11%`, mucho peor que `ADD≈27–28%`. Suma y resta modular deberían ser de dificultad comparable cuando los roles de ambos operandos están bien representados. Esa asimetría apunta primero a **pérdida del orden de operandos, orientación invertida de SUB o interferencia del readout**, no a falta general de capacidad.

No tengo adjunto el código PyTorch de T1, por lo que no puedo certificar cuál de esas posibilidades aparece en el source. Pero el diagnóstico se puede resolver con pruebas muy pequeñas.

# 1. Primer sospechoso: los operandos podrían estar fusionándose simétricamente

Hay que revisar cómo se construye la entrada del MLP.

Si hacen algo equivalente a:

$$
u=E(v)+E(a)+E_{\text{op}}(o)
$$

donde \(v\) es el registro y \(a\) el operando, entonces:

$$
u(v,a,o)=u(a,v,o)
$$

Eso no afecta a:

$$
v+a=a+v
$$

ni a:

$$
v\cdot a=a\cdot v
$$

pero destruye la resta, porque generalmente:

$$
v-a\neq a-v\pmod {32}
$$

Incluso esto sigue siendo simétrico:

$$
(E(v)+r_{\text{left}})
+
(E(a)+r_{\text{right}})
$$

Al intercambiar \(v\) y \(a\), la suma final contiene exactamente los mismos cuatro términos. **Agregar role embeddings antes de sumar no conserva los roles.**

Hace falta una operación no simétrica:

$$
u=
\operatorname{concat}
\left(
W_L E(v),
W_R E(a),
E_{\text{op}}(o)
\right)
$$

con:

$$
W_L\neq W_R
$$

o mantener dos slots separados hasta entrar al operador.

## Prueba sin entrenamiento

Para los \(1024\) pares de SUB:

```python
z1 = operator_input(lhs=a, rhs=b, op=SUB)
z2 = operator_input(lhs=b, rhs=a, op=SUB)
```

Registrar:

```text
max_abs(z1-z2)
cosine(z1,z2)
cantidad de colisiones exactas
cantidad de colisiones con targets diferentes
```

Si `z1 == z2`, SUB es matemáticamente irresoluble con esa representación.

Bajo pares uniformes y representación completamente simétrica, el máximo teórico de SUB sería aproximadamente:

$$
\frac{544}{1024}=53.125\%
$$

No el 100%. Como el resultado observado es mucho menor —9–11%—, probablemente existe además otro problema, pero este test debe ejecutarse primero.

# 2. Segundo sospechoso: dirección u orientación de SUB

Hay que comparar cada predicción SUB contra varias funciones alternativas:

```text
correcta:       (registro - operando) mod 32
invertida:      (operando - registro) mod 32
suma:           (registro + operando) mod 32
identidad L:    registro
identidad R:    operando
negación:       (-registro) mod 32
```

El informe debería dar:

```text
accuracy_vs_lhs_minus_rhs
accuracy_vs_rhs_minus_lhs
accuracy_vs_add
accuracy_vs_lhs
accuracy_vs_rhs
```

Esto puede detectar inmediatamente:

* operands intercambiados;
* mapping incorrecto de `op_id`;
* off-by-one en el índice de operación;
* target generado con una convención y forward con otra;
* modelo que ignora el operando.

También hay que imprimir una tabla de diez ejemplos SUB con:

```text
registro inicial
operando
op_id del generador
op_id visto por el modelo
target
predicción
target invertido
```

Con la asimetría actual, esta prueba vale más que otras 50,000 actualizaciones.

# 3. El softmax contra embeddings sirve muy bien para copiar, no necesariamente para calcular

Las tareas anteriores producen un valor que ya existe en la memoria:

$$
\text{reader output}\approx E(\text{respuesta})
$$

Entonces un readout como:

$$
\ell_k=z^\top E(k)
$$

funciona naturalmente: el estado recuperado ya está cerca del embedding correcto.

En aritmética, el MLP debe fabricar un vector:

$$
z=F(E(v),E(a),E(o))
$$

que caiga exactamente en la región de Voronoi correspondiente a:

$$
E((v\circ a)\bmod32)
$$

Eso impone dos trabajos simultáneos:

1. computar la función;
2. aprender la geometría del codebook.

Un MLP genérico puede ajustar aritmética, pero las redes convencionales suelen tener dificultades para aprender operadores numéricos sistemáticos, razón por la que se han investigado unidades aritméticas especializadas. ([arXiv][1])

## Ablación decisiva

Reemplazar temporalmente el softmax contra embeddings por una cabeza independiente:

$$
\text{logits}=W_{\text{class}}h+b,
\qquad
W_{\text{class}}\in\mathbb R^{32\times d_h}
$$

Sin weight tying y sin utilizar `value_embedding.weight` como clasificador.

Comparar:

```text
A. cabeza independiente Linear(hidden, 32)
B. dot-product contra embeddings aprendidos
C. cosine contra embeddings normalizados
```

Si A llega a 100% y B permanece en 20%, el problema no está en el operador: está en el codebook/readout.

También deben registrarse:

```text
norma de cada value embedding
frecuencia predicha de cada clase
correlación norma ↔ frecuencia predicha
distancia del output al embedding correcto
margen entre clase correcta y segunda clase
```

Una clase con embedding de norma mayor puede dominar un readout por producto punto cuando el vector sintetizado todavía no está bien canonicalizado.

# 4. La tabla elemental es demasiado pequeña para justificar “más datos” como primera respuesta

El operador completo contiene:

$$
32\times32\times3=3072
$$

casos posibles.

Antes de volver a datos procedurales, deben enumerar el truth table completo:

```text
ADD: 1024 casos
SUB: 1024 casos
MUL: 1024 casos
```

Y entrenar **sin recurrencia**, exclusivamente H1.

La primera pregunta no es si generaliza. Es:

> ¿Puede la parametrización ajustar las 3072 transiciones elementales?

Gate:

```text
train accuracy por operación ≥99.9%
```

Si el modelo no puede sobreajustar esa tabla, no es un problema de grokking, semillas o generalización. Es uno de estos:

* colisión de representación;
* operación mal indexada;
* head inadecuado;
* ancho insuficiente;
* gradiente desconectado;
* interferencia entre operaciones;
* target incorrecto.

Solo después de obtener prácticamente 100% de train tiene sentido dividir combinaciones entre train/test y estudiar generalización.

# 5. La distribución de MUL puede engañar

Para operandos uniformes en módulo 32:

* ADD tiene outputs uniformes;
* SUB tiene outputs uniformes;
* MUL no los tiene.

En MUL:

$$
P(y=0)=\frac{112}{1024}=10.9375\%
$$

Por tanto, un modelo que siempre responda cero ya obtiene casi 11%. Su `18–20%` en MUL está por encima de ese baseline, pero no demuestra que haya aprendido multiplicación modular.

Hay que reportar por operación:

```text
accuracy
majority baseline
balanced accuracy
confusion matrix
entropía de predicciones
frecuencia de output 0, 8, 16, 24
```

La multiplicación módulo 32 es además estructuralmente distinta de ADD/SUB: 32 es compuesto y la multiplicación contiene divisores de cero, de modo que no forma una operación globalmente invertible como la suma cíclica. Eso refuerza la idea de no exigir que las tres operaciones compartan exactamente toda su parametrización. ([arXiv][2])

Sin embargo, eso no explica que SUB sea la peor. **SUB sigue apuntando a roles, orientación o readout.**

# 6. Un MLP único para las tres operaciones puede estar creando interferencia

El hecho de que la operación sea conocida por índice significa que no hay ninguna razón física para obligar al mismo operador exacto a representar las tres funciones.

Se puede conservar el núcleo principal compartido y añadir componentes diminutos por operación:

$$
h=
F_\theta
\left(
W_LE(v),
W_RE(a),
E_o
\right)
$$

$$
\ell=
H_o(h)
$$

donde:

```text
Fθ: trunk compartido
HADD: cabeza pequeña
HSUB: cabeza pequeña
HMUL: cabeza pequeña
```

Otra opción:

$$
F_o=F_\theta+A_oB_o
$$

con adaptadores de rango bajo por operación.

Eso continúa siendo:

* compartido entre profundidades;
* pequeño;
* residente;
* compatible con el diseño CPU-native.

Lo único que deja de asumir es que una única transformación homogénea debe ejecutar semánticas algebraicas diferentes. Esa es precisamente la misma lección que apareció con los slots:

> La heterogeneidad semántica debe reflejarse en la regla de transición.

# 7. Arquitectura mínima recomendada para `sequential-update`

Primera versión diagnóstica:

$$
e_L=W_LE_{\text{value}}(v)
$$

$$
e_R=W_RE_{\text{operand}}(a)
$$

$$
u=
[
e_L;
e_R;
E_{\text{op}}(o);
e_L-e_R;
e_L\odot e_R
]
$$

$$
h=\operatorname{SwiGLU}(W_1u)
$$

$$
\text{logits}=H_o(h)
$$

Después se canonicaliza el registro:

$$
p=\operatorname{softmax}(\text{logits}/\tau)
$$

$$
p_{\text{hard}}
=
\operatorname{onehot}(\arg\max p)
$$

Durante entrenamiento puede usarse straight-through:

$$
p_{\text{ST}}
=
p_{\text{hard}}-p.\operatorname{detach}()+p
$$

Y el estado siguiente:

$$
r_{t+1}
=
p_{\text{ST}}E_{\text{value}}
$$

En pseudocódigo:

```python
left = left_proj(value_embedding[register])
right = right_proj(operand_embedding[operand])
op = op_embedding[op_id]

features = torch.cat(
    [left, right, op, left - right, left * right],
    dim=-1,
)

hidden = operator_mlp(features)
logits = operation_heads[op_id](hidden)

probs = logits.softmax(dim=-1)
hard = F.one_hot(
    probs.argmax(dim=-1),
    num_classes=32,
).to(probs.dtype)

state_distribution = hard - probs.detach() + probs
next_register = state_distribution @ value_embedding.weight
```

Para la primera prueba quitaría incluso `left*right`. Empezaría con concatenación y proyecciones de rol distintas. El producto se añade solo si MUL continúa fallando.

# 8. Escalera de ablaciones: cambia una sola cosa cada vez

## SU-0 — Oráculo y auditoría

Enumerar las 3072 combinaciones y verificar:

$$
y_{\text{ADD}}=(v+a)\bmod32
$$

$$
y_{\text{SUB}}=(v-a)\bmod32
$$

$$
y_{\text{MUL}}=(v\cdot a)\bmod32
$$

Probar además las etiquetas alternativas de SUB descritas arriba.

## SU-1 — Baseline que debe memorizar

Entrada:

```text
one-hot registro: 32
one-hot operando: 32
one-hot operación: 3
concatenación total: 67
```

Modelo:

```text
Linear(67, 128)
SiLU
Linear(128, 128)
SiLU
Linear(128, 32)
```

Sin embeddings compartidos, recurrencia ni readout atado.

Criterio:

```text
ADD train ≥99.9%
SUB train ≥99.9%
MUL train ≥99.9%
```

Si esto falla, hay un problema en datos, targets, optimizador o evaluación.

## SU-2 — Operaciones separadas

Entrenar:

```text
ADD-only
SUB-only
MUL-only
```

Interpretación:

| Resultado                              | Diagnóstico                                         |
| -------------------------------------- | --------------------------------------------------- |
| SUB-only falla                         | roles/orientación/representación                    |
| Cada una pasa, joint falla             | interferencia entre operaciones                     |
| Independent head pasa, tied head falla | codebook/readout                                    |
| One-hot pasa, embeddings fallan        | geometría de embeddings                             |
| Todas fallan                           | bug de pipeline o capacidad claramente insuficiente |

## SU-3 — Cabeza independiente con embeddings

Mantener embeddings, pero conservar:

```text
left_proj != right_proj
head Linear(hidden, 32)
```

Esto prueba si las representaciones aprendidas son adecuadas una vez eliminado el tying.

## SU-4 — Shared trunk + heads por operación

Solo después de que cada operación pase por separado.

## SU-5 — Canonicalización recurrente

Congelar inicialmente el operador H1 ya aprendido y ejecutar secuencias:

```text
H=1,2,3,4
R=1,2,4
```

Con teacher forcing primero:

```text
registro verdadero como input de cada paso
```

Luego free-running:

```text
registro predicho y canonicalizado
```

Esto separa:

```text
error del operador
vs.
acumulación de error recurrente
```

# 9. Hay una inconsistencia métrica que debe aclararse en el próximo reporte

Se afirma simultáneamente:

```text
una operación/ronda 1: ~19–20%
R≥H en secuencias: 59–68%
```

Esas cifras no parecen corresponder a la misma distribución o al mismo punto de decodificación.

El reporte debe incluir la matriz exacta:

```text
accuracy[H][R][operation]
```

y distinguir:

```text
final task accuracy
pointer/register accuracy después de cada ronda
teacher-forced one-step accuracy
free-running one-step accuracy
```

Si `H=1,R=1` final realmente es 20%, entonces cualquier agregado de 59–68% debe estar utilizando:

* otra combinación de H/R;
* otra distribución de targets;
* rondas adicionales para una sola operación;
* o una métrica distinta.

No implica necesariamente otro bug, pero **no se puede llamar frontera causal limpia hasta normalizar esas mediciones**.

También hay que comprobar no-overshoot:

```text
H=1,R=4
H=2,R=4
```

y declarar si las rondas sobrantes:

* están enmascaradas;
* repiten la última operación;
* ejecutan identidad;
* refinan el mismo resultado.

# 10. ¿Hace falta currículo?

**Sí, pero como herramienta de aislamiento, no como intento de rescatar a ciegas el modelo actual.**

Orden correcto:

```text
1. ADD-only H1 hasta 100%
2. SUB-only H1 hasta 100%
3. MUL-only H1 hasta 100%
4. mezcla balanceada de las tres en H1
5. congelar operador
6. composición H2–H4
7. descongelar y afinar conjuntamente
```

Las redes que aprenden suma modular pueden descubrir representaciones Fourier y circuitos trigonométricos, pero esa transición puede requerir una dinámica de entrenamiento específica; también se han observado mecanismos Fourier en redes recurrentes. ([arXiv][3]) Eso justifica estudiar representaciones estructuradas más adelante, pero no explica un modelo que todavía no puede ajustar limpiamente el operador elemental.

No usaría todavía NALU como sustitución directa: está orientado principalmente a cantidades numéricas continuas y extrapolación. Para valores categóricos en \(\mathbb Z_{32}\), una cabeza bilineal, features cíclicas o un operador por tipo encajan mejor.

# Decisión

```text
Más semillas:                  NO
Más steps sin cambios:         NO
Aumentar D inmediatamente:     NO
Currículo por operación:       SÍ
Cabeza independiente:          SÍ
Auditar roles de operandos:    PRIORIDAD 1
Auditar orientación de SUB:    PRIORIDAD 1
Canonicalización hard/ST:      SÍ
Tres operadores completamente
separados para siempre:        NO todavía
Shared trunk + adapters/heads: CANDIDATO PRINCIPAL
```

La primera corrida siguiente debe ser **SU-0/SU-1**, no otra campaña recurrente.

La lectura arquitectónica es positiva, aunque el gate falle:

> Las tareas anteriores probaron una máquina de memoria direccionable. Sequential-update está revelando que falta una ALU aprendible y tipada.

La arquitectura resultante probablemente no será:

```text
un mismo MLP genérico para cualquier transición
```

sino:

```text
núcleo compartido
+ estado tipado
+ canonicalización
+ pequeños operadores/adaptadores especializados
```

Ese cambio no contradice la idea CPU-native. La vuelve más coherente: **el núcleo conserva pesos compartidos en profundidad, mientras una cantidad diminuta de parámetros determina qué transformación semántica se aplica al estado.**

[1]: https://arxiv.org/abs/1808.00508?utm_source=chatgpt.com "Neural Arithmetic Logic Units"
[2]: https://arxiv.org/abs/2607.07066?utm_source=chatgpt.com "Multiplication Beyond Groups: Stratified Fourier Mechanisms in Transformer Circuits"
[3]: https://arxiv.org/abs/2301.05217?utm_source=chatgpt.com "Progress measures for grokking via mechanistic ..."

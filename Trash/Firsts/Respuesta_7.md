## Veredicto

**No es todavía una interferencia misteriosa del núcleo unificado. Hay un desvío semántico central en el código: la implementación U0-A no conserva la canonicalización del registro que hizo funcionar SU-4.**

Además encontré tres desviaciones secundarias:

1. El adapter actual no replica realmente la arquitectura de SU-4.
2. El entrenamiento unificado no usa el mismo objetivo ni el mismo currículo que SU-4.
3. Quitar `weight_decay` no impide que Adam siga moviendo heads/adapters inactivos mediante momentum.

Audité `unified.py`, `sequential_update_su4.py`, `train_u0a.py`, los tests y los JSON de resultados incluidos en el ZIP. 

# 1. Desvío dominante: el registro no vuelve al codebook después de cada operación

## Lo que hacía SU-4

En `sequential_update_su4.py:42–48`:

```python
hidden = self.trunk(features)

all_logits = torch.stack(
    [head(hidden) for head in self.operation_heads],
    dim=1,
)

selected_logits = all_logits[
    torch.arange(register.shape[0]),
    operation,
]

probabilities = torch.softmax(selected_logits, dim=-1)
next_register = probabilities @ self.value_embedding.weight
```

La operación produce:

$$
\text{logits}\in\mathbb R^{32}
$$

y después vuelve a colocar el registro en el manifold canónico:

$$
R_{r+1}
=
\operatorname{softmax}(\ell_r)E_{\text{value}}
$$

Por tanto, cada nueva ronda recibe nuevamente una representación válida de uno de los 32 valores.

## Lo que hace el unificado

En `unified.py:310–319`, las heads son:

```python
ALU_ADD: Linear(64, 64)
ALU_SUB: Linear(64, 64)
ALU_MUL: Linear(64, 64)
```

Y en `unified.py:351–353`:

```python
candidate = self.operation_heads[name](
    values[:, SLOT_R, :]
)
write(opcode == OPCODE_IDS[name], candidate)
```

El resultado continuo de dimensión 64 se escribe directamente en `R`.

No existe:

```python
softmax
argmax
proyección a las 32 clases
re-embedding al codebook
```

El decoder atado se usa solamente **después de todas las rondas**, en `train_u0a.py:502–510`.

Así que la trayectoria real es:

```text
embedding canónico
→ operación
→ vector continuo no canónico
→ operación sobre vector fuera del manifold
→ más deriva
→ decoder final
```

No es la trayectoria validada en SU-4:

```text
embedding canónico
→ operación
→ logits 32
→ canonicalización
→ embedding canónico
→ siguiente operación
```

Esto basta para explicar por qué el sistema parece aprender algo, pero no compone exactamente.

# 2. Lo reproduje con el código exacto del ZIP

Ejecuté un control adicional usando directamente `UnifiedT1U0` del archivo, no una maqueta.

Entrené el camino unificado actual sobre la tabla elemental completa:

$$
32\times32\times3=3072
$$

El modelo llegó a:

```text
H1 / teacher-forced: 100%
```

aproximadamente en el step 2000. Por tanto:

> **La arquitectura actual sí posee capacidad para representar las tres operaciones de un paso.**

Después ejecuté el operador libremente durante seis rondas, sin canonicalizar `R`.

| Ronda | Accuracy free-running |
| ----: | --------------------: |
|     1 |                100.0% |
|     2 |                 34.5% |
|     3 |                 15.9% |
|     4 |                 12.6% |
|     5 |                  9.9% |
|     6 |                  9.3% |

El estado de la primera ronda tenía la clase correcta, pero su coseno promedio con el embedding canónico correcto era solo aproximadamente:

$$
0.594
$$

Es decir, estaba en la región correcta del decoder, pero no era una representación reutilizable del valor.

Luego mantuve exactamente los mismos pesos y añadí solamente una canonicalización entre rondas:

### Soft

$$
R_{r+1}
=
\operatorname{softmax}(\text{decoder}(R_{r+1}))
E_{\text{value}}
$$

### Hard

$$
R_{r+1}
=
E_{\arg\max \text{decoder}(R_{r+1})}
$$

Resultado con ambas:

```text
r1: 100%
r2: 100%
r3: 100%
r4: 100%
r5: 100%
r6: 100%
```

Esto localiza la causa con mucha fuerza:

$$
\boxed{
\text{el operador elemental puede aprenderse;
lo que falta es cierre del estado bajo composición}
}
$$

No hace falta atribuir todavía el fallo a capacidad, SlotMix o interferencia general.

# 3. El adapter no replica SU-4

El comentario del código dice que reproduce la asimetría de SU-4, pero estructuralmente no es equivalente.

## SU-4

```text
left projection:   64 → 64
right projection:  64 → 64
features:          256
trunk:             256 → 256 → SiLU → 64
heads:             64 → 32 por operación
canonicalización:  softmax × codebook
```

## Adapter unificado actual

En `unified.py:202–226`:

```text
left projection:    64 → 16
right projection:   64 → 16
opcode projection:  64 → 16
features:            64
adapter down:        64 → 8
adapter up:           8 → 64
```

Y la operación es literalmente:

```python
adapter["up"](adapter["down"](alu_features))
```

No hay activación entre ambas capas.

Matemáticamente se reduce a una sola transformación lineal:

$$
A_oB_o x
$$

de rango máximo 8.

SU-4 utiliza:

$$
W_2\operatorname{SiLU}(W_1x)
$$

con una representación intermedia de 256 dimensiones.

Por tanto, el adapter actual es:

> una perturbación lineal de rango 8 que introduce asimetría,

no:

> una réplica pequeña del operador SU-4.

Este desvío puede volver el aprendizaje multitarea más frágil, pero mi control muestra que **no es la causa primaria**: incluso el camino actual ajusta H1 al 100%. Primero debe corregirse la canonicalización; después se decide si hace falta más capacidad no lineal.

# 4. SlotMix no es el culpable principal en esta tarea

`parse_sequential()` declara:

```python
presence = (False, True, False, False)
```

Solo está presente `R`.

Con una sola key válida en la atención de slots:

$$
\operatorname{softmax}([s])=1
$$

Por tanto, en sequential-update no existe mezcla real entre `P`, `R`, `E` y `W`. La atención se reduce esencialmente a una transformación `value/output` del único slot presente.

Así que el fallo del canario secuencial no puede atribuirse a que varios slots se estén contaminando dentro de SlotMix. Puede existir interferencia **paramétrica** porque esas matrices también se entrenan en otras tareas, pero no mezcla simultánea de contenido entre slots en ese forward.

# 5. MUL aparece mejor por una razón matemática, no porque su head sea mejor

El archivo agrupa accuracy por la **última operación de toda la secuencia**. Eso no mide aisladamente la calidad del operador final.

ADD y SUB son biyectivas respecto al registro:

$$
x\mapsto x+a\pmod{32}
$$

$$
x\mapsto x-a\pmod{32}
$$

Si el registro anterior está equivocado, el resultado final también permanece equivocado.

MUL no es biyectiva para la mayoría de operandos en \(\mathbb Z_{32}\):

$$
x\mapsto ax\pmod{32}
$$

Ejemplos:

```text
×0:
todos los registros terminan en 0

×16:
solo importa la paridad

×8:
solo importa el valor módulo 4

×4:
solo importa el valor módulo 8
```

La probabilidad de que dos registros distintos colisionen después de multiplicar por \(a\) es:

$$
P(ax=ay)
=
\frac{\gcd(a,32)}{32}
$$

Promediando operandos uniformes:

$$
P(\text{colisión por MUL})=10.9375\%
$$

Para ADD o SUB:

$$
P(\text{misma salida desde registros distintos})=0
$$

Por eso una última multiplicación puede **ocultar un error acumulado en rondas anteriores**. Una secuencia que termina en ADD o SUB no tiene ese mecanismo de recuperación accidental.

Además, los targets de MUL son mucho menos uniformes:

```text
output 0:  112/1024
output 16: 80/1024
output 8:   64/1024
output 24:  64/1024
```

ADD y SUB tienen los 32 outputs exactamente uniformes.

En mi control con el mismo source y un operador H1 perfecto, pero dejando `R` sin canonicalizar, apareció la misma inversión:

| Longitud |    ADD |    SUB |    MUL |
| -------: | -----: | -----: | -----: |
|       H3 | 10.96% | 10.46% | 24.67% |
|       H4 |  7.20% |  7.11% | 19.49% |

Así que el patrón:

```text
MUL > ADD ≈ SUB
```

es precisamente lo esperado cuando el registro está derivando fuera del codebook.

No indica que MUL esté mejor aprendida. Indica que MUL:

* tiene targets más comprimidos;
* puede utilizar reglas parciales de paridad/divisibilidad;
* puede borrar errores previos;
* es evaluada mediante una métrica que atribuye a la última operación todos los errores anteriores.

# 6. La comparación de entrenamiento tampoco era equivalente

SU-4 se entrenó con:

```text
3072 transiciones elementales
H=1
supervisión directa del operador
head de 32 clases
canonicalización inmediata
```

El unificado se entrena con:

```text
secuencias H=3..6
solo pérdida final
sin target intermedio
sin canonicalización por ronda
cobertura incompleta de la tabla elemental
```

Además, ya habían medido previamente que el dataset secuencial solo cubría:

```text
1106 / 3072 transiciones
MUL: 112 / 1024
```

Por tanto, no se estaba comprobando “el mismo mecanismo dentro del pipeline unificado”. Se estaba pidiendo algo bastante más difícil:

> descubrir simultáneamente la ALU elemental y aprender a estabilizar una representación continua durante hasta seis aplicaciones, usando solamente el target final.

Eso no es el gate de coexistencia U0-A. U0-A debería conservar las primitivas que T1 ya demostró.

# 7. Quitar weight decay no terminó de arreglar los módulos inactivos

En el código actual, todos los adapters se calculan siempre:

```python
for name, adapter in self.alu_adapters.items():
    mask = ...
    alu_delta += mask * adapter(...)
```

Y todas las operation heads también:

```python
for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL"):
    candidate = self.operation_heads[name](...)
    write(mask, candidate)
```

En un batch `READ_P`, las heads ALU reciben:

```text
grad tensor = 0
```

no:

```text
grad = None
```

Eliminar `weight_decay` evita una fuente de movimiento, pero Adam mantiene momentum.

Lo confirmé directamente con el código del ZIP y el learning rate real `3e-4`:

1. construí momentum mediante actualizaciones `ALU_ADD`;
2. ejecuté cinco batches `READ_P`;
3. los gradientes ALU eran exactamente cero;
4. aun así cambiaron:

```text
ADD head:   max |Δθ| ≈ 7.17e-4
ADD adapter:max |Δθ| ≈ 7.73e-4
```

La magnitud exacta depende del historial, pero la propiedad es inequívoca:

$$
g_t=0
\quad\not\Rightarrow\quad
\Delta\theta_t=0
$$

con Adam si existe momentum previo.

Esto afecta especialmente a sequential-update porque entre dos batches secuenciales hay cinco optimizer steps de otras tareas.

## Corrección

No calcular las heads/adapters inactivos.

Para batches homogéneos sin ALU:

```python
if not is_alu_batch:
    skip all ALU modules
```

Para batches con varios opcodes:

```python
for opcode in active_opcodes:
    indices = opcode_mask.nonzero()
    output[indices] = selected_module(input[indices])
```

Las heads no seleccionadas deben quedar con:

```python
parameter.grad is None
```

También puede usarse un optimizador separado para los módulos tipados y ejecutarlo solo cuando la operación correspondiente esté activa.

# 8. El test que debía proteger los adapters es vacuo

En `tests/test_unified.py:140–164`, el test crea el opcode:

```python
opcode = torch.full(...)
```

pero llama a:

```python
model.core(...)
```

sin pasar:

```python
opcode=opcode
```

Como `SharedRecurrentCore.forward()` tiene:

```python
opcode: Tensor | None = None
```

el adapter nunca se activa durante ese test.

Después el test modifica sus pesos y comprueba que retrieval no cambió. Naturalmente no cambia: el código bajo prueba nunca ejecutó los adapters.

Ese test debe corregirse para pasar explícitamente el opcode y añadir controles positivos:

```text
mutar ADD adapter:
- debe cambiar ALU_ADD
- no debe cambiar ALU_SUB
- no debe cambiar ALU_MUL
- no debe cambiar READ_P/READ_E/ACCUM_W
```

También falta un test de inmovilidad después de `optimizer.step()`, no solo igualdad del forward.

# 9. `train_u0a.py` todavía viola el protocolo iso-update

El source incluido continúa haciendo:

```python
for step in range(30_000):
    task = round_robin[step]
    loss.backward()
    optimizer.step()
```

Eso produce:

```text
5000 batches por tarea
30000 actualizaciones del trunk
```

El protocolo acordado era:

```text
5000 supersteps

en cada superstep:
    1 batch por cada tarea
    acumular las 6 pérdidas/gradientes
    1 optimizer.step()
```

Eso produce:

```text
5000 batches por tarea
5000 actualizaciones del trunk
```

No explica por sí solo el canario sequential-only, pero debe corregirse antes de repetir U0-A completo.

# 10. Corrección exacta del registro

Cambiar las heads de commit:

```python
nn.Linear(D, D)
```

por heads de clases:

```python
nn.Linear(D, 32)
```

Y añadir al candidato:

```python
@dataclass
class CandidateState:
    values: Tensor
    alu_logits: Tensor | None
```

Después:

```python
all_logits = torch.stack(
    [
        add_head(hidden),
        sub_head(hidden),
        mul_head(hidden),
    ],
    dim=1,
)

selected_logits = select_by_opcode(all_logits, opcode)

probabilities = torch.softmax(selected_logits, dim=-1)

next_register = probabilities @ register_codebook
```

Finalmente:

```python
write(is_alu, next_register)
```

La ecuación correcta es:

$$
\ell_r=H_{o_r}(C_r^R)
$$

$$
p_r=\operatorname{softmax}(\ell_r)
$$

$$
R_{r+1}=p_rE_{\mathbb Z_{32}}
$$

No:

$$
R_{r+1}=H_{o_r}(C_r^R)\in\mathbb R^{64}
$$

Para inferencia dura puede utilizarse:

$$
R_{r+1}=E_{\arg\max\ell_r}
$$

Durante entrenamiento comenzaría con la mezcla soft, porque es exactamente el mecanismo que ya pasó SU-4 y mi control composicional.

Esto incluso reduce parámetros:

```text
heads actuales 64→64:
~12,480 parámetros

heads correctas 64→32:
~6,240 parámetros
```

# 11. Qué hacer con el adapter

No aumentaría todavía su tamaño.

Orden correcto:

### A. Canonicalización solamente

Mantener el adapter actual y corregir el commit.

### B. Tabla completa H1

Dentro de `UnifiedT1U0`, no usando el modelo SU-4 externo:

```text
ADD: 100%
SUB: 100%
MUL: 100%
```

### C. Composición free-running

```text
H=1..6
teacher forcing = free-running
```

### D. Solo si H1 no llega a 100%

Convertir:

```python
up(down(features))
```

en:

```python
up(F.silu(down(features)))
```

y barrer:

```text
rank = 8, 16, 32, 64
```

Una aproximación más fiel y todavía pequeña sería:

```python
alu_front = nn.Sequential(
    nn.Linear(4 * feature_width, hidden),
    nn.SiLU(),
    nn.Linear(hidden, D),
)
```

compartida por ADD/SUB/MUL, con las tres heads de 32 clases al final.

No copiaría inmediatamente todo el trunk SU-4 porque U0 intenta conservar un trunk pesado común. Primero hay que comprobar si canonicalización + supervisión correcta bastan.

# 12. Supervisión que debe añadirse

El parser conoce el registro verdadero después de cada operación. Guardar:

```text
intermediate_register_targets[r]
```

y entrenar:

$$
\mathcal L
=
\mathcal L_{\text{final}}
+
\lambda
\frac1H
\sum_{r=1}^{H}
\operatorname{CE}
(\ell_r,y_r)
$$

Inicialmente:

```text
λ = 1
```

Esto no es teacher forcing ni fuga hacia el estado. El modelo sigue ejecutando free-running; solamente recibe señal de error en cada transición.

Alternativamente, preentrenar dentro del modelo unificado sobre la tabla completa de 3072 casos y después ejecutar composición. Ese es el control más fiel a SU-4/SU-5.

# Secuencia mínima siguiente

No más semillas ni más steps todavía.

## Paso 1 — sin reentrenar

Tomar el checkpoint actual y canonicalizar `R` después de cada ronda usando el decoder atado.

Registrar:

```text
accuracy raw
accuracy soft-canonical
accuracy hard-canonical
```

Un salto grande confirmará directamente el diagnóstico sobre ese checkpoint concreto.

## Paso 2 — one-step audit

Evaluar el checkpoint sobre las 3072 transiciones:

```text
ADD teacher-forced
SUB teacher-forced
MUL teacher-forced
```

Y registrar:

```text
nearest-codebook cosine
decoder margin
distancia a embedding canónico
```

## Paso 3 — retraining seed101

Con:

```text
heads D→32
canonicalización por ronda
pérdida intermedia
adapter actual sin modificar
```

Gate:

```text
H1 exacto:                100%
TF vs free-running:       idénticos
H3-H6, R≥H:               100%
R<H:                      frontera causal
```

## Paso 4 — U0 completo

Solo después:

```text
5000 supersteps
una actualización por conjunto de seis tareas
dispatch real de módulos tipados
grad=None cuando están inactivos
```

# Gap de reproducibilidad del ZIP

El archivo contiene los JSON del diagnóstico, pero no contiene:

* `best.pt`;
* `metrics.jsonl`;
* el script exacto que produjo la campaña `u0a_sequential_alu_adapter_real_seed101`.

El `train_u0a.py` incluido siempre entrena las seis tareas durante 30,000 steps; no produce directamente el resumen `"task": "sequential_update"` de 5,000 steps incluido en resultados.

Por tanto, pude:

* auditar el modelo;
* localizar los desvíos;
* reproducir la dinámica causal con el mismo source;

pero no recomputar exactamente ese checkpoint específico.

## Ledger

```text
Cinco rutas U0-A:
PASS tras correcciones reportadas.

Sequential-update actual:
REJECTED AS A VALID SU-4 REPLICATION.

Causa dominante:
falta de canonicalización del registro entre rondas.

Causas adicionales:
adapter lineal rank-8 ≠ SU-4;
final-only H3-H6 ≠ entrenamiento H1 completo;
momentum modifica módulos inactivos;
test de aislamiento del adapter es vacuo;
runner completo todavía no usa supersteps.
```

La lectura principal es:

$$
\boxed{
\text{el núcleo unificado puede aprender la ALU elemental,
pero su commit no devuelve }R\text{ al espacio discreto}
}
$$

Y la aparente superioridad de MUL es una consecuencia predecible de esa deriva: **la multiplicación módulo 32 comprime y borra errores; suma y resta los conservan**. No deben ajustar el adapter alrededor de ese patrón. Deben restablecer primero la canonicalización que ya había hecho exacta a SU-4.

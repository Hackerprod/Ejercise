## Veredicto

**U0-A falla de verdad, pero no por una simple falta de steps ni porque `D=64` sea necesariamente insuficiente.** El patrón indica tres fallos diferentes que se están mezclando:

1. **Interferencia dentro del reader y de las representaciones canónicas.**
2. **Errores pequeños por ronda que se multiplican con la profundidad.**
3. **Contaminación de caminos que debían ser exactos y estar protegidos del trunk.**

No lancen otra semilla ni aumenten a `D=128` todavía. El checkpoint actual contiene suficiente información para localizar la causa mediante ablaciones sin volver a entrenar.

---

# 1. Workspace es la evidencia más fuerte de un problema estructural

La tarea de workspace debería ejecutar:

$$
W_{r+1}=W_r+Y_r
$$

La ruta identidad ya fue validada como exacta en T1-W. En U0-A, mientras no se esté probando explícitamente una transformación contextual, la corrección aprendida debería permanecer desactivada:

$$
C_r^W=0
$$

Por tanto, obtener error `0.05–0.08` no puede atribuirse a falta de capacidad. Una suma exacta no necesita más parámetros ni más entrenamiento.

Esto apunta a una de estas desviaciones:

* la rama `correction_mlp` está activa y el trunk compartido la contaminó;
* la proyección final de `W` se comparte con candidatos de otros slots;
* parámetros inactivos continúan recibiendo weight decay o momentum;
* el payload no llega directamente al commit;
* la máscara de presencia no bloquea completamente la rama aprendida;
* se está normalizando o transformando el payload antes de la suma exacta.

### Prueba inmediata

Sobre el checkpoint actual:

```text
evaluación normal
vs.
evaluación con C_W forzado exactamente a cero
```

Resultado esperado:

```text
C_W = 0 → error exactamente 0
```

Si vuelve a cero, la acumulación funciona y el fallo es cross-talk de la rama correctora. Para U0-A, esa rama debe quedar **hard-disabled o congelada**. Su activación correspondía a U0-C, no al primer gate de coexistencia.

Si no vuelve a cero, existe un desvío en `READ → payload → COMMIT`.

---

# 2. Variable-binding a azar señala la interfaz entre tipos de memoria

Variable-binding ejecuta una composición:

```text
ASSIGN(variable → reference)
        ↓
ATTR(reference, attribute → value)
```

El primer reader puede funcionar y aun así la tarea caer a azar si la representación producida por `ASSIGN` no pertenece exactamente al espacio que espera la clave `reference` de `ATTR`.

La identidad debe ser canónica e independiente del tipo de fila:

$$
\operatorname{EncodeValue}_{ASSIGN}(ref)
=
\operatorname{EncodeKeyRef}_{ATTR}(ref)
$$

Del mismo modo:

$$
\operatorname{EncodeValue}_{REL}(id)
=
\operatorname{EncodeKeySrc}_{REL}(id)
$$

No conviene representar el payload como:

$$
\phi_V(\text{row type},\text{value})
$$

cuando el valor es una identidad que después se reutilizará como dirección. El `row_type` puede participar en la **clave de lectura**, pero no debe cambiar la identidad del payload.

Debe existir un codec canónico por tipo semántico:

```text
EntityCodec
ReferenceCodec
DiscreteValueCodec
VectorCodec
```

No un codec de identidad distinto para `REL`, `ASSIGN` y `ATTR`.

### Ablación por etapas

Evaluar variable-binding de tres maneras:

```text
A. reader normal → reader normal
B. primera lectura ASSIGN oráculo → segunda lectura normal
C. primera normal → segunda lectura ATTR oráculo
D. ambas lecturas oráculo
```

Interpretación:

| Resultado          | Causa                              |
| ------------------ | ---------------------------------- |
| B recupera, C no   | ASSIGN/query inicial               |
| C recupera, B no   | ATTR/composición de clave          |
| Solo D recupera    | ambos readers degradados           |
| D tampoco recupera | commit, decoder o instruction tape |

También deben registrar por separado:

```text
accuracy de ASSIGN
accuracy de ATTR condicionada a reference correcto
accuracy end-to-end
reader margin por etapa
entropía por etapa
```

`0.129≈azar` por sí solo no dice cuál de las dos lecturas falló.

---

# 3. Multi-hop no parece simplemente “subentrenado”

La curva:

```text
H1 = 1.000
H2/R2 = 0.967
H3/R3 = 0.500
H4/R4 = 0.300
```

no parece una degradación suave causada únicamente por capacidad limitada. Existe un **precipicio después de la segunda ronda**.

Hay que comprobar cuatro cosas.

## Canonicalización después de cada lectura

El nuevo puntero debe ser convertido otra vez a una identidad canónica:

$$
P_{r+1}
=
\operatorname{Canonicalize}
\left(
Y_r
\right)
$$

No basta con conservar el weighted sum blando del reader. Un pequeño componente de otras keys puede no afectar el primer decode, pero al reutilizarse como query se amplifica.

Registrar:

```text
pointer accuracy por ronda
reader margin por ronda
reader entropy por ronda
old-pointer mass por ronda
distancia al embedding canónico más cercano
```

Si ocurre algo parecido a:

```text
r1: 100%
r2: 97%
r3: 52%
r4: 31%
```

el problema está en el cierre del estado bajo aplicación repetida.

## Instruction tape

Comparar por ronda, contra la implementación aislada:

```text
opcode
source slot
destination slot
row-type mask
immediate
active-round mask
```

El salto brusco después de `r=2` también puede ser un error de:

* off-by-one;
* `EMIT` prematuro;
* máscara de halting;
* round embedding;
* índice de instrucción;
* reutilización del payload anterior.

## Embedding de profundidad

Si todavía existe un `depth_embedding[r]`, hacer una evaluación con él puesto a cero.

Para operaciones homogéneas como pointer chasing, cada ronda debería aplicar la misma función:

$$
P_{r+1}=f(P_r)
$$

No necesita una semántica distinta por índice de profundidad. Un embedding de ronda compartido por seis tareas puede convertirse en otro punto de interferencia.

## Reader oráculo

Reemplazar la salida del reader por el valor correcto manteniendo el resto del núcleo y commit.

Si multi-hop vuelve inmediatamente a 100%, el núcleo recurrente no es el problema.

---

# 4. Pointer-chasing y multi-hop no deberían divergir tanto sin una razón concreta

Ambos utilizan esencialmente:

```text
READ_P sobre relaciones
```

Sin embargo:

```text
pointer-chasing ≈ 0.27–0.35
multi-hop H1 = 1.0
```

Esto exige comparar:

* cantidad de identidades;
* codebook utilizado;
* decoder;
* key/value codecs;
* número de filas;
* temperatura del reader;
* distribución de distractores;
* mapeo local contra global de clases.

Si pointer-chasing usa 256 identidades y multi-hop un espacio menor, el resultado puede reflejar margen insuficiente del codebook. Pero debe demostrarse mediante `reader top-1`, no inferirse desde la accuracy final.

Especialmente hay que comprobar que los decoders compartidos usan **IDs globalmente consistentes**. Una clase local `3` no puede significar entidades diferentes según la tarea si no existe una transformación canónica previa.

---

# 5. Sequential-update parece error elemental compuesto

Resultados alrededor de `0.46–0.53` en secuencias profundas son compatibles con un operador elemental cercano al 90%:

$$
0.9^6\approx0.53
$$

No asumiría que la recurrencia volvió a fallar. Evaluaría el checkpoint unificado sobre la tabla elemental completa:

$$
32\times32\times3=3072
$$

y reportaría:

```text
ADD one-step
SUB one-step
MUL one-step
teacher-forced por ronda
free-running por ronda
```

Si teacher forcing conserva 90–95% y free-running cae conforme aumenta H, el problema es simplemente que la interferencia redujo la ALU elemental de 100% a un valor insuficiente para composición exacta.

Aquí el sospechoso es el trunk compartido, no las heads por operación, aunque también debe comprobarse que las heads inactivas no se modifican durante batches de otras tareas.

---

# 6. El protocolo de entrenamiento no fue realmente equivalente al aislado

La frase:

```text
30k total = 5k por tarea
```

es correcta en cantidad de minibatches, pero no en cantidad de **actualizaciones sobre los pesos compartidos**.

En aislado:

```text
5,000 batches
5,000 optimizer.step() sobre el trunk
```

En U0-A:

```text
30,000 batches
30,000 optimizer.step() sobre el trunk compartido
```

El trunk recibió seis veces más actualizaciones, seis veces más aplicaciones de weight decay y un historial de Adam mezclado entre tareas.

Además, un parámetro “inactivo” puede continuar cambiando si su gradiente es un tensor de ceros en lugar de `None`. AdamW puede aplicar:

* weight decay;
* decaimiento del momentum;
* incremento del contador de pasos.

Esto afecta especialmente:

* heads por opcode;
* embeddings de opcode;
* embeddings de row type;
* adapters;
* correction head de workspace;
* decoders tipados.

### Test obligatorio de aislamiento del optimizador

Para cada tipo de batch:

1. Copiar los parámetros de los módulos que deberían estar inactivos.
2. Ejecutar `backward()` y `optimizer.step()`.
3. Comprobar que sean bit-idénticos.

También registrar:

```python
parameter.grad is None
```

No basta con:

```python
parameter.grad.norm() == 0
```

Los módulos inactivos deben tener `grad=None` o quedar excluidos del step correspondiente.

Normas, biases, embeddings, gates y pequeñas heads tipadas deberían comenzar con:

```text
weight_decay = 0
```

hasta demostrar que el decay ayuda.

---

# 7. Revisar el scheduler

Hay que confirmar que el scheduler fue configurado para:

```text
total_steps = 30,000
```

y no heredó:

```text
total_steps = 5,000
```

Si el learning rate llegó al mínimo después de 5,000 pasos globales, cada tarea habría recibido aproximadamente:

$$
5000/6\approx833
$$

actualizaciones con learning rate útil.

También deben guardar por tarea:

```text
learning rate cuando apareció el batch
loss train
accuracy train
accuracy validation
```

Sin train accuracy no podemos distinguir:

```text
no puede ajustar las seis tareas
```

de:

```text
ajusta train pero no generaliza
```

---

# 8. El control decisivo sigue siendo entrenar dentro de `unified.py` una tarea a la vez

No contra las implementaciones antiguas. Debe usarse:

```text
mismo UnifiedModel
mismo SharedMemoryReader
mismo SharedRecurrentCore
mismo TypedCommit
mismos codecs
mismo decoder
```

pero proporcionando batches de una sola tarea.

No hace falta comenzar con seis campañas. Usaría tres canarios:

```text
Workspace:
porque debe ser exactamente 0.

Variable-binding:
porque es la peor y prueba ASSIGN→ATTR.

Pointer-chasing:
porque prueba cierre recurrente en 256 identidades.
```

Interpretación:

| Resultado                        | Conclusión                                       |
| -------------------------------- | ------------------------------------------------ |
| Fallan también individualmente   | Regresión de implementación/canonicalización     |
| Pasan individualmente            | Interferencia multitarea confirmada              |
| Workspace pasa, retrieval falla  | Reader/codecs                                    |
| Retrieval pasa, sequential falla | Core/ALU                                         |
| Todas pasan individualmente      | Optimización conjunta, no arquitectura elemental |

---

# 9. Matriz de gradientes, después de las ablaciones

Con un batch fijo por tarea, calcular para reader y core:

$$
\cos(g_i,g_j)
=
\frac{g_i^\top g_j}
{\|g_i\|\|g_j\|}
$$

Separado por módulo:

```text
SharedMemoryReader
query composer
key codec
value codec
SharedRecurrentCore
candidate heads
ALU trunk
opcode embeddings
```

Además:

```text
norma de gradiente por tarea
porcentaje de pares negativos
ratio max/min de normas
```

Esto revelará si:

* una tarea domina al resto;
* retrieval y ALU empujan el trunk en direcciones opuestas;
* PAIR y ATTR interfieren;
* ACCUM_W recibe gradientes innecesarios.

No usaría todavía PCGrad o GradNorm. Primero hay que observar la matriz.

---

# 10. La corrección arquitectónica más probable

El resultado sugiere que **typed commit no es suficiente**. La tipificación debe existir también en la formación de direcciones y en la salida del núcleo.

No propongo seis modelos distintos. Propongo:

## Reader

```text
un motor compartido de atención
+
codecs canónicos por tipo semántico
+
compositores pequeños por esquema de dirección
```

Por ejemplo:

```text
unary identity key
binary reference+attribute key
vector-index key
```

No:

```text
reader por tarea
```

## Core

```text
un trunk pesado compartido
+
FiLM/LoRA/adapters pequeños por opcode
+
heads tipadas
```

Por ejemplo:

$$
F_o(x)
=
F_\theta(x)
+
A_oB_ox
$$

Los adapters deben ser pequeños respecto al trunk y compartidos por cualquier tarea que use el mismo opcode.

## Commit

Las primitivas exactas quedan protegidas:

$$
P_{r+1}=\operatorname{Canon}(Y_r)
$$

$$
E_{r+1}=\operatorname{Canon}(Y_r)
$$

$$
W_{r+1}=W_r+Y_r
$$

En U0-A, esas operaciones **no necesitan una corrección del core**.

Solo ALU necesita obligatoriamente la ruta aprendida:

$$
R_{r+1}=H_{o_r}(F_\theta(\cdots))
$$

---

# 11. El siguiente entrenamiento debe ser iso-update

Después de cerrar los diagnósticos, la comparación limpia no debería usar 30,000 optimizer steps secuenciales.

Usaría 5,000 **supersteps multitarea**:

```text
por cada superstep:
    batch pointer
    batch multi-hop
    batch associative
    batch variable-binding
    batch sequential
    batch workspace

    normalizar pérdidas o gradientes
    acumular
    un solo optimizer.step()
```

Esto proporciona:

```text
5,000 batches por tarea
5,000 actualizaciones del trunk compartido
```

igualando la cantidad de updates del baseline aislado.

Si la matriz muestra conflictos fuertes, entonces probar:

* normalización de gradiente por tarea;
* PCGrad;
* adapters por opcode.

No todo simultáneamente.

---

## Respuesta directa al reporte

El resultado actual debe registrarse así:

```text
T1 primitives isolated:
CLOSED / PASS_STRONG

U0-A monolithic full sharing, D=64:
FAIL

Causa:
no localizada todavía, pero el patrón indica
reader/canonical-codec interference,
error compuesto por ronda
y contaminación de caminos exactos.

Más steps:
NO AUTORIZADO

Más semillas:
NO AUTORIZADO

D=128:
NO todavía

Ablaciones sobre checkpoint actual:
AUTORIZADAS

Single-task through unified.py:
AUTORIZADO

U0-A1 con typed adapters/iso-update:
solo después del diagnóstico
```

La conclusión arquitectónica provisional es:

> **Unificar no puede significar que todas las semánticas atraviesen indiscriminadamente la misma transformación. El trunk puede ser único, pero las identidades, direcciones, escrituras y operaciones deben conservar interfaces tipadas y caminos exactos.**

El dato del workspace demuestra que la integración actual dejó que el aprendizaje compartido tocara una operación que debía permanecer invariante. Variable-binding muestra que la composición entre tipos de memoria no está cerrada. Esos dos problemas deben resolverse antes de interpretar el resultado como falta de capacidad del núcleo.

No puedo señalar una línea concreta porque `unified.py` no está adjunto aquí; el source actualmente disponible es el laboratorio C++ de T0-M para Windows/AVX2, con sus dimensiones, slots y profundidad, no la implementación PyTorch de U0-A. 

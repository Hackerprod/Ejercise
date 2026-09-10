## Decisión

**Sí: T1-U0-A queda `CLOSED / PASS_STRONG`.**
Autorizo **U0-B primero** y **U0-C inmediatamente después**. No los ejecutaría en paralelo porque U0-B debe certificar causalmente el checkpoint antes de modificarlo o añadir capacidad.

## Qué quedó demostrado en U0-A

El resultado ya no es “seis modelos que funcionan por separado”. Existe:

* un único checkpoint;
* un único `SharedMemoryReader`;
* un único núcleo recurrente;
* un único `TypedCommit`;
* routing oráculo por operación, no por tarea;
* entrenamiento multitarea con un solo `optimizer.step()` por superstep;
* seis tareas coexistiendo;
* cinco semillas válidas.

Además, las fallas iniciales quedaron explicadas por causas concretas:

| Falla inicial                | Causa real                                                      |
| ---------------------------- | --------------------------------------------------------------- |
| Workspace contaminado        | Corrección no congelada + AdamW sobre componentes tipados       |
| Pointer/multi-hop degradados | PAD aleatorio, reader difuso y decoder aprendido innecesario    |
| Variable-binding a azar      | Bug del evaluador; el modelo siempre funcionó                   |
| Sequential incompleto        | Registro sin canonicalización + cobertura insuficiente + olvido |
| Joint a 5k insuficiente      | Currículo y presupuesto inadecuados                             |

La arquitectura no necesitó aumentar `D=64` para cerrar. Eso es una señal importante: el primer colapso no era evidencia de falta de capacidad general del trunk.

### Precisión para el ledger

Workspace debe registrarse como:

```text
PASS, sin degradación sistemática frente al baseline aislado post-refactor
error H6 por seed: 0.00029–0.00144
```

No como “exactamente cero” ni “bit-exacto”. Una semilla supera ligeramente el antiguo umbral de `1e-3`, pero el criterio de U0-A era conservar el baseline aislado dentro del checkpoint conjunto, y eso sí se cumplió.

También debe constar:

```text
U0-A functional coexistence: PASS_STRONG
U0-A training efficiency: todavía no evaluada
```

Necesitar 12,000 supersteps, cobertura completa de la ALU y replay 5:1 es válido para demostrar coexistencia, pero todavía no demuestra que esta arquitectura sea más barata de entrenar.

---

# U0-B — ablaciones tipadas

U0-B no requiere entrenamiento. Debe ejecutarse sobre **los cinco checkpoints congelados** de U0-A y sobre los mismos test sets sellados.

Antes de empezar, archiven:

```text
hash de los 5 checkpoints
commit del código
configuración exacta
datasets/test hashes
currículo
conteo de parámetros por módulo
resultado baseline sin ablación
```

## Matriz mínima de ablaciones

### B1 — puntero: reemplazo → residual

Cambiar temporalmente:

$$
P_{r+1}=\operatorname{Canon}(Y_r)
$$

por algo como:

$$
P_{r+1}=\operatorname{Norm}(P_r+Y_r)
$$

Resultado esperado:

* pointer-chasing profundo debe perder la frontera;
* multi-hop H3/H4 debe degradar;
* `old_pointer_mass` debe aumentar;
* associative, ALU y workspace deben permanecer prácticamente iguales.

Esta ablation debe reproducir la firma que ya se observó en P1.

### B2 — puntero congelado

$$
P_{r+1}=P_r
$$

Resultado esperado:

```text
pointer-chasing/multi-hop:
H>0 cerca del baseline imposible o azar

resto:
sin degradación material
```

### B3 — evidencia: desactivar `WRITE_E`

Para `READ_E`:

$$
E_{r+1}=E_r
$$

o escribir cero.

Resultado esperado:

* associative-recall colapsa;
* la etapa ATTR de variable-binding colapsa;
* ASSIGN en \(P\) continúa funcionando;
* pointer, ALU y workspace no cambian.

En variable-binding deben reportarse separadamente:

```text
ASSIGN accuracy
ATTR | reference correcta
end-to-end
```

### B4 — registro: eliminar canonicalización

Cambiar:

$$
R_{r+1}
=
\operatorname{softmax}(\ell_r)E_{\mathbb Z_{32}}
$$

por el vector continuo sin re-embedding.

Resultado esperado:

* H1 puede seguir alto;
* teacher forcing puede seguir alto;
* free-running debe caer rápidamente con H;
* ADD/SUB deben degradar más que MUL;
* ninguna tarea no-ALU debe cambiar.

Esta es una ablation especialmente fuerte porque posee una firma ya conocida.

### B5 — registro: usar head incorrecta

Permutar cíclicamente:

```text
ADD → SUB
SUB → MUL
MUL → ADD
```

Resultado esperado:

* tabla H1 y composición ALU colapsan;
* retrieval y workspace permanecen iguales.

### B6 — workspace: residual → reemplazo

Cambiar:

$$
W_{r+1}=W_r+Y_r
$$

por:

$$
W_{r+1}=Y_r
$$

Para evidencias gaussianas independientes, el coseno esperado respecto a la suma completa cae aproximadamente hacia:

$$
\frac{1}{\sqrt H}
$$

Por ejemplo:

```text
H=4 → ~0.50
H=6 → ~0.41
```

El resto de las tareas debe conservarse.

### B7 — workspace congelado

$$
W_{r+1}=W_r
$$

Resultado esperado:

```text
workspace cosine ≈ 0
```

con las demás tareas intactas.

### B8 — eliminar la ruta identidad payload→workspace

Mantener congelada la corrección y retirar:

$$
+Y_r
$$

Resultado esperado:

* workspace falla por completo;
* demuestra que la precisión no procede de una ruta oculta del trunk.

### B9 — payload de reader puesto a cero

Para instrucciones `READ_P`, `READ_E` y `ACCUM_W`:

$$
Y_r=0
$$

Resultado esperado:

* retrieval y workspace colapsan;
* sequential-update permanece exacto, porque sus operandos pertenecen al instruction tape y no al reader.

## Criterio de paso de U0-B

No todas las ablaciones tienen que llevar exactamente al azar. Deben producir la **firma causal prevista**.

El gate pasa si:

1. Cada ablation degrada fuertemente únicamente las tareas que dependen de esa primitiva.
2. Las tareas no relacionadas cambian como máximo aproximadamente `0.5 pp`.
3. Las cinco semillas presentan el mismo patrón cualitativo.
4. No se reentrena después de la ablation.
5. Ninguna cabeza final obtiene acceso adicional al input para compensarla.

U0-B cerrará una pregunta diferente a U0-A:

$$
\boxed{
\text{el checkpoint no solo funciona;
usa causalmente las reglas tipadas declaradas}
}
$$

---

# U0-C — activar la corrección aprendida de W

Después de U0-B, debe activarse la rama actualmente congelada:

$$
W_{r+1}
=
W_r+Y_r+C_\theta(Y_r,\operatorname{Norm}(W_r),I_r)
$$

Pero la nueva tarea debe hacer imposible resolver todo con la identidad.

## Tarea recomendada: acumulación transformada

Cada ronda recupera un vector gaussiano:

$$
e_r\in\mathbb R^{64}
$$

La instrucción `ACCUM_W` incluye un `transform_id`, y el objetivo es:

$$
W_H
=
\sum_{r=1}^{H}A_{\tau_r}e_r
$$

donde \(A_\tau\) son transformaciones fijas, por ejemplo:

```text
τ=0: identidad
τ=1: negación
τ=2: permutación circular de coordenadas
τ=3: permutación por pares con cambios de signo
```

Conviene que sean transformaciones ortogonales y preserven norma. Así se evita reabrir problemas de escala.

El commit continúa siendo único:

$$
W_{r+1}
=
W_r+e_r+C_\theta(e_r,\operatorname{Norm}(W_r),\tau_r)
$$

La corrección ideal es:

$$
C_\theta
=
(A_{\tau_r}-I)e_r
$$

Para identidad:

$$
C_\theta=0
$$

Para las demás transformaciones, la rama debe actuar.

## Restricciones

* Un solo `correction_mlp`.
* Condicionado por `transform_id` o su embedding.
* No crear un MLP completo por transformación.
* No implementar \(A_\tau\) directamente dentro del commit.
* El commit solo conoce que debe acumular.
* Las matrices objetivo existen únicamente en el generador/evaluador.
* El reader continúa haciendo una sola lectura por ronda.
* `R<H` sigue impidiendo observar las evidencias no leídas.

## U0-C0 — learnability aislada

Partir de uno de los checkpoints U0-A y congelar:

```text
reader
core principal
typed commits
ALU
codebooks
decoders
```

Entrenar solo:

```text
correction_mlp
embedding de transform_id
```

Primero una semilla.

Criterios:

```text
Identidad:
conservar el error del workspace original

Transformaciones:
cosine > 0.999

error normalizado:
≤0.01 como gate causal inicial

R<H:
degradación predecible

correction=0:
solo identidad funciona
```

El umbral `1e-3` puede conservarse como meta posterior, pero no lo usaría para rechazar causalidad de una transformación aprendida no trivial.

## U0-C1 — coexistencia conjunta

Cuando C0 funcione, añadir esa tarea a los supersteps multitarea:

```text
6 tareas actuales
+ acumulación transformada
= 7 batches por superstep
→ un optimizer.step()
```

La corrección debe activarse únicamente en su opcode/configuración. En las seis tareas originales:

```text
correction grad debe ser None o exactamente aislado por dispatch
```

Criterio:

* Las seis tareas U0-A conservan sus resultados.
* La nueva tarea supera el gate.
* Ablacionar la corrección destruye únicamente transformaciones no-identidad.
* Permutar `transform_id` degrada la salida.
* No aparece drift del workspace en `ACCUM_RAW`.

Primero seed 101; las otras cuatro semillas solo después de obtener una corrida limpia.

---

# Contabilidad arquitectónica obligatoria

Antes de U0-C, registren:

```text
P_core
P_reader
P_codebooks
P_decoders
P_ALU_heads
P_typed_adapters
P_correction
```

Y las proporciones:

$$
\frac{P_{\text{typed}}}{P_{\text{core}}}
\qquad
\frac{P_{\text{total no compartido}}}{P_{\text{total}}}
$$

El pass de U0-A no debe convertirse gradualmente en varios especialistas grandes escondidos detrás de opcodes.

Las heads por operación son válidas porque representan semánticas reutilizables. La condición es:

```text
sin parámetros por tarea;
solo parámetros por primitiva/opcode;
módulos tipados pequeños frente al trunk;
mismos módulos en todas las rondas y programas.
```

---

# Qué viene después

Cuando U0-B y U0-C pasen, no corresponde todavía aprender el router. El siguiente gate debe ser composición mixta con routing oráculo dentro del mismo ejemplo:

```text
READ_P
→ READ_E
→ ALU_SUB
→ ACCUM_W transformado
→ EMIT
```

Train y test deben contener **programas distintos**, no únicamente datos distintos. Esa será la primera demostración de que el checkpoint puede cambiar de semántica durante una sola trayectoria.

## Ledger actualizado

```text
T0:
CLOSED / PASS

T1 primitivas aisladas:
CLOSED / PASS_STRONG

T1-U0-A coexistencia en un checkpoint:
CLOSED / PASS_STRONG

T1-U0-A eficiencia de entrenamiento:
OPEN

T1-U0-B causalidad tipada por ablación:
AUTHORIZED / NEXT

T1-U0-C corrección aprendida de workspace:
AUTHORIZED AFTER U0-B

Programas mixtos con routing oráculo:
PENDING

Router aprendido:
NO iniciar todavía
```

La conclusión correcta de U0-A es:

> **Un núcleo, reader y commit compartidos pueden conservar simultáneamente memoria direccionable, pointer traversal, binding, ALU composicional y workspace continuo, siempre que las representaciones discretas se canonicalicen, las escrituras exactas estén protegidas y el entrenamiento use cobertura y replay adecuados.**

Eso ya es un resultado sustancial. U0-B debe demostrar ahora que cada primitiva es causalmente necesaria; U0-C, que la parte neuronal aprendida puede extender las primitivas exactas sin contaminar el resto.

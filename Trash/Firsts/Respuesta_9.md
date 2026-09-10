**Mantendría C0-reader pendiente de precisión, pero no lo interpretaría como una falla del mecanismo ni entrenaría más el corrector todavía.** La siguiente acción debe ser una auditoría de inferencia acotada.

Hay tres puntos que cambian la lectura:

1. **Atención estable no implica errores independientes.**
2. **El error combinado no demuestra por sí solo que el corrector “no generaliza al ruido”.**
3. **Encontré en el código publicado un riesgo concreto: el reader usado podría no ser el checkpoint entrenado de U0-A.** Esto debe comprobarse primero.

## 1. Alcance de la revisión del repositorio

Pude acceder al repositorio, pero el commit `a1588c4` no fue resoluble mediante GitHub durante esta revisión. La rama publicada que pude consultar apunta a `c4a66d89…`; en ella aparecen las campañas C0-oracle, pero no `u0c_c0_reader_seed101`. Por tanto, **revisé el reader y el entrenamiento C0-oracle disponibles, pero no pude auditar el ejecutable de evaluación ni recomputar los cuatro resultados de tu campaña nueva**.

Las conclusiones numéricas siguientes usan tu reporte. Los hallazgos de código corresponden explícitamente a esa versión accesible.

## 2. Primera comprobación: ¿“reader real” significa realmente el reader validado de U0-A?

En `train_u0c_c0_oracle.py`, `C0OracleModel` hace:

```python
self.frozen_base = UnifiedT1U0(dimension)

for parameter in self.frozen_base.parameters():
    parameter.requires_grad_(False)
```

Eso **construye una base nueva y la congela**. No carga automáticamente el checkpoint que pasó U0-A. En el entrenamiento publicado tampoco encontré una carga posterior del checkpoint U0-A: se construye `C0OracleModel`, se entrena la corrección y se guarda todo su `state_dict`.

Para C0-oracle esto no invalida el resultado: la base congelada no participa en el cálculo y el payload llega del oráculo.

Pero para C0-reader existe esta posibilidad:

```text
Cargar final.pt de C0-oracle
→ utilizar model.frozen_base.memory_reader
→ evaluar un reader inicializado y congelado,
  no el reader entrenado que cerró U0-A.
```

**No estoy afirmando que eso haya ocurrido en tu nueva campaña**, porque no pude leer su script. Estoy señalando una posibilidad concreta que el código publicado deja abierta.

Antes de interpretar cualquier crecimiento con H, comprobaría igualdad tensor por tensor contra el checkpoint U0-A seed101 para:

```text
reader.query
reader.input_norm
reader.condition_projection
embeddings usados por el reader
codecs/codebooks que construyen claves y consultas
```

El contrato correcto es:

```text
Reader y codecs: checkpoint U0-A validado.
Corrector: checkpoint C0-oracle aprobado.
```

Si esa igualdad ya está comprobada, este punto queda descartado inmediatamente. Si no, es la comprobación de mayor prioridad: **congelado no significa preentrenado**.

## 3. El 99.27% constante no descarta mezcla correlacionada

Tienes razón en algo: ese dato **no parece la firma de un reader cuya atención se vuelve progresivamente más difusa**.

Pero mi hipótesis anterior de compensación entre lecturas no exigía que la nitidez empeorara con H. Puede ocurrir con exactamente la misma masa sobre el target en cada ronda.

### Contraejemplo exacto

Sean dos evidencias \(e_1,e_2\), con fuga constante:

$$
\varepsilon=1-0.9927=0.0073.
$$

El reader devuelve:

$$
Y_1=(1-\varepsilon)e_1+\varepsilon e_2,
$$

$$
Y_2=(1-\varepsilon)e_2+\varepsilon e_1.
$$

**Ambas lecturas tienen exactamente 99.27% de atención sobre su evidencia correcta.**

Para suma sin transformar:

$$
Y_1+Y_2=e_1+e_2.
$$

El error final es cero: las contaminaciones se compensan.

Ahora pedimos identidad en la primera ronda y negación en la segunda:

$$
W^*=e_1-e_2.
$$

La salida es:

$$
\widehat W=Y_1-Y_2
=(1-2\varepsilon)(e_1-e_2).
$$

Su error relativo es:

$$
\frac{\|\widehat W-W^*\|}{\|W^*\|}
=2\varepsilon
=0.0146.
$$

**Resultado: 1.46% de error, sin ninguna degradación de la atención con la profundidad.**

No afirmo que tus lecturas tengan exactamente esta estructura. El ejemplo demuestra que:

> **Una diagonal de atención estable no permite distinguir ruido independiente de contaminación cruzada entre las mismas evidencias.**

Tampoco significa que se haya perdido catastróficamente toda la información. Con atención tan concentrada, podemos estar ante una distorsión pequeña y corregible, no ante el caso extremo de mezclar todo uniformemente.

## 4. Los errores independientes no justifican automáticamente subir el umbral con H

Definamos el error de lectura de una ronda:

$$
\eta_r=Y_r-e_{q_r}.
$$

Después de la transformación exacta:

$$
u_r=A_{\tau_r}\eta_r.
$$

La condición 3 produce un error final:

$$
E_H=\sum_{r=1}^{H}u_r.
$$

Su energía cumple:

$$
\mathbb E\|E_H\|^2
=
\sum_r\mathbb E\|u_r\|^2
+
2\sum_{r<s}\mathbb E\langle u_r,u_s\rangle.
$$

Si los errores fueran centrados, no correlacionados y de magnitud similar, el error absoluto crecería aproximadamente como \(\sqrt H\).

**Pero el target también crece aproximadamente como \(\sqrt H\)** cuando sumamos evidencias gaussianas independientes transformadas ortogonalmente. Bajo esas condiciones:

$$
\frac{
\operatorname{RMS}(\text{error})
}{
\operatorname{RMS}(\text{target})
}
$$

debería permanecer aproximadamente constante.

Por eso:

```text
H1: 0.0084
H6: 0.0131
```

no queda explicado simplemente diciendo “se acumulan errores independientes”.

Puede deberse a correlación entre errores, distinta distribución de fuga hacia otras filas, sesgo, cambios en el conjunto de filas permitido o en la distribución de transformaciones. También hay que distinguir la media de errores relativos del cociente entre energías agregadas.

El evaluador C0-oracle publicado calcula **la media del cociente por muestra**:

$$
\operatorname{mean}_n
\frac{\|\widehat W_n-W_n^*\|}{\|W_n^*\|},
$$

no un cociente de energías. Eso debe conservarse como métrica del gate, añadiendo la segunda como diagnóstico, sin sustituirla silenciosamente.

## 5. La condición 4 no prueba todavía mala generalización del corrector

El corrector ya tiene aproximadamente 0.7% de error incluso con payloads perfectos. El reader introduce aproximadamente 1.3% en H6.

Si ambos componentes de error fueran aproximadamente ortogonales, su combinación tendría magnitud:

$$
\sqrt{0.0131^2+0.007^2}
\approx0.01485.
$$

Eso está en el mismo orden que el máximo reportado de 0.0152.

Es una **ilustración**, no una recomputación válida a partir de medias agregadas. Pero basta para mostrar que:

> “El sistema completo es peor que reader + transformación exacta” no demuestra una incapacidad adicional para procesar payloads ruidosos.

Puede ser simplemente la combinación de dos aproximaciones.

### El código publicado permite una comprobación todavía más directa

La configuración de `u0c_c0_oracle_seed101` declara que se entrenaron la base del payload y el embedding de transformación, mientras la rama MLP residual permaneció en cero y congelada.

En `TransformCorrectionMLP`, esa base es una proyección lineal aplicada al payload condicionado multiplicativamente por el embedding. **Para una transformación fija, la actualización aprendida es afín en el payload**:

$$
\widehat A_\tau Y+b.
$$

La dependencia no lineal de \(W\) queda anulada si la rama residual sigue exactamente en cero.

También existe una campaña separada `_network_trainable`. Por eso hay que registrar el hash y la configuración exacta del corrector usado en C0-reader, sin mezclar ambas variantes.

Si se utilizó el C0-oracle oficial con rama residual congelada, podemos escribir:

$$
Y=e+\eta,
\qquad
\widehat A_\tau=A_\tau+D_\tau.
$$

Entonces:

$$
\widehat A_\tau Y+b-A_\tau e
=
\underbrace{A_\tau\eta}_{\text{lectura}}
+
\underbrace{D_\tau e+b}_{\text{aproximación del operador}}
+
\underbrace{D_\tau\eta}_{\text{interacción}}.
$$

No hace falta atribuirlo a una nueva falla de generalización antes de medir esos términos.

---

# 6. La siguiente auditoría: sin entrenamiento y sobre el mismo checkpoint

Haría **tres comprobaciones**, no otra campaña de optimización.

## A. Procedencia y selección real de filas

Después de verificar que el reader procede de U0-A, registrar por ronda:

| Métrica                                                  | Qué distingue                                        |
| -------------------------------------------------------- | ---------------------------------------------------- |
| `argmax(attention) == requested_row`                     | Selección de dirección correcta                      |
| Masa sobre la fila correcta: media, mínimo y percentiles | Fugas pequeñas frente a casos excepcionalmente malos |
| \(\|Y_r-e_{q_r}\|/\|e_{q_r}\|\)                          | Error real del payload                               |
| Masa asignada a cada fila no solicitada                  | Contaminación recurrente por las mismas evidencias   |

**99.27% de masa media no equivale a 99.27% de accuracy de selección**, ni demuestra que no exista ninguna lectura con top-1 incorrecto.

La pregunta decisiva es: ¿la fila correcta ya es el máximo en todas las lecturas, pero el softmax la mezcla innecesariamente?

## B. Ablación soft frente a top-1

El reader publicado calcula:

```python
attention = torch.softmax(safe_logits, dim=-1)
payload = torch.einsum("bm,bmd->bd", attention, values)
```

Por tanto, aunque seleccione correctamente el máximo, el payload contiene contribuciones de otras filas.

Evaluar una variante diagnóstica:

```python
selected = attention.argmax(dim=-1)
payload_hard = memory_values[
    torch.arange(memory_values.shape[0], device=selected.device),
    selected,
]
```

**La selección sale de los scores del modelo, nunca del índice target del evaluador.** Se conservan las mismas filas legales y la misma consulta.

Repetir únicamente las condiciones 3 y 4:

| Resultado                                      | Interpretación                                              |
| ---------------------------------------------- | ----------------------------------------------------------- |
| Top-1 correcto y condición 3 cae cerca de cero | Selección aprendida correcta; el problema es mezcla soft    |
| Condición 4 vuelve al nivel C0-oracle          | No hay evidencia de fragilidad adicional del corrector      |
| Top-1 falla en algunas rondas                  | Hay un problema de direccionamiento, no solo de nitidez     |
| Top-1 correcto pero condición 3 mantiene error | Revisar values, escalas, transformaciones, padding o target |

Esto **no convierte top-1 en el nuevo reader de producción automáticamente**. Es una intervención para localizar el error.

Si después se prueba sharpening suave, hay un detalle de implementación: en el código publicado, `attention_temperature` **multiplica** los logits. Aumentarla concentra la atención; reducirla la difumina. No aplicar la convención inversa solo por el nombre del parámetro. Cualquier valor elegido debe fijarse con validación antes del test final.

## C. Descomposición exacta de los errores

Sobre cada trayectoria real, calcular:

$$
u_r=A_{\tau_r}(Y_r-e_{q_r}),
$$

$$
v_r=
Y_r+C_\psi(Y_r,W_r,\tau_r)-A_{\tau_r}Y_r.
$$

Entonces, con suma pura y el mismo enmascarado de rondas:

$$
E_{\text{completo}}=\sum_r u_r+\sum_r v_r.
$$

Registrar las tres cantidades:

$$
\left\|\sum_r u_r\right\|^2,\qquad
\left\|\sum_r v_r\right\|^2,\qquad
2\left\langle\sum_r u_r,\sum_r v_r\right\rangle.
$$

Su suma debe reconstruir el error cuadrático completo. Este test también detecta discrepancias entre las condiciones del evaluador.

Para comprobar independencia entre rondas del reader:

$$
\kappa_{\text{lectura}}
=
\sqrt{
\frac{
\sum_n\left\|\sum_r u_{n,r}\right\|^2
}{
\sum_n\sum_r\|u_{n,r}\|^2
}
}.
$$

* Cerca de 1: poca contribución neta de correlaciones cruzadas.
* Mayor que 1: los errores se refuerzan.
* Menor que 1: se compensan.

Esta es la medición que permite discutir acumulación independiente. **La nitidez media de atención no la sustituye.**

Si se está usando el corrector afín, añadiría una inspección de sus cuatro matrices efectivas y su bias. Pueden reconstruirse evaluando cada vector base y restando la salida para entrada cero. Así se mide directamente \(\|\widehat A_\tau-A_\tau\|\), sin entrenamiento ni hipótesis sobre capacidad.

## 7. ¿Hay que cambiar el criterio?

**No cambiaría ahora el umbral final de 0.01.**

Con los valores reportados:

```text
C0-oracle:
PASS en seed101.

C0-reader + transformación exacta:
no satisface precisión en H4/H6.

C0 completo:
no satisface precisión.

Causa:
todavía por separar entre procedencia del reader,
mezcla soft, correlación y error propio del operador.
```

Esto es una **falla de precisión pequeña y localizada**, no un colapso del sistema. No bloquea el diagnóstico ni obliga a rediseñar la arquitectura. Sí impide declarar cerrado el gate que se fijó.

Además, el gate final necesita un presupuesto de error para sus componentes: un corrector que consume por sí solo aproximadamente 0.7% deja poco margen a un reader que añade 1.3%. Que ambos sean “buenos” individualmente no garantiza que su composición quede por debajo de 1%.

Si una aplicación futura admite más error, se puede formular otro criterio explícito para esa aplicación. Pero no usaría “H es mayor” como justificación retrospectiva sin medir primero las covarianzas y la normalización.

## Decisión concreta

**Mantener los checkpoints congelados. No añadir steps ni semillas todavía.**

El orden es:

$$
\boxed{
\text{confirmar reader U0-A}
\;\rightarrow\;
\text{soft frente a top-1}
\;\rightarrow\;
\text{descomponer error}
}
$$

Mi hipótesis principal, condicionada a que el reader cargado sea el correcto, es **una lectura con dirección correcta pero payload ligeramente contaminado, combinada con el error de aproximación que el corrector ya tenía**. Los datos no demuestran todavía ni errores independientes benignos ni mala generalización del corrector.

La prueba top-1 puede resolver la parte central sin entrenar: **si recupera el payload exacto y el sistema completo vuelve al rendimiento C0-oracle, el siguiente cambio debe estar en la precisión de lectura, no en la capacidad del corrector.**

# Contrato experimental

## Símbolos

- `D`: dimensión de entrada y, en recurrencia real, número total de filas de salida.
- `S`: slots latentes procesados simultáneamente.
- `R`: rondas de profundidad.
- `O_i`: filas del shard del worker `i`.
- `N`: repeticiones cronometradas.
- `Q`: secuencias independientes por repetición.

Para int8:

```text
base_weight_bytes = Σ_i O_i × D
MAC_total = Σ_i O_i × D × S × R × Q × N
one_pass_weight_bytes = base_weight_bytes × R × Q × N
```

`one_pass_weight_bytes` representa una lectura de cada bloque por ronda y corrige el error recurrente de usar `GMAC/s` como `GB/s` con varios slots:

```text
one_pass_weight_GB/s = one_pass_weight_bytes / elapsed
```

Para un kernel fused que procesa `T` slots por carga:

```text
logical_weight_load_bytes = base_weight_bytes × passes(S,T) × R × Q × N
```

`logical_weight_load_bytes` describe lecturas del hot loop a través de la jerarquía; no afirma que hayan llegado desde DRAM.

## T0-R: residencia por profundidad

Configuración canónica:

```text
S = 1
transition = frozen
kernel = avx2-fused o avx2-repeat
A = shared
B = clone (bytes idénticos, direcciones distintas)
C = cold (clflush fuera del timer)
```

Invariantes obligatorios:

1. A y B tienen el mismo `weight_hash_signature`.
2. A y B producen los mismos `output_checksum` y `round_sink`.
3. En `R=1`, A y B deben estar cerca de paridad.
4. Antes de interpretar A/B como DRAM↔caché, B debe acercarse al techo read-only medido mediante `one_pass_weight_gb_per_second`.
5. El orden A/B se alterna externamente.

Umbrales históricos usados para decidir inversión:

```text
A/B >= 2.0×  → pasa
A/B >= 3.0×  → señal fuerte
```

No son leyes universales; se reporta también separación min(A)>max(B).

## T0-M: matrixización por slots

Configuración:

```text
transition = frozen
S ∈ {1,2,4,8,16}
kernel ∈ {repeat,fused}
slot_tile predeterminado = 4
```

Métricas:

```text
G(S) = MAC/s_fused(S) / MAC/s_fused(S=1)
F(S) = MAC/s_fused(S) / MAC/s_repeat(S)
```

Criterio:

```text
max(G(8), G(16)) >= 1.5× → pasa
max(G(8), G(16)) >= 2.0× → señal fuerte
```

A/B puede acercarse a 1 al crecer `S`: eso puede significar que B dejó de estar limitado por DRAM debido a la reutilización intrarronda. No se exige conservar el ratio de T0-R.

## T0-RM: recurrencia real

Requiere:

```text
Σ_i O_i = D
X(r+1) depende de Y(r)
doble buffer state_a/state_b
misma función run_kernel que T0-R/T0-M
```

Puentes obligatorios dentro del mismo ejecutable:

1. `frozen`, `S=1`: reproduce T0-R con matriz cuadrada.
2. `frozen`, `S>1`: reproduce T0-M.
3. `fixed-point`: transición local mínima.
4. `group-rms`: normalización por shard.
5. `global-rms`: normalización global paralela.

No existe un umbral único end-to-end antes de conocer el modelo. Se reportan:

- `MAC/s` y tiempo total;
- A/Bclone;
- coste de transición por worker en modo `--profile`;
- clipping;
- varianza externa;
- checksums exactos shared/Bclone.

## Semántica de Bclone

`clone` contiene `R` bloques físicamente distintos dentro de un slab alineado. Cada bloque se crea con `memcpy` desde el primero. El programa rechaza la ejecución si:

```text
hash(round) != hash(0)
address(round) == address(0)
```

`untied` genera contenidos diferentes y no se usa para medir residencia.

## Temporización

- Los workers se crean y fijan antes del timer.
- El estado se reinicia antes de cada ventana cronometrada.
- No se ejecutan autotests ni asignaciones de memoria dentro del hot loop.
- `full-repetition` mide la secuencia completa, excluyendo reset.
- `round-window` mide ronda por ronda y permite expulsar C fuera del timer.
- El modo `--profile` añade instrumentación y no debe reemplazar el resultado principal.

## Hardware objetivo inicial

Ryzen AI 5 330:

```text
4 núcleos físicos / 8 SMT
1 Zen5 + 3 Zen5c
L2 total 4 MiB ≈ 1 MiB por núcleo físico
L3 compartida 8 MiB
DDR5-5600 single-channel en la validación original
```

Para recurrencia cuadrada int8:

| D | matriz total | promedio por 4 cores |
|---:|---:|---:|
| 1280 | 1.56 MiB | 400 KiB |
| 1472 | 2.07 MiB | 529 KiB |
| 1600 | 2.44 MiB | 625 KiB |
| 1728 | 2.85 MiB | 729 KiB |

El reparto real debe seguir la calibración por núcleo, no ser uniforme.


## Invariancia del contenido frente al sharding

La matriz lógica se genera como función stateless de:

```text
(seed, global_row, column, round_key)
```

Cambiar el reparto de filas entre Zen5/Zen5c no puede cambiar ningún peso. El `weight_hash_signature` se calcula recorriendo las filas globales en orden y debe ser idéntico para cualquier partición contigua de la misma matriz.

## Selección física y sharding

El baseline usa un logical processor por núcleo físico. Dos logical processors con el mismo `physical_core_index` se rechazan salvo que la fila declare explícitamente `allow_smt_siblings=true`.

En los gates estáticos, `--average-weight-kib-per-core` expresa un presupuesto promedio, no un tamaño idéntico para cada worker. El reparto proporcional usa granularidad de una fila por defecto. La alineación del slab de pesos se resuelve mediante `block_stride_bytes` redondeado a 64 bytes; no se deben redondear las filas a múltiplos de 64 porque eso altera innecesariamente el balance Zen5/Zen5c. Para rates `19.3,18.1,10.9,17.0` y `D=1472` el reparto esperado es:

```text
435,408,246,383
```

## Escala de transición

`projection_shift` pertenece al contrato numérico, no al hardware. Debe reportarse junto con `clipped_cells`. Una comparación recurrente se rechaza si A y Bclone usan escalas distintas o si la saturación no se publica. Los scripts usan 12 como punto inicial para `D≈1472`; no se interpreta como valor entrenado u óptimo.

## Group-RMS y portabilidad

`group-rms` normaliza cada shard por separado. Es una transición válida para explorar una arquitectura alineada al hardware, pero su función matemática cambia si cambian el número o los límites de los shards. No se debe comparar calidad entre máquinas o layouts distintos usando el mismo checkpoint sin entrenarlo para esos grupos. `global-rms` conserva una definición independiente del sharding y es el baseline portable.

## Benchmark aislado de transición

`cnrl_transition_bench` ejecuta exactamente las funciones de `transitions.cpp`, con los mismos shards, double buffer y workers. No ejecuta el kernel matricial. Su función es separar:

```text
coste de transición local
coste de reducción global
coste de barreras
clipping
```

No sustituye T0-RM: una transición barata aislada puede interactuar mal con caché o sincronización cuando se alterna con GEMM.

## Revalidación de contabilidad

`analyze_results.py` no confía en los contadores emitidos. Recalcula desde `D,S,R,total_rows,sequences,timed_repetitions,slot_tile,rows`:

```text
MAC_total
base_weight_bytes
one_pass_weight_bytes
logical_weight_load_bytes
allocated_weight_bytes
rates derivadas del tiempo
```

Una discrepancia —incluido un factor accidental por repeticiones— rechaza estructuralmente la corrida.

## Q4

Q4 no forma parte del gate autoritativo actual. Consulta `Q4_STATUS.md`. Reintroducirlo antes de demostrar que su kernel puede alcanzar el régimen de memoria volvería a confundir dequantización con residencia.

# Arquitectura del proyecto

```text
Topology/Affinity ─┐
Calibration        ├── ShardSpec ── WeightBank
                   │                  ├─ shared
                   │                  ├─ Bclone
                   │                  ├─ untied
                   │                  └─ cold
                   │
State X[S,D] ── Kernel registry ── Y[S,O]
                   │
                   ├─ scalar oracle
                   ├─ AVX2 repeat
                   └─ AVX2 fused (tiles 1/2/4/8)
                                      │
                                      ▼
                          Transition registry
                          ├─ frozen
                          ├─ fixed-point local
                          ├─ group-RMS local
                          └─ global-RMS parallel
                                      │
                                      ▼
                           State next[S,D]
```

## Regla de sustitución

Cada componente tiene una interfaz estable:

- `KernelCall` recibe vistas sin ownership. Un kernel nuevo puede reemplazarse sin tocar el runner.
- `WeightBank` separa layout físico de contenido matemático.
- Las transiciones operan sobre rangos definidos por `ShardSpec`.
- `run_benchmark` no sabe cómo se implementa matemáticamente el kernel ni la transición.
- Los scripts consumen CSV, no texto humano.

## Recurrencia paralela

### fixed-point / group-RMS

```text
worker i: GEMM shard i
worker i: transición shard i
barrera
swap local de current/next
```

### global-RMS

```text
worker i: GEMM shard i
worker i: residual shard i + suma parcial[slot]
barrera
worker 0: reduce workers×slots escalares y calcula inv_rms[slot]
barrera
worker i: normaliza/recuantiza shard i
barrera
swap local de current/next
```

El worker 0 no recorre el estado completo. Su sección serial es `O(workers×S)`, mientras el trabajo repartido es `O(S×D)`.

## Layout

- Estado: slot-major, `state[slot*D + d]`.
- Output: slot-major, `output[slot*O + row]`.
- Shards: filas contiguas, permanentes por worker.
- Pesos Bclone: bloques contiguos, cada bloque con stride redondeado a 64 bytes.
- Partial sums: cada worker ocupa una estructura alineada a 64 bytes.

## Kernel fused

El tile predeterminado de cuatro slots mantiene cuatro acumuladores YMM nombrados durante toda `D`, hace una sola reducción horizontal al terminar el producto y no usa arrays indexados dinámicamente.

El tile de ocho slots permanece disponible para medir el límite, pero GCC puede derramar uno de los ocho acumuladores. Por ello no es el baseline de release.

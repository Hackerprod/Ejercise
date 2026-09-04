# Ledger de desvíos y prevención

| Desvío histórico | Por qué invalidaba | Prevención en CNRL |
|---|---|---|
| Shift 12 de fixed-point tratado como conservador | En T0-RM podía saturar una fracción material del estado | fixed usa 14 por defecto; RMS conserva 12; sweep 12–15 y clipping por trayectoria |
| Clipping de 1000 updates sintéticos interpretado como clipping de R=8 | Confundía deriva acumulativa con estabilidad del gate real | cadenas independientes con `chain_length`; reset fuera del timer; clipping T0-RM autoritativo |
| Saves XMM6–XMM15 del ABI marcados como spills | Falso FAIL del ensamblado MSVC | auditoría limitada a la ventana aritmética entre opcodes del hot loop |
| Scripts frozen fijados a D=512 | Impedía un puente standalone literalmente idéntico a D=1472 | `-D`, `-SquareOutput` y `run_exact_bridges.ps1` |
| Build fijado a VS2022/v143 | Fallaba con Build Tools/toolsets posteriores | autodetección de `vcvarsall`, Ninja y layouts single/multi-config |
| Kernel Q4→FP32 compute-bound | A/B no llegaba a medir DRAM | int8 AVX2 como oráculo físico; Q4 queda para fase posterior |
| `mac_count` multiplicado por repeticiones no cronometradas | Producía 284–823 GB/s imposibles | una sola función `calculate_mac_total`; tests con fórmula cerrada |
| Orden A→B fijo | Sesgo térmico y boost | scripts alternan el orden por repeat externo |
| 8 SMT tratados como 8 L2 | Duplicaba capacidad ficticia | topología física mediante API; un logical por core |
| A y B con pesos distintos | En recurrencia generaban trayectorias diferentes | `clone` usa `memcpy`; `untied` es una variante separada |
| Reescribir el kernel al activar recurrencia | Cambiaba la pregunta física | todos los gates validan el mismo `KernelCall` una vez y llaman `run_kernel_unchecked(config.kernel, call)` en el hot loop |
| Acumulador reducido cada 16 elementos | Añadía cadena escalar e int64 | AVX2 acumula a través de toda D; el oracle usa int64 fuera del hot path |
| Array dinámico de acumuladores | Spills a stack | especializaciones 1/2/4/8; tile 4 auditado y predeterminado |
| D=512 cuadrado en T0-RM | Solo 64 KiB/core; no reproducía L2 | default recurrente D=1472; scripts cubren 1280–1728 |
| Transición completa en hilo principal | 4 workers esperaban | transición local por shard; líder solo reduce workers×S escalares |
| Estado sobrescrito in-place | Riesgo de leer valores de la ronda siguiente | `state_a/state_b` y swap después de barrera |
| RMS “vectorizado” con checks escalares/double redundante | Coste no representativo | camino AVX2 único, sin overflow checks por fragmento |
| `Y + X` sin escala | Residual dominado por acumulador | `projection_shift`, multiplicadores y target RMS explícitos |
| `clflush` dentro del timer | C medía el desalojo, no el kernel frío | variante cold exige `round-window`; flush antes del timestamp |
| Autotest de decenas de GiB antes de cada fila | Calentamiento y throttling | `cnrl_tests` es un ejecutable independiente |
| Checksum solo de última ronda | Rondas anteriores podían no ser observables | `round_sink` consume una celda de cada ronda; recurrencia consume Y completo |
| GMAC/s interpretado como GB/s con S>1 | Error de factor S | `one_pass_weight_gb_per_second` separado de carga lógica |
| Benchmark `memcpy` llamado techo de lectura | Contaba read+write | `cnrl_bandwidth --mode read` mide payload read-only; copy se etiqueta aparte |
| Sharding uniforme en Zen5/Zen5c | El worker lento fijaba la ronda | `cnrl_calibrate` + reparto proporcional por rates |
| Instrumentación mezclada con resultado principal | Inflaba transición y varianza | `--profile` opt-in; métricas principales no usan timers por fase |
| Comparación T0-R y T0-RM en binarios distintos | Harness confound | un solo `cnrl_gate` y un solo runner |
| Pesos dependientes del sharding | Cambiar filas por core cambiaba la matriz matemática | generación stateless por `(seed, global_row, column, round)`; test de invariancia entre particiones |
| Orden de slots siempre ascendente | Sesgo térmico podía contaminar G(S) | orden de tamaños, profundidades y slots rotado/revertido por repeat externo |
| Puentes frozen y recurrencia en fases separadas | Sesgo térmico entre tablas de retención | T0-RM intercala puente y recurrencia por cada celda `(S,R)` |

## Desvíos deliberadamente no ocultos

- Tile 8 puede derramar un acumulador en algunos compiladores; se mantiene como estrés y se reporta. Tile 4 es el baseline.
- `group-rms` depende del particionado de shards y no es matemáticamente portable entre layouts; `global-rms` es el control portable.
- `global-rms` conserva dependencia secuencial entre rondas; el objetivo es medirla correctamente, no fingir que puede eliminarse.
- El tráfico DRAM real no se infiere desde caché. `one_pass` es demanda lógica corregida; para tráfico físico se requieren contadores uProf.
- El proyecto valida hardware. No afirma todavía calidad lingüística, estabilidad de entrenamiento ni equivalencia con Transformers.

| Desvío | Efecto | Prevención actual |
|---|---|---|
| Forzar `rows` a múltiplos de 64 | Distorsionaba el reparto Zen5/Zen5c | granularidad predeterminada de 1 fila; stride del slab alineado aparte |
| Seleccionar dos siblings SMT como si fueran dos núcleos | Duplicaba workers sin duplicar L2 | `physical_core_index` en CSV y rechazo por defecto |
| Confiar en `mac_total` emitido | Un bug de repeticiones podía inflarlo 8× | analizador recalcula toda la contabilidad desde dimensiones crudas |
| Calibrar núcleos una sola vez en orden fijo | Sesgo térmico en rates y sharding | tres pases: directo, inverso y rotado; se usa mediana |
| Medir transición solo dentro del sistema completo | No aislaba coste ni barreras | `cnrl_transition_bench` independiente |
| Reintroducir Q4→FP32 como gate | Kernel compute-bound ocultaba memoria | Q4 archivado hasta cumplir el contrato de `Q4_STATUS.md` |

# Auditoría de entrega — v0.4.0

Esta entrega es un laboratorio T0 para validar hardware y runtime. No es todavía un modelo lingüístico, un entrenador ni una implementación de memoria semántica externa.

## Matriz de construcción ejecutada desde cero

| Toolchain | Configuración | Resultado |
|---|---|---|
| GCC 14 | Release, `-O3`, AVX2/FMA, warnings como errores | PASS |
| Clang 17 | Release, `-O3`, AVX2/FMA, warnings como errores | PASS |
| GCC 14 | Debug, ASan + UBSan, warnings como errores | PASS |

En las tres configuraciones, CTest aprobó:

1. `cnrl_tests`;
2. `cnrl_analyzer_tests`;
3. `cnrl_cli_pipeline_tests`.

## Corrección numérica y estructural verificada

- scalar, `avx2-repeat` y `avx2-fused` producen salidas exactas para dimensiones con y sin cola SIMD y `S={1,2,4,8,16}`;
- `fused` y `repeat` usan la misma matriz y el mismo layout de estado;
- `shared` y `Bclone` producen exactamente los mismos output/state/round checksums bajo frozen, fixed-point, group-RMS y global-RMS;
- `Bclone` usa bytes idénticos, hashes iguales y direcciones físicas distintas;
- `untied` es una variante separada y no puede entrar accidentalmente en una tabla de residencia;
- la matriz lógica depende de `(seed, global_row, column, round)`, no del worker: cambiar el particionado de shards no cambia ningún peso;
- el estado recurrente usa doble buffer;
- ninguna transición S×D se ejecuta en el hilo coordinador;
- global-RMS distribuye el trabajo vectorial y serializa únicamente la reducción de `workers×S` escalares;
- group-RMS está documentado como dependiente del layout; global-RMS es el control portable;
- el reset del estado, la construcción de pesos y los self-tests permanecen fuera de la ventana cronometrada;
- el control cold usa `clflush` fuera de la ventana de kernel por contrato.

## Contabilidad y pipeline de datos

El analizador recomputa desde campos primarios:

- `mac_total`;
- bytes base y asignados;
- pasadas físicas del kernel `repeat/fused`;
- `logical_weight_load_bytes`;
- `one_pass_weight_bytes`;
- tasas derivadas;
- celdas recurrentes actualizadas y clipping.

También rechaza:

- MAC inflados;
- pares incompletos o desbalanceados shared/Bclone;
- pares incompletos o desbalanceados repeat/fused;
- checksums divergentes;
- versiones de proyecto mezcladas;
- listas CPU/core/rows malformadas;
- siblings SMT no declarados.

`test_cli_pipeline.py` ejecuta el binario real, genera CSV, pasa el analizador estricto y confirma que una copia manipulada es rechazada.

## Auditoría de ensamblado ejecutada

Con objetos GCC y Clang independientes:

- `fused4` contiene expansión signed-int8 y `vpmaddwd`;
- `fused4` no contiene accesos XMM/YMM contra stack;
- las transiciones contienen las rutas AVX2 de multiplicación/shift, packing saturado, conversión RMS y redondeo.

Tile 4 es el baseline. Tile 8 queda disponible como estrés porque puede derramar registros dependiendo del compilador.

## Correcciones encontradas durante la revisión final

1. Una versión intermedia imponía granularidad de 64 filas al sharding. Se eliminó; el baseline reparte con granularidad de una fila.
2. Una versión intermedia generaba pesos a partir de `worker_index`; eso cambiaba la matriz cuando cambiaba el sharding. Se sustituyó por generación stateless por coordenada global y se añadió un test específico.
3. Los sweeps recorrían slots siempre en el mismo orden. Ahora rotan/revierten tamaños, profundidades, slots, variantes y kernels entre repeats.
4. Los puentes frozen y T0-RM estaban en fases térmicas separadas. Ahora se intercalan por celda `(S,R)` y se invierte el orden entre repeats.
5. Las auditorías MSVC suponían una ruta fija de objetos CMake. Ahora localizan recursivamente el objeto único de la configuración solicitada y resuelven `dumpbin.exe` mediante PATH o `vswhere`.
6. El CSV no identificaba por completo la trayectoria recurrente. Ahora registra versión, seed, política de afinidad y todas las escalas de transición.
7. El analizador aceptaba ciertas entradas malformadas mediante excepción no controlada. Ahora las convierte en rechazo estructural explícito.

## Lo que no puede certificarse en este entorno

No se ejecutó MSVC ni el Ryzen AI 5 330 desde el contenedor Linux. La ruta autoritativa de Windows está incluida en `scripts/run_all_gates.ps1`, pero debe ejecutarse en la laptop para certificar:

- compilación MSVC `/O2 /arch:AVX2`;
- topología real 1×Zen5 + 3×Zen5c;
- afinidad y sharding de esa sesión;
- ensamblado `dumpbin` de los objetos MSVC;
- ancho de banda DDR5 single-channel;
- tasas T0-R/T0-M/T0-RM y varianza térmica reales.

Tampoco se afirma todavía:

- que T0-RM conserve las ganancias aisladas;
- calidad lingüística o de coding;
- estabilidad de entrenamiento;
- Q4 de producción;
- recuperación semántica externa;
- equivalencia con un Transformer.

## Condición de aceptación en la laptop

La corrida se considera evidencia únicamente si se archivan juntos:

- hash del ZIP o commit;
- build MSVC limpio y `ctest` aprobado;
- auditorías `dumpbin` de kernels y transiciones;
- topología, calibración y bandwidth de la misma sesión;
- CSV crudos con repeats alternados;
- `analysis.md` sin errores estructurales;
- modo de energía, temperatura y configuración de RAM.

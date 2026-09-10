# CPU Native Recurrence Lab (CNRL)

Laboratorio reproducible para validar, en CPU convencional, los dos mecanismos físicos que motivan una arquitectura recurrente residente:

1. **T0-R — reutilización en profundidad:** el mismo bloque de pesos se reutiliza entre rondas y permanece en caché.
2. **T0-M — matrixización por slots:** una carga de pesos alimenta varios estados latentes, transformando GEMV en small-GEMM.
3. **T0-RM — recurrencia real:** `X(r+1)` depende de `Y(r)` mediante una transición explícita, paralela y doble-buffered.

Este repositorio **no es todavía un LLM**. No incluye tokenizer, entrenamiento, memoria externa semántica ni LM head. Su función es cerrar correctamente el contrato de hardware antes de invertir en esas fases.

## Qué corrige

El proyecto se escribió desde cero para impedir los desvíos que invalidaron probes anteriores:

- `shared` y `Bclone` usan exactamente los mismos bytes; `Bclone` solo cambia las direcciones físicas.
- `untied` es una variante distinta y nunca se usa como control de residencia.
- T0-R, T0-M y T0-RM llaman al **mismo registro de kernels prevalidado**; la validación se hace una vez antes de crear los workers y el hot loop usa `run_kernel_unchecked` sin cambiar de implementación.
- La transición ocurre dentro de los workers fijados a núcleos físicos; el hilo coordinador no procesa `S×D` datos.
- El estado es doble-buffered: ningún worker sobrescribe datos aún leídos por otro.
- `global-rms` solo serializa la reducción de `workers×slots` escalares; toda la transición vectorial permanece repartida.
- El control frío usa `clflush` fuera de la ventana cronometrada.
- Los autotests nunca se ejecutan dentro del benchmark.
- El conteo de MAC y bytes se calcula una vez en la biblioteca y se prueba automáticamente.
- `GMAC/s` no se interpreta como `GB/s` cuando `S>1`; el CSV incluye `one_pass_weight_gb_per_second`.
- El tile AVX2 predeterminado es 4. El tile 8 existe como estrés, pero puede derramar un acumulador según compilador.

## Estructura

```text
include/cnrl/       Interfaces sustituibles
src/                Kernels, pesos, transiciones, topología y runner
apps/               Gates, calibración, ancho de banda, transiciones y topología
tests/              Oráculos exactos e integración shared/Bclone
scripts/            Build y sweeps reproducibles para Windows
tools/              Auditoría de fuente y ensamblado
docs/               Contrato, fórmulas, ledger de desvíos y runbook
legacy/              Probes originales archivados; no se compilan
```

## Compilar en Windows 11

Desde una Developer PowerShell de Visual Studio 2022:

```powershell
./scripts/build_windows.ps1 -Clean
```

El script configura CMake, compila Release con `/O2 /arch:AVX2 /W4 /WX`, ejecuta `ctest` y rechaza warnings. También hay presets `windows-release` y `windows-asan`.

## Validación mínima

```powershell
./build-windows/Release/cnrl_topology.exe --json
./build-windows/Release/cnrl_tests.exe
./build-windows/Release/cnrl_bandwidth.exe --mode read --mib 256 --repetitions 4
./build-windows/Release/cnrl_calibrate.exe --passes 3
./build-windows/Release/cnrl_transition_bench.exe --D 1472 --S 8 --transition global-rms
```

Después usa las tasas de calibración reales en los sweeps. En T0-R/T0-M, `--average-weight-kib-per-core` fija un presupuesto **promedio**; el reparto proporcional puede asignar más bytes al núcleo rápido y menos al lento:

```powershell
./scripts/run_t0r.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
./scripts/run_t0m.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
./scripts/run_t0rm.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
python ./scripts/analyze_results.py ./results --output ./results/analysis.md --strict-structure
```

Para ejecutar la cadena completa y derivar automáticamente `Cpus` y `Rates` de la calibración:

```powershell
./scripts/run_all_gates.ps1
```

## Ejemplo manual: puente recurrente correcto

```powershell
./build-windows/Release/cnrl_gate.exe `
  --gate t0rm --D 1472 --S 8 --R 8 `
  --kernel fused --slot-tile 4 `
  --variant clone --transition global-rms `
  --cpus 0,2,4,6 --rates 19.3,18.1,10.9,17.0 `
  --projection-shift 12 --target-rms 32 `
  --warmup 2 --repetitions 10
```

`D=1472` produce una matriz cuadrada de aproximadamente 2.07 MiB int8 total. Con rates `19.3,18.1,10.9,17.0`, el reparto exacto es `435,408,246,383` filas. No se redondea a múltiplos arbitrarios de 64; cada slab de pesos ya tiene stride físico alineado a 64 bytes.


## Selección física de workers

El gate rechaza por defecto CPUs lógicas que pertenezcan al mismo núcleo físico. `--allow-smt-siblings` existe únicamente para un experimento SMT explícito y queda registrado en el CSV. El reparto por rates usa granularidad de una fila; `--row-alignment` solo debe cambiarse deliberadamente.

## Benchmark independiente de transición

```powershell
./scripts/run_transition_bench.ps1 `
  -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
```

Este ejecutable usa las mismas funciones de transición, double buffer y barreras de workers, pero no ejecuta GEMM. Permite medir o reemplazar `fixed-point`, `group-rms` y `global-rms` sin atribuir su coste al kernel.

## Variantes de pesos

| Variante | Contenido por ronda | Dirección por ronda | Uso |
|---|---|---|---|
| `shared` | idéntico | misma | A: residencia recurrente |
| `clone` | idéntico | distinta | Bclone: control físico limpio |
| `untied` | distinto | distinta | experimento arquitectónico, no control |
| `cold` | idéntico | misma, expulsada | C: control causal de caché |

## Transiciones

| Transición | Sincronización | Propósito |
|---|---:|---|
| `frozen` | 1 barrera/ronda | T0-R/T0-M aislados |
| `fixed-point` | 1 barrera/ronda | puente recurrente local mínimo |
| `group-rms` | 1 barrera/ronda | normalización local alineada con shards; depende del particionado físico |
| `global-rms` | 3 barreras/ronda | RMS global exacto, reducción pequeña |

`group-rms` es deliberadamente hardware-layout-dependent: cambiar el número o los límites de shards cambia los grupos matemáticos. Para comparar checkpoints o máquinas con distinto particionado, `global-rms` es el baseline portable.

Todas usan escalas explícitas. Nunca se suma directamente un acumulador `int32` a un estado `int8` sin `projection_shift`. El valor no es una constante universal: debe registrarse junto con `clipped_cells`; los scripts recurrentes usan 12 como punto inicial conservador para `D≈1472`.

## Auditoría

```bash
python tools/audit_source.py
python tools/audit_assembly.py build/CMakeFiles/cnrl_core.dir/src/kernels.cpp.o
python tools/audit_transition_assembly.py build/CMakeFiles/cnrl_core.dir/src/transitions.cpp.o
```

La auditoría de ensamblado exige `vpmovsxbw` y `vpmaddwd` en `fused4`, rechaza accesos XMM/YMM a la pila dentro de esa función y verifica por separado la ruta AVX2 de escala, saturación y RMS de las transiciones.

Lee primero:

- [`docs/EXPERIMENT_CONTRACT.md`](docs/EXPERIMENT_CONTRACT.md)
- [`docs/DEVIATION_LEDGER.md`](docs/DEVIATION_LEDGER.md)
- [`docs/WINDOWS_RUNBOOK.md`](docs/WINDOWS_RUNBOOK.md)
- [`docs/AUDIT_CHECKLIST.md`](docs/AUDIT_CHECKLIST.md)
- [`docs/Q4_STATUS.md`](docs/Q4_STATUS.md)
- [`docs/DELIVERY_AUDIT.md`](docs/DELIVERY_AUDIT.md)

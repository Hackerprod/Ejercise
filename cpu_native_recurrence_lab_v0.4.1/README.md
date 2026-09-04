# CPU Native Recurrence Lab (CNRL)

Laboratorio reproducible para validar, en CPU convencional, los dos mecanismos físicos que motivan una arquitectura recurrente residente:

1. **T0-R — reutilización en profundidad:** el mismo bloque de pesos se reutiliza entre rondas y permanece en caché.
2. **T0-M — matrixización por slots:** una carga de pesos alimenta varios estados latentes, transformando GEMV en small-GEMM.
3. **T0-RM — recurrencia real:** `X(r+1)` depende de `Y(r)` mediante una transición explícita, paralela y doble-buffered.

Este repositorio **no es todavía un LLM**. No incluye tokenizer, entrenamiento, memoria externa semántica ni LM head. Su función es cerrar correctamente el contrato de hardware antes de invertir en esas fases.

## Estado de validación

La primera corrida bare-metal externa sobre un Ryzen AI 5 330 reportó **PASS** en T0-R, **PASS_STRONG** en T0-M y conservación de la separación A/Bclone en T0-RM real para `D=1472,S=1,R=8` con fixed-point, group-RMS y global-RMS. Los CSV originales de esa sesión no se incluyen en este paquete, por lo que el detalle se conserva como evidencia externa reportada en [`docs/FIRST_BARE_METAL_VALIDATION.md`](docs/FIRST_BARE_METAL_VALIDATION.md). La v0.4.1 corrige únicamente tooling, auditoría y calibración numérica detectados por esa corrida; no cambia el mecanismo físico ya validado.

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

Desde PowerShell con Visual Studio **o Build Tools** instalados:

```powershell
./scripts/build_windows.ps1 -Clean -Generator Auto
```

El script localiza `vcvarsall.bat`, importa el entorno MSVC y prefiere Ninja cuando está disponible. No fija VS2022/v143: también funciona con toolsets posteriores. Compila Release con `/O2 /arch:AVX2 /W4 /WX`, ejecuta `ctest` y escribe `cnrl-build-info.json` con generador, compilador y directorio binario. Para forzar una ruta: `-Generator Ninja` o `-Generator VisualStudio -VisualStudioGenerator "..."`.

## Validación mínima

La ubicación de los `.exe` depende del generador: Ninja los deja en `build-windows/`; un generador multi-configuración suele dejarlos en `build-windows/Release/`. Los scripts los resuelven automáticamente. La corrida autoritativa incluye por defecto los puentes standalone `D=1472`; `-SkipExactStandaloneBridges` solo existe para una ejecución abreviada.

```powershell
./scripts/run_all_gates.ps1 -Generator Auto
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

Obtén primero la ruta de `cnrl_gate.exe` desde `build-windows/cnrl-build-info.json`; después:

```powershell
& $Gate `
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

Este ejecutable usa las mismas funciones de transición, double buffer y barreras de workers, pero no ejecuta GEMM. `analyze_transition_results.py --strict` genera `transition_analysis.md` y rechaza contabilidad o semántica de reset alteradas. Cada cadena empieza desde el mismo estado inicial **fuera del timer**. `-ChainLengths 1,8` separa coste aislado y estabilidad de una trayectoria de ocho rondas. El output sintético se reutiliza dentro de cada cadena; por tanto, clipping acumulado en una cadena larga es una prueba de estrés, no sustituto del clipping observado dentro de T0-RM real.

## Puentes standalone exactos

Para reproducir fuera de `run_t0rm.ps1` exactamente la geometría cuadrada `D=1472` usada en recurrencia:

```powershell
./scripts/run_exact_bridges.ps1 -D 1472 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

`run_t0r.ps1` y `run_t0m.ps1` exponen ahora `-D` y `-SquareOutput`; el tamaño promedio en KiB queda desactivado cuando se solicita una matriz cuadrada.

## Validación dirigida de v0.4.1

Para verificar únicamente las correcciones posteriores a la primera corrida bare-metal —build adaptable, auditoría ABI, puentes D=1472, clipping por trayectoria y shifts fixed 12–15— sin repetir toda la campaña original:

```powershell
./scripts/run_v041_patch_validation.ps1 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

Los resultados quedan en `results-v041-patch/`. Este script vuelve a ejecutar T0-RM en `R=8`, porque cambia la escala predeterminada de fixed-point; no altera los kernels T0-R/T0-M ya validados.

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

Todas usan escalas explícitas. Nunca se suma directamente un acumulador `int32` a un estado `int8` sin `projection_shift`. El valor no es una constante universal: debe registrarse junto con `clipped_cells`; los scripts recurrentes usan 14 para `fixed-point` y 12 para las variantes RMS como puntos iniciales **no entrenados** para `D≈1472`; la validez numérica se decide con el clipping de la trayectoria T0-RM, no con una cadena sintética arbitrariamente larga.

## Calibración de la transición fixed-point

`fixed-point` debe aceptarse por su clipping en la trayectoria T0-RM real, no por su velocidad aislada. El sweep dedicado mantiene shared/Bclone byte-idénticos y prueba shifts 12–15:

```powershell
./scripts/run_fixed_shift_sweep.ps1 -D 1472 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

El valor predeterminado 14 evita presentar como numéricamente válida la configuración 12 que saturaba una fracción material del estado en la trayectoria sintética auditada. Sigue siendo un parámetro de probe, no una escala aprendida.

## Auditoría

```bash
python tools/audit_source.py
python tools/audit_assembly.py build/CMakeFiles/cnrl_core.dir/src/kernels.cpp.o
python tools/audit_transition_assembly.py build/CMakeFiles/cnrl_core.dir/src/transitions.cpp.o
```

La auditoría de ensamblado exige `vpmovsxbw` y `vpmaddwd` en `fused4`, rechaza accesos XMM/YMM a la pila **dentro de la ventana aritmética** y permite los saves/restores XMM6–XMM15 del prólogo/epílogo exigidos por el ABI x64 de Windows. También verifica por separado la ruta AVX2 de escala, saturación y RMS de las transiciones.

Lee primero:

- [`docs/EXPERIMENT_CONTRACT.md`](docs/EXPERIMENT_CONTRACT.md)
- [`docs/DEVIATION_LEDGER.md`](docs/DEVIATION_LEDGER.md)
- [`docs/WINDOWS_RUNBOOK.md`](docs/WINDOWS_RUNBOOK.md)
- [`docs/AUDIT_CHECKLIST.md`](docs/AUDIT_CHECKLIST.md)
- [`docs/Q4_STATUS.md`](docs/Q4_STATUS.md)
- [`docs/DELIVERY_AUDIT.md`](docs/DELIVERY_AUDIT.md)
- [`docs/FIRST_BARE_METAL_VALIDATION.md`](docs/FIRST_BARE_METAL_VALIDATION.md)

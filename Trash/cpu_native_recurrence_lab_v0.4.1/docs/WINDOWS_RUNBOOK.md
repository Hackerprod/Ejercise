# Runbook Windows 11 — Ryzen AI 5 330

## 1. Preparación

Requisitos:

- Visual Studio o Build Tools con el workload C++; no se exige VS2022/v143.
- CMake disponible en PATH.
- PowerShell 7 o Windows PowerShell 5.1.
- Plan de energía estable; laptop conectada.
- Cerrar cargas sostenidas y registrar temperatura/energía si AMD uProf está disponible.

No ejecutar el gate principal dentro de WSL2. La topología, afinidad y temporización autoritativas son las nativas de Windows.

## 2. Build limpio

```powershell
./scripts/build_windows.ps1 -Clean -Generator Auto
```

El modo Auto importa `vcvarsall.bat` y prefiere Ninja; también acepta toolsets posteriores a v143. Debe finalizar con todos los tests aprobados y escribir `build-windows/cnrl-build-info.json`. Los binarios pueden quedar en `build-windows/` o `build-windows/Release/`; los scripts los resuelven automáticamente.

## 3. Confirmar topología

```powershell
$BuildInfo = Get-Content ./build-windows/cnrl-build-info.json | ConvertFrom-Json
$Bin = $BuildInfo.binary_directory
& (Join-Path $Bin "cnrl_topology.exe") --json
```

En el hardware objetivo deben aparecer cuatro `physical_cores`, cada uno con dos logical processors. No asumir de antemano que `0,2,4,6` son correctos: usar la salida real.

## 4. Ancho de banda

```powershell
& (Join-Path $Bin "cnrl_bandwidth.exe") --mode read --mib 256 --repetitions 4 --cpus 0,2,4,6
& (Join-Path $Bin "cnrl_bandwidth.exe") --mode copy --mib 256 --repetitions 4 --cpus 0,2,4,6
```

- `read`: referencia autoritativa para lectura secuencial.
- `copy`: informa payload y tráfico estimado read+write por separado.

## 5. Calibración heterogénea

```powershell
& (Join-Path $Bin "cnrl_calibrate.exe") --cpus 0,2,4,6 > results/calibration.csv
```

Usar `mac_per_second` en el mismo orden como `--rates`. No etiquetar manualmente Zen5/Zen5c; el reparto se deriva de rendimiento.

## 6. Gates aislados

```powershell
./scripts/run_t0r.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
./scripts/run_t0m.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
```

Los scripts rotan/revierten tamaños, profundidades, slots y variantes entre repeats externos; conservan CSV crudo más stderr. `--average-weight-kib-per-core` es un presupuesto promedio bajo sharding proporcional.

## 6.1 Puentes standalone exactos

Para repetir literalmente los puentes frozen con la misma matriz cuadrada de T0-RM:

```powershell
./scripts/run_exact_bridges.ps1 -D 1472 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

Los scripts `run_t0r.ps1` y `run_t0m.ps1` aceptan `-D` y `-SquareOutput`. `run_all_gates.ps1` ejecuta estos puentes por defecto; una corrida abreviada puede usar `-SkipExactStandaloneBridges`.

## 7. Recurrencia real

```powershell
./scripts/run_t0rm.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0" -D 1472
```

Cada celda `(S,R)` intercala sus puentes `frozen` y sus corridas recurrentes; el orden de fase se invierte entre repeats. Así la retención recurrente no se compara contra una fase térmica distante.

## 8. Análisis estructural

```powershell
python ./scripts/analyze_results.py ./results --output ./results/analysis.md --strict-structure
```

El análisis rechaza:

- filas inválidas;
- afinidad fallida;
- Bclone no idéntico o no físicamente distinto;
- divergencia de checksums entre shared/Bclone.

## 9. uProf opcional

El CSV no afirma tráfico físico de DRAM. Para confirmar:

1. Ejecutar una celda larga, por ejemplo `R=16`, `repetitions=20`.
2. Perfilar el proceso con AMD uProf/PCM si el driver expone Memory BW, L2 y L3.
3. Comparar A y Bclone manteniendo el mismo orden alternado.
4. Archivar versión de uProf, comando, contador y unidad.

## 10. Reglas de aceptación

- No aceptar un reporte sin CSV crudo.
- No aceptar GB/s calculados desde GMAC/s sin considerar S.
- No cambiar kernel, D, sharding o contenido de pesos entre A y B.
- No mezclar `--profile` con la tabla principal.
- No añadir un autotest dentro del mismo proceso antes del timer.

## 11. Benchmark de transición aislada

```powershell
./scripts/run_transition_bench.ps1 `
  -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
```

Úsalo para responder cuánto cuesta cada transición sin GEMM. No uses sus tasas para afirmar rendimiento T0-RM. El run completo invoca `analyze_transition_results.py --strict` y produce `transition_analysis.md`.

## 12. Auditoría de ensamblado MSVC

```powershell
./scripts/audit_windows_assembly.ps1
./scripts/audit_windows_transitions.ps1
```

Los scripts localizan recursivamente los objetos de `cnrl_core` dentro del árbol CMake; no dependen de una ruta Visual Studio supuesta. El script de kernels aísla `fused4`, exige `vpmovsxbw`/`vpmaddwd` y rechaza accesos XMM/YMM contra stack dentro de la ventana aritmética; saves/restores ABI en prólogo/epílogo se permiten. `fused8` se reporta aparte; no sustituye el baseline tile 4.

## 13. Sharding exacto

No pases `--row-alignment 64` por rutina. Con `D=1472`, la granularidad de una fila conserva el reparto heterogéneo. La memoria ya se redondea a 64 bytes dentro del `WeightBank`.

## 14. Semántica del benchmark de transición

`run_transition_bench.ps1` ejecuta por defecto `ChainLengths = 1,8`. Cada cadena reinicia el estado fuera del timer y reutiliza un output sintético fijo dentro de la cadena.

- `chain_length=1`: coste aislado, sin deriva acumulativa.
- `chain_length=8`: estrés numérico comparable a R=8.
- Una cadena continua de 1000 actualizaciones idénticas no es evidencia de que 94% de las celdas clipeen en T0-RM; solo demuestra deriva bajo un forzamiento constante.
- El clipping autoritativo para aceptar una transición se toma de las filas T0-RM reales con la misma D/S/R.

La ruta fixed-point siempre ejecuta comparación, saturación y packing; una celda clipeada no toma un atajo computacional más barato. Su ventaja frente a RMS procede de evitar reducción, raíz/división y barreras adicionales.

## 15. Auditoría MSVC del hot loop

La auditoría permite saves/restores de XMM6–XMM15 en prólogo/epílogo, obligatorios por el ABI x64. Solo rechaza accesos vectoriales a stack dentro de la ventana delimitada por los opcodes aritméticos del kernel.


## 16. Calibrar fixed-point

```powershell
./scripts/run_fixed_shift_sweep.ps1 -D 1472 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

El sweep prueba shifts 12–15 sobre T0-RM real. El resultado autoritativo es `clipped_cells / transition_cells` por condición. El script principal usa 14 para fixed-point y 12 para group/global RMS.


## 17. Validación dirigida de la revisión

```powershell
./scripts/run_v041_patch_validation.ps1 `
  -Cpus "0,2,4,6" -Rates "42.9,30.4,33.7,33.7"
```

Esta ruta recompila limpio y revalida auditorías ABI, puentes standalone, T0-RM R=8, cadenas de transición 1/8 y el sweep fixed 12–15. No pretende sustituir una futura campaña completa cuando cambien kernels o transiciones.

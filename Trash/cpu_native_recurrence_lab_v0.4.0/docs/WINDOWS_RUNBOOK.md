# Runbook Windows 11 — Ryzen AI 5 330

## 1. Preparación

Requisitos:

- Visual Studio 2022 con Desktop development with C++.
- CMake disponible en PATH.
- PowerShell 7 o Windows PowerShell 5.1.
- Plan de energía estable; laptop conectada.
- Cerrar cargas sostenidas y registrar temperatura/energía si AMD uProf está disponible.

No ejecutar el gate principal dentro de WSL2. La topología, afinidad y temporización autoritativas son las nativas de Windows.

## 2. Build limpio

```powershell
./scripts/build_windows.ps1 -Clean
```

Debe finalizar con todos los tests aprobados.

## 3. Confirmar topología

```powershell
./build-windows/Release/cnrl_topology.exe --json
```

En el hardware objetivo deben aparecer cuatro `physical_cores`, cada uno con dos logical processors. No asumir de antemano que `0,2,4,6` son correctos: usar la salida real.

## 4. Ancho de banda

```powershell
./build-windows/Release/cnrl_bandwidth.exe --mode read --mib 256 --repetitions 4 --cpus 0,2,4,6
./build-windows/Release/cnrl_bandwidth.exe --mode copy --mib 256 --repetitions 4 --cpus 0,2,4,6
```

- `read`: referencia autoritativa para lectura secuencial.
- `copy`: informa payload y tráfico estimado read+write por separado.

## 5. Calibración heterogénea

```powershell
./build-windows/Release/cnrl_calibrate.exe --cpus 0,2,4,6 > results/calibration.csv
```

Usar `mac_per_second` en el mismo orden como `--rates`. No etiquetar manualmente Zen5/Zen5c; el reparto se deriva de rendimiento.

## 6. Gates aislados

```powershell
./scripts/run_t0r.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
./scripts/run_t0m.ps1 -Cpus "0,2,4,6" -Rates "19.3,18.1,10.9,17.0"
```

Los scripts rotan/revierten tamaños, profundidades, slots y variantes entre repeats externos; conservan CSV crudo más stderr. `--average-weight-kib-per-core` es un presupuesto promedio bajo sharding proporcional.

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

Úsalo para responder cuánto cuesta cada transición sin GEMM. No uses sus tasas para afirmar rendimiento T0-RM.

## 12. Auditoría de ensamblado MSVC

```powershell
./scripts/audit_windows_assembly.ps1
./scripts/audit_windows_transitions.ps1
```

Los scripts localizan recursivamente los objetos de `cnrl_core` dentro del árbol CMake; no dependen de una ruta Visual Studio supuesta. El script de kernels aísla `fused4`, exige `vpmovsxbw`/`vpmaddwd` y rechaza accesos XMM/YMM contra stack. `fused8` se reporta aparte; no sustituye el baseline tile 4.

## 13. Sharding exacto

No pases `--row-alignment 64` por rutina. Con `D=1472`, la granularidad de una fila conserva el reparto heterogéneo. La memoria ya se redondea a 64 bytes dentro del `WeightBank`.

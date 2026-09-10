# Changelog

## 0.4.1

- Build Windows desacoplado de VS2022/v143: detección de Build Tools, `vcvarsall`, Ninja y layout single/multi-configuración.
- `run_t0r.ps1` y `run_t0m.ps1` exponen `-D` y `-SquareOutput`; nuevo `run_exact_bridges.ps1`, ejecutado por defecto en el run completo.
- Auditoría MSVC corregida para inspeccionar la ventana aritmética y no confundir saves/restores ABI con spills del hot loop.
- `cnrl_transition_bench` mide cadenas independientes: resetea estado fuera del timer y declara `chain_length`; `analyze_transition_results.py` audita su contabilidad por separado.
- El CSV de transición publica `clipping_rate` y distingue clipping por trayectoria de clipping acumulativo artificial.
- El analizador T0-RM publica clipping mediano y validez numérica por condición; shared/Bclone también deben coincidir en clipping.
- `fixed-point` usa shift 14 en los scripts recurrentes; las rutas RMS conservan shift 12; nuevo sweep explícito 12–15.
- Scripts resuelven binarios tanto en `build-windows/` (Ninja) como en `build-windows/Release/` (multi-configuración).
- `run_v041_patch_validation.ps1` revalida únicamente los cambios posteriores a la primera campaña bare-metal.
- Documentación actualizada con la validación conceptual: T0-R/T0-M/T0-RM físicos pueden pasar aunque fixed-point requiera calibración numérica separada.
- Registro explícito de la primera validación bare-metal externa y de sus límites de procedencia.
- El CSV del microbenchmark de transición añade versión, seed, política de afinidad y epsilon; su analizador rechaza versiones mezcladas.
- La tabla T0-RM muestra `projection_shift` para que un sweep 12–15 no produzca filas ambiguas.

## 0.4.0

- Reconstrucción modular completa de T0-R, T0-M y T0-RM.
- Un único registro de kernels para gates frozen y recurrentes.
- `Bclone` byte-idéntico en direcciones distintas; `untied` separado.
- Generación de matriz invariante al sharding mediante coordenadas globales.
- Kernels scalar, AVX2 repeat y AVX2 fused con tiles especializados.
- Transiciones fixed-point, group-RMS y global-RMS ejecutadas por workers.
- Estado doble-buffered y barrera spin worker-only.
- Topología/afinidad física, calibración heterogénea y bandwidth read-only.
- CSV auditable y analizador estricto con rechazo de manipulación.
- Sweeps térmicamente alternados e intercalación frozen/recurrente.
- Tests unitarios, pipeline end-to-end, sanitizadores y auditorías de ensamblado.
- Q4 previo archivado y explícitamente excluido del camino autoritativo.

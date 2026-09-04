# Changelog

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

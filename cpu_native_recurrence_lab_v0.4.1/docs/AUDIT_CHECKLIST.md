# Checklist de auditoría antes de aceptar un resultado

## Build

- [ ] Generador/toolset reales registrados en `cnrl-build-info.json`; ninguna adaptación local silenciosa.
- [ ] `project_version`, seed y parámetros de transición presentes en cada CSV.
- [ ] Test end-to-end CLI→CSV→analizador aprobado, incluida corrupción deliberada rechazada.
- [ ] Auditoría de ensamblado de kernels **y** transiciones aprobada.

- [ ] Release limpio con warnings como errores.
- [ ] `cnrl_tests` aprobado.
- [ ] `tools/audit_source.py` aprobado.
- [ ] Ensamblado contiene `vpmovsxbw` y `vpmaddwd`.
- [ ] Baseline tile 4 sin accesos XMM/YMM a stack dentro de la ventana aritmética; saves/restores ABI fuera de ella permitidos.

## Hardware

- [ ] Topología obtenida por API, no inferida por índices pares.
- [ ] Un worker por núcleo físico en baseline.
- [ ] Afinidad verdadera en todas las filas.
- [ ] Rates calibrados en la misma sesión.
- [ ] RAM/canales registrados.

## A/Bclone

- [ ] Matriz lógica invariante al cambiar el reparto de shards.
- [ ] Mismo número de repeats shared/clone en cada condición.
- [ ] Mismo número de repeats repeat/fused en cada condición T0-M.

- [ ] Misma D, S, R, kernel, tile, transición, filas y seed.
- [ ] Hash base idéntico.
- [ ] Bloques clone en direcciones distintas.
- [ ] Output/state/round checksums idénticos.
- [ ] `untied` no aparece en una tabla de residencia.

## Tiempo

- [ ] Orden de slots/tamaños/profundidades rotado, no siempre ascendente.
- [ ] Puente frozen y recurrencia intercalados por celda para comparaciones de retención.

- [ ] Warmup excluido.
- [ ] Reset del estado excluido.
- [ ] Threads y afinidad creados antes del timer.
- [ ] Cold usa `round-window`.
- [ ] Orden alternado.
- [ ] Al menos cinco corridas externas para una afirmación fuerte.
- [ ] Varianza y superposición min/max reportadas.

## Contabilidad

- [ ] `mac_total = rows×D×S×R×sequences×timed_repetitions`.
- [ ] `one_pass_weight_bytes = base×R×sequences×timed_repetitions`.
- [ ] No usar GMAC/s como GB/s con S>1.
- [ ] Read-only bandwidth, no solo memcpy.
- [ ] Tráfico físico se etiqueta como contador, no inferencia.

## Recurrencia

- [ ] G(S) y F(S) publicados por separado; G no se confunde con ventaja fused/repeat.
- [ ] `group-rms` se etiqueta como dependiente del layout; `global-rms` se usa como baseline portable.

- [ ] sum(rows)=D.
- [ ] State double-buffered.
- [ ] Transición ejecutada por workers.
- [ ] Global leader reduce solo workers×S valores.
- [ ] Escala residual explícita.
- [ ] Clipping reportado por trayectoria T0-RM y shared/Bclone idénticos.
- [ ] Fixed-point usa una escala calibrada o queda marcado como no válido numéricamente.
- [ ] Puentes frozen reproducen gates aislados con la misma D, filas, seed, kernel y tile.

## Auditoría adicional de entrega

- [ ] `physical_cores` no contiene duplicados salvo experimento SMT declarado.
- [ ] Sharding proporcional usa granularidad de una fila en el baseline.
- [ ] `analyze_results.py --strict-structure` recomputa y acepta todos los contadores.
- [ ] Calibración contiene al menos tres pases y reporta dispersión.
- [ ] El coste de transición se midió por separado cuando se atribuye un colapso a la transición.
- [ ] Benchmark de transición declara `chain_length` y resetea el estado fuera del timer entre cadenas.
- [ ] Q4 no se usa como evidencia hasta pasar `docs/Q4_STATUS.md`.

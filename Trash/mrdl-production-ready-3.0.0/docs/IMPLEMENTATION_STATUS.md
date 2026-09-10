# Estado de implementación frente a MRDL v3.0

Este mapa separa lo implementado y verificado de lo que sigue siendo una hipótesis experimental. La referencia normativa es `docs/MRDL-v3.0-source.md`; este documento describe el código entregado.

| Parte del documento base | Implementación | Verificación disponible | Estado |
|---|---|---|---|
| Embedding base congelado | `FrozenEmbeddingStore`, Q8 por fila, mmap, random/random-indexing/externo | checksum, repetibilidad, corrupción, vocabulario compatible | implementado |
| Relaciones vectoriales explícitas | `RelationRecord`, `RelationVector`, prototipos por par | roundtrip, actualización local, índices | implementado |
| Operador monomial y Compose | `MonomialOperator` | composición contra aplicación serial, benchmark O(d) | implementado |
| Contexto como cápsulas | `RouteCapsule`, expectativas, energía, historial | invariantes de beam, ciclos y ejecución | implementado |
| Puertos contextuales | `OnePassPortRouter` | una pasada, capacidad, máximo dos puertos, conservación de energía | implementado |
| Fold_B | `DiverseBeamFold` | cota de estado activo, diversidad y replay | implementado |
| FULL/CLEAN | dos `LaneEngine` e índices físicos separados | A/B/C, ataque M1 extremo, métricas por carril | implementado |
| M0/M1/M2 | estado contextual, escrow y grafo consolidado | promoción automática e inexistencia M1 en CLEAN | implementado |
| Replay sin árbol de evidencia | `ReplayStep` y `ReplayClosure` versionados | high-watermark, snapshots históricos, hashes y reproducción exacta | implementado |
| TTL + reserva atómica | `PromotionManager` + SQLite CAS | carrera con 64 hilos, vencimiento durante auditoría, `UNREPLAYABLE` | implementado |
| Modo B | delta local sobre relaciones M1, `stopgrad` estructural por API | no actualización de parámetros antes de promoción | implementado |
| Controlador lento | `Controller` actualizado solo con `PromotionPermit` | prueba de aislamiento y persistencia | implementado, inicial |
| Roles autoinducidos | `RoleInducer` por slots estructurales promovidos | serialización, replay histórico y permiso de promoción | implementado, inicial |
| Repetición y cierre | historia de token/ruta/arista, señales y EOS competitivo | cubierto dentro de ejecución y smoke | implementado, requiere corpus |
| Túneles efímeros consolidados | composición de transformaciones dentro de cápsula y metadata derivada | invalidación de derivados al promover | parcial: no hay caché global de túneles efímeros |
| Modo A diferenciable | no se mezcla con el camino persistente | configuración lo rechaza explícitamente | no implementado |
| Átomos no lineales 2-canales | deliberadamente ausentes | benchmark monomial conserva el límite afín | no implementado por diseño |
| Calidad lingüística | entrenamiento/eval/generate y baselines n-gram | smoke y comandos de benchmark | experimental, no validada a escala |

## Resultado mínimo reproducido en el smoke de entrega

El corpus de humo confirma el flujo operacional completo: `prepare → train → promotion → eval → generate → audit → gc → inspect → backup`. El arranque desde M2 vacío promueve relaciones mediante auditoría contrafactual; CLEAN deja de estar vacío.

Ese resultado no demuestra competitividad lingüística. En el corpus diminuto del smoke, los n-gram memorizan mejor que MRDL, lo cual es esperable y se conserva en los logs como control negativo honesto. La campaña real debe medir abstracción, sustitución, composición profunda, generalización fuera de distribución, costo dual y salud CLEAN sobre un corpus separado.

## Criterios para escalar en el VPS

1. Ejecutar primero `scripts/run_validation.sh` en el VPS sin modificar el código.
2. Preparar un corpus de entrenamiento y otro de validación sin solapamiento.
3. Registrar FULL, CLEAN, n-gram y Transformer pequeño con igual presupuesto de memoria/cómputo.
4. Separar filas `clean_degenerate` o `clean_empty`; no promediarlas como rendimiento válido.
5. Escalar dimensión, Top-K y beam de una variable por vez.
6. No habilitar átomos no lineales hasta que un benchmark de corpus confirme el límite afín como cuello de botella real.

# Validación del paquete

Este documento se actualiza con `scripts/run_validation.sh`. Los números de rendimiento son específicos de la máquina de construcción y no sustituyen el benchmark en el VPS.

## Comandos

```bash
cmake --preset debug-sanitize
cmake --build --preset debug-sanitize
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  ./build/debug/mrdl_tests

cmake --preset release-vps
cmake --build --preset release-vps
ctest --preset release-vps

./build/release/mrdl_bench --quick --suite dual
./build/release/mrdl_bench --quick --suite monomial
scripts/smoke_test.sh ./build/release/mrdl
```

## Cobertura funcional

La suite contiene 21 casos principales:

1. posición de fuente en escrow y serialización compatible;
2. rollback de artefactos ante fallo de `prepare`;
3. tokenizador reversible UTF-8 y persistencia;
4. embeddings congelados, checksum y reproducibilidad;
5. composición monomial exacta y serialización;
6. vector/registro relacional cuantizado;
7. índices físicos FULL/CLEAN y protección contra duplicados;
8. A — equivalencia con referencia filtrada;
9. B/C — no-interferencia CLEAN y aislamiento del controlador;
10. puertos de una pasada, energía conservada y límites;
11. D — beam/replay acotados;
12. replay persistente y high-watermark tras reinicio;
13. E — promoción, TTL y entrada automática en CLEAN;
14. E — dependencia faltante produce `UNREPLAYABLE`;
15. E — 64 hilos compiten y solo una reserva gana;
16. baseline n-gram persistido;
17. transacción SQLite RAII: commit, rollback explícito y rollback por destructor;
18. lock excluye segundo escritor;
19. configuración estricta, persistencia y validación;
20. arranque desde CLEAN vacío promueve únicamente mediante control contrafactual vacuo certificado;
21. flujo end-to-end prepare→train→eval→backup→reopen.

## Regresiones capturadas durante implementación

- ID de replay reutilizado tras reinicio.
- Deserialización de pares dependiente del orden de evaluación de C++.
- Baseline n-gram cargado con pares token/count intercambiables.
- Pérdida de snapshot histórico tras varias escrituras sobre el mismo M1.
- Estado activo sin cota en la primera ronda.
- Conteo de diversidad antes de elegir sobrevivientes.
- Scope guard ejecutado después de commit exitoso.
- Sanitizadores aplicados a la biblioteca pero no al ejecutable final.
- Consulta de adyacencia que copiaba y ordenaba todo el vecindario por inferencia.
- Transacciones de token accesibles solo mediante métodos privados y sin exclusión de hilos durante toda la unidad de escritura.
- Deadlock de arranque epistémico: CLEAN vacío nunca podía certificar la primera promoción M1→M2.
- Conversiones implícitas con signo en bytes UTF-8/JSON y semilla temporal, detectadas por el build limpio con Clang 17 y `-Werror`.

## Resultado de entrega

El resultado final exacto, hashes y logs están en `artifacts/validation/` dentro del paquete generado. Vuelva a ejecutar la validación en el VPS porque `-march=native`, scheduler, caché y almacenamiento cambian los ratios.

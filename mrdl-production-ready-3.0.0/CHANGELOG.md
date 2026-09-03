# Changelog

## 3.0.0 — Production-core handoff

- Implementado núcleo C++20 CPU-first con componentes sustituibles.
- Añadidos tokenizador reversible y embeddings Q8 congelados mediante mmap.
- Implementada álgebra monomial exacta, vectores relacionales y prototipos múltiples.
- Implementados cápsulas, puertos de una pasada, propagación competitiva y Fold_B.
- Implementados índices físicos FULL/CLEAN y aislamiento total de decisiones por carril.
- Implementado replay reconstruible, snapshots históricos y auditoría contrafactual exacta.
- Integrada promoción M1→M2 con actualización automática del índice CLEAN.
- Implementados TTL con pin atómico, estados `UNREPLAYABLE` y GC separado.
- Añadido entrenamiento Modo B, evaluación dual, generación certificada y n-gram baselines.
- Añadidos SQLite WAL, checksums, locking, doctor, backup y CLI operativa.
- Añadidas pruebas A–E, ASan/UBSan, benchmark dual y benchmark de operadores.
- Agrupada cada escritura de entrenamiento en una transacción RAII por token, con exclusión de hilos durante toda la unidad.
- Corregido el bootstrap M2: CLEAN genuinamente vacío puede ser control contrafactual vacuo, pero CLEAN colapsado no.
- Añadidas regresiones de transacción, rollback de preparación, replay histórico, posición causal y arranque en frío.
- Corregidas conversiones de byte/tiempo detectadas por Clang 17 y añadida una matriz CI de portabilidad GCC/Clang con `-Werror`.

# MRDL 3.0 — Modelo Relacional Disperso de Lenguaje

Implementación C++20, CPU-first y modular de la arquitectura descrita en `docs/MRDL-v3.0-source.md`.

MRDL no es un Transformer reducido ni una memoria auxiliar conectada a otro LLM. Su estado principal es un grafo disperso de relaciones vectoriales explícitas. Cada token posee un embedding base congelado; las relaciones contienen un vector multidimensional y un operador monomial componible en `O(d)`. La inferencia mantiene cápsulas de ruta activas y compite sobre un Top-K físico, sin atención densa.

## Estado real

Este repositorio está preparado para **desplegar, entrenar, auditar, medir y depurar el núcleo experimental en un VPS Linux**. Incluye persistencia transaccional, recuperación, bloqueo entre procesos, backups, validación con sanitizadores, carriles FULL/CLEAN aislados, promoción M1→M2 y benchmarks reproducibles.

No debe interpretarse como una afirmación de que la arquitectura ya alcanzó calidad lingüística competitiva. El documento base establece que el núcleo sobre corpus real era la siguiente etapa experimental. Esta implementación permite ejecutar esa etapa sin reconstruir la infraestructura ya validada.

## Propiedades implementadas

- Tokenizador reversible híbrido: piezas frecuentes + fallback por byte; nunca pierde texto UTF-8.
- Embeddings base congelados, Q8 por fila y `mmap`; modos aleatorio, random-indexing o matriz float32 externa.
- Relaciones con hasta 4 prototipos por par de nodos.
- Operador `T(z) = a ⊙ P(z) + b`, con permutación con signo, composición exacta y costo lineal.
- Vector relacional por canales: semántica, rol, tiempo, composición, continuación, cierre y confianza.
- Cápsulas de ruta, expectativas abiertas, energía, procedencia local y control de ciclos/repetición.
- Puertos contextuales con routing de una sola pasada; sin agreement iterativo.
- `Fold_B` con beam acotado y diversidad estructural.
- Índices físicos independientes: FULL contiene M1+M2; CLEAN contiene solo M2.
- M1 nunca consume Top-K, gate, puerto, normalización o beam de CLEAN.
- Carriles con estado mutable, decisiones y métricas completamente separados.
- Reutilización opcional únicamente de aplicaciones puras exactamente idénticas.
- Replay versionado sin árbol de evidencia, con snapshots de relaciones, controlador, roles y semillas.
- Reserva atómica de auditoría, pin de la clausura completa, TTL seguro y estado `UNREPLAYABLE`.
- Integración automática de promoción: al aprobarse, la relación entra en M2/CLEAN e invalida derivados.
- Modo B: escritura rápida en memoria relacional M1; parámetros compartidos solo cambian tras promoción.
- SQLite WAL, `synchronous=FULL`, checksums, backup consistente y `PRAGMA integrity_check`.
- CLI para preparar, entrenar, evaluar, generar, auditar, inspeccionar, respaldar y comparar baselines.

## Estructura modular

```text
include/mrdl/             interfaces públicas
src/tokenizer.cpp         tokenización reversible
src/embeddings.cpp        conocimiento base congelado Q8/mmap
src/relation.cpp          operadores monomiales y vectores relacionales
src/graph.cpp             almacenamiento e índices físicos FULL/CLEAN
src/controller.cpp        gate, puntuación y roles autoinducidos
src/routing.cpp           cápsulas, puertos y Fold_B
src/engine.cpp            propagación competitiva y ejecución dual
src/replay.cpp            registro reconstruible y versionado
src/promotion.cpp         M1/M2, TTL, reservas y promoción
src/persistence.cpp       SQLite/WAL y backups
src/training.cpp          entrenamiento Modo B, auditoría y generación
src/baselines.cpp         unigram/bigram/trigram/... comparables
src/diagnostics.cpp       doctor operativo
benchmarks/               costo dual y kernels monomiales
tests/                    invariantes A–E y regresiones
```

Cada bloque depende de interfaces (`IEmbeddingStore`, `IRelationStore`, `IFoldPolicy`, persistencia y recorder desacoplados). Un backend puede sustituirse sin reescribir los demás módulos.

## Requisitos

Plataforma soportada por esta versión:

- Linux little-endian de 64 bits, probado con GCC y SQLite.
- CMake 3.24 o superior.
- Compilador C++20, Ninja, SQLite3 development headers y pthreads.
- VPS recomendado para el perfil incluido: 4 vCPU, 8 GB RAM y almacenamiento local SSD/NVMe.

En Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libsqlite3-dev
```

## Compilación optimizada para el VPS

```bash
cmake --preset release-vps
cmake --build --preset release-vps -j4
ctest --preset release-vps
```

El preset `release-vps` activa `-march=native`, LTO, tests y benchmarks. El binario queda en `build/release/mrdl`.

Para instalar:

```bash
sudo cmake --install build/release --prefix /usr/local
```

Para una compilación portable entre máquinas, desactive `MRDL_NATIVE_ARCH`:

```bash
cmake -S . -B build/portable -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMRDL_NATIVE_ARCH=OFF \
  -DMRDL_BUILD_TESTS=ON
cmake --build build/portable -j4
ctest --test-dir build/portable --output-on-failure
```

## Flujo inicial

### 1. Configurar rutas

`config/vps.ini` está preparado para ejecutarse desde el repositorio y guarda el modelo en `model/`. Para una instalación del sistema use `config/vps-system.ini`, que apunta a `/var/lib/mrdl/model`.

```bash
cp config/vps.ini config/local.ini
# Ajuste vocabulario, dimensiones, beam y rutas antes de preparar el modelo.
```

Los parámetros estructurales del modelo no deben cambiar después de `prepare`. El runtime compara la configuración efectiva persistida y rechaza incompatibilidades.

### 2. Preparar tokenizador y embeddings congelados

```bash
build/release/mrdl prepare \
  --config config/local.ini \
  --corpus /ruta/corpus-train.txt \
  --embeddings random-indexing
```

Modos:

- `random-indexing`: deriva geometría distribucional del corpus una sola vez y la congela.
- `random`: control experimental para separar conocimiento relacional de semántica importada.
- `external`: consume una matriz float32 raw, row-major, con forma exacta `[vocab_size_real, embedding_dim]`.

Ejemplo externo:

```bash
build/release/mrdl prepare \
  --config config/local.ini \
  --corpus train.txt \
  --embeddings external \
  --external embeddings.f32
```

`prepare` se niega a sobrescribir un modelo existente. `--force` elimina los artefactos de ese modelo; haga backup antes.

### 3. Validar artefactos

```bash
build/release/mrdl doctor --config config/local.ini
build/release/mrdl inspect --config config/local.ini
```

### 4. Entrenar Modo B

```bash
build/release/mrdl train \
  --config config/local.ini \
  --corpus /ruta/corpus-train.txt \
  --json
```

La escritura rápida crea o actualiza relaciones M1 y sus registros en custodia. Predicción, replay, relación y escrow de cada token se confirman como una sola transacción; una excepción revierte la unidad completa. No actualiza el controlador compartido, claves persistentes de puertos ni roles globales. Las actualizaciones persistentes del controlador ocurren únicamente durante una promoción aprobada.

El primer M2 no se inserta por una regla especial. Se audita CLEAN vacío contra CLEAN+e; el control vacío solo se acepta cuando no existió relación elegible, gate, operador ni frontera podada. Un CLEAN que colapsó por falta de beam no puede usar esta vía.

### 5. Evaluar FULL y CLEAN

```bash
build/release/mrdl eval \
  --config config/local.ini \
  --corpus /ruta/corpus-validation.txt \
  --max-tokens 100000
```

La salida informa pérdida, perplexity, exactitud y vacíos de CLEAN por separado. No mezcle resultados donde `clean_empty_ratio` sea alto con filas sanas.

### 6. Generar

```bash
build/release/mrdl generate \
  --config config/local.ini \
  --prompt "Había una vez" \
  --max-tokens 80 \
  --temperature 0.7 \
  --seed 42 \
  --json
```

Cada token queda clasificado como `clean`, `provisional`, `fragile` o `empty` según la comparación FULL/CLEAN.

## Auditoría, TTL y mantenimiento

```bash
# Auditar hasta 64 candidatos M1
build/release/mrdl audit --config config/local.ini --max 64

# Expirar M1 no fijado y limpiar rechazados/irreproducibles antiguos
build/release/mrdl gc --config config/local.ini

# Persistir controlador/roles y truncar WAL
build/release/mrdl checkpoint --config config/local.ini

# Backup consistente con manifiesto y hashes
build/release/mrdl backup \
  --config config/local.ini \
  --output /ruta/backups/mrdl-$(date -u +%Y%m%dT%H%M%SZ)
```

Una auditoría reserva atómicamente el registro y toda su clausura de replay. Si el TTL vence durante la auditoría, el contenido permanece fijado hasta completarla. Si falta una dependencia histórica, la promoción termina en `UNREPLAYABLE`; nunca se reconstruye silenciosamente con la versión actual.

## Inspección y depuración

```bash
build/release/mrdl relation --config config/local.ini --id 123
build/release/mrdl neighbors --config config/local.ini --text "casa" --lane full --limit 20
build/release/mrdl neighbors --config config/local.ini --text "casa" --lane clean --limit 20
build/release/mrdl tokenize --config config/local.ini --text "área verde" --bos --eos
```

La comparación de `neighbors ... --lane full` y `--lane clean` permite comprobar directamente que una relación M1 no está materializada en el índice CLEAN.

## Baselines

Los n-gram usan exactamente el mismo tokenizador:

```bash
build/release/mrdl baseline \
  --config config/local.ini \
  --train-corpus train.txt \
  --eval-corpus validation.txt \
  --orders 1,2,3 \
  --output-dir baselines
```

El Transformer pequeño de comparación está aislado en `tools/tiny_transformer_baseline.py`; no forma parte del runtime ni del camino caliente C++.

## Benchmarks de arquitectura

```bash
build/release/mrdl_bench --quick --suite dual
build/release/mrdl_bench --quick --suite monomial
build/release/mrdl_bench --suite all --json > benchmark.jsonl
```

El benchmark dual reporta:

- `R_runtime_isolated`: FULL+CLEAN aislados / FULL-only.
- `R_runtime_parallel_reuse`: ejecución paralela con reutilización pura exacta / FULL-only.
- `R_ops`, operaciones por carril y p50/p95.
- picos de estado activo y entradas de replay.
- `clean_health`, `clean_degenerate` y `clean_empty`.

`clean_degenerate=true` significa que CLEAN conserva menos de la mitad de la actividad esperada. Esa fila es un fallo de capacidad, no una optimización.

## Validación

```bash
scripts/run_validation.sh
```

La suite cubre:

- A: equivalencia con la ejecución filtrada de referencia.
- B: no-interferencia ante M1 hostil.
- C: no-lavado mediante controlador, gates, puertos o beam.
- D: cotas de beam, estado activo y replay sin `k^D`.
- E: promoción, TTL, replay, reinicio, condición de carrera y `UNREPLAYABLE`.
- serialización/checksums, tokenizador, embeddings, n-gram, lock y flujo end-to-end.

Consulte `docs/VALIDATION.md` para los comandos y resultados del paquete entregado.

## Despliegue

- Instalación directa: `docs/VPS_DEPLOYMENT.md`.
- Persistencia y restauración: `docs/STORAGE_AND_RECOVERY.md`.
- Invariantes no negociables: `docs/INVARIANTS.md`.
- Mapa técnico: `docs/ARCHITECTURE.md`.
- Migración desde `/root/mrdl`: `docs/MIGRATION.md`.

## Límites actuales

1. La calidad lingüística real debe medirse sobre corpus, no deducirse de los tests sintéticos.
2. La familia monomial es afín; no representa XOR/producto sin átomos no lineales adicionales. Este repositorio no los añade sin un benchmark que los exija.
3. Modo A no está habilitado en este build. El camino de producción implementado es Modo B, coherente con one-shot M1 + promoción auditada.
4. El costo dual FULL+CLEAN es permanente. Debe medirse en el VPS con su densidad M1 real.
5. Los formatos persistidos son versionados pero esta versión se limita a plataformas little-endian de 64 bits.
6. No existe migración automática desde la implementación parcial anterior; se necesita su esquema/código exacto para crear un migrador seguro.

## Licencia

No se impone una licencia desde este paquete. Antes de publicar el repositorio, añada la licencia elegida por el propietario del proyecto.

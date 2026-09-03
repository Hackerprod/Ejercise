# Arquitectura implementada

## 1. Dependencias y fronteras

```text
Corpus ──> HybridTokenizer ──> TokenId
                         └──> FrozenEmbeddingStore (Q8, mmap, inmutable)

TokenId/contexto
  └──> DualLaneEngine
       ├── LaneEngine(FULL)  ──> GraphStore.full_index  (M1 + M2)
       └── LaneEngine(CLEAN) ──> GraphStore.clean_index (solo M2)
             │
             ├── RouteCapsule / ContextPort / Fold_B
             ├── Controller (solo lectura durante inferencia)
             └── MonomialOperator + RelationVector

Entrenamiento Modo B
  ├── escritura rápida ──> RelationRecord(M1) + EscrowRecord
  └── auditoría ──> replay histórico CLEAN / CLEAN+e
                     └── PromotionManager ──> RelationRecord(M2) + Controller/Role update

Durabilidad
  ├── SQLite WAL: relaciones, versiones, escrow, replay, controlador y roles
  ├── .mrdltok: tokenizador con checksum
  └── .mrdlemb: embeddings Q8 congelados con checksum
```

Las interfaces públicas permiten sustituir componentes:

- `IEmbeddingStore`: backend de embeddings.
- `IRelationStore`: backend de grafo/índices.
- `IFoldPolicy`: política de reducción del beam.
- `ReplayRecorder`: recorder en memoria o persistido.
- `SqliteModelStore`: frontera de durabilidad.

## 2. Token y conocimiento base

`HybridTokenizer` conserva cuatro tokens especiales, 256 tokens byte y un presupuesto de piezas frecuentes. Una pieza no conocida se codifica byte a byte, por lo que la operación encode→decode es reversible incluso con texto UTF-8 fuera del vocabulario.

`FrozenEmbeddingStore` mantiene una fila por token. El archivo usa escala float32 + `int8[dimension]` por fila. Se carga mediante `mmap`; no se copia toda la matriz a heap. Los embeddings nunca se modifican durante entrenamiento principal.

Modos de inicialización:

- aleatorio normalizado;
- random indexing sobre coocurrencia local del corpus;
- matriz float32 externa, normalizada antes de cuantizar.

## 3. Relación

Cada `RelationRecord` contiene:

```text
(source, destination, prototype, level, lane mask,
 support, confidence, version, timestamps, escrow state,
 MonomialOperator, RelationVector, derivation metadata)
```

### Operador monomial

```text
T(z) = a ⊙ P(z) + b
```

`P` se guarda como permutación + signo; `a` y `b` son vectores. Aplicar y componer cuesta `O(d)`. La composición implementa exactamente:

```text
P21 = P2 P1
a21 = a2 ⊙ Q2(a1)
b21 = a2 ⊙ P2(b1) + b2
```

La actualización rápida `update_delta` modifica escalas y sesgos de una relación activa; no crea un gradiente global.

### Vector relacional

`RelationChannelLayout` divide un vector compacto en siete regiones lógicas:

1. delta semántico;
2. firma de rol;
3. firma temporal;
4. firma de composición;
5. señal de continuación;
6. señal de cierre;
7. estado de confianza.

Se persiste cuantizado y se actualiza con observaciones locales.

## 4. Grafo e índices epistémicos

`GraphStore` mantiene objetos de relación versionados y tres índices:

- adyacencia FULL, ordenada por confianza/soporte/id;
- adyacencia CLEAN, materializada únicamente para M2;
- búsqueda por `(source, destination, prototype)`.

Una relación M1 no aparece en `clean_index`. La exclusión ocurre al escribir/actualizar el grafo, antes de cualquier capacidad limitada. No existe `operator=ZERO` para CLEAN.

Una promoción actualiza el mismo `RelationRecord` a M2, persiste la nueva versión y lo inserta automáticamente en CLEAN. Las relaciones derivadas que dependan de la versión anterior se invalidan.

## 5. Estado contextual

La unidad activa es `RouteCapsule`, no una activación escalar por nodo. Incluye:

- nodo actual;
- estado contextual vectorial;
- bindings de rol;
- expectativas abiertas;
- energía;
- firma de ruta;
- padres y contribuciones locales;
- historial corto para ciclos.

Dos cápsulas pueden ocupar el mismo nodo sin fusionar contexto ni procedencia.

## 6. Routing y puertos

`ContextPortRouter` calcula una clave contextual por cápsula y asigna en una sola pasada al puerto compatible. Las claves están congeladas durante la ronda y solo reciben EMA al finalizar.

Reglas:

- no existe routing-by-agreement iterativo;
- una cápsula puede duplicarse como máximo en dos puertos;
- la energía total se conserva al duplicar;
- capacidad y máximo de puertos son límites duros;
- overflow se decide por utilidad local, no mediante negociación global.

## 7. Propagación y Fold_B

Cada ronda:

1. recupera Top-K desde el índice físico del carril;
2. calcula gate y compatibilidad relación-contexto;
3. aplica el operador monomial;
4. actualiza cápsula, roles, expectativas y energía;
5. inhibe repetición, ciclos y saturación;
6. agrupa por puertos;
7. fusiona duplicados estructurales;
8. conserva un beam diverso de tamaño B.

`BoundedDiverseFoldPolicy` evita materializar `k^D`. El estado activo queda acotado por el beam de cada carril.

## 8. FULL y CLEAN

`DualLaneEngine` contiene dos instancias de `LaneEngine`. No comparten:

- candidate set;
- decisiones de gate;
- puertos o cargas;
- normalización;
- beam y sobrevivientes;
- profundidad o parada;
- métricas mutables.

Pueden compartir datos inmutables: embeddings, controlador consolidado y tablas M2.

`PureComputeCache` solo reutiliza el resultado de una función pura si coinciden relación, versión, versión del controlador, hash del input y hash del operador. La selección Top-K y el resto de decisiones siguen ejecutándose por carril.

## 9. Certificación

La salida dual se clasifica así:

- `clean`: FULL y CLEAN eligen el mismo token y CLEAN supera el margen mínimo;
- `provisional`: FULL depende de contenido que CLEAN no sostiene;
- `fragile`: coinciden, pero el margen CLEAN es insuficiente;
- `empty`: CLEAN no conserva una hipótesis válida.

`clean_health_ratio` combina actividad y operaciones de CLEAN respecto de su presupuesto. Menos de 0.5 se considera degenerado.

## 10. Replay y auditoría

Cada ronda puede persistir un `ReplayStep` con:

- `operation_id`, profundidad y versiones;
- IDs padre;
- decisiones por carril;
- presupuesto y sobrevivientes;
- bounds de sombra;
- hash del candidate set;
- semilla determinista.

`ReplayClosure` fija:

- pasos de replay;
- versiones y snapshots binarios de todas las relaciones necesarias;
- snapshot/hash del controlador;
- snapshot/hash del inductor de roles;
- observaciones, contextos, bindings y semillas.

La auditoría crea un `SnapshotRelationStore` histórico. Primero exige reproducir CLEAN exactamente. Luego ejecuta CLEAN+e temporal y mide la diferencia causal. Si la primera ejecución no coincide, la relación es `UNREPLAYABLE` y no puede promoverse.

En el arranque, un CLEAN realmente vacío es un control contrafactual válido solo cuando no recuperó relaciones, no ejecutó gates/operadores y no descartó ramas. En ese caso no existe frontera oculta y el control se certifica vacuamente. Esto permite la primera promoción sin relajar la auditoría. Un CLEAN vacío por colapso de beam, energía o capacidad no cumple esas condiciones y permanece sin certificar. Una relación con influencia exactamente cero nunca se promueve aunque el umbral configurado sea cero.

## 11. Máquina de estados M1

```text
ACTIVE ──reserve──> AUDIT_RESERVED ──begin──> AUDITING
   │                                      ├──> PROMOTED
   └──expire (solo sin pins)──> EXPIRED   ├──> ACTIVE (defer)
                                          ├──> REJECTED
                                          └──> UNREPLAYABLE
```

La reserva y el pin se realizan dentro de una transacción SQLite. El TTL vencido durante auditoría establece `expiry_pending`; no elimina la clausura. Al liberar un registro diferido, la expiración pendiente se procesa de forma determinista.

## 12. Entrenamiento Modo B

Por token objetivo, replay + mutación M1 + escrow forman una sola transacción RAII de SQLite:

1. ejecutar FULL/CLEAN;
2. calcular pérdida dispersa;
3. elegir fuentes contextuales limitadas;
4. crear o actualizar prototipos M1 mediante delta local;
5. registrar observación y clausura histórica;
6. auditar solo al superar soporte/contextos configurados;
7. tras promoción, y solo entonces, emitir `PromotionPermit` para actualizar controlador y roles;
8. confirmar la unidad completa; una excepción revierte todo el token en SQLite.

El tipo `PromotionPermit` no puede construirse desde el camino de escritura M1. Esto convierte el no-lavado por parámetros en una restricción de API, no en una convención.

## 13. Persistencia

SQLite almacena semántica mutable y versiones; los dos archivos grandes e inmutables quedan fuera de la base. Esto evita reescribir embeddings y permite `mmap`.

Las escrituras sensibles usan transacciones y WAL. Un bloqueo `flock` por modelo impide dos escritores concurrentes desde procesos separados. Operaciones de lectura usan lock compartido.

# Invariantes no negociables

Estas condiciones deben conservarse al optimizar o sustituir módulos.

## I-1. Embedding base inmutable

Una vez ejecutado `prepare`, ninguna ruta de entrenamiento puede escribir `embeddings.mrdlemb`. Para cambiar embeddings se crea un modelo nuevo.

## I-2. M1 no existe en CLEAN

La elegibilidad se decide antes de recuperar candidatos:

```text
FULL  -> full_index[node]  = M1 ∪ M2
CLEAN -> clean_index[node] = M2
```

Está prohibido recorrer FULL y filtrar M1 después. También está prohibido representar M1 como operador cero, porque contaminaría conteos, normalización y capacidad.

## I-3. Estado mutable por carril

FULL y CLEAN no comparten candidate set, gate, puerto, beam, condición de parada ni métricas. Una optimización puede compartir solamente resultados puros con clave exacta de versión/input/operador.

## I-4. No promoción implícita

Compose, Fold, routing, gate, túneles, repetición y recuperación no cambian M1→M2. Solo `PromotionManager::complete` con una auditoría aceptada puede hacerlo.

## I-5. No-lavado por parámetros

Una escritura M1 no puede modificar controlador, roles persistentes, claves persistentes de puerto, codebooks, estadísticas de normalización o cachés semánticas compartidas. Las APIs de actualización exigen `PromotionPermit`.

## I-6. Replay histórico exacto

Una auditoría usa las versiones fijadas en su clausura. Nunca usa la relación/controlador actuales como reemplazo. Snapshot faltante, checksum incorrecto o divergencia de decisiones produce `UNREPLAYABLE`.

## I-7. TTL subordinado al pin

Un registro `AUDIT_RESERVED` o `AUDITING` y toda su clausura no se eliminan. El vencimiento solo marca `expiry_pending`. La transición a `EXPIRED` exige `pin_count==0`.

## I-8. Beam independiente

`beam_full` y `beam_clean` son presupuestos separados. Una rama M1 no puede desplazar provisionalmente una M2 y ser filtrada después.

## I-9. Routing de una pasada

No se agrega un bucle de agreement o reasignación dentro de la ronda. Las claves de puerto se congelan durante la ronda; la EMA es posterior.

## I-10. Estado acotado

El motor no materializa la enumeración de rutas. El estado activo es `O(G·B)` y el registro reconstruible es `O(G·D·B·k)` bajo los límites configurados.

## I-11. Métricas degeneradas no se promedian

Una fila con `clean_degenerate=true` o `clean_empty=true` se reporta separada. Un costo dual bajo debido a colapso de CLEAN no cuenta como mejora.

## I-12. Formatos y configuración versionados

Cambiar dimensión, vocabulario, prototipos o layout requiere migración explícita o nuevo modelo. Nunca se interpreta un blob de otra versión con el esquema actual.

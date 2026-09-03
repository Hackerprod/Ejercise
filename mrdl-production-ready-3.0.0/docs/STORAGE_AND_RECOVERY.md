# Almacenamiento, backup y recuperación

## Artefactos del modelo

```text
model/
├── mrdl.db                 relaciones, escrow, replay, controlador y roles
├── mrdl.db-wal             WAL activo de SQLite
├── mrdl.db-shm             coordinación WAL
├── tokenizer.mrdltok       vocabulario reversible, versionado y con checksum
├── embeddings.mrdlemb      embeddings congelados Q8, versionados y con checksum
├── config.effective.ini    configuración estructural usada al preparar
├── manifest.json           opcional en backups
└── .mrdl.lock              lock advisory entre procesos
```

## Política de durabilidad

El perfil VPS usa:

- journal mode WAL;
- `synchronous=FULL`;
- busy timeout de 10 segundos;
- una transacción RAII por token objetivo que agrupa replay, relación M1 y escrow;
- transacciones para cambios de estado y reservas;
- reemplazo atómico de tokenizador/configuración/embeddings;
- checksums de payload en archivos binarios y snapshots.

La unidad de durabilidad del entrenamiento es un token objetivo completo: sus `ReplayStep`, versiones de relación, escritura M1 y `EscrowRecord` se confirman juntos. La transacción mantiene el mutex de la conexión durante toda su vida, por lo que otro hilo no puede incorporar escrituras accidentalmente. Una excepción provoca rollback automático.

Reducir `synchronous_full=false` mejora escritura pero puede perder transacciones recientes ante corte de energía. No se recomienda durante las primeras campañas de entrenamiento.

## Backup consistente

No copie `mrdl.db`, `-wal` y `-shm` manualmente mientras el proceso escribe. Use:

```bash
mrdl backup --config /etc/mrdl/mrdl.ini --output /backup/mrdl-YYYYMMDDTHHMMSSZ
```

El comando:

1. toma lock exclusivo;
2. guarda controlador y roles;
3. ejecuta checkpoint WAL;
4. usa la API de backup SQLite;
5. copia tokenizador y embeddings;
6. escribe `manifest.json` con tamaño y hash de cada artefacto.

## Verificación del backup

```bash
find /backup/mrdl-... -maxdepth 1 -type f -ls
cat /backup/mrdl-.../manifest.json
```

Para una verificación completa, cree una configuración temporal que apunte al backup y ejecute:

```bash
mrdl doctor --config /tmp/restore-check.ini
mrdl inspect --config /tmp/restore-check.ini
```

## Restauración

1. Detenga jobs de entrenamiento/auditoría.
2. Mueva el modelo actual fuera de la ruta; no lo sobrescriba parcialmente.
3. Copie todos los artefactos del mismo backup a un directorio vacío.
4. Ajuste la sección `[persistence]` para apuntar al directorio restaurado.
5. Ejecute `doctor` e `inspect`.
6. Reanude jobs solo si ambos terminan correctamente.

Ejemplo:

```bash
sudo systemctl stop mrdl-audit.timer mrdl-gc.timer mrdl-backup.timer
sudo mv /var/lib/mrdl/model /var/lib/mrdl/model.failed
sudo mkdir -p /var/lib/mrdl/model
sudo cp -a /backup/mrdl-20260815T120000Z/. /var/lib/mrdl/model/
sudo chown -R mrdl:mrdl /var/lib/mrdl/model
sudo -u mrdl mrdl doctor --config /etc/mrdl/mrdl.ini
```

## Recuperación tras caída

SQLite recupera el WAL al abrir. Luego ejecute:

```bash
mrdl doctor --config /etc/mrdl/mrdl.ini
mrdl inspect --config /etc/mrdl/mrdl.ini
mrdl checkpoint --config /etc/mrdl/mrdl.ini
```

Si `sqlite_integrity` falla, no entrene ni ejecute GC. Restaure el último backup válido y conserve el directorio fallido para análisis.

## Retención sugerida

En un VPS de recursos limitados:

- 7 backups diarios;
- 4 semanales;
- 3 mensuales;
- una copia fuera del VPS.

La política debe considerar que embeddings y tokenizador pueden dominar el tamaño. Deduplicación a nivel de filesystem es segura porque esos archivos son inmutables después de `prepare`.

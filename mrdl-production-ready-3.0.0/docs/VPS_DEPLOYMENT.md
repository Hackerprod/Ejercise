# Despliegue en VPS 4 vCPU / 8 GB

## 1. Preparación del sistema

En Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libsqlite3-dev sqlite3
```

Se recomienda un filesystem local ext4/xfs sobre SSD/NVMe. Evite almacenar `mrdl.db-wal` en NFS o volúmenes de red con semántica de locking incierta.

## 2. Compilación nativa

```bash
cd /ruta/mrdl-production
scripts/build_release.sh
```

El script compila con `-march=native`, LTO y ejecuta tests release. Para validar memoria/undefined behavior:

```bash
scripts/run_validation.sh
```

## 3. Instalación del binario

```bash
sudo cmake --install build/release --prefix /usr/local
mrdl version
```

## 4. Usuario y directorios

```bash
sudo useradd --system --home /var/lib/mrdl --shell /usr/sbin/nologin mrdl || true
sudo install -d -o mrdl -g mrdl -m 0750 \
  /var/lib/mrdl/model \
  /var/lib/mrdl/corpus \
  /var/backups/mrdl
sudo install -d -o root -g mrdl -m 0750 /etc/mrdl
sudo install -o root -g mrdl -m 0640 config/vps-system.ini /etc/mrdl/mrdl.ini
```

Ajuste la configuración antes de preparar el modelo. Los valores estructurales quedan fijados en `config.effective.ini`.

## 5. Corpus

```bash
sudo install -o mrdl -g mrdl -m 0640 train.txt /var/lib/mrdl/corpus/train.txt
sudo install -o mrdl -g mrdl -m 0640 validation.txt /var/lib/mrdl/corpus/validation.txt
```

Use archivos de texto UTF-8, una unidad documental por línea cuando sea posible. El tokenizador preserva bytes arbitrarios, pero corpus limpio produce relaciones más interpretables.

## 6. Inicialización

```bash
sudo -u mrdl mrdl doctor --config /etc/mrdl/mrdl.ini --allow-unprepared
sudo -u mrdl mrdl prepare \
  --config /etc/mrdl/mrdl.ini \
  --corpus /var/lib/mrdl/corpus/train.txt \
  --embeddings random-indexing
sudo -u mrdl mrdl doctor --config /etc/mrdl/mrdl.ini
```

`random-indexing` hace dos recorridos de corpus para tokenización y otro para geometría contextual. Vigile RAM y disco durante `prepare`, pero el archivo final queda Q8/mmap.

## 7. Primera campaña controlada

No comience con todo el corpus. Use un shard y mida:

```bash
sudo -u mrdl mrdl train \
  --config /etc/mrdl/mrdl.ini \
  --corpus /var/lib/mrdl/corpus/train-shard.txt \
  --json 2> /tmp/mrdl-progress.jsonl

sudo -u mrdl mrdl eval \
  --config /etc/mrdl/mrdl.ini \
  --corpus /var/lib/mrdl/corpus/validation.txt \
  --max-tokens 100000

sudo -u mrdl mrdl baseline \
  --config /etc/mrdl/mrdl.ini \
  --train-corpus /var/lib/mrdl/corpus/train-shard.txt \
  --eval-corpus /var/lib/mrdl/corpus/validation.txt \
  --orders 1,2,3
```

Registre como mínimo:

- tokens/s de entrenamiento, evaluación y generación;
- pérdida/perplexity FULL y CLEAN;
- `clean_empty_ratio`;
- relaciones M1/M2;
- promociones, diferidos y `UNREPLAYABLE`;
- tamaño de DB/WAL y RAM pico;
- comparación con bigram/trigram y Transformer de presupuesto comparable.

## 8. Beam y salud CLEAN

El documento base ya detectó colapso CLEAN con beam insuficiente a densidad M1 alta. No reduzca `beam_clean` solo para ganar velocidad.

Antes de una campaña larga:

```bash
mrdl_bench --suite dual --json > /var/lib/mrdl/dual-benchmark.jsonl
```

Criterio operativo mínimo sugerido para investigar, no para declarar éxito científico:

- ninguna fila objetivo con `clean_empty=true`;
- `clean_degenerate=false` en la densidad M1 esperada;
- p95 estable sin crecimiento sostenido de WAL;
- RAM pico con al menos 1–1.5 GB libres para SO/page cache.

## 9. Systemd

Puede instalar unidades conservadoras:

```bash
sudo deploy/install_systemd.sh
```

Esto instala timers de auditoría, GC y backup, pero no inicia entrenamiento automático. Revise primero:

```bash
systemctl cat mrdl-audit.service
systemctl list-timers 'mrdl-*'
```

El entrenamiento se ejecuta manualmente:

```bash
sudo systemctl start mrdl-train.service
journalctl -u mrdl-train.service -f
```

## 10. Observabilidad

Comandos útiles:

```bash
watch -n 10 'mrdl inspect --config /etc/mrdl/mrdl.ini'
watch -n 5 'du -h /var/lib/mrdl/model/* 2>/dev/null'
pidstat -r -u -d -p $(pgrep -n mrdl) 5
sqlite3 /var/lib/mrdl/model/mrdl.db 'PRAGMA wal_checkpoint(PASSIVE);'
```

No ejecute escrituras SQL manuales sobre la base. Las consultas de solo lectura son aceptables sobre un backup.

## 11. Ajustes para 8 GB

Empiece con el perfil incluido. Si hay presión de memoria:

1. reduzca `vocab_size` antes de `prepare`;
2. reduzca `embedding_dim` antes de `prepare`;
3. reduzca `max_source_capsules`;
4. reduzca `top_k_full`, pero mantenga `top_k_clean`/beam suficientes para salud CLEAN;
5. reduzca `batch_tokens` — es una cadencia de checkpoint/progreso, no un batch denso;
6. mantenga swap pequeño de emergencia, no como memoria de trabajo normal.

No cambie dimensiones ni vocabulario dentro de un modelo preparado.

## 12. Docker

El contenedor es útil para reproducibilidad, no necesariamente para máximo rendimiento nativo:

```bash
docker compose build
docker compose run --rm mrdl doctor --config /etc/mrdl/mrdl.ini --allow-unprepared
docker compose run --rm mrdl prepare --config /etc/mrdl/mrdl.ini \
  --corpus /data/train.txt --embeddings random-indexing
```

Compose usa el volumen nombrado `mrdl-model`, de modo que conserva la propiedad del usuario no privilegiado del contenedor. Coloque los corpus del host en `data/corpus/`; se montan en `/data` como solo lectura.

La imagen portable compila sin `-march=native`. Para el mejor rendimiento del VPS use compilación directa.

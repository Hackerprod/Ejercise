# Formato de embeddings externos

`mrdl prepare --embeddings external` acepta un archivo binario raw de float32.

## Forma

```text
rows    = número real de tokens producido por el tokenizador preparado
columns = model.embedding_dim
layout  = row-major
bytes   = rows * columns * 4
endian  = little-endian
```

No incluye cabecera, nombres de tokens ni padding. La fila `i` corresponde exactamente al TokenId `i` del tokenizador que `prepare` acaba de construir.

Antes de cuantizar, MRDL normaliza cada fila a norma L2. Después almacena una escala float32 y `dimension` valores int8 por fila.

## Generación segura

La forma debe derivarse del tokenizador del mismo corpus/configuración. Para evitar discrepancias:

1. use exactamente el mismo corpus y opciones de tokenización;
2. no cambie `lowercase`, `vocab_size` o `heavy_hitter_multiplier`;
3. incluya filas para `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` y los 256 bytes;
4. escriba float32 finitos; filas cero se conservan como cero.

El comando valida el tamaño exacto y aborta antes de crear un modelo parcial si no coincide.

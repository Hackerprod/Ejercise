# Ejemplos

`tiny-corpus.txt` sirve únicamente para comprobar el pipeline. No es suficiente para evaluar capacidad lingüística.

```bash
./build/release/mrdl prepare --config config/vps.ini \
  --corpus examples/tiny-corpus.txt --embeddings random-indexing
./build/release/mrdl train --config config/vps.ini \
  --corpus examples/tiny-corpus.txt --json
./build/release/mrdl generate --config config/vps.ini \
  --prompt "Había una vez" --max-tokens 20 --json
```

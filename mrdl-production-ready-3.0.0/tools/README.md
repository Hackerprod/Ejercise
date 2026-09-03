# Herramientas opcionales

## Transformer pequeño comparable

No se importa en el runtime C++. Usa el tokenizador MRDL y puede inicializar/fijar exactamente los embeddings Q8 del modelo:

```bash
python3 -m venv .venv-baseline
. .venv-baseline/bin/activate
pip install -r tools/requirements-transformer.txt

python tools/tiny_transformer_baseline.py \
  --tokenizer model/tokenizer.mrdltok \
  --mrdl-embeddings model/embeddings.mrdlemb \
  --freeze-input-embeddings \
  --train-corpus train.txt \
  --eval-corpus validation.txt \
  --d-model 96 --heads 4 --layers 2 --feed-forward 256 \
  --context 64 --batch-size 8 --epochs 1 --threads 4 \
  --output baselines/tiny-transformer.pt
```

Para una comparación responsable reporte parámetros totales/entrenables, RAM pico, tokens procesados, contexto, pérdida, perplexity, exactitud y tokens/s.

## Exportar embeddings externos

```bash
python tools/export_external_embeddings.py embeddings.npy embeddings.f32 \
  --rows 20000 --dimension 96
```

`rows` debe coincidir con el tamaño real del tokenizador, no únicamente con el presupuesto configurado.

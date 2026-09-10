#!/usr/bin/env bash
set -Eeuo pipefail

MRDL_BIN="${1:-mrdl}"
WORK="$(mktemp -d -t mrdl-smoke-XXXXXX)"
trap 'rm -rf -- "$WORK"' EXIT
mkdir -p "$WORK/model" "$WORK/backup"

cat > "$WORK/corpus.txt" <<'CORPUS'
El gato mira la casa verde.
El perro mira la pelota roja.
La casa verde tiene una puerta.
El gato persigue la pelota.
La puerta de la casa está abierta.
El perro descansa junto a la casa.
CORPUS

cat > "$WORK/config.ini" <<EOF_CONFIG
[model]
embedding_dim = 32
relation_dim = 16
max_relation_prototypes = 3
seed = 123456

[tokenizer]
vocab_size = 1024
heavy_hitter_multiplier = 2
lowercase = false

[engine]
top_k_full = 8
top_k_clean = 8
beam_full = 12
beam_clean = 12
max_rounds = 3
max_ports_per_node = 4
port_capacity = 6
port_similarity_threshold = 0.60
port_pressure_threshold = 1.0
branch_energy_floor = 0.001
clean_margin = 0.05
clean_health_threshold = 0.50
repetition_penalty = 0.35
cycle_penalty = 0.50
saturation_penalty = 0.20
length_log_penalty = 0.08
confidence_epsilon = 0.05
exact_pure_reuse = true
parallel_lanes = true

[memory]
m1_ttl_seconds = 3600
m1_confidence_cap = 0.45
promotion_min_support = 2
promotion_min_contexts = 1
promotion_min_influence = 0.0
promotion_stability_ratio = 0.0
audit_top_m = 4

[training]
mode = B
context_tokens = 8
max_source_capsules = 3
epochs = 1
batch_tokens = 64
checkpoint_every_tokens = 64
fast_learning_rate = 0.04
controller_learning_rate = 0.001
relation_weight_decay = 0.0001
negative_samples = 4
auto_audit = true
trusted_source = false

[persistence]
model_dir = $WORK/model
database = $WORK/model/mrdl.db
tokenizer = $WORK/model/tokenizer.mrdltok
embeddings = $WORK/model/embeddings.mrdlemb
sqlite_busy_timeout_ms = 5000
synchronous_full = true

[runtime]
threads = 2
max_generation_tokens = 12
temperature = 0.0
top_p_candidates = 8
EOF_CONFIG

"$MRDL_BIN" doctor --config "$WORK/config.ini" --allow-unprepared --json
"$MRDL_BIN" prepare --config "$WORK/config.ini" --corpus "$WORK/corpus.txt" --embeddings random-indexing --json
"$MRDL_BIN" doctor --config "$WORK/config.ini" --json
"$MRDL_BIN" tokenize --config "$WORK/config.ini" --text 'área verde' --bos --eos
TRAIN_JSON="$("$MRDL_BIN" train --config "$WORK/config.ini" --corpus "$WORK/corpus.txt" --quiet --json)"
printf '%s\n' "$TRAIN_JSON"
grep -Eq '"promotions":[1-9][0-9]*' <<<"$TRAIN_JSON"
EVAL_JSON="$("$MRDL_BIN" eval --config "$WORK/config.ini" --corpus "$WORK/corpus.txt" --max-tokens 64 --json)"
printf '%s\n' "$EVAL_JSON"
grep -Eq '"clean_empty":0([,}])' <<<"$EVAL_JSON"
"$MRDL_BIN" baseline --config "$WORK/config.ini" --train-corpus "$WORK/corpus.txt" \
  --eval-corpus "$WORK/corpus.txt" --orders 1,2,3 --max-tokens 64 --output-dir "$WORK/baselines"
"$MRDL_BIN" generate --config "$WORK/config.ini" --prompt 'El gato' --max-tokens 8 --seed 7 --json
"$MRDL_BIN" audit --config "$WORK/config.ini" --max 8 --json
"$MRDL_BIN" gc --config "$WORK/config.ini" --json
"$MRDL_BIN" inspect --config "$WORK/config.ini" --json
"$MRDL_BIN" backup --config "$WORK/config.ini" --output "$WORK/backup" --json

# Typos and missing option values must fail rather than silently using defaults.
if "$MRDL_BIN" doctor --config "$WORK/config.ini" --beam-celan 12 >/dev/null 2>&1; then
  echo 'unknown CLI option was accepted' >&2
  exit 1
fi
if "$MRDL_BIN" doctor --config >/dev/null 2>&1; then
  echo 'missing CLI option value was accepted' >&2
  exit 1
fi

test -s "$WORK/backup/mrdl.db"
test -s "$WORK/backup/tokenizer.mrdltok"
test -s "$WORK/backup/embeddings.mrdlemb"
test -s "$WORK/backup/manifest.json"
printf 'smoke_test=pass workdir=%s\n' "$WORK"

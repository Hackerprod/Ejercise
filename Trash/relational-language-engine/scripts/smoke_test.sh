#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${RLMCTL:-$ROOT/build/rlmctl}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/data"
"$BIN" embeddings build --input "$ROOT/fixtures/embeddings.txt" --output "$TMP/data/embeddings.rle"
cat > "$TMP/smoke.toml" <<CONFIG
[storage]
state_dir = "$TMP/data/state"
embedding_file = "$TMP/data/embeddings.rle"
shards = 4
durability = "data"
[search]
beam_width = 4
candidate_k = 8
max_depth = 2
replay_reopen_limit = 128
state_decay = 0.65
relation_mix = 0.65
target_mix = 0.35
[scoring]
confidence_weight = 2.0
support_weight = 0.25
relation_weight = 0.5
target_weight = 0.5
context_weight = 0.2
repetition_penalty = 0.1
[audit]
min_exact_cases = 2
max_cases = 4
max_unknown_cases = 2
causal_margin = 0.01
min_pass_fraction = 0.5
allow_empty_clean_bootstrap = true
[training]
batch_tokens = 32
context_radius = 3
evidence_cases_per_edge = 3
auto_promote_per_batch = 8
min_support_for_promotion = 3
min_confidence_for_promotion = 0.4
m1_ttl_seconds = 604800
max_m1_edges = 10000
checkpoint_every_batches = 1
reject_unknown_tokens = false
[replay]
max_records = 1000
[runtime]
parallel_lanes = true
worker_threads = 4
queue_capacity = 16
service_port = 19087
CONFIG
"$BIN" doctor --config "$TMP/smoke.toml"
"$BIN" train --config "$TMP/smoke.toml" --corpus "$ROOT/fixtures/corpus.txt"
"$BIN" infer --config "$TMP/smoke.toml" --text "the cat" --lane both --depth 1
"$BIN" benchmark --config "$TMP/smoke.toml" --corpus "$ROOT/fixtures/corpus.txt" --samples 30
"$BIN" checkpoint --config "$TMP/smoke.toml"
"$BIN" doctor --config "$TMP/smoke.toml"
echo "smoke_test=PASS"

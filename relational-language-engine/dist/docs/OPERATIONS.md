# VPS operations

## Suggested layout

```text
/opt/relational-language-engine/
  bin/rlmctl
  config/production.toml
  data/embeddings.rle
  data/state/
  log/
```

Use `config/vps-4vcpu-8gb.toml` as the initial profile. Its beam/candidate limits and replay retention are intentionally below the generic production profile.

## Installation

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake zstd
./scripts/install.sh
sudo useradd --system --home /opt/relational-language-engine --shell /usr/sbin/nologin rlm || true
sudo chown -R rlm:rlm /opt/relational-language-engine
sudo cp deploy/relational-language.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now relational-language
```

Install the real frozen embedding file before starting. The fixture is only for smoke validation.

## Health and metrics

```bash
curl -fsS http://127.0.0.1:9087/healthz
curl -fsS http://127.0.0.1:9087/metrics
journalctl -u relational-language -f
```

Important gauges:

- `rlm_m1_edges`: provisional growth; sustained growth means promotion or TTL cannot keep up;
- `rlm_m2_edges`: durable promoted knowledge;
- `rlm_replay_records`: bounded by configuration;
- inference errors and cumulative latency;
- promotion failures.

## Capacity controls

For 4 vCPU / 8 GB:

- begin with `beam_width=8`, `candidate_k=16`, `max_depth=3`;
- keep `worker_threads=4`; more threads do not create more CPU;
- keep the HTTP worker count and lane pool equal to vCPU initially;
- use `parallel_lanes=false` when only one response lane is needed or CPU saturation matters more than simultaneous control output;
- measure FULL+CLEAN/FULL runtime ratio with `benchmark`, rather than assuming a fixed 2× number;
- reduce `max_records` before reducing audit exactness limits;
- monitor M1 edge count and resident memory before raising `max_m1_edges`.

## Rolling deployment

A state directory is single-writer. For a one-node VPS:

1. stop the service;
2. run the new binary's `doctor` against a copy of state;
3. back up embeddings and state;
4. replace the binary;
5. start and check `/healthz` and logs.

Do not run old and new binaries against the same state directory concurrently.

## Repair policy

- Partial EOF tails are repaired automatically.
- Mid-file CRC corruption is never skipped.
- Preserve the damaged state and restore the latest checkpoint/archive.
- Do not delete M1, M2 or promotion files independently; their transactional relationship matters.

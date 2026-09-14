# OMEGA persistent-volume layout

Persistent volume: `q4t3-vol` (`networkVolumeId=6yrppoqpkz`), 50 GB, `US-MO-2`.
Do not create another volume. The pod image is versioned separately from this
volume; the image owns Python/PyTorch/CUDA/SSH startup, while the volume owns
experiment state.

## Exact layout

```text
/workspace/omega/
├── code/                         # checked-out OMEGA code and scripts
├── locks/                        # copied requirements/lock files and image manifest
├── cache/
│   └── huggingface/              # HF_HOME/HF_HUB_CACHE; model and dataset cache
├── checkpoints/                  # discardable and promoted checkpoints
├── logs/                         # JSONL events, heartbeats, stdout/stderr
├── results/                      # consolidated reports and metrics
└── manifests/                    # revisions, SHA-256s, run/config manifests
```

Suggested environment variables inside future pod:

```text
OMEGA_ROOT=/workspace/omega
HF_HOME=/workspace/omega/cache/huggingface
HF_HUB_CACHE=/workspace/omega/cache/huggingface/hub
TRANSFORMERS_CACHE=/workspace/omega/cache/huggingface/transformers
```

Credentials such as `HF_TOKEN` belong only in pod environment/secret injection
(or the provider's secret facility). Never put tokens in this repository,
volume manifests, JSON artifacts, command lines, or logs.

## Safe pre-write procedure (documented, not executed here)

Run these read-only checks after the provider mounts the existing volume and
before writing code, caches, checkpoints, or logs:

```bash
mountpoint -q /workspace/omega
findmnt --target /workspace/omega --output SOURCE,FSTYPE,OPTIONS,TARGET
df -hT /workspace/omega
find /workspace/omega -maxdepth 2 -mindepth 1 -printf '%y %p\n' | sort
du -sh /workspace/omega/* 2>/dev/null || true
```

`50GB` is capacity, not free space. Do not delete, move, or overwrite existing
volume content to make room. If mount identity, free space, or existing
content is unexpected, stop for maintainer review.

After review, create only missing directories with `mkdir -p`; preserve
existing files and verify their hashes before any replacement. Record mount
source, free bytes, and pre-write tree/hash inventory in `/workspace/omega/manifests`.

## Future pod lifecycle notes

When a pod is explicitly approved, attach `networkVolumeId=6yrppoqpkz` in
`US-MO-2`. Do not create a replacement volume or transfer the already-fixed
teacher/dataset again; reuse the recorded revisions and hashes. Select GPU
memory/price to fit the authorized measurement, not the largest available GPU.
This preparation unit creates no pod and performs no volume transfer.

# HTTP API

The native server is intentionally small and dependency-free. Put it behind a TLS reverse proxy and authentication layer.

## `GET /healthz`

Returns readiness, frozen-embedding identity, vocabulary/dimension, M1/M2 counts, replay count and semantic repository epoch.

## `GET /metrics`

Prometheus text exposition.

## `POST /v1/infer?lane=both&depth=1`

Request body: UTF-8 plain text. Maximum body size is 4 MiB. Supported lane values:

- `full`: M2 ∪ M1;
- `clean`: M2 only;
- `both`: executes both, optionally in parallel.

Example:

```bash
curl -sS -X POST 'http://127.0.0.1:9087/v1/infer?lane=both&depth=1' \
  -H 'Content-Type: text/plain' \
  --data-binary 'the cat'
```

Representative response:

```json
{
  "unknown_tokens": 0,
  "full": {
    "has_prediction": true,
    "exact_within_beam": true,
    "stable_epoch": true,
    "repository_epoch": 21,
    "token_id": 6,
    "token": "sleeps",
    "score": 2.31,
    "path": [{"edge_id": 123, "token_id": 6, "token": "sleeps"}]
  },
  "clean": {
    "has_prediction": false,
    "exact_within_beam": true,
    "stable_epoch": true,
    "repository_epoch": 21
  }
}
```

The service closes each HTTP/1.1 connection after one request. Nginx handles public keepalive, TLS, access control and rate limiting.

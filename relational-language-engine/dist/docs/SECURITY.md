# Security notes

- The native HTTP listener has no TLS or authentication. Bind it behind Nginx/firewall rules.
- Request headers are capped at 64 KiB and bodies at 4 MiB.
- Binary readers validate lengths before allocation.
- The service runs as a non-login user with a strict systemd sandbox.
- State is single-writer through `flock`; do not share one state directory over an unreliable network filesystem.
- Frozen embedding and corpus paths are operator-controlled; untrusted users should not be allowed to replace them.
- Prometheus metrics may reveal model dimensions and state sizes; restrict access.

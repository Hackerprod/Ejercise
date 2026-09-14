#!/usr/bin/env python3
"""Read-only-supervision helper for a future OMEGA pod.

Stdlib only. It records heartbeats/events and audits JSONL in streaming mode.
It has no RunPod client, pod lifecycle API, retry loop, or relaunch authority.
It is prepared for deployment, not deployed by this unit.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def stream_audit(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    invalid_lines = 0
    total_lines = 0
    max_rss: int | None = None
    min_available: int | None = None
    if not path.is_file():
        return {"path": str(path), "exists": False, "total_lines": 0, "invalid_lines": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            total_lines += 1
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("event is not object")
                phase = event.get("phase") or event.get("kind") or "untyped"
                counts[str(phase)] += 1
                if "variant" in event:
                    variants[str(event["variant"])] += 1
                memory = event.get("memory")
                if isinstance(memory, dict):
                    rss = memory.get("rss_bytes")
                    available = memory.get("available_system_bytes")
                    if isinstance(rss, int):
                        max_rss = rss if max_rss is None else max(max_rss, rss)
                    if isinstance(available, int):
                        min_available = available if min_available is None else min(min_available, available)
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid_lines += 1
    return {"path": str(path), "exists": True, "total_lines": total_lines, "invalid_lines": invalid_lines, "phase_counts": dict(counts), "variant_counts": dict(variants), "maximum_rss_bytes": max_rss, "minimum_available_system_bytes": min_available}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_heartbeat(path: Path) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("kind") == "heartbeat":
                latest = event
    return latest


class ControlHandler(http.server.BaseHTTPRequestHandler):
    events_path: Path
    timeout_seconds: float

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/event":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("event must be JSON object")
            payload = dict(payload)
            payload["received_at"] = utc_now()
            payload["received_epoch"] = time.time()
            payload["source"] = self.client_address[0]
            append_jsonl(self.events_path, payload)
            self.send_json(202, {"accepted": True, "authority": "record-only; no pod lifecycle actions"})
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.send_json(400, {"accepted": False, "error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/status":
            self.send_json(404, {"error": "not_found"})
            return
        heartbeat = latest_heartbeat(self.events_path)
        now = time.time()
        age = None
        stale = True
        if heartbeat and isinstance(heartbeat.get("received_epoch"), (int, float)):
            age = max(0.0, now - float(heartbeat["received_epoch"]))
            stale = age > self.timeout_seconds
        self.send_json(200, {"heartbeat": heartbeat, "age_seconds": age, "stale": stale, "timeout_seconds": self.timeout_seconds, "authority": "record-only; human intervention required for pod create/stop/relaunch"})


def serve(args: argparse.Namespace) -> int:
    class Handler(ControlHandler):
        events_path = args.events
        timeout_seconds = args.timeout_minutes * 60.0

    server = http.server.ThreadingHTTPServer((args.bind, args.port), Handler)
    print(json.dumps({"listening": f"{args.bind}:{args.port}", "events": str(args.events), "timeout_minutes": args.timeout_minutes, "authority": "record-only"}, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--bind", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--timeout-minutes", type=float, default=5.0)
    serve_parser.add_argument("--events", type=Path, default=Path("logs/vps-control.events.jsonl"))
    audit_parser = subparsers.add_parser("audit-jsonl")
    audit_parser.add_argument("path", type=Path)
    hash_parser = subparsers.add_parser("sha256")
    hash_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        return serve(args)
    if args.command == "audit-jsonl":
        print(json.dumps(stream_audit(args.path), sort_keys=True))
        return 0
    print(json.dumps({"path": str(args.path), "sha256": sha256_file(args.path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

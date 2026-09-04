"""Run remaining typed multi-hop P1-only seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_multihop_pointer.py"
SEEDS = (202, 303, 404, 505)


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "campaign" / "multihop_pointer_replication")
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for index, seed in enumerate(SEEDS, start=1):
        output_dir = args.output_root / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(TRAIN), "--seed", str(seed), "--max-steps", str(args.max_steps), "--output-dir", str(output_dir)]
        with (output_dir / "launcher.log").open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(" ".join(command) + "\n")
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        final_path = output_dir / "final.json"
        final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
        final["process_returncode"] = result.returncode
        save_json(final_path, final)
        if result.returncode != 0 or final.get("status") != "completed" or final.get("finite") is not True:
            raise RuntimeError(f"typed multi-hop replication failed for seed {seed}")
        save_json(args.output_root / "state.json", {"status": "running", "completed": index, "total": len(SEEDS), "last_seed": seed, "elapsed_seconds": time.perf_counter() - started})
    save_json(args.output_root / "state.json", {"status": "completed", "completed": len(SEEDS), "total": len(SEEDS), "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({"status": "completed", "runs": len(SEEDS), "elapsed_seconds": time.perf_counter() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

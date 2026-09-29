"""Run one clean native backward process for the FC2 M=8 candidate matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import run_omega_recurrent_backward_replay_timing as timing


HERE = Path(__file__).resolve().parent
FIXTURE_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "fixtures"
FIXTURE_NAMES = (
    "replay_K4_update2_window0.bin",
    "replay_K4_update3_window1.bin",
)
VARIANT_ROUTES = {
    1: ("baseline", "candidate"),
    2: ("candidate", "baseline"),
    3: ("baseline", "candidate"),
}
ENV_VARS_REQUIRED_UNSET = timing.ENV_VARS_REQUIRED_UNSET


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--pair-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--position", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--expected-dll-sha256", required=True)
    parser.add_argument("--fixture", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    expected_variant = VARIANT_ROUTES[args.pair_id][args.position - 1]
    if args.variant != expected_variant:
        parser.error(f"pair {args.pair_id} position {args.position} must be {expected_variant}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite process report: {args.output}")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"report parent must exist: {args.output.parent}")
    if "torch" in sys.modules:
        raise RuntimeError("native comparison worker must not load PyTorch")

    # Reuse canonical timing implementation while assigning explicit variant
    # identity and hash gate inside this fresh child process only.
    timing.PAIR_ROUTES = VARIANT_ROUTES
    timing.ACCEPTED_DLL_SHA256 = args.expected_dll_sha256.lower()
    environment = timing._assert_canonical_process_environment()
    fixture_paths = args.fixture or [FIXTURE_ROOT / name for name in FIXTURE_NAMES]
    replay, fixtures = timing._load_two_fixtures(fixture_paths)
    del replay
    if timing._assert_canonical_process_environment() != environment:
        raise RuntimeError("fixture setup changed canonical timing environment")

    report = timing._run_native_process(
        fixtures,
        args.pair_id,
        args.position,
        args.dll,
        args.output,
        report_route=args.variant,
    )
    if report["torch_loaded"] is not False or "torch" in sys.modules:
        raise RuntimeError("native timing worker loaded PyTorch")
    report["variant"] = args.variant
    report["expected_dll_sha256"] = args.expected_dll_sha256.lower()
    timing._write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reject transition objects that lost the required AVX2 hot-path operations."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_transition_assembly.py <transitions-object>", file=sys.stderr)
        return 2
    object_path = Path(sys.argv[1])
    if not object_path.is_file():
        print(f"object not found: {object_path}", file=sys.stderr)
        return 2
    text = subprocess.check_output(
        ["objdump", "-d", "-M", "intel", str(object_path)], text=True
    ).lower()
    required_all = {
        "signed int8 expansion": ("vpmovsxbd",),
        "integer scaling": ("vpmulld",),
        "rounded variable shift": ("vpsrlvd",),
        "saturating int32→int8 packing": ("vpackssdw", "vpacksswb"),
        "RMS int32→double": ("vcvtdq2pd",),
        "RMS rounding": ("vroundpd",),
        "RMS double→int32": ("vcvtpd2dq",),
    }
    required_any = {
        "RMS square root": ("vsqrtsd", "vsqrtpd"),
    }
    failed: list[str] = []
    for name, opcodes in required_all.items():
        if not all(re.search(rf"\b{re.escape(opcode)}\b", text) for opcode in opcodes):
            failed.append(f"{name}: missing required set {', '.join(opcodes)}")
    for name, opcodes in required_any.items():
        if not any(re.search(rf"\b{re.escape(opcode)}\b", text) for opcode in opcodes):
            failed.append(f"{name}: missing every alternative {', '.join(opcodes)}")
    if failed:
        for failure in failed:
            print(f"FAIL {failure}")
        return 1
    print("PASS transitions contain AVX2 scale/shift, saturated packing, and RMS vector paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

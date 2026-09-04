#!/usr/bin/env python3
"""Audit the spill-safe AVX2 fused4 kernel in an ELF build.

Windows builds should use scripts/audit_windows_assembly.ps1 for opcode presence;
this script gives the stronger stack-spill assertion when nm/objdump are available.
"""
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("object", type=Path)
args = parser.parse_args()

nm = subprocess.check_output(["nm", str(args.object)], text=True)
match = re.search(r"^[0-9a-fA-F]+\s+[a-zA-Z]\s+(\S*fused4\S*)$", nm, re.MULTILINE)
if not match:
    raise SystemExit("fused4 symbol not found")
symbol = match.group(1)
asm = subprocess.check_output(
    ["objdump", "-d", "-M", "intel", f"--disassemble={symbol}", str(args.object)], text=True
)
required = ["vpmaddwd", "vpmovsxbw"]
missing = [opcode for opcode in required if opcode not in asm]
vector_stack = [line.strip() for line in asm.splitlines()
                if re.search(r"\b[xyz]mm\d*\b", line) and "[rsp" in line]
if missing:
    print(asm)
    raise SystemExit("missing required opcodes: " + ", ".join(missing))
if vector_stack:
    print("Vector stack accesses found in fused4:")
    print("\n".join(vector_stack[:30]))
    raise SystemExit(1)
print("PASS fused4 contains signed int8 expansion and pairwise integer dot-product opcodes")
print("PASS fused4 contains no XMM/YMM stack accesses")

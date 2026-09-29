from __future__ import annotations

import hashlib
import json
from pathlib import Path


DIAG_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = DIAG_ROOT.parents[1] / "results" / "omega_v2_1b_kernel_qualification" / "candidate_01" / "diagnostics" / "run_02"
REPORT = RESULTS_ROOT / "CANDIDATE01_COMPONENT_DIAGNOSTIC.md"
SIDECAR = RESULTS_ROOT / "CANDIDATE01_COMPONENT_DIAGNOSTIC.md.sha256"
MANIFEST = RESULTS_ROOT / "artifact_hashes.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not (RESULTS_ROOT / "candidate01_component_diagnostic.json").is_file():
        raise FileNotFoundError("completed native candidate_01 diagnostic is missing")
    if not MANIFEST.is_file() or not REPORT.is_file():
        raise FileNotFoundError("initial diagnostic artifact manifest/report missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["artifacts"]
    analysis_tool = Path(__file__).resolve()
    entries[str(analysis_tool)] = {"sha256": sha256_file(analysis_tool), "size_bytes": analysis_tool.stat().st_size}
    excluded = {str(REPORT.resolve()), str(SIDECAR.resolve())}
    lines = REPORT.read_text(encoding="utf-8").split("## SHA-256", 1)[0].rstrip().splitlines()
    lines.extend(["", "## SHA-256 (complete artifact table)"])
    for path, row in sorted(entries.items()):
        if path not in excluded:
            lines.append(f"- `{path}`: `{row['sha256']}` ({row['size_bytes']} bytes)")
    lines.extend([
        "", "`artifact_hashes.json` excludes its own self-hash to avoid a circular digest; all measured inputs, outputs, source files, binaries, and this analysis tool are listed.", "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    report_hash = sha256_file(REPORT)
    SIDECAR.write_text(report_hash + "\n", encoding="ascii", newline="\n")
    entries[str(REPORT.resolve())] = {"sha256": report_hash, "size_bytes": REPORT.stat().st_size}
    entries[str(SIDECAR.resolve())] = {"sha256": sha256_file(SIDECAR), "size_bytes": SIDECAR.stat().st_size}
    manifest["report_self_sha256"] = report_hash
    manifest["artifacts"] = entries
    MANIFEST.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    verified = all(record["sha256"] == sha256_file(Path(path)) for path, record in entries.items())
    verified = verified and report_hash == SIDECAR.read_text(encoding="ascii").strip()
    if not verified:
        raise RuntimeError("candidate_01 diagnostic final report/artifact hashes did not verify")
    print(json.dumps({
        "diagnostic_report_abs": str(REPORT.resolve()),
        "report_sha256": report_hash,
        "artifact_hashes_abs": str(MANIFEST.resolve()),
        "artifact_count": len(entries),
        "hashes_verified": verified,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

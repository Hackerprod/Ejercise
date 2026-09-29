"""Analyze exact marker-bounded Native-K4 WPT samples, all worker threads."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
NATIVE_WPT_ROOT = HERE / "results" / "recurrent_backward_replay_final" / "profile" / "native_wpt"
DEFAULT_ETL = Path(r"C:\Users\danil\AppData\Local\Temp\opencode\recurrent-backward-replay-wpt\native_recurrent_backward_replay.etl")
DEFAULT_TEMP_ROOT = Path(r"C:\Users\danil\AppData\Local\Temp\opencode\recurrent-backward-replay-wpt\analysis_qpc_verified")
DEFAULT_OUTPUT_ROOT = NATIVE_WPT_ROOT / "wpt_analysis_qpc_verified"
DEFAULT_MARKERS = NATIVE_WPT_ROOT / "native_replay_markers.json"
DEFAULT_WINDOWS = NATIVE_WPT_ROOT / "wpt_time_windows_qpc_verified.json"
DEFAULT_BUILD_IDENTITY = NATIVE_WPT_ROOT / "diagnostic_build_identity.json"
DEFAULT_XPERF = Path(r"C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\xperf.exe")
DEFAULT_SYMBOLIZER = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\llvm-symbolizer.exe")
IMAGE_PROVIDER_GUID = "{2cb15d1d-5fc1-11d2-abe1-00a0c911f518}"
PROFILE_PERIOD_US = 1000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _run_xperf(xperf: Path, args: Sequence[str], log_root: Path, label: str, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(xperf), *[str(arg) for arg in args]],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env),
        check=False,
    )
    (log_root / f"{label}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"xperf command failed ({completed.returncode}): {label}; {completed.stderr}")
    return completed


def _clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _html_table_rows(text: str, table_anchor: str) -> list[list[str]]:
    anchor = f"<a id='{table_anchor}'"
    anchor_at = text.find(anchor)
    if anchor_at < 0:
        return []
    table_start = text.find("<table", anchor_at)
    table_end = text.find("</table>", table_start)
    if table_start < 0 or table_end < 0:
        return []
    table = text[table_start : table_end + len("</table>")]
    rows: list[list[str]] = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
        if cells:
            rows.append([_clean_html(cell) for cell in cells])
    return rows


def _parse_stack_function_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = _html_table_rows(text, "TblSE")
    parsed: list[dict[str, Any]] = []
    for cells in rows:
        if len(cells) < 7:
            continue
        try:
            exclusive_hits = int(cells[1].replace(",", ""))
            inclusive_hits = int(cells[3].replace(",", ""))
            base_rva = int(cells[4], 16)
            limit_rva = int(cells[5], 16)
            size_bytes = int(cells[6], 16)
        except (ValueError, IndexError):
            continue
        parsed.append({
            "function": cells[0],
            "exclusive_hits": exclusive_hits,
            "inclusive_hits": inclusive_hits,
            "exclusive_percent": cells[2],
            "base_rva": base_rva,
            "limit_rva": limit_rva,
            "size_bytes": size_bytes,
            "derived_exclusive_sample_weight_us": exclusive_hits * PROFILE_PERIOD_US,
        })
    return parsed


def _parse_call_relationships(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = _html_table_rows(text, "TblSN")
    lines: list[str] = []
    for cells in rows:
        if not cells:
            continue
        first = cells[0]
        if any(token in first for token in (
            "recurrent_backward_impl_body",
            "OmegaRuntime::run_worker",
            "OmegaRuntime::worker_loop",
            "std::thread::_Invoke",
            "gelu_derivative",
            "state_prelude_write",
            "::gelu",
        )):
            lines.append(first)
    joined = " ".join(lines)
    worker_chain_present = all(name in text for name in (
        "recurrent_backward_impl_body",
        "OmegaRuntime::run_worker",
        "OmegaRuntime::worker_loop",
    ))
    return {
        "relevant_call_relationship_rows": lines,
        "worker_call_chain_status": "OBSERVED" if worker_chain_present else "PARTIAL_OR_UNRESOLVED",
        "worker_call_chain": [
            "std::thread::_Invoke<OmegaRuntime::worker_loop>",
            "OmegaRuntime::worker_loop",
            "OmegaRuntime::run_worker",
            "recurrent_backward_impl_body",
        ] if worker_chain_present else [],
        "evidence_note": "Extracted from per-TID xperf Stack report; caller rows remain in the raw HTML artifact.",
    }


def _parse_profile_weights(path: Path, pid: int) -> list[dict[str, Any]]:
    process_re = re.compile(rf"^\s*python\.exe\s*\(\s*{pid}\s*\),\s*(\d+),\s*([\d.]+),\s*(.+?)\s*$", re.IGNORECASE)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = process_re.match(line)
        if not match:
            continue
        function = match.group(3).strip()
        if "omega_recurrent.dll!" not in function.lower():
            continue
        rows.append({
            "function": function,
            "xperf_profile_weight_us": int(match.group(1)),
            "system_usage_percent": float(match.group(2)),
            "weight_semantics": "flat Profile sample weight assigned to sampled instruction symbol; xperf unit microseconds",
        })
    return sorted(rows, key=lambda item: item["xperf_profile_weight_us"], reverse=True)


def _parse_raw_profile_events(path: Path, pid: int, module_base: int, range_us: Sequence[int]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], Counter[tuple[int, int]]]:
    events: list[dict[str, Any]] = []
    by_tid: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "profile_sample_count": 0,
        "omega_recurrent_sample_count": 0,
        "sampled_cpu_ids": set(),
        "first_timestamp_us": None,
        "last_timestamp_us": None,
    })
    pc_by_tid: Counter[tuple[int, int]] = Counter()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("SampledProfile,"):
            continue
        fields = line.split(",", 9)
        if len(fields) < 9:
            continue
        process = fields[2].strip()
        process_match = re.search(r"\((\d+)\)", process)
        if process_match is None or int(process_match.group(1)) != pid:
            continue
        timestamp_us = int(fields[1].strip())
        if timestamp_us < int(range_us[0]) or timestamp_us > int(range_us[1]):
            continue
        tid = int(fields[3].strip())
        pc = int(fields[4].strip(), 16)
        cpu = int(fields[5].strip())
        image_function = fields[7].strip()
        count = int(fields[8].strip())
        module_name = image_function.split("!", 1)[0].replace('"', "").strip().split("\\")[-1]
        rva = pc - module_base if module_name.lower() == "omega_recurrent.dll" else None
        row = {
            "timestamp_us_from_etl_start": timestamp_us,
            "pid": pid,
            "tid": tid,
            "pc": pc,
            "pc_hex": f"0x{pc:x}",
            "rva": rva,
            "rva_hex": f"0x{rva:x}" if rva is not None else None,
            "cpu": cpu,
            "thread_start_image_function": fields[6].strip(),
            "image_function": image_function,
            "module": module_name,
            "count": count,
            "sample_type": fields[9].strip() if len(fields) > 9 else None,
        }
        events.append(row)
        summary = by_tid[tid]
        summary["profile_sample_count"] += count
        summary["sampled_cpu_ids"].add(cpu)
        summary["first_timestamp_us"] = timestamp_us if summary["first_timestamp_us"] is None else min(summary["first_timestamp_us"], timestamp_us)
        summary["last_timestamp_us"] = timestamp_us if summary["last_timestamp_us"] is None else max(summary["last_timestamp_us"], timestamp_us)
        if module_name.lower() == "omega_recurrent.dll":
            summary["omega_recurrent_sample_count"] += count
            pc_by_tid[(tid, int(rva))] += count
    for summary in by_tid.values():
        summary["sampled_cpu_ids"] = sorted(summary["sampled_cpu_ids"])
        summary["nominal_profile_sample_weight_us"] = summary["profile_sample_count"] * PROFILE_PERIOD_US
        summary["weight_note"] = "Derived nominal sampled CPU weight = Profile sample count × verified 1000us sample period; keep count and WPT Profile Weight separate."
    return events, dict(by_tid), pc_by_tid


def _parse_image_base(image_dump: Path, pid: int, dll_path: Path) -> dict[str, Any]:
    expected_name = dll_path.name.lower()
    entries: list[dict[str, Any]] = []
    for line in image_dump.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith(("I-Start,", "I-DCStart,")):
            continue
        fields = line.split(",", 9)
        if len(fields) < 9:
            continue
        process_match = re.search(r"\((\d+)\)", fields[2])
        if process_match is None or int(process_match.group(1)) != pid:
            continue
        file_name = fields[8].strip().strip('"')
        if Path(file_name.replace("\\Device\\HarddiskVolume3", "")).name.lower() != expected_name:
            if expected_name not in file_name.lower():
                continue
        entries.append({
            "event": fields[0].strip(),
            "timestamp_us_from_etl_start": int(fields[1].strip()),
            "process": fields[2].strip(),
            "base_address": int(fields[3].strip(), 16),
            "end_address": int(fields[4].strip(), 16),
            "filename": file_name,
        })
    if not entries:
        raise RuntimeError(f"could not find image-load base for {dll_path.name}, PID {pid}")
    bases = {item["base_address"] for item in entries}
    if len(bases) != 1:
        raise RuntimeError(f"multiple DLL load bases found for one process: {entries}")
    return {"base_address": entries[0]["base_address"], "events": entries, "module_path": entries[0]["filename"]}


def _symbolize_pc(symbolizer: Path, dll_path: Path, rva: int, env: Mapping[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        [str(symbolizer), f"--obj={dll_path}", "--relative-address", "--functions", "--inlines", "--demangle", f"0x{rva:x}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env),
        check=False,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    source_lines = []
    for line in lines:
        match = re.search(r"(.+\.(?:cpp|cc|c|h|hpp)):(\d+):(\d+)$", line, re.IGNORECASE)
        if match:
            source_lines.append({"file": match.group(1), "line": int(match.group(2)), "column": int(match.group(3))})
    return {
        "rva": rva,
        "rva_hex": f"0x{rva:x}",
        "status": "RESOLVED" if source_lines else ("SYMBOL_ONLY" if lines else "UNRESOLVED"),
        "symbolizer_lines": lines,
        "source_locations": source_lines,
        "exit_code": int(result.returncode),
        "stderr": result.stderr.strip(),
    }


def _parse_profile_weight(path: Path, pid: int) -> tuple[list[dict[str, Any]], int]:
    pattern = re.compile(rf"^\s*python\.exe\s*\(\s*{pid}\s*\),\s*(\d+),\s*([\d.]+),\s*(.+?)\s*$", re.IGNORECASE)
    rows: list[dict[str, Any]] = []
    total_weight = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        weight = int(match.group(1))
        symbol = match.group(3).strip()
        total_weight += weight
        if "omega_recurrent.dll!" in symbol.lower():
            rows.append({
                "symbol": symbol,
                "profile_weight_us": weight,
                "system_usage_percent": float(match.group(2)),
                "weight_note": "Flat WPT Profile Weight reported by xperf, microseconds; this is not wall time or profiler time.",
            })
    return sorted(rows, key=lambda item: item["profile_weight_us"], reverse=True), total_weight


def _parse_stack_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    anchor = text.find("<a id='TblSE'")
    table_start = text.find("<table", anchor)
    table_end = text.find("</table>", table_start)
    if anchor < 0 or table_start < 0 or table_end < 0:
        return [], text
    table = text[table_start : table_end + len("</table>")]
    rows: list[dict[str, Any]] = []
    for raw_row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", raw_row, flags=re.IGNORECASE | re.DOTALL)
        cleaned = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", cell))).strip() for cell in cells]
        if len(cleaned) < 7:
            continue
        try:
            rows.append({
                "function": cleaned[0],
                "exclusive_hits": int(cleaned[1].replace(",", "")),
                "exclusive_percent": cleaned[2],
                "inclusive_hits": int(cleaned[3].replace(",", "")),
                "rva_start": int(cleaned[4], 16),
                "rva_end": int(cleaned[5], 16),
                "function_size_bytes": int(cleaned[6], 16),
            })
        except (ValueError, IndexError):
            continue
    return rows, text


def _parse_profile_sample(line: str, pid: int) -> dict[str, Any] | None:
    if not line.lstrip().startswith("SampledProfile,"):
        return None
    fields = line.split(",", 9)
    if len(fields) < 9:
        return None
    process = fields[2].strip()
    pid_match = re.search(r"\((\d+)\)", process)
    if pid_match is None or int(pid_match.group(1)) != pid:
        return None
    image_function = fields[7].strip()
    module = image_function.split("!", 1)[0].replace('"', "").strip().split("\\")[-1]
    pc = int(fields[4].strip(), 16)
    return {
        "timestamp_us": int(fields[1].strip()),
        "tid": int(fields[3].strip()),
        "pc": pc,
        "pc_hex": f"0x{pc:x}",
        "rva": None,
        "cpu": int(fields[5].strip()),
        "image_function": image_function,
        "module": module,
        "count": int(fields[8].strip()),
        "sample_type": fields[9].strip() if len(fields) > 9 else "",
    }


def run_analysis(
    *,
    etl: Path,
    marker_json: Path,
    time_windows: Path,
    build_identity_path: Path,
    output_root: Path,
    temp_root: Path,
    xperf: Path,
    symbolizer: Path,
) -> dict[str, Any]:
    for path in (etl, marker_json, time_windows, build_identity_path, xperf, symbolizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite WPT analysis output: {output_root}")
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"analysis output parent missing: {output_root.parent}")
    if not temp_root.parent.is_dir():
        raise FileNotFoundError(f"temporary analysis parent missing: {temp_root.parent}")
    output_root.mkdir(parents=False)
    command_root = output_root / "commands"
    command_root.mkdir()
    temp_root.mkdir(parents=False, exist_ok=False)
    command_records: list[dict[str, Any]] = []

    session = json.loads((marker_json.parent / "wpr_capture_session.json").read_text(encoding="utf-8"))
    markers = json.loads(marker_json.read_text(encoding="utf-8"))
    window_report = json.loads(time_windows.read_text(encoding="utf-8"))
    identity = json.loads(build_identity_path.read_text(encoding="utf-8"))
    if session.get("status") != "PASS" or int(session.get("worker_exit_code", -1)) != 0 or int(session.get("wpr_helper_exit_code", -1)) != 0:
        raise RuntimeError("WPR capture session or native marker worker did not pass")
    if identity.get("OMEGA_PROFILE_INTERNAL") is not False or identity.get("OMEGA_P2R_DIAGNOSTIC") is not False:
        raise RuntimeError("symbol build must keep profile and runtime diagnostic macros disabled")

    pid = int(markers["pid"])
    main_tid = int(markers["main_tid"])
    dll_path = Path(str(identity["dll_path"]))
    pdb_path = Path(str(identity["pdb_path"]))
    if _sha256(dll_path.read_bytes()) != identity["dll_sha256"]:
        raise RuntimeError("diagnostic DLL identity drift before WPT analysis")
    if _sha256(pdb_path.read_bytes()) != identity["pdb_sha256"]:
        raise RuntimeError("diagnostic PDB identity drift before WPT analysis")

    env = os.environ.copy()
    env["_NT_SYMBOL_PATH"] = str(pdb_path.parent)
    image_dump = temp_root / "image_loads.txt"
    image_cmd = [
        str(xperf), "-i", str(etl), "-o", str(image_dump), "-a", "dumper",
        "-provider", IMAGE_PROVIDER_GUID, "-add_rawdata", "-add_fieldnames",
    ]
    completed = subprocess.run(image_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    (command_root / "image_loads.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (command_root / "image_loads.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"xperf image-load dump failed: {completed.stderr}")
    command_records.append({"label": "image_loads", "argv": image_cmd, "exit_code": int(completed.returncode), "output_path": image_dump.as_posix()})
    image_events: list[dict[str, Any]] = []
    for line in image_dump.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith(("I-Start,", "I-DCStart,")):
            continue
        fields = line.split(",", 9)
        if len(fields) < 9:
            continue
        process_match = re.search(r"\((\d+)\)", fields[2])
        if process_match is None or int(process_match.group(1)) != pid:
            continue
        filename = fields[8].strip().strip('"')
        if dll_path.name.lower() not in filename.lower():
            continue
        image_events.append({
            "event": fields[0].strip(),
            "timestamp_us": int(fields[1].strip()),
            "base_address": int(fields[3].strip(), 16),
            "end_address": int(fields[4].strip(), 16),
            "filename": filename,
        })
    if not image_events:
        raise RuntimeError("ETL has no image-load event for the diagnostic DLL")
    base_addresses = {item["base_address"] for item in image_events}
    if len(base_addresses) != 1:
        raise RuntimeError(f"diagnostic DLL was loaded at multiple bases in target process: {image_events}")
    module_base = next(iter(base_addresses))

    trace_stats = subprocess.run([str(xperf), "-i", str(etl), "-a", "tracestats"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    if trace_stats.returncode != 0:
        raise RuntimeError(f"xperf tracestats failed: {trace_stats.stderr}")
    (output_root / "trace_stats.txt").write_text(trace_stats.stdout, encoding="utf-8")
    command_records.append({"label": "tracestats", "argv": [str(xperf), "-i", str(etl), "-a", "tracestats"], "exit_code": int(trace_stats.returncode), "output_path": (output_root / "trace_stats.txt").as_posix()})
    if "Lost Buffers : 0" not in trace_stats.stdout and "# Lost Buffers       : 0" not in trace_stats.stdout:
        raise RuntimeError("trace loss counters are not confirmed zero")
    if "Lost Events  : 0" not in trace_stats.stdout and "# Lost Events        : 0" not in trace_stats.stdout:
        raise RuntimeError("trace loss counters are not confirmed zero")

    freq = subprocess.run([str(xperf), "-i", str(etl), "-a", "profile", "-freq"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    if freq.returncode != 0:
        raise RuntimeError(f"xperf profile-frequency query failed: {freq.stderr}")
    (output_root / "profile_frequency.txt").write_text(freq.stdout, encoding="utf-8")
    command_records.append({"label": "profile_frequency", "argv": [str(xperf), "-i", str(etl), "-a", "profile", "-freq"], "exit_code": int(freq.returncode), "output_path": (output_root / "profile_frequency.txt").as_posix()})
    freq_rows = []
    for line in freq.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and parts[0] == "FreqZone":
            try:
                freq_rows.append({"start_us": int(parts[1]), "end_us": int(parts[2]), "period_us": float(parts[3])})
            except ValueError:
                continue
    if not freq_rows or any(row["period_us"] != 1000.0 for row in freq_rows):
        raise RuntimeError(f"unexpected/non-constant Profile sampling frequency: {freq_rows}")

    interval_results: list[dict[str, Any]] = []
    for interval in window_report["analyzed_intervals"]:
        replay_number = int(interval["replay"])
        start_us, end_us = [int(value) for value in interval["xperf_range_us"]]
        prefix = f"replay{replay_number}"
        raw_dump = temp_root / f"{prefix}_raw_events.txt"
        profile_detail = output_root / f"{prefix}_profile_detail.txt"
        stack_pid = output_root / f"{prefix}_stack_pid{pid}.html"
        raw_cmd = [str(xperf), "-i", str(etl), "-o", str(raw_dump), "-a", "dumper", "-range", str(start_us), str(end_us), "-add_rawdata", "-add_fieldnames"]
        profile_cmd = [str(xperf), "-i", str(etl), "-o", str(profile_detail), "-symbols", "-a", "profile", "-detail", "-range", str(start_us), str(end_us)]
        stack_cmd = [str(xperf), "-i", str(etl), "-o", str(stack_pid), "-symbols", "-a", "stack", "-butterfly", "1", "-pid", str(pid), "-event", "Profile", "-range", str(start_us), str(end_us)]
        for label, command in ((f"{prefix}_dumper", raw_cmd), (f"{prefix}_profile", profile_cmd), (f"{prefix}_stack_pid", stack_cmd)):
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
            (command_root / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
            (command_root / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"xperf {label} failed: {result.stderr}")
            command_records.append({"label": label, "argv": command, "exit_code": int(result.returncode), "output_path": str(command[command.index("-o") + 1])})

        profile_events: list[dict[str, Any]] = []
        tid_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {
            "sample_count": 0,
            "omega_recurrent_sample_count": 0,
            "cpus": set(),
            "pc_counts": Counter(),
        })
        for line in raw_dump.read_text(encoding="utf-8", errors="replace").splitlines():
            sample = _parse_profile_sample(line, pid)
            if sample is None:
                continue
            if sample["timestamp_us"] < start_us or sample["timestamp_us"] > end_us:
                continue
            profile_events.append(sample)
            tid_stats[sample["tid"]]["sample_count"] += sample["count"]
            tid_stats[sample["tid"]]["cpus"].add(sample["cpu"])
            if sample["module"].lower() == dll_path.name.lower():
                tid_stats[sample["tid"]]["omega_recurrent_sample_count"] += sample["count"]
                rva = sample["pc"] - module_base
                sample["rva"] = rva
                tid_stats[sample["tid"]]["pc_counts"][rva] += sample["count"]

        profile_weight_rows, pid_profile_weight_us = _parse_profile_weight(profile_detail, pid)
        process_tids = sorted(tid_stats)
        if not process_tids:
            raise RuntimeError(f"no target PID Profile events in marker interval {prefix}")

        all_tid_stack = _parse_stack_function_rows(stack_pid)
        recurrent_body = next((row for row in all_tid_stack if "recurrent_backward_impl_body" in row["function"]), None)
        if recurrent_body is None:
            raise RuntimeError(f"PDB symbol recurrent_backward_impl_body<0> absent in xperf stack report: {prefix}")
        body_start, body_end = int(recurrent_body["base_rva"]), int(recurrent_body["limit_rva"])

        per_thread: list[dict[str, Any]] = []
        hot_pcs: Counter[tuple[int, int]] = Counter()
        for tid in process_tids:
            tid_stack = output_root / f"{prefix}_stack_pid{pid}_tid{tid}.html"
            tid_cmd = [str(xperf), "-i", str(etl), "-o", str(tid_stack), "-symbols", "-a", "stack", "-butterfly", "1", "-pid", str(pid), "-tid", str(tid), "-event", "Profile", "-range", str(start_us), str(end_us)]
            tid_label = f"{prefix}_stack_tid{tid}"
            tid_result = subprocess.run(tid_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
            (command_root / f"{tid_label}.stdout.txt").write_text(tid_result.stdout, encoding="utf-8")
            (command_root / f"{tid_label}.stderr.txt").write_text(tid_result.stderr, encoding="utf-8")
            if tid_result.returncode != 0:
                raise RuntimeError(f"xperf per-thread stack failed for TID {tid}: {tid_result.stderr}")
            command_records.append({"label": tid_label, "argv": tid_cmd, "exit_code": int(tid_result.returncode), "output_path": tid_stack.as_posix()})
            function_rows, stack_text = _parse_stack_rows(tid_stack)
            recurrent_row = next((row for row in function_rows if "recurrent_backward_impl_body" in row["function"]), None)
            stats = tid_stats[tid]
            is_worker = "OmegaRuntime::worker_loop" in stack_text and "OmegaRuntime::run_worker" in stack_text
            tid_pcs = [(int(rva), int(count)) for rva, count in stats["pc_counts"].items()]
            tid_pcs.sort(key=lambda item: item[1], reverse=True)
            for rva, count in tid_pcs:
                if body_start <= rva < body_end:
                    hot_pcs[(rva, tid)] += count
            per_thread.append({
                "tid": tid,
                "role": "native_worker" if is_worker else "other_target_process_thread",
                "omega_runtime_worker_symbols_observed": is_worker,
                "sampled_cpus": sorted(stats["cpus"]),
                "profile_sample_count": int(stats["sample_count"]),
                "omega_recurrent_sample_count": int(stats["omega_recurrent_sample_count"]),
                "nominal_sample_weight_us": int(stats["sample_count"]) * PROFILE_PERIOD_US,
                "weight_derivation": "Profile sample count × constant 1000us ETL sampling period; nominal per-thread distribution.",
                "recurrent_backward_impl_body": recurrent_row,
                "exclusive_functions_top": function_rows[:12],
                "caller_chain_evidence": {
                    "worker_loop": "OmegaRuntime::worker_loop" in stack_text,
                    "run_worker": "OmegaRuntime::run_worker" in stack_text,
                    "recurrent_backward_impl_body": "recurrent_backward_impl_body<0>" in stack_text,
                    "report": tid_stack.name,
                },
                "top_pcs_inside_recurrent_backward_impl": [
                    {"rva": rva, "rva_hex": f"0x{rva:x}", "sample_count": count}
                    for rva, count in tid_pcs if body_start <= rva < body_end
                ][:10],
            })

        recurrent_hot_pcs = Counter()
        recurrent_hot_pcs_by_tid: dict[int, Counter[int]] = defaultdict(Counter)
        for tid, stats in tid_stats.items():
            for rva, count in stats["pc_counts"].items():
                if body_start <= rva < body_end:
                    recurrent_hot_pcs[int(rva)] += int(count)
                    recurrent_hot_pcs_by_tid[int(tid)][int(rva)] += int(count)
        top_rvas = recurrent_hot_pcs.most_common(30)
        symbolized: list[dict[str, Any]] = []
        for rva, count in top_rvas:
            symbol = _symbolize_pc(symbolizer, dll_path, rva, env)
            symbolized.append({
                **symbol,
                "sample_count": int(count),
                "nominal_sample_weight_us": int(count) * PROFILE_PERIOD_US,
                "sample_count_by_tid": {
                    str(tid): int(per_tid_counts[rva])
                    for tid, per_tid_counts in recurrent_hot_pcs_by_tid.items()
                    if per_tid_counts[rva]
                },
                "weight_note": "Sampled instruction hits × verified fixed profile period; do not treat profiler samples as wall time.",
            })

        interval_results.append({
            "fixture": str(interval["fixture"]),
            "replay": replay_number,
            "phase": "analyzed",
            "pid": pid,
            "main_tid": main_tid,
            "marker_begin_utc_ns": int(interval["begin_utc_ns"]),
            "marker_end_utc_ns": int(interval["end_utc_ns"]),
            "marker_elapsed_perf_counter_ns": int(interval["clean_backward_call_elapsed_ns"]),
            "xperf_range_us": [start_us, end_us],
            "xperf_range_quantization": "outward to microsecond boundaries; each edge extends less than 1us from exact paired marker UTC boundary",
            "sample_period_us": PROFILE_PERIOD_US,
            "profile_samples_total_for_pid": len(profile_events),
            "profile_sample_count_by_tid": {
                str(tid): int(stats["sample_count"])
                for tid, stats in sorted(tid_stats.items())
            },
            "omega_recurrent_samples_by_tid": {
                str(tid): int(stats["omega_recurrent_sample_count"])
                for tid, stats in sorted(tid_stats.items())
            },
            "thread_distribution": per_thread,
            "worker_tids": [item["tid"] for item in per_thread if item["omega_runtime_worker_symbols_observed"]],
            "native_worker_count_observed": sum(item["omega_runtime_worker_symbols_observed"] for item in per_thread),
            "recurrent_backward_function_range": {
                "symbol": recurrent_body["function"],
                "rva_start": recurrent_body["base_rva"],
                "rva_start_hex": f"0x{recurrent_body['base_rva']:x}",
                "rva_end_exclusive": recurrent_body["limit_rva"],
                "rva_end_hex": f"0x{recurrent_body['limit_rva']:x}",
                "size_bytes": recurrent_body["size_bytes"],
                "function_table_evidence": stack_pid.name,
            },
            "xperf_profile_weight_by_native_function": profile_weight_rows,
            "pid_total_xperf_profile_weight_us": pid_profile_weight_us,
            "xperf_profile_detail_report": profile_detail.name,
            "xperf_stack_all_target_threads_report": stack_pid.name,
            "recurrent_backward_hot_pc_ranges": symbolized,
            "raw_profile_event_dump_temporary": raw_dump.as_posix(),
            "raw_sample_event_count_for_target_pid": len(profile_events),
        })

        # Retain a compact per-interval sample inventory; the ETL remains source of truth.
        _write_json(output_root / f"{prefix}_sample_inventory.json", {
            "pid": pid,
            "replay": replay_number,
            "fixture": str(interval["fixture"]),
            "xperf_range_us": [start_us, end_us],
            "sample_period_us": PROFILE_PERIOD_US,
            "sampled_profile_events": [
                {
                    "timestamp_us": event["timestamp_us"],
                    "tid": event["tid"],
                    "pc": event["pc_hex"],
                    "rva": event.get("rva"),
                    "cpu": event["cpu"],
                    "image_function": event["image_function"],
                    "count": event["count"],
                }
                for event in profile_events
            ],
        })

    main_sample_counts = {
        f"replay{item['replay']}": item["profile_sample_count_by_tid"].get(str(main_tid), 0)
        for item in interval_results
    }
    report = {
        "schema": "omega-recurrent-backward-replay-native-wpt-analysis-v1",
        "status": "NATIVE_WPT_ANALYSIS_PASS",
        "route": "native",
        "etl_path": etl.as_posix(),
        "etl_bytes": int(etl.stat().st_size),
        "trace_stats": trace_stats.stdout,
        "etl_image_load_for_native_dll": image_events,
        "native_dll_base": module_base,
        "native_dll_base_hex": f"0x{module_base:x}",
        "native_dll": {
            "path": dll_path.as_posix(),
            "sha256": identity["dll_sha256"],
            "pdb_path": identity["pdb_path"],
            "pdb_sha256": identity["pdb_sha256"],
            "source_commit": identity["source_commit"],
            "source_sha256": identity["source_sha256"],
            "configuration": identity["configuration"],
            "compiler_flags": identity["compiler_flags"],
            "OMEGA_PROFILE_INTERNAL": identity["OMEGA_PROFILE_INTERNAL"],
            "OMEGA_P2R_DIAGNOSTIC": identity["OMEGA_P2R_DIAGNOSTIC"],
            "accepted_benchmark_dll_sha256": identity["accepted_benchmark_dll_sha256"],
            "accepted_benchmark_dll_replaced": False,
        },
        "analysis_tooling": {
            "xperf": str(xperf),
            "llvm_symbolizer": str(symbolizer),
            "profile_filter": "Kernel SampledProfile only, exact per-replay xperf range, target PID, all TIDs",
            "symbols": "local PDB via _NT_SYMBOL_PATH; optimizations/inlining unchanged",
            "symbolized_instruction_samples": True,
        },
        "timebase": window_report["timebase_validation"],
        "profile_sampling": {
            "frequency_zones": freq_rows,
            "profile_period_us": PROFILE_PERIOD_US,
            "profile_weight_unit": "WPT xperf Profile Weight in microseconds",
            "per_tid_nominal_weight_unit": "Profile sample count × 1000us constant period; labeled nominal",
        },
        "all_thread_scope": {
            "target_pid": pid,
            "main_tid": main_tid,
            "main_tid_profile_sample_counts_by_interval": main_sample_counts,
            "worker_tids_by_interval": {f"replay{item['replay']}": item["worker_tids"] for item in interval_results},
            "note": "Every target-PID TID with Profile samples is analyzed. Main-thread zero samples, if any, are not interpreted as zero work; C-ABI dispatch/final reduction remain inside marker intervals and are reported as unsampled if not captured.",
        },
        "intervals": interval_results,
        "clean_external_R_backward_reference": 1.7363279572184258,
        "profiler_ratio_recomputed": False,
        "raw_analysis_commands": command_records,
        "raw_profile_event_dump_policy": "Large per-interval dumper files remain in separate Temp analysis folder; workspace stores compact sample inventories and xperf reports.",
    }
    _write_json(output_root / "native_wpt_analysis_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etl", type=Path, default=DEFAULT_ETL)
    parser.add_argument("--marker-json", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--time-windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--build-identity", type=Path, default=DEFAULT_BUILD_IDENTITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--xperf", type=Path, default=DEFAULT_XPERF)
    parser.add_argument("--symbolizer", type=Path, default=DEFAULT_SYMBOLIZER)
    args = parser.parse_args(argv)
    report = run_analysis(
        etl=args.etl.resolve(),
        marker_json=args.marker_json.resolve(),
        time_windows=args.time_windows.resolve(),
        build_identity_path=args.build_identity.resolve(),
        output_root=args.output_root.resolve(),
        temp_root=args.temp_root.resolve(),
        xperf=args.xperf.resolve(),
        symbolizer=args.symbolizer.resolve(),
    )
    print(json.dumps({
        "status": report["status"],
        "interval_count": len(report["intervals"]),
        "worker_counts": {key: len(value) for key, value in report["all_thread_scope"]["worker_tids_by_interval"].items()},
        "report": (args.output_root.resolve() / "native_wpt_analysis_report.json").as_posix(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
checks: list[tuple[str, bool, str]] = []

def require(name: str, condition: bool, detail: str) -> None:
    checks.append((name, condition, detail))

benchmark = (ROOT / "src/benchmark.cpp").read_text(encoding="utf-8")
weights = (ROOT / "src/weights.cpp").read_text(encoding="utf-8")
kernels = (ROOT / "src/kernels.cpp").read_text(encoding="utf-8")
types = (ROOT / "include/cnrl/types.hpp").read_text(encoding="utf-8")
csv = (ROOT / "src/csv.cpp").read_text(encoding="utf-8")
cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
bandwidth = (ROOT / "apps/cnrl_bandwidth.cpp").read_text(encoding="utf-8")
calibrate = (ROOT / "apps/cnrl_calibrate.cpp").read_text(encoding="utf-8")
run_all = (ROOT / "scripts/run_all_gates.ps1").read_text(encoding="utf-8")
analyzer = (ROOT / "scripts/analyze_results.py").read_text(encoding="utf-8")
gate_app = (ROOT / "apps/cnrl_gate.cpp").read_text(encoding="utf-8")
state_source = (ROOT / "src/state.cpp").read_text(encoding="utf-8")
t0r_script = (ROOT / "scripts/run_t0r.ps1").read_text(encoding="utf-8")
t0m_script = (ROOT / "scripts/run_t0m.ps1").read_text(encoding="utf-8")
t0rm_script = (ROOT / "scripts/run_t0rm.ps1").read_text(encoding="utf-8")
run_all_script = (ROOT / "scripts/run_all_gates.ps1").read_text(encoding="utf-8")
experiment_contract = (ROOT / "docs/EXPERIMENT_CONTRACT.md").read_text(encoding="utf-8")
transition_bench = (ROOT / "apps/cnrl_transition_bench.cpp").read_text(encoding="utf-8")
build_windows = (ROOT / "scripts/build_windows.ps1").read_text(encoding="utf-8")
assembly_windows = (ROOT / "scripts/audit_windows_assembly.ps1").read_text(encoding="utf-8")

require("legacy excluded", "legacy/original_probes" not in cmake,
        "Original probes must remain archival and unlinked.")
require("no hidden self-test", "run_self_test" not in benchmark,
        "Benchmark execution must not run correctness campaigns before timing.")
require("worker spin barrier", "SpinBarrier barrier(worker_count)" in benchmark and "std::barrier" not in benchmark,
        "Microsecond recurrence phases use worker-only spinning, not a coordinator barrier.")
require("double-buffered state", "state_a" in benchmark and "state_b" in benchmark and "std::swap(current, next)" in benchmark,
        "Real recurrence must never overwrite the state still read by another worker.")
require("Bclone is byte-identical", "std::memcpy(shard.block(round), shard.block(0)" in weights,
        "Clone rounds copy the same bytes into distinct addresses.")
require("untied is separate", "WeightVariant::untied" in weights and "WeightVariant::clone" in weights,
        "Physical residency control and architectural untied weights cannot share a variant.")
require("clone invariants exported", "clone_hashes_equal" in csv and "clone_addresses_distinct" in csv,
        "Every raw row exposes the Bclone proof fields.")
require("same kernel registry", "run_kernel_unchecked(config.kernel, call)" in benchmark and
        "validate_kernel_call(validation_call)" in benchmark,
        "Static and recurrent gates invoke the same prevalidated kernel registry.")
require("no int64 hot-dot checks", "checked_add_i64" not in kernels and kernels.count("std::int64_t sum") == 1,
        "Only the scalar oracle may use int64; AVX2 hot loops must accumulate across D in int32.")
require("default spill-safe tile", "slot_tile = 4" in types,
        "Tile 4 is the default because the audited AVX2 tile-8 path can spill on some compilers.")
require("transition runs in workers", "transition_fixed_point_local" in benchmark and "threads.emplace_back" in benchmark,
        "The coordinator never transforms S×D state.")
require("global reduction is scalar-only", "if (worker == 0)" in benchmark and "transition_global_rms_reduce" in benchmark,
        "The only leader work is reduction of workers×slots partial sums.")
require("cold excluded by contract", "cold control must use round-window timing" in benchmark,
        "clflush cannot be included in the kernel comparison window.")
require("slot-correct bandwidth metric", "one_pass_weight_gb_per_second" in csv and "one_pass_weight_bytes" in benchmark,
        "Raw GMAC/s is never reused as DRAM GB/s when S>1.")
require("explicit transition scale", "projection_shift" in types and "state_multiplier" in types,
        "Residual input and dot-product output cannot be added without a declared scale.")
require("physical-core selection", "find_physical_core_index" in benchmark and "allow_smt_siblings" in benchmark,
        "Production gates reject SMT siblings unless explicitly requested.")
require("exact proportional sharding", "row_alignment = 1" in (ROOT / "apps/cnrl_gate.cpp").read_text(encoding="utf-8"),
        "Heterogeneous row allocation is not coarsened to arbitrary 64-row units.")
require("accounting recomputed from raw fields", "expected_mac" in (ROOT / "scripts/analyze_results.py").read_text(encoding="utf-8") and
        "expected_logical" in (ROOT / "scripts/analyze_results.py").read_text(encoding="utf-8"),
        "The analyzer independently rejects inflated MAC/byte counts.")
require("standalone transition benchmark", (ROOT / "apps/cnrl_transition_bench.cpp").exists(),
        "Transition implementations can be measured without the matrix kernel.")

require("recurrence scale recorded", "projection_shift = 12" in types and "clipped_cells" in csv,
        "Transition scales are explicit and clipping is exported; fixed=14 and RMS=12 remain probe defaults, not learned constants.")
require("bandwidth first-touch before timing",
        bandwidth.find("destination.fill_zero") < bandwidth.find("initialized.arrive_and_wait") and
        bandwidth.find("read_stream(buffers[worker].source.data(), bytes)") < bandwidth.find("initialized.arrive_and_wait"),
        "Read/copy pages are touched and warmed under worker affinity before the timer.")
require("auxiliary physical-core checks",
        "allow_smt_siblings" in bandwidth and "allow_smt_siblings" in calibrate and
        "find_physical_core_index" in bandwidth and "find_physical_core_index" in calibrate,
        "Bandwidth and calibration cannot silently feed SMT siblings into production gates.")
require("both Windows assembly audits in full run",
        "audit_windows_assembly.ps1" in run_all and "audit_windows_transitions.ps1" in run_all,
        "The authoritative Windows workflow audits both the matrix kernel and recurrent transition.")

require("weights invariant to shard partition",
        "deterministic_weight" in weights and "global_row" in weights and
        "round_key" in weights and "spec.worker_index" not in weights,
        "The logical matrix must depend on global coordinates, not the worker receiving a row.")
require("global weight signature",
        "fnv1a64_update(signature, shard.block(round)" in weights,
        "The base matrix signature is streamed in global row order across shards.")
require("state reset module",
        "initialize_state" in state_source and "copy_state_shard" in state_source and
        "clear_state_shard" in state_source,
        "State generation/reset is reusable and remains outside benchmark timing.")
require("CSV schema identity",
        "project_version" in csv and "seed" in csv and "projection_shift" in csv,
        "Every result records the code schema and numerical trajectory controls.")
require("analyzer balanced pairs",
        "unbalanced shared/Bclone repetitions" in analyzer and
        "unbalanced repeat/fused repetitions" in analyzer,
        "Equal-length A/B and repeat/fused samples are mandatory.")
require("CLI pipeline test wired",
        "cnrl_cli_pipeline_tests" in cmake and (ROOT / "tests/test_cli_pipeline.py").exists(),
        "The executable-to-CSV-to-analyzer path and tamper rejection are tested.")
require("transition analyzer wired",
        (ROOT / "scripts/analyze_transition_results.py").exists() and
        "analyze_transition_results.py" in cmake and
        "transition_analysis.md" in run_all_script,
        "Transition CSVs are audited independently instead of being silently skipped by the gate analyzer.")
require("transition benchmark CLI test wired",
        "cnrl_transition_bench_cli_tests" in cmake and
        (ROOT / "tests/test_transition_bench_cli.py").exists(),
        "Independent-chain reset, default scale, and cell accounting are executable tests.")
require("slot order thermal control",
        "Get-CycledOrder" in t0m_script and "$slotOrder" in t0m_script,
        "T0-M does not always execute slot counts in ascending order.")
require("bridge/recurrent interleaving",
        "Interleave the frozen bridge and recurrent cell" in t0rm_script and
        "Invoke-BridgeCell" in t0rm_script and "Invoke-RecurrentCell" in t0rm_script,
        "T0-RM retention comparisons are not separated into distant thermal phases.")
require("average static weight target",
        "average-weight-kib-per-core" in gate_app,
        "Static weight size is described as an average under proportional sharding.")
require("group RMS portability warning",
        "función matemática cambia" in experiment_contract and
        "global-rms" in experiment_contract,
        "Group-RMS cannot be presented as invariant across shard layouts.")

require("Windows build generator is not pinned",
        "Visual Studio 17 2022" not in build_windows and 'Generator = "Auto"' in build_windows and
        "Import-CnrlVcVars" in build_windows,
        "Build Tools, Ninja, and later MSVC toolsets must be supported without a silent local patch.")
require("exact standalone bridges exposed",
        "[int]$D = 512" in t0r_script and "[switch]$SquareOutput" in t0r_script and
        "[int]$D = 512" in t0m_script and "[switch]$SquareOutput" in t0m_script and
        (ROOT / "scripts/run_exact_bridges.ps1").exists(),
        "Frozen gates can reproduce the D=1472 square recurrent geometry literally.")
require("exact bridges run by default",
        "SkipExactStandaloneBridges" in run_all_script and
        "if (-not $SkipExactStandaloneBridges)" in run_all_script,
        "The authoritative full run closes the standalone D=1472 bridge unless explicitly shortened.")
require("transition chains are reset outside timing",
        "chain_length" in transition_bench and
        transition_bench.find("reset_local_state();") < transition_bench.find("chain_begin = clock.now()") and
        "state_reset_between_chains" in transition_bench,
        "Transition cost and R-step numerical drift cannot be confused with one 1000-step synthetic chain.")
require("fixed-point shift is separated from RMS",
        "FixedProjectionShift = 14" in t0rm_script and "RmsProjectionShift = 12" in t0rm_script and
        "projection_shift = 14U" in gate_app and "projection_shift = 14U" in transition_bench and
        (ROOT / "scripts/run_fixed_shift_sweep.ps1").exists(),
        "Fixed-point saturation must be calibrated independently of RMS transitions.")
require("targeted patch validation",
        (ROOT / "scripts/run_v041_patch_validation.ps1").exists(),
        "The user can revalidate only build/ABI/bridge/clipping changes without rerunning unrelated gates.")
require("bare-metal validation provenance",
        (ROOT / "docs/FIRST_BARE_METAL_VALIDATION.md").exists() and
        "evidencia externa reportada" in (ROOT / "docs/FIRST_BARE_METAL_VALIDATION.md").read_text(encoding="utf-8"),
        "External laptop results are recorded without being misrepresented as locally recomputed evidence.")
require("transition CSV provenance",
        "project_version,seed,require_affinity,allow_smt_siblings" in transition_bench and
        "mixed or missing project versions" in (ROOT / "scripts/analyze_transition_results.py").read_text(encoding="utf-8"),
        "Transition microbench rows carry version, seed and affinity semantics.")
require("MSVC audit ignores ABI-only saves",
        "Get-ArithmeticWindow" in assembly_windows and "outside the arithmetic window" in assembly_windows,
        "Windows x64 XMM6-XMM15 prologue/epilogue saves are not classified as hot-loop spills.")

failed = [item for item in checks if not item[1]]
for name, passed, detail in checks:
    print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
print(f"\n{len(checks)-len(failed)}/{len(checks)} source invariants passed")
if failed:
    sys.exit(1)

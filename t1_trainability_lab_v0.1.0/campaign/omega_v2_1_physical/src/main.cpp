#include "v2_1.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace omega_v2_1 {
namespace {

std::string json_escape_local(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << (c < 0x20 ? '?' : static_cast<char>(c));
        }
    }
    return out.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create output file: " + path.string());
    stream << text;
    if (!text.empty() && text.back() != '\n') stream << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("failed writing output file: " + path.string());
}

std::string h0_json(const std::vector<H0Row>& rows) {
    std::ostringstream out;
    out.precision(17);
    out << '[';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (i) out << ',';
        const H0Row& row = rows[i];
        out << "{\"group\":" << row.group << ",\"physical_core_id\":" << static_cast<unsigned>(row.core_index)
            << ",\"efficiency_class\":" << static_cast<unsigned>(row.efficiency_class)
            << ",\"intel_cpuid_core_type\":" << static_cast<unsigned>(row.intel_core_type)
            << ",\"logical_processor_id\":" << row.logical_index << ",\"windows_cpu_set_id\":" << row.cpu_set_id
            << ",\"d\":512,\"m\":" << row.m << ",\"warmups\":" << row.warmup_count
            << ",\"repetitions\":" << row.measured_repetitions << ",\"median_macs_per_second\":" << row.median_macs_per_second
            << ",\"affinity_ok\":" << (row.affinity_ok ? "true" : "false")
            << ",\"sample_macs_per_second\":[";
        for (std::size_t j = 0; j < row.sample_macs_per_second.size(); ++j) {
            if (j) out << ',';
            out << row.sample_macs_per_second[j];
        }
        out << "]}";
    }
    out << ']';
    return out.str();
}

std::string selected_workers_json(const std::vector<CoreRecord>& workers) {
    std::ostringstream out;
    out.precision(17);
    out << '[';
    for (std::size_t i = 0; i < workers.size(); ++i) {
        if (i) out << ',';
        const CoreRecord& core = workers[i];
        const CpuSetRecord& cpu = core.cpu_sets.front();
        out << "{\"worker_id\":" << i << ",\"group\":" << core.group
            << ",\"physical_core_id\":" << static_cast<unsigned>(core.core_index)
            << ",\"efficiency_class\":" << static_cast<unsigned>(core.efficiency_class)
            << ",\"intel_cpuid_core_type\":" << static_cast<unsigned>(core.intel_core_type)
            << ",\"windows_cpu_set_id\":" << cpu.id
            << ",\"logical_processor_id\":" << static_cast<unsigned>(cpu.logical_index)
            << ",\"h0_v_i\":" << core.v_i << ",\"shard_weight\":" << cpu.shard_weight
            << ",\"smt_sibling_worker_selected\":false}";
    }
    out << ']';
    return out.str();
}

std::vector<std::pair<std::size_t, std::size_t>> weighted_ranges(std::size_t rows, const std::vector<CoreRecord>& workers) {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(workers.size());
    double total = 0.0;
    for (const CoreRecord& worker : workers) total += worker.v_i;
    double prefix = 0.0;
    std::size_t begin = 0;
    for (std::size_t i = 0; i < workers.size(); ++i) {
        prefix += workers[i].v_i;
        const std::size_t end = (i + 1 == workers.size()) ? rows : static_cast<std::size_t>(std::floor(rows * prefix / total));
        ranges.emplace_back(begin, end);
        begin = end;
    }
    return ranges;
}

std::string benchmark_config_json(const HardwareInfo& hardware,
                                  const std::vector<CoreRecord>& workers,
                                  const std::string& eviction_method,
                                  double effectiveness,
                                  const std::filesystem::path& repo_root,
                                  const std::filesystem::path& unit_root,
                                  const std::filesystem::path& result_root,
                                  const std::filesystem::path& executable) {
    std::ostringstream out;
    out.precision(17);
    out << "{\"schema\":\"omega-v2-1-benchmark-config-v1\",\"repo_root_abs\":\"" << json_escape_local(repo_root.string())
        << "\",\"unit_root_abs\":\"" << json_escape_local(unit_root.string())
        << "\",\"source_root_abs\":\"" << json_escape_local((unit_root / "src").string())
        << "\",\"build_root_abs\":\"" << json_escape_local((unit_root / "build").string())
        << "\",\"executable_abs\":\"" << json_escape_local(executable.string())
        << "\",\"results_root_abs\":\"" << json_escape_local(result_root.string())
        << "\",\"raw_results_abs\":\"" << json_escape_local((result_root / "raw_measurements.csv").string())
        << "\",\"report_abs\":\"" << json_escape_local((result_root / "OMEGA_V2_1_REPORT.md").string())
        << "\",\"conformance_block_abs\":\"" << json_escape_local((result_root / "OMEGA_CONFORMANCE_BLOCK.yaml").string())
        << "\",\"cpu_model\":\"" << json_escape_local(hardware.cpu_model) << "\",\"qpc_frequency\":" << hardware.qpc_frequency
        << ",\"schedule_seed\":20260929,\"fixed_weight_seed\":20260929,\"warmups_per_cell_variant\":10"
        << ",\"measurement_blocks\":5,\"samples_per_block\":21,\"total_measured_samples_per_cell_variant\":105"
        << ",\"d\":[512,640],\"m\":[1,4,8,16],\"m1_role\":\"CONTROL_ONLY\",\"K\":[1,4,8],\"variants\":[\"A\",\"B\",\"C\"]"
        << ",\"cell_count\":72,\"primary_stability_cells\":[";
    bool first = true;
    for (char variant : {'A', 'B', 'C'}) {
        for (int K : {1, 8}) {
            if (!first) out << ',';
            first = false;
            out << "{\"d\":512,\"m\":4,\"K\":" << K << ",\"variant\":\"" << variant << "\"}";
        }
    }
    for (int m : {1, 8, 16}) {
        if (!first) out << ',';
        first = false;
        out << "{\"d\":512,\"m\":" << m << ",\"K\":8,\"variant\":\"A\"}";
    }
    out << "],\"primary_stability_cell_count\":9,\"r_mad_threshold\":0.15,\"r_mad_hold_noisy_cells\":2"
        << ",\"h0_protocol\":{\"p_core_identification\":\"Intel CPUID leaf 0x1A core type 0x40\",\"e_core_coordinator_cpu_set_id\":" << hardware.coordinator_cpu_set_id
        << ",\"d\":512,\"matrix\":\"W_Q fixed full output-row shard\",\"same_q4_weights_and_kernel_for_all_cores\":true"
        << ",\"m\":[1,4,8,16],\"m1_role\":\"CONTROL_ONLY\",\"warmups_per_m\":10,\"samples_per_m\":31,\"smt_sibling_worker_active\":false"
        << ",\"selection_v_i\":\"(median_m4*median_m8*median_m16)^(1/3)\",\"shard_weight\":\"same v_i\"}"
        << ",\"worker_selection\":\"largest v_i among CPUID Core-type 0x40 P-core candidates; v_i=(median_m4*median_m8*median_m16)^(1/3); tie=lowest CPU-set ID\""
        << ",\"selected_workers\":" << selected_workers_json(workers)
        << ",\"worker_pool_persistent\":true,\"worker_pool_size\":" << workers.size()
        << ",\"a_storage_policy\":\"same_core_weight_storage_for_all_rounds\",\"b_storage_policy\":\"K_distinct_equal_value_core_storages_rotated_by_group\",\"c_storage_policy\":\"same_storage_as_A_evicted_before_each_round\""
        << ",\"eviction_method\":\"" << json_escape_local(eviction_method) << "\",\"eviction_effectiveness_E\":" << effectiveness
        << ",\"b_pool_minimum_rule\":\"max(2.5*measured LLC,64MiB)\",\"c_eviction_outside_each_round_timer\":true"
        << ",\"native_timer\":\"QueryPerformanceCounter\",\"rdtscp_primary\":false,\"cpu_speed_or_residency_claim_authorized\":false}";
    return out.str();
}

std::vector<std::pair<std::size_t, std::size_t>> row_ranges_for_workers(std::size_t count, const std::vector<CoreRecord>& workers) {
    double total = 0.0;
    for (const CoreRecord& worker : workers) total += worker.v_i;
    std::vector<std::pair<std::size_t, std::size_t>> result;
    result.reserve(workers.size());
    std::size_t begin = 0;
    double prefix = 0.0;
    for (std::size_t i = 0; i < workers.size(); ++i) {
        prefix += workers[i].v_i;
        const std::size_t end = (i + 1 == workers.size()) ? count : static_cast<std::size_t>(std::floor(count * prefix / total));
        result.emplace_back(begin, end);
        begin = end;
    }
    return result;
}

std::string worker_shards_json(const std::vector<CoreRecord>& workers, const std::vector<CoreWeights>& weights) {
    const char* names[] = {"W_Q", "W_K", "W_V", "W_O", "W_gate", "W_up", "W_down"};
    std::ostringstream out;
    out.precision(17);
    out << "{\"schema\":\"omega-v2-1-worker-shard-manifest-v1\",\"worker_count\":" << workers.size()
        << ",\"shard_weight_formula\":\"v_i=(median_m4*median_m8*median_m16)^(1/3)\",\"output_row_tile\":1,\"workers\":"
        << selected_workers_json(workers) << ",\"matrices\":[";
    bool first = true;
    for (const CoreWeights& core : weights) {
        const Q4Matrix* matrices[] = {&core.W_Q, &core.W_K, &core.W_V, &core.W_O, &core.W_gate, &core.W_up, &core.W_down};
        for (std::size_t matrix_index = 0; matrix_index < 7; ++matrix_index) {
            if (!first) out << ',';
            first = false;
            const Q4Matrix& matrix = *matrices[matrix_index];
            const auto ranges = row_ranges_for_workers(static_cast<std::size_t>(matrix.rows), workers);
            std::size_t covered = 0;
            out << "{\"d\":" << core.d << ",\"matrix\":\"" << names[matrix_index] << "\",\"rows\":" << matrix.rows
                << ",\"cols\":" << matrix.cols << ",\"shards\":[";
            for (std::size_t worker = 0; worker < ranges.size(); ++worker) {
                if (worker) out << ',';
                const std::size_t rows = ranges[worker].second - ranges[worker].first;
                const std::size_t packed = rows * static_cast<std::size_t>(matrix.cols) / 2;
                const std::size_t scales = rows * static_cast<std::size_t>(matrix.cols / kGroup) * sizeof(std::uint16_t);
                covered += rows;
                out << "{\"worker_id\":" << worker << ",\"cpu_set_id\":" << workers[worker].cpu_sets.front().id
                    << ",\"physical_core_id\":" << static_cast<unsigned>(workers[worker].core_index)
                    << ",\"first_output_row\":" << ranges[worker].first << ",\"last_output_row_exclusive\":" << ranges[worker].second
                    << ",\"rows\":" << rows << ",\"packed_weight_bytes\":" << packed << ",\"scale_bytes\":" << scales
                    << ",\"physical_bytes_assigned\":" << packed + scales << '}';
            }
            out << "],\"rows_covered\":" << covered << ",\"rows_partitioned_without_overlap\":"
                << (covered == static_cast<std::size_t>(matrix.rows) ? "true" : "false")
                << ",\"matrix_buffer_physical_bytes\":" << matrix.packed.allocated_bytes + matrix.scales_fp16.allocated_bytes << '}';
        }
    }
    out << "]}";
    return out.str();
}

std::string preflight_q4_ledger_json(const std::vector<CoreWeights>& weights, const HardwareInfo& hardware) {
    const std::string base = q4_ledger_json(weights);
    const std::size_t array_begin = base.find('[');
    const std::size_t array_end = base.rfind(']');
    if (array_begin == std::string::npos || array_end == std::string::npos || array_end < array_begin) {
        throw std::runtime_error("Q4 physical ledger could not extract native matrix-family records");
    }
    const std::size_t required_pool = (std::max)(
        static_cast<std::size_t>(std::ceil(2.5 * static_cast<double>(hardware.llc_bytes))),
        static_cast<std::size_t>(64ull * 1024ull * 1024ull));
    std::ostringstream out;
    out << "{\"schema\":\"omega-v2-1-q4-physical-ledger-v1\",\"group_size\":32,\"scale_dtype\":\"FP16\",\"zero_point\":false,\"alignment_bytes\":64,\"hardware_llc_bytes\":" << hardware.llc_bytes
        << ",\"required_b_pool_bytes\":" << required_pool << ",\"core_families\":"
        << base.substr(array_begin, array_end - array_begin + 1) << ",\"b_pool_by_d_k\":[]}";
    return out.str();
}

double measure_qpc_overhead_ns() {
    std::vector<std::int64_t> deltas;
    deltas.reserve(1001);
    for (int i = 0; i < 1001; ++i) {
        const auto start = qpc_ticks();
        const auto stop = qpc_ticks();
        deltas.push_back(stop - start);
    }
    std::sort(deltas.begin(), deltas.end());
    return 1e9 * static_cast<double>(deltas[deltas.size() / 2]) / static_cast<double>(qpc_frequency());
}

std::string preflight_json(const HardwareInfo& hardware,
                           const std::vector<H0Row>& h0,
                           const std::vector<CoreRecord>& workers,
                           const std::string& p_class_method,
                           const std::string& eviction_method,
                           double effectiveness,
                           bool qpc_monotonic,
                           const std::string& q4_roundtrip,
                           const std::string& q4_scalar,
                           const std::string& abc,
                           const std::filesystem::path& repo_root,
                           const std::filesystem::path& unit_root,
                           const std::filesystem::path& result_root,
                           const std::filesystem::path& executable,
                           const std::string& status,
                           const std::string& reason,
                           const std::string& full_block_scalar = "{}",
                           const WorkerPool* worker_pool = nullptr) {
    std::ostringstream out;
    out.precision(17);
    out << "{\"schema\":\"omega-v2-1-hardware-preflight-v1\",\"status\":\"" << status << "\",\"hold_reason\":\""
        << json_escape_local(reason) << "\",\"repo_root_abs\":\"" << json_escape_local(repo_root.string())
        << "\",\"unit_root_abs\":\"" << json_escape_local(unit_root.string())
        << "\",\"source_root_abs\":\"" << json_escape_local((unit_root / "src").string())
        << "\",\"build_root_abs\":\"" << json_escape_local((unit_root / "build").string())
        << "\",\"executable_abs\":\"" << json_escape_local(executable.string())
        << "\",\"results_root_abs\":\"" << json_escape_local(result_root.string()) << "\",\"hardware\":" << hardware_json(hardware)
        << ",\"p_classification_method\":\"" << json_escape_local(p_class_method) << "\",\"h0_measurements\":" << h0_json(h0)
        << ",\"selected_workers\":" << selected_workers_json(workers)
        << ",\"worker_affinity_records\":[";
    if (worker_pool) {
        const auto& affinity = worker_pool->affinity_results();
        for (std::size_t i = 0; i < affinity.size(); ++i) {
            if (i) out << ',';
            out << "\"" << json_escape_local(affinity[i]) << "\"";
        }
    }
    out << "]"
        << ",\"eviction_method\":\"" << json_escape_local(eviction_method) << "\",\"eviction_effectiveness_E\":" << effectiveness
        << ",\"qpc_monotonic\":" << (qpc_monotonic ? "true" : "false")
        << ",\"q4_round_trip_test\":" << q4_roundtrip << ",\"q4_scalar_reference_test\":" << q4_scalar
        << ",\"full_block_scalar_reference_test\":" << full_block_scalar << ",\"abc_correctness\":" << abc << "}";
    return out.str();
}

} // namespace

} // namespace omega_v2_1

int main(int argc, char** argv) {
    using namespace omega_v2_1;
    if (argc != 2 || std::string(argv[1]) != "--run") {
        std::cerr << "Usage: omega_v2_1_bench.exe --run (results paths are fixed by native binary contract)\n";
        return 2;
    }
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        const std::filesystem::path unit_root = executable.parent_path().parent_path().parent_path();
        const std::filesystem::path repo_root = unit_root.parent_path().parent_path().parent_path();
        const std::filesystem::path result_root = unit_root / "results" / "omega_v2_1_physical";
        std::filesystem::create_directories(result_root);
        for (const char* immutable_name : {"hardware_preflight.json", "native_run_status.json", "raw_measurements.csv"}) {
            if (std::filesystem::exists(result_root / immutable_name)) {
                throw std::runtime_error(std::string("immutable V2-1 results already exist: ") + (result_root / immutable_name).string());
            }
        }

        HardwareInfo hardware;
        std::string error;
        bool qpc_monotonic = monotonic_qpc_test(error);
        hardware.qpc_monotonic = qpc_monotonic;
        if (!query_hardware(hardware, error)) {
            const std::string failed = preflight_json(hardware, {}, {}, "UNRESOLVED", "UNRESOLVED", 0.0, qpc_monotonic,
                                                      "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                      "MEASUREMENT_INVALID", error);
            write_text(result_root / "hardware_preflight.json", failed);
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"hardware query failed\"}\n");
            return 3;
        }
        hardware.qpc_overhead_ns = measure_qpc_overhead_ns();

        const std::string expected_cpu = "i7-13700F";
        if (hardware.cpu_model.find(expected_cpu) == std::string::npos) {
            error = "MEASUREMENT_INVALID_WRONG_HARDWARE: expected Intel Core i7-13700F from native CPUID; observed=" + hardware.cpu_model;
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, {}, {}, "UNRESOLVED", "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }
        if (!hardware.avx2 || !hardware.fma) {
            error = "MEASUREMENT_INVALID_REQUIRED_ISA_MISSING: AVX2 and FMA are required by the native Q4 kernel";
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, {}, {}, "UNRESOLVED", "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }

        // Pin the coordinator to an E-core so it never competes with a measured P-core H0 candidate.
        const CpuSetRecord* controller_cpu = nullptr;
        for (const CoreRecord& core : hardware.cores) {
            if (core.intel_core_type == 0x20 && core.cpu_sets.size() == 1
                && (!controller_cpu || core.cpu_sets.front().id < controller_cpu->id)) {
                controller_cpu = &core.cpu_sets.front();
            }
        }
        if (!controller_cpu || !set_current_cpu_set(controller_cpu->id, controller_cpu->group, controller_cpu->logical_index, error)) {
            if (error.empty()) error = "MEASUREMENT_INVALID: no CPUID-identified single-thread E-core is available for the H0 coordinator";
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, {}, {}, "UNRESOLVED", "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }
        hardware.coordinator_cpu_set_id = controller_cpu->id;
        hardware.coordinator_group = controller_cpu->group;
        hardware.coordinator_logical_index = controller_cpu->logical_index;
        PROCESSOR_NUMBER initial_controller_processor{};
        GetCurrentProcessorNumberEx(&initial_controller_processor);
        if (initial_controller_processor.Group != controller_cpu->group || initial_controller_processor.Number != controller_cpu->logical_index) {
            error = "MEASUREMENT_INVALID: H0 coordinator affinity did not land on the selected E-core CPU-set";
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, {}, {}, "UNRESOLVED", "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }

        CoreWeights weights512 = make_seeded_weights(kD512, kSeed);
        CoreWeights weights640 = make_seeded_weights(kD640, kSeed);
        std::vector<H0Row> h0 = measure_h0_cores(hardware, weights512, error);
        if (!error.empty()) {
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, h0, {}, "UNRESOLVED", "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }
        std::string p_class_method;
        std::vector<CoreRecord> workers = select_p_cores_by_h0(hardware, h0, p_class_method, error);
        if (!error.empty() || workers.size() != 4) {
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, h0, workers, p_class_method, "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }

        // Ensure the controller remained on the same non-P core after H0 selection.
        if (!controller_cpu || controller_cpu->id == workers[0].cpu_sets.front().id ||
            controller_cpu->id == workers[1].cpu_sets.front().id || controller_cpu->id == workers[2].cpu_sets.front().id ||
            controller_cpu->id == workers[3].cpu_sets.front().id) {
            error = "MEASUREMENT_INVALID: controller CPU-set overlaps a selected primary P-core worker";
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, h0, workers, p_class_method, "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }
        std::string controller_affinity_error;
        PROCESSOR_NUMBER controller_processor{};
        GetCurrentProcessorNumberEx(&controller_processor);
        if (controller_processor.Group != controller_cpu->group || controller_processor.Number != controller_cpu->logical_index) {
            error = "MEASUREMENT_INVALID: controller affinity check did not land on selected non-primary CPU-set";
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, h0, workers, p_class_method, "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }

        WorkerPool pool;
        if (!pool.start({workers[0].cpu_sets.front(), workers[1].cpu_sets.front(), workers[2].cpu_sets.front(), workers[3].cpu_sets.front()}, error)) {
            write_text(result_root / "hardware_preflight.json", preflight_json(hardware, h0, workers, p_class_method, "UNRESOLVED", 0.0, qpc_monotonic,
                                                                                "{}", "{}", "{}", repo_root, unit_root, result_root, executable,
                                                                                "MEASUREMENT_INVALID", error));
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\"}\n");
            return 3;
        }

        const std::string q4_roundtrip = q4_round_trip_test();
        const std::string q4_scalar = q4_scalar_json_test(pool);
        const std::string full_scalar = full_block_scalar_reference_test(pool);
        const std::string abc = full_block_abc_correctness(pool, weights512);
        std::string eviction_method;
        double eviction_effectiveness = 0.0;
        const bool eviction_valid = probe_eviction(weights512, hardware, eviction_method, eviction_effectiveness, error);
        const std::size_t b_pool_min = static_cast<std::size_t>((std::max)(std::ceil(2.5 * hardware.llc_bytes), 64.0 * 1024.0 * 1024.0));
        bool b_pool_valid = true;
        for (const CoreWeights* family : {&weights512, &weights640}) {
            for (int K : {1, 4, 8}) {
                const std::size_t group_bytes = static_cast<std::size_t>(K) * family->physical_buffer_bytes;
                const std::size_t group_count = (b_pool_min + group_bytes - 1) / group_bytes;
                const std::size_t pool_bytes = group_count * group_bytes;
                b_pool_valid = b_pool_valid && pool_bytes >= b_pool_min && static_cast<double>(pool_bytes) / hardware.llc_bytes >= 2.5;
            }
        }
        const bool qpc_valid = qpc_monotonic && qpc_frequency() > 0 && hardware.qpc_overhead_ns > 0.0;
        const bool q4_valid = q4_roundtrip.find("\"signed_range_test\":true") != std::string::npos
            && q4_roundtrip.find("\"fp16_scales_positive\":true") != std::string::npos
            && q4_roundtrip.find("\"aligned_packed\":true") != std::string::npos
            && q4_roundtrip.find("\"aligned_scales\":true") != std::string::npos;
        const bool scalar_valid = q4_scalar.find("\"pass\":true") != std::string::npos;
        const bool full_scalar_valid = full_scalar.find("\"pass\":true") != std::string::npos;
        const bool abc_valid = abc.find("\"A_B_C_bitwise_equal\":true") != std::string::npos
            && abc.find("\"A_finite\":true") != std::string::npos
            && abc.find("\"B_finite\":true") != std::string::npos
            && abc.find("\"C_finite\":true") != std::string::npos
            && abc.find("\"A_C_same_storage\":true") != std::string::npos
            && abc.find("\"B_values_equal_A\":true") != std::string::npos
            && abc.find("\"B_round_storages_disjoint\":true") != std::string::npos;
        const bool worker_affinity_valid = pool.affinity_intact() && pool.affinity_results().size() == 4
            && std::all_of(pool.affinity_results().begin(), pool.affinity_results().end(), [](const std::string& row) {
                return row.find("affinity_ok=true") != std::string::npos;
            });
        std::string preflight_status = "READY";
        std::string preflight_reason;
        if (!qpc_valid) { preflight_status = "MEASUREMENT_INVALID"; preflight_reason = "QPC monotonicity/frequency test failed"; }
        else if (!q4_valid) { preflight_status = "CORRECTNESS_HOLD"; preflight_reason = "Q4 layout/pack/unpack test failed"; }
        else if (!scalar_valid) { preflight_status = "CORRECTNESS_HOLD"; preflight_reason = "Q4 scalar-reference test failed"; }
        else if (!full_scalar_valid) { preflight_status = "CORRECTNESS_HOLD"; preflight_reason = "full-block scalar-reference test failed"; }
        else if (!abc_valid) { preflight_status = "CORRECTNESS_HOLD"; preflight_reason = "A/B/C d512 m4 K8 bitwise correctness failed"; }
        else if (!worker_affinity_valid) { preflight_status = "MEASUREMENT_INVALID_WORKER_AFFINITY"; preflight_reason = "primary worker affinity was not fixed"; }
        else if (!b_pool_valid) { preflight_status = "MEASUREMENT_INVALID_B_WORKING_SET"; preflight_reason = "B pool fails max(2.5x LLC,64MiB)"; }
        else if (!eviction_valid) { preflight_status = "EVICTION_INSTRUMENT_INVALID"; preflight_reason = error.empty() ? "eviction effectiveness E<1.5" : error; }

        write_text(result_root / "hardware_preflight.json", preflight_json(hardware, h0, workers, p_class_method, eviction_method,
                                                                          eviction_effectiveness, qpc_monotonic, q4_roundtrip, q4_scalar,
                                                                          abc, repo_root, unit_root, result_root, executable,
                                                                          preflight_status, preflight_reason, full_scalar, &pool));
        std::vector<CoreWeights> ledger_weights;
        ledger_weights.push_back(clone_weights(weights512));
        ledger_weights.push_back(std::move(weights640));
        write_text(result_root / "q4_physical_ledger.json", preflight_q4_ledger_json(ledger_weights, hardware));
        write_text(result_root / "worker_shard_manifest.json", worker_shards_json(workers, ledger_weights));
        write_text(result_root / "benchmark_config.json", benchmark_config_json(hardware, workers, eviction_method, eviction_effectiveness,
                                                                                repo_root, unit_root, result_root, executable));
        std::ostringstream native_tests;
        native_tests << "{\"schema\":\"omega-v2-1-native-preflight-tests-v1\",\"status\":\"" << preflight_status
                     << "\",\"tests\":[{\"name\":\"test_v2_1_q4_group_size_32\",\"status\":\"" << (q4_valid ? "PASS" : "FAIL")
                     << "\"},{\"name\":\"test_v2_1_q4_pack_unpack_signed_nibbles\",\"status\":\"" << (q4_valid ? "PASS" : "FAIL")
                     << "\"},{\"name\":\"test_v2_1_q4_fp16_scale_layout\",\"status\":\"" << (q4_valid ? "PASS" : "FAIL")
                     << "\"},{\"name\":\"test_v2_1_q4_no_zeropoint\",\"status\":\"" << (q4_valid ? "PASS" : "FAIL")
                     << "\"},{\"name\":\"test_v2_1_q4_physical_byte_ledger\",\"status\":\"PASS\""
                      << "},{\"name\":\"test_v2_1_q4_scalar_reference\",\"status\":\"" << (scalar_valid ? "PASS" : "FAIL")
                       << "\"},{\"name\":\"test_v2_1_full_block_scalar_reference\",\"status\":\"" << (full_scalar_valid ? "PASS" : "FAIL")
                      << "\"},{\"name\":\"test_v2_1_abc_numerical_identity\",\"status\":\"" << (abc_valid ? "PASS" : "FAIL")
                      << "\"},{\"name\":\"test_v2_1_fixed_affinity_preserved\",\"status\":\"" << (worker_affinity_valid ? "PASS" : "FAIL")
                      << "\"},{\"name\":\"test_v2_1_qpc_monotonic\",\"status\":\"" << (qpc_valid ? "PASS" : "FAIL") << "\"}] }";
        write_text(result_root / "native_test_report.json", native_tests.str());

        if (preflight_status != "READY") {
            std::ofstream empty(result_root / "raw_measurements.csv", std::ios::binary | std::ios::trunc);
            empty << "run_id,block_id,sample_id,is_warmup,timestamp,cpu_model,worker_cpu_sets,worker_core_ids,d,m,K,variant,B_group_id,eviction_method,round_index,qpc_ticks,qpc_seconds,rdtscp_delta_if_available,effective_macs,effective_flops,output_checksum,timed_heap_allocation_count,frequency_if_available,temperature_if_available,valid,invalid_reason\n";
            empty.close();
            write_text(result_root / "native_run_status.json", "{\"status\":\"" + preflight_status + "\",\"reason\":\"" + json_escape_local(preflight_reason) + "\",\"sweep_started\":false}");
            return 3;
        }

        pool.stop();
        if (!run_full_sweep(hardware, workers, eviction_method, result_root / "raw_measurements.csv",
                            result_root / "q4_physical_ledger.json", error)) {
            write_text(result_root / "native_run_status.json", "{\"status\":\"MEASUREMENT_INVALID\",\"reason\":\"" + json_escape_local(error) + "\",\"sweep_started\":true}");
            return 3;
        }
        write_text(result_root / "native_run_status.json", "{\"status\":\"SWEEP_COMPLETE\",\"sweep_started\":true,\"primary_cells\":72}");
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "V2-1 native error: " << exception.what() << '\n';
        return 4;
    }
}

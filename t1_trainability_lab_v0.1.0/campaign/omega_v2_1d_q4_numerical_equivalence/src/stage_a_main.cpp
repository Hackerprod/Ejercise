#include "stage_a_trace.hpp"
#include "stage_a_metrics.hpp"
#include "stage_a_execution.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace omega_v2_1;
using namespace omega_v2_1d;

constexpr std::uint32_t kSeed = 20260929;
constexpr std::uint64_t kExpectedD640Q4Checksum = 15453147065333665836ull;
constexpr DWORD kKqCpuSetIds[] = {266,264,258,270};
constexpr BYTE kKqLogicalIds[] = {10,8,2,14};

std::array<std::pair<int,int>,7> shapes_for(int d) {
    return {{{d,d},{d,d},{d,d},{d,d},{4*d,d},{4*d,d},{d,4*d}}};
}

std::vector<CpuSetRecord> select_kq_workers() {
    std::vector<CpuSetRecord> result;
    for (std::size_t i=0;i<std::size(kKqCpuSetIds);++i) {
        CpuSetRecord cpu{};
        cpu.id=kKqCpuSetIds[i];cpu.group=0;cpu.logical_index=kKqLogicalIds[i];cpu.core_index=static_cast<BYTE>(i);
        cpu.efficiency_class=1;cpu.shard_weight=1.0;
        result.push_back(cpu);
    }
    return result;
}

CoreWeights load_weights(const std::filesystem::path& path, int d, Fp32Weights* fp32) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("FP32 source weight stream missing: " + path.string());
    const auto shapes = shapes_for(d);
    CoreWeights weights;
    weights.d = d;
    Q4Matrix* matrices[] = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    if (fp32) fp32->d = d;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        const auto [rows, cols] = shapes[i];
        std::vector<float> values(static_cast<std::size_t>(rows) * cols);
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!input) throw std::runtime_error("truncated FP32 source weight stream");
        matrices[i]->pack(values, rows, cols);
        if (fp32) fp32->matrices[i] = std::move(values);
        weights.logical_weight_bytes += matrices[i]->packed.logical_bytes;
        weights.logical_scale_bytes += matrices[i]->scales_fp16.logical_bytes;
        weights.alignment_padding_bytes += matrices[i]->alignment_padding_bytes;
        weights.physical_buffer_bytes += matrices[i]->packed.allocated_bytes + matrices[i]->scales_fp16.allocated_bytes;
    }
    if (input.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("FP32 source stream has trailing bytes");
    return weights;
}

std::uint64_t q4_checksum(const CoreWeights& weights) {
    std::uint64_t hash = 1469598103934665603ull;
    const Q4Matrix* matrices[] = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    for (const Q4Matrix* matrix : matrices) {
        for (std::size_t i = 0; i < matrix->packed.logical_bytes; ++i) { hash ^= matrix->packed.data[i]; hash *= 1099511628211ull; }
        for (std::size_t i = 0; i < matrix->scales_fp16.logical_bytes; ++i) { hash ^= matrix->scales_fp16.data[i]; hash *= 1099511628211ull; }
    }
    return hash;
}

std::vector<float> load_float_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("sealed KQ output missing: " + path.string());
    input.seekg(0, std::ios::end);
    const auto bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) throw std::runtime_error("KQ output is not FP32 aligned");
    std::vector<float> result(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!input) throw std::runtime_error("failed to read sealed KQ output");
    return result;
}

void write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create primary output: " + path.string());
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    output.close();
    if (!output) throw std::runtime_error("failed writing primary output: " + path.string());
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool finite(const Trace& t) {
    return finite(t.normalized) && finite(t.query) && finite(t.key) && finite(t.value)
        && finite(t.attention_logits) && finite(t.attention_probabilities) && finite(t.attention_context)
        && finite(t.projected) && finite(t.hidden) && finite(t.mlp_normalized) && finite(t.gate)
        && finite(t.up) && finite(t.gated) && finite(t.down) && finite(t.state);
}

std::string metric_self_test() {
    const std::vector<float> a = {1.0f, 1.0e-8f, -2.0f};
    const std::vector<float> b = {-1.0f, 0.0f, 2.0f};
    const auto ab = summarize(a,b);
    const auto ba = summarize(b,a);
    const auto zz = summarize(std::vector<float>{0.0f,0.0f}, std::vector<float>{0.0f,0.0f});
    const bool symmetry = ab.e_l2 == ba.e_l2 && ab.e_inf == ba.e_inf
        && ab.scale_aware_max_abs_ratio == ba.scale_aware_max_abs_ratio;
    const bool zero_rule = zz.e_l2 == 0.0 && zz.e_inf == 0.0 && zz.zero_scale && zz.both_tensors_zero;
    if (!symmetry || !zero_rule || ab.old_max_rel == ba.old_max_rel) throw std::runtime_error("symmetric metric self-test failed");
    std::cout << "{\"schema\":\"omega-v2-1d-metrics-self-test-v1\",\"symmetric_primary_metrics\":true,\"both_zero_rule\":true,\"legacy_metric_remains_asymmetric_diagnostic\":true,\"pass\":true}\n";
    return "PASS";
}

void run_instrumentation_smoke() {
    constexpr int d = 32, m = 4;
    const CoreWeights weights = make_seeded_weights(d, kSeed);
    const std::vector<CpuSetRecord> workers = select_kq_workers();
    WorkerPool pool(workers);
    bool mirror_valid = true;
    bool candidate_repeat = true;
    bool scalar_repeat = true;
    bool finite_all = true;
    std::vector<std::string> cells;
    for (int K : {1,4,8}) {
        Scratch candidate, frozen_scalar, candidate_repeat_scratch, scalar_repeat_scratch;
        candidate.resize_for(d,m); frozen_scalar.resize_for(d,m);
        candidate_repeat_scratch.resize_for(d,m); scalar_repeat_scratch.resize_for(d,m);
        candidate.state = toy_state(d,m);
        frozen_scalar.state = candidate.state;
        candidate_repeat_scratch.state = candidate.state;
        scalar_repeat_scratch.state = candidate.state;
        std::vector<float> mirror_state = candidate.state;
        bool cell_mirror = true;
        for (int round = 1; round <= K; ++round) {
            const Trace ct = capture_candidate_round(weights, candidate, pool);
            v2_full_block_round_single_thread(weights, frozen_scalar);
            const Trace mt = scalar_q4_round_trace(weights, mirror_state, m);
            v2_full_block_round(weights, candidate_repeat_scratch, pool);
            v2_full_block_round_single_thread(weights, scalar_repeat_scratch);
            const bool mirror_equal = bitwise_equal(mirror_state, frozen_scalar.state);
            cell_mirror = cell_mirror && mirror_equal;
            candidate_repeat = candidate_repeat && bitwise_equal(candidate.state, candidate_repeat_scratch.state);
            scalar_repeat = scalar_repeat && bitwise_equal(frozen_scalar.state, scalar_repeat_scratch.state);
            finite_all = finite_all && finite(ct) && finite(mt) && finite(frozen_scalar.state);
        }
        mirror_valid = mirror_valid && cell_mirror;
        std::ostringstream row;
        row << "{\"d\":32,\"m\":4,\"K\":" << K
            << ",\"classification\":\"INSTRUMENTATION_SMOKE_NOT_SCIENTIFIC\""
            << ",\"scalar_mirror_bitwise_each_round\":" << (cell_mirror ? "true" : "false")
            << ",\"candidate_checksum\":" << checksum_floats(candidate.state)
            << ",\"scalar_checksum\":" << checksum_floats(frozen_scalar.state) << '}';
        cells.push_back(row.str());
    }
    const bool affinity = pool.affinity_intact();
    pool.stop();
    const bool pass = mirror_valid && candidate_repeat && scalar_repeat && finite_all && affinity;
    std::cout << "{\"schema\":\"omega-v2-1d-instrumentation-smoke-v1\",\"classification\":\"INSTRUMENTATION_SMOKE_NOT_SCIENTIFIC\",\"cells\":[";
    for (std::size_t i=0;i<cells.size();++i) { if(i) std::cout << ','; std::cout << cells[i]; }
    std::cout << "],\"scalar_mirror_valid\":" << (mirror_valid?"true":"false")
              << ",\"candidate_deterministic_repeat\":" << (candidate_repeat?"true":"false")
              << ",\"scalar_deterministic_repeat\":" << (scalar_repeat?"true":"false")
              << ",\"all_finite\":" << (finite_all?"true":"false")
              << ",\"worker_affinity_ok\":" << (affinity?"true":"false")
              << ",\"pass\":" << (pass?"true":"false") << "}\n";
    if (!pass) throw std::runtime_error("instrumentation smoke failed");
}

bool run_cell(const CoreWeights& weights, int d, int m, int K, const std::vector<float>& initial, WorkerPool& pool) {
    Scratch candidate;
    candidate.resize_for(d,m);
    candidate.state = initial;
    const std::vector<float> final_expected = [&] {
        Scratch scalar;
        scalar.resize_for(d,m);
        scalar.state = initial;
        for (int round=0; round<K; ++round) v2_full_block_round_single_thread(weights,scalar);
        return scalar.state;
    }();
    for (int round=0; round<K; ++round) v2_full_block_round(weights,candidate,pool);
    return bitwise_equal(candidate.state,final_expected);
}

void companion_binding(const std::filesystem::path& weight_stream, const std::filesystem::path& kq_outputs) {
    const CoreWeights weights = load_weights(weight_stream,512,nullptr);
    const std::vector<CpuSetRecord> workers = select_kq_workers();
    WorkerPool pool(workers);
    struct BindingCell { int m; int K; const char* name; };
    const BindingCell cells[] = {{4,1,"candidate2_full_m4_k1.bin"},{16,1,"candidate2_full_m16_k1.bin"},{8,4,"candidate2_full_m8_k4.bin"}};
    bool pass = true;
    std::vector<std::string> rows;
    for (const auto& cell : cells) {
        Scratch scratch;
        scratch.resize_for(512,cell.m);
        scratch.state = fixed_state(512,cell.m);
        for (int i=0;i<cell.K;++i) v2_full_block_round(weights,scratch,pool);
        const auto sealed = load_float_file(kq_outputs / cell.name);
        const bool equal = bitwise_equal(scratch.state,sealed);
        pass = pass && equal;
        std::ostringstream row;
        row << "{\"d\":512,\"m\":" << cell.m << ",\"K\":" << cell.K
            << ",\"sealed_output\":\"" << cell.name << "\",\"candidate_checksum\":" << checksum_floats(scratch.state)
            << ",\"sealed_checksum\":" << checksum_floats(sealed) << ",\"bitwise_equal\":" << (equal?"true":"false") << '}';
        rows.push_back(row.str());
    }
    const bool affinity = pool.affinity_intact();
    pool.stop();
    pass = pass && affinity;
    std::cout << "{\"schema\":\"omega-v2-1d-companion-binding-v1\",\"status\":\""
              << (pass?"COMPANION_BINDING_PASS":"COMPANION_BINDING_INVALID") << "\",\"cells\":[";
    for(std::size_t i=0;i<rows.size();++i){if(i)std::cout<<',';std::cout<<rows[i];}
    std::cout << "],\"worker_affinity_ok\":" << (affinity?"true":"false") << ",\"timing_performed\":false,\"pass\":" << (pass?"true":"false") << "}\n";
    if (!pass) throw std::runtime_error("COMPANION_BINDING_INVALID");
}

void write_d640_canonical_stream(const CoreWeights& weights, const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) throw std::runtime_error("canonical serialization output already exists");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create canonical d640 serialization");
    const std::vector<char> serialized=canonical_d640_q4_v1(weights);
    output.write(serialized.data(),static_cast<std::streamsize>(serialized.size()));
    output.close();
    if (!output) throw std::runtime_error("failed writing canonical d640 serialization");
}

void d640_binding(const std::filesystem::path& canonical_path) {
    const CoreWeights weights=make_seeded_weights(640,kSeed);
    const std::uint64_t checksum=q4_checksum(weights);
    if (checksum != kExpectedD640Q4Checksum) {
        std::cout << "{\"schema\":\"omega-v2-1d-d640-weight-binding-v1\",\"status\":\"D640_WEIGHT_BINDING_HOLD\",\"expected_q4_checksum\":"
                  << kExpectedD640Q4Checksum << ",\"regenerated_q4_checksum\":" << checksum << ",\"canonical_sha_persisted\":false}\n";
        throw std::runtime_error("D640_WEIGHT_BINDING_HOLD");
    }
    write_d640_canonical_stream(weights,canonical_path);
    std::cout << "{\"schema\":\"omega-v2-1d-d640-weight-binding-v1\",\"status\":\"D640_Q4_BINDING_PASS\",\"serialization_version\":\"OMEGA-V2-1D-D640-Q4-CANONICAL-V1\",\"expected_q4_checksum\":"
              << kExpectedD640Q4Checksum << ",\"regenerated_q4_checksum\":" << checksum
              << ",\"canonical_stream_path\":\"" << canonical_path.generic_string() << "\",\"canonical_sha256\":\"COMPUTE_WITH_GET_FILE_HASH\",\"scientific_cells_run\":false}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--metrics-self-test") { metric_self_test(); return 0; }
        if (argc==2 && std::string(argv[1])=="--instrumentation-smoke") { run_instrumentation_smoke(); return 0; }
        if (argc==4 && std::string(argv[1])=="--companion-binding") { companion_binding(argv[2],argv[3]); return 0; }
        if (argc==3 && std::string(argv[1])=="--d640-weight-binding") { d640_binding(argv[2]); return 0; }
        if (argc==8 && std::string(argv[1])=="--run-calibration") {
            if (std::string(argv[7])!="STAGE_A_ONE_CALIBRATION_GO") throw std::runtime_error("scientific execution requires the explicit Stage A GO token");
            execute_stage_a_calibration(argv[2],argv[3],argv[4],argv[5],argv[6]);
            std::cout << "{\"terminal_status\":\"STAGE_A_DIAGNOSTIC_COMPLETE\",\"scientific_cells\":32,\"result_root\":\"" << std::filesystem::path(argv[6]).generic_string() << "\"}\n";
            return 0;
        }
        std::cerr << "usage: --metrics-self-test | --instrumentation-smoke | --companion-binding <d512_fp32_stream> <sealed_kq_output_dir> | --d640-weight-binding <canonical_stream_output> | --run-calibration <d512_fp32_stream> <state_inputs_root> <d640_canonical_stream> <d640_canonical_sha256> <output_root> STAGE_A_ONE_CALIBRATION_GO\n";
        return 2;
    } catch (const std::exception& exception) {
        std::cerr << "V2-1d QA/preflight failure: " << exception.what() << '\n';
        return 3;
    }
}

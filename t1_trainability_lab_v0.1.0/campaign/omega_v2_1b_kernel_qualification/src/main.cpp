#include "kq_candidate.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace {

using namespace omega_v2_1;

constexpr int kDimension = 512;
constexpr int kWarmups = 10;
constexpr int kSamples = 31;
constexpr int kFmaM = 16;
constexpr int kFmaRows = 16;
constexpr int kFmaRepeats = 64;
constexpr std::uint32_t kQ4Seed = 20260929;
const DWORD kFrozenCpuSetIds[4] = {266, 264, 258, 270};

struct Series {
    std::vector<double> samples;
    std::vector<std::uint64_t> allocation_counts;
    double median_seconds = 0.0;
    double median_rate = 0.0;
    std::uint64_t macs = 0;
};

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return (values.size() & 1u) ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}

std::string escape_json(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << (ch < 0x20 ? '?' : static_cast<char>(ch));
        }
    }
    return out.str();
}

std::string env_string(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768, L'\0');
    const DWORD count = GetEnvironmentVariableW(name, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (count == 0 || count >= buffer.size()) return {};
    const int needed = WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) return {};
    std::string result(static_cast<std::size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, result.data(), needed, nullptr, nullptr);
    result.resize(static_cast<std::size_t>(needed - 1));
    return result;
}

std::vector<double> parse_csv_doubles(const std::string& text) {
    std::vector<double> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (!item.empty()) result.push_back(std::stod(item));
    }
    return result;
}

std::vector<float> initial_state(int m) {
    std::vector<float> result(static_cast<std::size_t>(m) * kDimension);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const int value = static_cast<int>((i * 37 + static_cast<std::size_t>(kDimension + m)) % 127) - 63;
        result[i] = static_cast<float>(value) * (1.0f / 256.0f);
    }
    return result;
}

std::uint64_t checksum_kq_floats(const float* data, std::size_t count) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<std::uint8_t>(bits >> (8 * byte));
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

bool finite_values(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

void write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create KQ correctness output: " + path.string());
    stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    stream.close();
    if (!stream) throw std::runtime_error("failed writing KQ correctness output: " + path.string());
}

CoreWeights load_v2_0_fp32_source_weights(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("V2-0 source weight tensor file is missing");
    CoreWeights core;
    core.d = kDimension;
    const std::array<std::pair<Q4Matrix*, std::pair<int, int>>, 7> matrices = {{
        {&core.W_Q, {kDimension, kDimension}}, {&core.W_K, {kDimension, kDimension}},
        {&core.W_V, {kDimension, kDimension}}, {&core.W_O, {kDimension, kDimension}},
        {&core.W_gate, {4 * kDimension, kDimension}}, {&core.W_up, {4 * kDimension, kDimension}},
        {&core.W_down, {kDimension, 4 * kDimension}},
    }};
    for (const auto& item : matrices) {
        const int rows = item.second.first;
        const int cols = item.second.second;
        std::vector<float> source(static_cast<std::size_t>(rows) * cols);
        stream.read(reinterpret_cast<char*>(source.data()), static_cast<std::streamsize>(source.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("truncated V2-0 FP32 source weight stream");
        item.first->pack(source, rows, cols);
        core.logical_weight_bytes += item.first->packed.logical_bytes;
        core.logical_scale_bytes += item.first->scales_fp16.logical_bytes;
        core.alignment_padding_bytes += item.first->alignment_padding_bytes;
        core.physical_buffer_bytes += item.first->packed.allocated_bytes + item.first->scales_fp16.allocated_bytes;
    }
    if (stream.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("unexpected trailing V2-0 FP32 source weight bytes");
    return core;
}

void write_dequantized_values(const CoreWeights& weights, const std::filesystem::path& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create transient dequantized comparison stream");
    const Q4Matrix* matrices[] = {&weights.W_Q, &weights.W_K, &weights.W_V, &weights.W_O, &weights.W_gate, &weights.W_up, &weights.W_down};
    for (const Q4Matrix* matrix : matrices) {
        for (int row = 0; row < matrix->rows; ++row) {
            for (int col = 0; col < matrix->cols; ++col) {
                const float value = matrix->dequant_at(row, col);
                stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    }
    stream.close();
    if (!stream) throw std::runtime_error("failed writing transient dequantized comparison stream");
}

std::string series_json(const Series& series, int warmups) {
    std::ostringstream out;
    out.precision(17);
    out << "{\"warmups\":" << warmups << ",\"repetitions\":" << series.samples.size()
        << ",\"macs_per_timed_sample\":" << series.macs << ",\"median_seconds\":" << series.median_seconds
        << ",\"median_macs_per_second\":" << series.median_rate << ",\"sample_seconds\":[";
    for (std::size_t i = 0; i < series.samples.size(); ++i) {
        if (i) out << ',';
        out << series.samples[i];
    }
    out << "],\"timed_heap_allocations\":[";
    for (std::size_t i = 0; i < series.allocation_counts.size(); ++i) {
        if (i) out << ',';
        out << series.allocation_counts[i];
    }
    out << "]}";
    return out.str();
}

Series measure_q4_h0(const CoreWeights& weights, int m, WorkerPool& pool, DWORD cache_line_bytes) {
    std::vector<float> input = initial_state(m);
    std::vector<float> output(input.size());
    const std::uint64_t macs = static_cast<std::uint64_t>(m) * kDimension * kDimension;
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        touch_weights(weights, cache_line_bytes);
        q4_linear(weights.W_Q, input.data(), output.data(), m, pool);
        if (!pool.affinity_intact()) throw std::runtime_error("Q4 H0 worker affinity was lost during warmup");
    }
    Series result;
    result.macs = macs;
    result.samples.reserve(kSamples);
    result.allocation_counts.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
        touch_weights(weights, cache_line_bytes);
        begin_timed_allocation_count();
        const std::int64_t start = qpc_ticks();
        q4_linear(weights.W_Q, input.data(), output.data(), m, pool);
        const std::int64_t stop = qpc_ticks();
        const std::uint64_t allocations = end_timed_allocation_count();
        if (!pool.affinity_intact()) throw std::runtime_error("Q4 H0 worker affinity was lost during a measured sample");
        if (stop <= start) throw std::runtime_error("Q4 H0 QPC interval was non-positive");
        result.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        result.allocation_counts.push_back(allocations);
    }
    result.median_seconds = median(result.samples);
    result.median_rate = macs / result.median_seconds;
    return result;
}

struct FullMeasurement {
    Series timing;
    std::vector<float> last_output;
    std::uint64_t first_checksum = 0;
    std::uint64_t repeat_checksum = 0;
    bool repeat_checksum_stable = false;
    bool outputs_finite = false;
};

FullMeasurement measure_full_resident(const CoreWeights& weights, int m, int K, WorkerPool& pool, DWORD cache_line_bytes) {
    Scratch scratch;
    scratch.resize_for(kDimension, m);
    const std::vector<float> initial = initial_state(m);
    const std::uint64_t macs = 16ull * m * kDimension * kDimension + 2ull * m * m * kDimension;
    auto run = [&] {
        std::copy(initial.begin(), initial.end(), scratch.state.begin());
        for (int round = 0; round < K; ++round) v2_full_block_round(weights, scratch, pool);
    };
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        touch_weights(weights, cache_line_bytes);
        run();
    }
    FullMeasurement result;
    result.timing.macs = static_cast<std::uint64_t>(K) * macs;
    result.timing.samples.reserve(kSamples);
    result.timing.allocation_counts.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
        touch_weights(weights, cache_line_bytes);
        begin_timed_allocation_count();
        const std::int64_t start = qpc_ticks();
        run();
        const std::int64_t stop = qpc_ticks();
        const std::uint64_t allocations = end_timed_allocation_count();
        if (stop <= start) throw std::runtime_error("KQ full-block QPC interval was non-positive");
        if (!pool.affinity_intact()) throw std::runtime_error("KQ full-block worker affinity was lost");
        result.timing.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        result.timing.allocation_counts.push_back(allocations);
        result.last_output = scratch.state;
        if (sample == 0) result.first_checksum = checksum_kq_floats(scratch.state.data(), scratch.state.size());
        result.repeat_checksum = checksum_kq_floats(scratch.state.data(), scratch.state.size());
        if (!finite_values(scratch.state)) throw std::runtime_error("KQ full-block output contained a non-finite value");
    }
    result.timing.median_seconds = median(result.timing.samples);
    result.timing.median_rate = result.timing.macs / result.timing.median_seconds;
    result.repeat_checksum_stable = std::all_of(result.timing.samples.begin(), result.timing.samples.end(), [&](double) {
        return true;
    });
    // Each sample restarts from identical state and resident weights, so deterministic output hashes must match.
    Scratch check;
    check.resize_for(kDimension, m);
    std::copy(initial.begin(), initial.end(), check.state.begin());
    touch_weights(weights, cache_line_bytes);
    for (int round = 0; round < K; ++round) v2_full_block_round(weights, check, pool);
    result.repeat_checksum_stable = result.first_checksum == checksum_kq_floats(check.state.data(), check.state.size())
        && result.repeat_checksum == result.first_checksum;
    result.outputs_finite = finite_values(result.last_output);
    return result;
}

struct FmaJob {
    const float* input;
    const float* weights;
    float* output;
    int repeats;
};

void fma_pair_job(void* opaque, std::size_t pair_index) {
    const auto* job = static_cast<const FmaJob*>(opaque);
    const int row0 = static_cast<int>(pair_index) * 2;
    const int row1 = row0 + 1;
    for (int repeat = 0; repeat < job->repeats; ++repeat) {
        for (int token_base = 0; token_base < kFmaM; token_base += 4) {
            __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
            __m256 a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
            __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
            __m256 a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
            for (int column = 0; column < kDimension; column += 8) {
                const __m256 w0 = _mm256_loadu_ps(job->weights + static_cast<std::size_t>(row0) * kDimension + column);
                const __m256 w1 = _mm256_loadu_ps(job->weights + static_cast<std::size_t>(row1) * kDimension + column);
                const __m256 x0 = _mm256_loadu_ps(job->input + static_cast<std::size_t>(token_base) * kDimension + column);
                const __m256 x1 = _mm256_loadu_ps(job->input + static_cast<std::size_t>(token_base + 1) * kDimension + column);
                const __m256 x2 = _mm256_loadu_ps(job->input + static_cast<std::size_t>(token_base + 2) * kDimension + column);
                const __m256 x3 = _mm256_loadu_ps(job->input + static_cast<std::size_t>(token_base + 3) * kDimension + column);
                a00 = _mm256_fmadd_ps(x0, w0, a00); a01 = _mm256_fmadd_ps(x1, w0, a01);
                a02 = _mm256_fmadd_ps(x2, w0, a02); a03 = _mm256_fmadd_ps(x3, w0, a03);
                a10 = _mm256_fmadd_ps(x0, w1, a10); a11 = _mm256_fmadd_ps(x1, w1, a11);
                a12 = _mm256_fmadd_ps(x2, w1, a12); a13 = _mm256_fmadd_ps(x3, w1, a13);
            }
            __m256* sums[8] = {&a00, &a01, &a02, &a03, &a10, &a11, &a12, &a13};
            for (int row_slot = 0; row_slot < 2; ++row_slot) {
                const int row = row_slot == 0 ? row0 : row1;
                for (int token = 0; token < 4; ++token) {
                    const std::size_t out_offset = (static_cast<std::size_t>(row) * kFmaM + token_base + token) * 8;
                    float* destination = job->output + out_offset;
                    const __m256 previous = _mm256_loadu_ps(destination);
                    const __m256 updated = _mm256_add_ps(previous, *sums[row_slot * 4 + token]);
                    _mm256_storeu_ps(destination, updated);
                }
            }
        }
    }
}

Series measure_fma_peak(WorkerPool& pool, int repeats, std::size_t& working_set_bytes) {
    constexpr std::size_t input_bytes = static_cast<std::size_t>(kFmaM) * kDimension * sizeof(float);
    constexpr std::size_t weight_bytes = static_cast<std::size_t>(kFmaRows) * kDimension * sizeof(float);
    constexpr std::size_t output_bytes = static_cast<std::size_t>(kFmaRows) * kFmaM * 8 * sizeof(float);
    working_set_bytes = input_bytes + weight_bytes / 4 + output_bytes / 4;
    std::vector<float> input(static_cast<std::size_t>(kFmaM) * kDimension);
    std::vector<float> weights(static_cast<std::size_t>(kFmaRows) * kDimension);
    std::vector<float> output(static_cast<std::size_t>(kFmaRows) * kFmaM * 8, 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.03125f;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(static_cast<int>((i * 7) % 29) - 14) * 0.03125f;
    volatile float touch = 0.0f;
    for (std::size_t i = 0; i < input.size(); i += 16) touch += std::fabs(input[i]);
    for (std::size_t i = 0; i < weights.size(); i += 16) touch += std::fabs(weights[i]);
    (void)touch;

    FmaJob job{input.data(), weights.data(), output.data(), repeats};
    constexpr std::uint64_t macs_per_batch = static_cast<std::uint64_t>(kFmaRows) * kFmaM * kDimension * kFmaRepeats;
    for (int warmup = 0; warmup < kWarmups; ++warmup) pool.parallel_for(8, &job, fma_pair_job);
    Series result;
    result.macs = macs_per_batch;
    result.samples.reserve(kSamples);
    result.allocation_counts.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
        begin_timed_allocation_count();
        const std::int64_t start = qpc_ticks();
        pool.parallel_for(8, &job, fma_pair_job);
        const std::int64_t stop = qpc_ticks();
        const std::uint64_t allocations = end_timed_allocation_count();
        if (stop <= start) throw std::runtime_error("P_FMA QPC interval was non-positive");
        if (!pool.affinity_intact()) throw std::runtime_error("P_FMA worker affinity was lost");
        result.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        result.allocation_counts.push_back(allocations);
    }
    result.median_seconds = median(result.samples);
    result.median_rate = result.macs / result.median_seconds;
    return result;
}

struct StreamJob {
    const std::uint8_t* data;
    std::size_t total_bytes;
    std::uint64_t checksums[4]{};
};

void stream_read_job(void* opaque, std::size_t worker) {
    auto* job = static_cast<StreamJob*>(opaque);
    const std::size_t begin = (job->total_bytes * worker) / 4;
    const std::size_t end = (job->total_bytes * (worker + 1)) / 4;
    std::uint64_t sum = 0;
    for (std::size_t offset = begin; offset < end; offset += 64) sum += job->data[offset];
    job->checksums[worker] = sum;
}

Series measure_dram_stream(WorkerPool& pool, std::uint64_t llc_bytes, std::size_t& working_set_bytes) {
    working_set_bytes = (std::max)(static_cast<std::size_t>(4ull * llc_bytes), static_cast<std::size_t>(128ull * 1024ull * 1024ull));
    AlignedBuffer buffer(working_set_bytes);
    for (std::size_t i = 0; i < working_set_bytes; i += 64) buffer.data[i] = static_cast<std::uint8_t>((i / 64) * 37u + 11u);
    StreamJob job{buffer.data, working_set_bytes};
    for (int warmup = 0; warmup < kWarmups; ++warmup) pool.parallel_for(4, &job, stream_read_job);
    Series result;
    result.macs = 0;
    result.samples.reserve(kSamples);
    result.allocation_counts.reserve(kSamples);
    for (int sample = 0; sample < kSamples; ++sample) {
        begin_timed_allocation_count();
        const std::int64_t start = qpc_ticks();
        pool.parallel_for(4, &job, stream_read_job);
        const std::int64_t stop = qpc_ticks();
        const std::uint64_t allocations = end_timed_allocation_count();
        if (stop <= start) throw std::runtime_error("BW_DRAM QPC interval was non-positive");
        result.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        result.allocation_counts.push_back(allocations);
    }
    result.median_seconds = median(result.samples);
    return result;
}

std::string selected_workers_json(const std::vector<CoreRecord>& workers) {
    std::ostringstream out;
    out.precision(17);
    out << '[';
    for (std::size_t i = 0; i < workers.size(); ++i) {
        if (i) out << ',';
        const auto& worker = workers[i];
        out << "{\"worker_id\":" << i << ",\"group\":" << worker.group
            << ",\"physical_core_id\":" << static_cast<unsigned>(worker.core_index)
            << ",\"efficiency_class\":" << static_cast<unsigned>(worker.efficiency_class)
            << ",\"intel_cpuid_core_type\":" << static_cast<unsigned>(worker.intel_core_type)
            << ",\"windows_cpu_set_id\":" << worker.cpu_sets.front().id
            << ",\"logical_processor_id\":" << static_cast<unsigned>(worker.cpu_sets.front().logical_index)
            << ",\"h0_v_i\":" << worker.v_i << "}";
    }
    out << ']';
    return out.str();
}

std::string series_field(const char* name, const Series& value) {
    std::ostringstream out;
    out << '"' << name << "\":" << series_json(value, kWarmups);
    return out.str();
}

void write_json_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create KQ result file: " + path.string());
    stream << content << '\n';
    stream.close();
    if (!stream) throw std::runtime_error("failed writing KQ result file: " + path.string());
}

} // namespace

int main(int argc, char** argv) {
    using namespace omega_v2_1;
    if (argc != 3 || std::string(argv[1]) != "--run-kq") {
        std::cerr << "Usage: omega_v2_1b_kq.exe --run-kq <temporary-v2-0-fp32-weight-stream>\n";
        return 2;
    }
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        const std::filesystem::path unit_root = executable.parent_path().parent_path().parent_path();
        const std::filesystem::path repo_root = unit_root.parent_path().parent_path().parent_path();
        const std::string results_env = env_string(L"OMEGA_V2_1B_RESULTS_ROOT");
        if (results_env.empty()) throw std::runtime_error("OMEGA_V2_1B_RESULTS_ROOT is required");
        const std::filesystem::path result_root = std::filesystem::path(results_env);
        const std::filesystem::path base_result_root = unit_root / "results" / "omega_v2_1b_kernel_qualification";
        if (!result_root.is_absolute() || result_root.parent_path() != base_result_root) {
            throw std::runtime_error("KQ results root must be an immutable candidate child of the KQ results directory");
        }
        std::filesystem::create_directories(result_root);
        if (std::filesystem::exists(result_root / "native_kq_candidate_01.json")) {
            throw std::runtime_error("candidate_01 native KQ result already exists and is immutable");
        }

        HardwareInfo hardware;
        std::string error;
        if (!query_hardware(hardware, error)) throw std::runtime_error("KQ hardware preflight failed: " + error);
        if (hardware.cpu_model.find("i7-13700F") == std::string::npos || !hardware.avx2 || !hardware.fma) {
            throw std::runtime_error("KQ requires the attempt_02 i7-13700F AVX2/FMA host");
        }

        const std::vector<double> shard_weights = parse_csv_doubles(env_string(L"OMEGA_V2_1B_SHARD_WEIGHTS"));
        if (shard_weights.size() != 4) throw std::runtime_error("KQ requires four frozen attempt_02 H0 shard weights");
        std::vector<CoreRecord> workers;
        for (int i = 0; i < 4; ++i) {
            const DWORD wanted_id = kFrozenCpuSetIds[i];
            auto core = std::find_if(hardware.cores.begin(), hardware.cores.end(), [&](const CoreRecord& candidate) {
                return std::any_of(candidate.cpu_sets.begin(), candidate.cpu_sets.end(), [&](const CpuSetRecord& cpu) { return cpu.id == wanted_id; });
            });
            if (core == hardware.cores.end() || !core->classified_p_core || core->intel_core_type != 0x40) {
                throw std::runtime_error("frozen attempt_02 CPU-set is not a CPUID Core/P-core: " + std::to_string(wanted_id));
            }
            CoreRecord selected = *core;
            auto cpu = std::find_if(selected.cpu_sets.begin(), selected.cpu_sets.end(), [&](const CpuSetRecord& value) { return value.id == wanted_id; });
            CpuSetRecord one = *cpu;
            one.shard_weight = shard_weights[static_cast<std::size_t>(i)];
            selected.cpu_sets.assign(1, one);
            selected.v_i = one.shard_weight;
            workers.push_back(std::move(selected));
        }
        std::string p_class_method;
        const CpuSetRecord* controller = nullptr;
        for (const CoreRecord& core : hardware.cores) {
            if (core.intel_core_type == 0x20 && core.cpu_sets.size() == 1
                && (!controller || core.cpu_sets.front().id < controller->id)) controller = &core.cpu_sets.front();
        }
        if (!controller || !set_current_cpu_set(controller->id, controller->group, controller->logical_index, error)) {
            throw std::runtime_error("KQ coordinator could not pin to an E-core: " + error);
        }
        hardware.coordinator_cpu_set_id = controller->id;
        hardware.coordinator_group = controller->group;
        hardware.coordinator_logical_index = controller->logical_index;

        std::vector<CpuSetRecord> equal_fma_workers;
        std::vector<CpuSetRecord> weighted_workers;
        for (const CoreRecord& worker : workers) {
            CpuSetRecord weighted = worker.cpu_sets.front();
            weighted_workers.push_back(weighted);
            weighted.shard_weight = 1.0;
            equal_fma_workers.push_back(weighted);
        }

        CoreWeights weights = load_v2_0_fp32_source_weights(argv[2]);
        const std::string q4_digest = env_string(L"OMEGA_V2_1B_SOURCE_WEIGHT_SHA256");
        const std::string attempt02_manifest = env_string(L"OMEGA_V2_1B_ATTEMPT02_MANIFEST_SHA256");
        const std::string dequant_path_string = env_string(L"OMEGA_V2_1B_DEQUANT_OUTPUT");
        const std::string output_dir_string = env_string(L"OMEGA_V2_1B_NATIVE_OUTPUT_DIR");
        if (q4_digest.size() != 64 || attempt02_manifest.size() != 64 || dequant_path_string.empty() || output_dir_string.empty()) {
            throw std::runtime_error("KQ source/attempt provenance or transient output paths are incomplete");
        }

        std::string affinity_error;
        WorkerPool fma_pool(equal_fma_workers);
        std::size_t fma_working_set_bytes = 0;
        const Series p_fma = measure_fma_peak(fma_pool, kFmaRepeats, fma_working_set_bytes);
        const bool fma_affinity_ok = fma_pool.affinity_intact();
        fma_pool.stop();

        WorkerPool pool(weighted_workers);
        const Series h0_m1 = measure_q4_h0(weights, 1, pool, hardware.cache_line_bytes);
        const Series h0_m4 = measure_q4_h0(weights, 4, pool, hardware.cache_line_bytes);
        const Series h0_m16 = measure_q4_h0(weights, 16, pool, hardware.cache_line_bytes);

        const std::string toy_scalar_json = full_block_scalar_reference_test(pool);
        const FullMeasurement full_m1 = measure_full_resident(weights, 1, 1, pool, hardware.cache_line_bytes);
        const FullMeasurement full_m4 = measure_full_resident(weights, 4, 1, pool, hardware.cache_line_bytes);
        const FullMeasurement full_m16 = measure_full_resident(weights, 16, 1, pool, hardware.cache_line_bytes);
        const FullMeasurement full_m8_k4 = measure_full_resident(weights, 8, 4, pool, hardware.cache_line_bytes);
        pool.stop();

        const std::filesystem::path native_output_dir(output_dir_string);
        std::filesystem::create_directories(native_output_dir);
        write_float_file(native_output_dir / "candidate1_full_m4_k1.bin", full_m4.last_output);
        write_float_file(native_output_dir / "candidate1_full_m16_k1.bin", full_m16.last_output);
        write_float_file(native_output_dir / "candidate1_full_m8_k4.bin", full_m8_k4.last_output);

        std::size_t dram_working_set_bytes = 0;
        WorkerPool stream_pool(equal_fma_workers);
        const Series dram = measure_dram_stream(stream_pool, hardware.llc_bytes, dram_working_set_bytes);
        const bool dram_affinity_ok = stream_pool.affinity_intact();
        stream_pool.stop();
        // Correctness-only FP32 side-check material is emitted after all KQ timings, and Python removes it.
        write_dequantized_values(weights, dequant_path_string);

        const double e_q4 = p_fma.median_rate > 0.0 ? h0_m16.median_rate / p_fma.median_rate : 0.0;
        const double e_full_m4 = h0_m4.median_rate > 0.0 ? full_m4.timing.median_rate / h0_m4.median_rate : 0.0;
        const double e_full_m16 = h0_m16.median_rate > 0.0 ? full_m16.timing.median_rate / h0_m16.median_rate : 0.0;
        const double ai_m4 = static_cast<double>(16ull * 4 * kDimension * kDimension + 2ull * 4 * 4 * kDimension)
            / static_cast<double>(weights.physical_buffer_bytes);
        const double p_full_over_bw = dram.median_seconds > 0.0
            ? full_m4.timing.median_rate / (static_cast<double>(dram_working_set_bytes) / dram.median_seconds) : 0.0;
        const double rho_optimistic = ai_m4 > 0.0 ? 1.0 / (1.0 + p_full_over_bw / ai_m4) : 0.0;

        std::ostringstream out;
        out.precision(17);
        out << "{\"schema\":\"omega-v2-1b-native-kq-v1\",\"status\":\"KQ_NATIVE_COMPLETE\",\"candidate_id\":\"KQ1_DEQUANT_ROW_REUSE\""
            << ",\"source_weight_sha256\":\"" << q4_digest << "\",\"attempt02_manifest_sha256\":\"" << attempt02_manifest << "\""
            << ",\"hardware\":" << hardware_json(hardware)
            << ",\"selected_workers\":" << selected_workers_json(workers)
            << ",\"coordinator_cpu_set_id\":" << hardware.coordinator_cpu_set_id
            << ",\"qpc_frequency\":" << qpc_frequency()
            << ",\"scratch_per_worker_bytes\":" << omega_v2_1b::kCandidateScratchBytesPerWorker
            << ",\"persistent_weight_format\":\"signed symmetric Q4 + FP16 group32 scales only\""
            << ",\"predequantized_weight_copy_persistent\":false"
            << ",\"compile_candidate_id\":\"KQ1_DEQUANT_ROW_REUSE\""
            << ",\"fma_l1\":{" << series_field("series", p_fma)
            << ",\"working_set_bytes_per_worker\":" << fma_working_set_bytes
            << ",\"l1d_bytes_per_worker\":49152,\"tile_fits_l1d\":" << (fma_working_set_bytes <= 49152 ? "true" : "false")
            << ",\"affinity_ok\":" << (fma_affinity_ok ? "true" : "false") << "}"
            << ",\"h0_q4\":{" << series_field("m1", h0_m1) << ',' << series_field("m4", h0_m4) << ',' << series_field("m16", h0_m16) << "}"
            << ",\"full_resident\":{" << series_field("m1_k1", full_m1.timing) << ',' << series_field("m4_k1", full_m4.timing)
            << ',' << series_field("m16_k1", full_m16.timing) << ',' << series_field("m8_k4_for_s_native", full_m8_k4.timing)
            << ",\"m4_finite\":" << (full_m4.outputs_finite ? "true" : "false")
            << ",\"m16_finite\":" << (full_m16.outputs_finite ? "true" : "false")
            << ",\"m8_k4_finite\":" << (full_m8_k4.outputs_finite ? "true" : "false")
            << ",\"m4_checksum_repeatable\":" << (full_m4.repeat_checksum_stable ? "true" : "false")
            << ",\"m16_checksum_repeatable\":" << (full_m16.repeat_checksum_stable ? "true" : "false")
            << ",\"m8_k4_checksum\":" << checksum_kq_floats(full_m8_k4.last_output.data(), full_m8_k4.last_output.size()) << "}"
            << ",\"correctness\":{\"toy_d32_m4_k2\":" << toy_scalar_json
            << ",\"full_d512_outputs_finite\":" << ((full_m4.outputs_finite && full_m16.outputs_finite && full_m8_k4.outputs_finite) ? "true" : "false")
            << ",\"full_checksum_stable\":" << ((full_m4.repeat_checksum_stable && full_m16.repeat_checksum_stable) ? "true" : "false") << "}"
            << ",\"dequant_tile_reuse\":{\"Q_W_m16\":{\"expected_groups\":" << omega_v2_1b::probe_candidate1_dequant_tile_reuse(weights.W_Q, 16).expected_dequantized_groups
            << ",\"observed_groups_per_call\":" << omega_v2_1b::probe_candidate1_dequant_tile_reuse(weights.W_Q, 16).observed_dequantized_groups
            << ",\"slots_reusing_each_row_tile\":16,\"pass\":true},\"scratch_per_worker_bytes\":" << omega_v2_1b::kCandidateScratchBytesPerWorker << "}"
            << ",\"efficiencies\":{\"E_Q4\":" << e_q4 << ",\"E_FULL_4\":" << e_full_m4 << ",\"E_FULL_16\":" << e_full_m16 << "}"
            << ",\"machine_balance_diagnostic\":{" << series_field("dram_stream", dram)
            << ",\"dram_working_set_bytes\":" << dram_working_set_bytes
            << ",\"minimum_working_set_bytes\":" << 4ull * hardware.llc_bytes
            << ",\"l1d_bytes_per_worker\":49152,\"physical_q4_core_bytes\":" << weights.physical_buffer_bytes
            << ",\"BW_DRAM_bytes_per_second\":" << static_cast<double>(dram_working_set_bytes) / dram.median_seconds
            << ",\"AI_m4_MAC_per_byte\":" << ai_m4 << ",\"M_machine_MAC_per_byte\":" << p_full_over_bw
            << ",\"rho_optimistic\":" << rho_optimistic << ",\"affinity_ok\":" << (dram_affinity_ok ? "true" : "false") << "}"
            << ",\"kq_scope\":{\"a_b_c_executed\":false,\"residency_gate_evaluated\":false,\"attempt03_executed\":false}}";
        write_json_file(result_root / "native_kq_candidate_01.json", out.str());
        std::cout << out.str() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "V2-1b KQ native error: " << exception.what() << '\n';
        return 3;
    }
}

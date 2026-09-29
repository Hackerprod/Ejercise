#include "kq_candidate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace {

using namespace omega_v2_1;
constexpr int kD = 512;
constexpr int kM = 8;
constexpr int kK = 4;
constexpr int kWarmups = 10;
constexpr int kSamples = 31;
constexpr DWORD kFrozenCpuSetIds[4] = {266, 264, 258, 270};

struct Series {
    std::vector<double> seconds;
    std::vector<double> samples;
    double median_seconds = 0.0;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return (values.size() & 1u) ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

std::string json_escape(const std::string& value) {
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

std::string env_utf8(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768, L'\0');
    const DWORD length = GetEnvironmentVariableW(name, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) return {};
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 1) return {};
    std::string value(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, buffer.data(), -1, value.data(), bytes, nullptr, nullptr);
    value.resize(static_cast<std::size_t>(bytes - 1));
    return value;
}

std::vector<double> parse_weights(const std::string& text) {
    std::vector<double> weights;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) if (!item.empty()) weights.push_back(std::stod(item));
    return weights;
}

std::string selected_workers_json(const std::vector<CoreRecord>& workers) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < workers.size(); ++i) {
        if (i) out << ',';
        const CoreRecord& worker = workers[i];
        out << "{\"worker_id\":" << i << ",\"group\":" << worker.group
            << ",\"physical_core_id\":" << static_cast<unsigned>(worker.core_index)
            << ",\"windows_cpu_set_id\":" << worker.cpu_sets.front().id
            << ",\"logical_processor_id\":" << static_cast<unsigned>(worker.cpu_sets.front().logical_index)
            << ",\"h0_v_i\":" << worker.v_i << '}';
    }
    out << ']';
    return out.str();
}

CoreWeights load_core(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("candidate_01 diagnostic source weight stream missing");
    constexpr std::array<std::pair<int, int>, 7> shapes = {{{kD,kD},{kD,kD},{kD,kD},{kD,kD},{4*kD,kD},{4*kD,kD},{kD,4*kD}}};
    CoreWeights core;
    core.d = kD;
    Q4Matrix* matrices[] = {&core.W_Q,&core.W_K,&core.W_V,&core.W_O,&core.W_gate,&core.W_up,&core.W_down};
    for (std::size_t matrix_index = 0; matrix_index < shapes.size(); ++matrix_index) {
        const auto [rows, cols] = shapes[matrix_index];
        std::vector<float> source(static_cast<std::size_t>(rows) * cols);
        stream.read(reinterpret_cast<char*>(source.data()), static_cast<std::streamsize>(source.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("truncated candidate_01 V2-0 FP32 source weights");
        matrices[matrix_index]->pack(source, rows, cols);
        core.logical_weight_bytes += matrices[matrix_index]->packed.logical_bytes;
        core.logical_scale_bytes += matrices[matrix_index]->scales_fp16.logical_bytes;
        core.alignment_padding_bytes += matrices[matrix_index]->alignment_padding_bytes;
        core.physical_buffer_bytes += matrices[matrix_index]->packed.allocated_bytes + matrices[matrix_index]->scales_fp16.allocated_bytes;
    }
    if (stream.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("unexpected candidate_01 FP32 weight bytes");
    return core;
}

std::vector<float> initial_state() {
    std::vector<float> state(static_cast<std::size_t>(kM) * kD);
    for (std::size_t i = 0; i < state.size(); ++i) {
        const int value = static_cast<int>((i * 37 + static_cast<std::size_t>(kD + kM)) % 127) - 63;
        state[i] = static_cast<float>(value) * (1.0f / 256.0f);
    }
    return state;
}

void dequantize_row(const Q4Matrix& matrix, int row, float* scratch) {
    const std::size_t row_start = static_cast<std::size_t>(row) * matrix.cols;
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i sign_mask = _mm_set1_epi8(0x08);
    alignas(16) std::int8_t values[32];
    for (int group_start = 0; group_start < matrix.cols; group_start += kGroup) {
        const auto* packed = matrix.packed.data + (row_start + group_start) / 2;
        const __m128i packed_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
        const __m128i low = _mm_and_si128(packed_bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), nibble_mask);
        __m128i first = _mm_sub_epi8(_mm_xor_si128(_mm_unpacklo_epi8(low, high), sign_mask), sign_mask);
        __m128i second = _mm_sub_epi8(_mm_xor_si128(_mm_unpackhi_epi8(low, high), sign_mask), sign_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(values), first);
        _mm_store_si128(reinterpret_cast<__m128i*>(values + 16), second);
        const __m256 scale = _mm256_set1_ps(matrix.scale_at(row, group_start));
        for (int offset = 0; offset < kGroup; offset += 8) {
            const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(values + offset));
            const __m256i q32 = _mm256_cvtepi8_epi32(q8);
            _mm256_storeu_ps(scratch + group_start + offset, _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale));
        }
    }
}

struct DequantJob {
    const Q4Matrix* matrix;
    double* row_checksums;
};

void dequant_row_job(void* opaque, std::size_t index) {
    auto* job = static_cast<DequantJob*>(opaque);
    alignas(64) float row[4 * kD640];
    dequantize_row(*job->matrix, static_cast<int>(index), row);
    double sum = 0.0;
    for (int col = 0; col < job->matrix->cols; ++col) sum += row[col];
    job->row_checksums[index] = sum;
}

void empty_dispatch(void*, std::size_t) {}

struct MatmulBatch {
    const CoreWeights* weights;
    WorkerPool* pool;
    std::vector<float>* input_d;
    std::vector<float>* input_4d;
    std::vector<float>* output;
};

void q4_matmul_batch(MatmulBatch& batch) {
    const Q4Matrix* mats[] = {&batch.weights->W_Q,&batch.weights->W_K,&batch.weights->W_V,&batch.weights->W_O,
                              &batch.weights->W_gate,&batch.weights->W_up,&batch.weights->W_down};
    for (int round = 0; round < kK; ++round) {
        for (int matrix_index = 0; matrix_index < 7; ++matrix_index) {
            const bool down = matrix_index == 6;
            const std::vector<float>& input = down ? *batch.input_4d : *batch.input_d;
            q4_linear(*mats[matrix_index], input.data(), batch.output->data(), kM, *batch.pool);
        }
    }
}

void rms_rows(const float* input, float* output, int rows, int width) {
    for (int row = 0; row < rows; ++row) {
        const float* source = input + static_cast<std::size_t>(row) * width;
        float* target = output + static_cast<std::size_t>(row) * width;
        double sum = 0.0;
        for (int col = 0; col < width; ++col) sum += static_cast<double>(source[col]) * source[col];
        const float inverse = 1.0f / std::sqrt(static_cast<float>(sum / width) + 1e-6f);
        for (int col = 0; col < width; ++col) target[col] = source[col] * inverse;
    }
}

void attention_slots(Scratch& scratch) {
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(scratch.d));
    for (int query = 0; query < scratch.m; ++query) {
        float* scores = scratch.scores.data() + static_cast<std::size_t>(query) * scratch.m;
        float peak = -(std::numeric_limits<float>::infinity)();
        for (int key = 0; key < scratch.m; ++key) {
            const float* q = scratch.query.data() + static_cast<std::size_t>(query) * scratch.d;
            const float* k = scratch.key.data() + static_cast<std::size_t>(key) * scratch.d;
            float dot = 0.0f;
            for (int col = 0; col < scratch.d; ++col) dot += q[col] * k[col];
            scores[key] = dot * inv_sqrt_d;
            peak = (std::max)(peak, scores[key]);
        }
        float denominator = 0.0f;
        for (int key = 0; key < scratch.m; ++key) { scores[key] = std::exp(scores[key] - peak); denominator += scores[key]; }
        for (int key = 0; key < scratch.m; ++key) scores[key] /= denominator;
        float* out = scratch.attention.data() + static_cast<std::size_t>(query) * scratch.d;
        for (int col = 0; col < scratch.d; ++col) {
            float value = 0.0f;
            for (int key = 0; key < scratch.m; ++key) value += scores[key] * scratch.value[static_cast<std::size_t>(key) * scratch.d + col];
            out[col] = value;
        }
    }
}

void silu_gate(const float* gate, const float* up, float* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
        output[i] = (gate[i] * sigmoid) * up[i];
    }
}

struct ComponentSample {
    double qkvo = 0.0;
    double swiglu = 0.0;
    double rms_residual = 0.0;
    double qkv = 0.0;
    double attention = 0.0;
    double output_projection = 0.0;
    double gate_up = 0.0;
    double silu = 0.0;
    double down = 0.0;
    double total = 0.0;
};

ComponentSample run_component_sample(const CoreWeights& weights, Scratch& scratch, WorkerPool& pool, const std::vector<float>& initial) {
    std::copy(initial.begin(), initial.end(), scratch.state.begin());
    ComponentSample row;
    const std::size_t md = static_cast<std::size_t>(scratch.m) * scratch.d;
    const std::size_t m4d = static_cast<std::size_t>(scratch.m) * 4 * scratch.d;
    const std::int64_t full_start = qpc_ticks();

    auto timed = [&](double& field, auto&& operation) {
        const std::int64_t start = qpc_ticks();
        operation();
        const std::int64_t stop = qpc_ticks();
        field += static_cast<double>(stop - start) / qpc_frequency();
    };

    for (int round = 0; round < kK; ++round) {
        timed(row.rms_residual, [&] { rms_rows(scratch.state.data(), scratch.normalized.data(), scratch.m, scratch.d); });
        timed(row.qkv, [&] {
            q4_linear(weights.W_Q, scratch.normalized.data(), scratch.query.data(), scratch.m, pool);
            q4_linear(weights.W_K, scratch.normalized.data(), scratch.key.data(), scratch.m, pool);
            q4_linear(weights.W_V, scratch.normalized.data(), scratch.value.data(), scratch.m, pool);
        });
        timed(row.attention, [&] { attention_slots(scratch); });
        timed(row.output_projection, [&] { q4_linear(weights.W_O, scratch.attention.data(), scratch.projected.data(), scratch.m, pool); });
        timed(row.rms_residual, [&] {
            for (std::size_t i = 0; i < md; ++i) scratch.hidden[i] = scratch.state[i] + scratch.projected[i];
            rms_rows(scratch.hidden.data(), scratch.mlp_normalized.data(), scratch.m, scratch.d);
        });
        timed(row.gate_up, [&] {
            q4_linear(weights.W_gate, scratch.mlp_normalized.data(), scratch.gate.data(), scratch.m, pool);
            q4_linear(weights.W_up, scratch.mlp_normalized.data(), scratch.up.data(), scratch.m, pool);
        });
        timed(row.silu, [&] { silu_gate(scratch.gate.data(), scratch.up.data(), scratch.gated.data(), m4d); });
        timed(row.down, [&] { q4_linear(weights.W_down, scratch.gated.data(), scratch.down.data(), scratch.m, pool); });
        timed(row.rms_residual, [&] {
            for (std::size_t i = 0; i < md; ++i) scratch.output[i] = scratch.hidden[i] + scratch.down[i];
            scratch.state.swap(scratch.output);
        });
    }
    const std::int64_t full_stop = qpc_ticks();
    row.total = static_cast<double>(full_stop - full_start) / qpc_frequency();
    row.qkvo = row.qkv + row.attention + row.output_projection;
    row.swiglu = row.gate_up + row.silu + row.down;
    return row;
}

} // namespace

int main(int argc, char** argv) {
    using namespace omega_v2_1;
    if (argc != 2 || std::string(argv[1]) != "--run-candidate01-component-diagnostic") {
        std::cerr << "Usage: candidate01_diag.exe --run-candidate01-component-diagnostic <weights-path-from-env>\n";
        return 2;
    }
    try {
        const std::filesystem::path executable = std::filesystem::absolute(argv[0]);
        const std::string result_env = env_utf8(L"OMEGA_V2_1B_CANDIDATE01_DIAG_RESULTS_ROOT");
        const std::string weight_env = env_utf8(L"OMEGA_V2_1B_CANDIDATE01_DIAG_WEIGHTS");
        const std::string shard_env = env_utf8(L"OMEGA_V2_1B_CANDIDATE01_DIAG_SHARD_WEIGHTS");
        if (result_env.empty() || weight_env.empty()) throw std::runtime_error("candidate01 diagnostic result/weight paths missing");
        const std::filesystem::path results_root(result_env);
        std::filesystem::create_directories(results_root);

        HardwareInfo hardware;
        std::string error;
        if (!query_hardware(hardware, error)) throw std::runtime_error("diagnostic host query failed: " + error);
        const std::vector<double> shard_weights = parse_weights(shard_env);
        if (shard_weights.size() != 4) throw std::runtime_error("candidate01 diagnostic needs four frozen attempt_02 shard weights");
        std::vector<CoreRecord> workers;
        for (std::size_t i = 0; i < 4; ++i) {
            auto core = std::find_if(hardware.cores.begin(), hardware.cores.end(), [&](const CoreRecord& c) {
                return std::any_of(c.cpu_sets.begin(), c.cpu_sets.end(), [&](const CpuSetRecord& cpu) { return cpu.id == kFrozenCpuSetIds[i]; });
            });
            if (core == hardware.cores.end() || !core->classified_p_core) throw std::runtime_error("candidate01 frozen P-core CPU-set missing");
            CoreRecord chosen = *core;
            auto cpu = std::find_if(chosen.cpu_sets.begin(), chosen.cpu_sets.end(), [&](const CpuSetRecord& c) { return c.id == kFrozenCpuSetIds[i]; });
            CpuSetRecord one = *cpu;
            one.shard_weight = shard_weights[i];
            chosen.cpu_sets.assign(1, one);
            chosen.v_i = one.shard_weight;
            workers.push_back(std::move(chosen));
        }
        const CpuSetRecord* coordinator = nullptr;
        for (const CoreRecord& core : hardware.cores) {
            if (core.intel_core_type == 0x20 && core.cpu_sets.size() == 1
                && (!coordinator || core.cpu_sets.front().id < coordinator->id)) coordinator = &core.cpu_sets.front();
        }
        if (!coordinator || !set_current_cpu_set(coordinator->id, coordinator->group, coordinator->logical_index, error)) {
            throw std::runtime_error("candidate01 diagnostic coordinator E-core affinity failed: " + error);
        }

        const CoreWeights weights = load_core(weight_env);
        WorkerPool pool({workers[0].cpu_sets.front(), workers[1].cpu_sets.front(), workers[2].cpu_sets.front(), workers[3].cpu_sets.front()});
        std::vector<const Q4Matrix*> matrices = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
        const std::size_t total_round_bytes = std::accumulate(matrices.begin(), matrices.end(), std::size_t{0}, [](std::size_t total, const Q4Matrix* matrix) {
            return total + matrix->packed.allocated_bytes + matrix->scales_fp16.allocated_bytes;
        });

        std::array<std::vector<double>, 7> row_sums;
        for (std::size_t i = 0; i < matrices.size(); ++i) row_sums[i].resize(static_cast<std::size_t>(matrices[i]->rows));
        auto dequant_only_batch = [&] {
            for (int round = 0; round < kK; ++round) {
                for (std::size_t matrix_index = 0; matrix_index < matrices.size(); ++matrix_index) {
                    const Q4Matrix* matrix = matrices[matrix_index];
                    DequantJob job{matrix, row_sums[matrix_index].data()};
                    pool.parallel_for(static_cast<std::size_t>(matrix->rows), &job, dequant_row_job);
                }
            }
        };
        auto consume_dequant_checksums = [&] {
            volatile double checksum = 0.0;
            for (const auto& matrix_sums : row_sums) checksum += std::accumulate(matrix_sums.begin(), matrix_sums.end(), 0.0);
            (void)checksum;
        };
        auto pretouch_core = [&] { touch_weights(weights, hardware.cache_line_bytes); };
        Series dequant_only;
        dequant_only.samples.reserve(kSamples);
        for (int warmup = 0; warmup < kWarmups; ++warmup) {
            pretouch_core();
            dequant_only_batch();
            consume_dequant_checksums();
        }
        for (int sample = 0; sample < kSamples; ++sample) {
            pretouch_core();
            const std::int64_t start = qpc_ticks();
            dequant_only_batch();
            const std::int64_t stop = qpc_ticks();
            if (stop <= start) throw std::runtime_error("dequant-only diagnostic timer failed");
            consume_dequant_checksums();
            dequant_only.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        }
        dequant_only.median_seconds = median(dequant_only.samples);

        Series empty_dispatch_28;
        empty_dispatch_28.samples.reserve(kSamples);
        for (int warmup = 0; warmup < kWarmups; ++warmup) for (int dispatch = 0; dispatch < 28; ++dispatch) pool.parallel_for(0, nullptr, empty_dispatch);
        for (int sample = 0; sample < kSamples; ++sample) {
            const std::int64_t start = qpc_ticks();
            for (int dispatch = 0; dispatch < 28; ++dispatch) pool.parallel_for(0, nullptr, empty_dispatch);
            const std::int64_t stop = qpc_ticks();
            if (stop <= start) throw std::runtime_error("empty-dispatch diagnostic timer failed");
            empty_dispatch_28.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        }
        empty_dispatch_28.median_seconds = median(empty_dispatch_28.samples);

        std::vector<float> input_d(static_cast<std::size_t>(kM) * kD);
        std::vector<float> input_4d(static_cast<std::size_t>(kM) * 4 * kD);
        std::vector<float> output(static_cast<std::size_t>(kM) * 4 * kD);
        for (std::size_t i = 0; i < input_d.size(); ++i) input_d[i] = static_cast<float>(static_cast<int>(i % 41) - 20) * 0.015625f;
        for (std::size_t i = 0; i < input_4d.size(); ++i) input_4d[i] = static_cast<float>(static_cast<int>(i % 47) - 23) * 0.0078125f;
        MatmulBatch batch{&weights,&pool,&input_d,&input_4d,&output};
        Series q4_gemm_28;
        q4_gemm_28.samples.reserve(kSamples);
        for (int warmup = 0; warmup < kWarmups; ++warmup) { pretouch_core(); q4_matmul_batch(batch); }
        for (int sample = 0; sample < kSamples; ++sample) {
            pretouch_core();
            const std::int64_t start = qpc_ticks();
            q4_matmul_batch(batch);
            const std::int64_t stop = qpc_ticks();
            if (stop <= start) throw std::runtime_error("dequant+FMA diagnostic timer failed");
            q4_gemm_28.samples.push_back(static_cast<double>(stop - start) / qpc_frequency());
        }
        q4_gemm_28.median_seconds = median(q4_gemm_28.samples);

        Scratch scratch;
        scratch.resize_for(kD, kM);
        const std::vector<float> initial = initial_state();
        std::vector<ComponentSample> component_samples;
        component_samples.reserve(kSamples);
        for (int warmup = 0; warmup < kWarmups; ++warmup) {
            touch_weights(weights, hardware.cache_line_bytes);
            component_samples.push_back(run_component_sample(weights, scratch, pool, initial));
        }
        component_samples.clear();
        for (int sample = 0; sample < kSamples; ++sample) {
            touch_weights(weights, hardware.cache_line_bytes);
            component_samples.push_back(run_component_sample(weights, scratch, pool, initial));
        }

        std::vector<double> full, qkvo, swiglu, rms_residual, qkv, attention, output_projection, gate_up, silu, down;
        for (const auto& row : component_samples) {
            full.push_back(row.total); qkvo.push_back(row.qkvo); swiglu.push_back(row.swiglu);
            rms_residual.push_back(row.rms_residual); qkv.push_back(row.qkv); attention.push_back(row.attention);
            output_projection.push_back(row.output_projection); gate_up.push_back(row.gate_up);
            silu.push_back(row.silu); down.push_back(row.down);
        }
        const double dequant_no_dispatch = (std::max)(0.0, dequant_only.median_seconds - empty_dispatch_28.median_seconds);
        const double q4_gemm_no_dispatch = (std::max)(0.0, q4_gemm_28.median_seconds - empty_dispatch_28.median_seconds);
        std::ostringstream out;
        out.precision(17);
        out << "{\"schema\":\"omega-v2-1b-candidate01-component-diagnostic-v1\",\"status\":\"DIAGNOSTIC_COMPLETE\""
            << ",\"candidate_id\":\"KQ1_DEQUANT_ROW_REUSE\",\"d\":512,\"m\":8,\"K\":4"
            << ",\"attempt02_cpu_sets\":" << selected_workers_json(workers)
            << ",\"attempt02_dependency_weight_seed\":20260929,\"qpc_frequency\":" << qpc_frequency()
            << ",\"repetitions\":31,\"warmups\":10,\"attempt03_executed\":false,\"a_b_c_executed\":false"
            << ",\"dequant_only_full_K4_28_dispatches\":{\"median_seconds_including_28_dispatches_and_row_checksum\":" << dequant_only.median_seconds
            << ",\"median_seconds_28_empty_dispatches\":" << empty_dispatch_28.median_seconds
            << ",\"estimated_dequant_plus_checksum_no_dispatch_seconds\":" << dequant_no_dispatch
            << ",\"dispatch_count_per_sample\":" << 7 * kK << ",\"observed_dispatch_count\":" << 7 * kK
            << ",\"q4_core_physical_bytes\":" << total_round_bytes << ",\"samples_seconds\":[";
        for (std::size_t i=0;i<dequant_only.samples.size();++i){if(i)out<<',';out<<dequant_only.samples[i];}
        out << "]}"
            << ",\"q4_dequant_plus_fma_full_K4_28_dispatches\":{\"median_seconds_including_dispatches\":" << q4_gemm_28.median_seconds
            << ",\"median_seconds_28_empty_dispatches\":" << empty_dispatch_28.median_seconds
            << ",\"estimated_dequant_plus_FMA_no_dispatch_seconds\":" << q4_gemm_no_dispatch
            << ",\"dequant_plus_FMA_includes_no_interstage_barrier\":true,\"samples_seconds\":[";
        for (std::size_t i=0;i<q4_gemm_28.samples.size();++i){if(i)out<<',';out<<q4_gemm_28.samples[i];}
        out << "]}"
            << ",\"dispatch_overhead_28_empty\":{\"median_seconds\":" << empty_dispatch_28.median_seconds << ",\"samples_seconds\":[";
        for (std::size_t i=0;i<empty_dispatch_28.samples.size();++i){if(i)out<<',';out<<empty_dispatch_28.samples[i];}
        out << "]}"
            << ",\"full_block_component_breakdown\":{\"median_total_K4_seconds\":" << median(full)
            << ",\"median_QKVO_seconds\":" << median(qkvo) << ",\"median_SwiGLU_seconds\":" << median(swiglu)
            << ",\"median_RMS_residual_seconds\":" << median(rms_residual)
            << ",\"median_QKV_matmul_seconds\":" << median(qkv) << ",\"median_attention_seconds\":" << median(attention)
            << ",\"median_WO_seconds\":" << median(output_projection) << ",\"median_gate_up_seconds\":" << median(gate_up)
            << ",\"median_SiLU_hadamard_seconds\":" << median(silu) << ",\"median_down_seconds\":" << median(down)
            << ",\"q4_linear_dispatches_per_full_K4\":28,\"samples\":[";
        for (std::size_t i=0;i<component_samples.size();++i){if(i)out<<',';const auto& r=component_samples[i];out<<"{\"total\":"<<r.total<<",\"QKVO\":"<<r.qkvo<<",\"SwiGLU\":"<<r.swiglu<<",\"RMS_residual\":"<<r.rms_residual<<",\"QKV\":"<<r.qkv<<",\"attention\":"<<r.attention<<",\"WO\":"<<r.output_projection<<",\"gate_up\":"<<r.gate_up<<",\"silu\":"<<r.silu<<",\"down\":"<<r.down<<'}';}
        out << "]},\"scope\":\"diagnostic_only\",\"residency_gates_evaluated\":false,\"attempt02_modified\":false}";
        const std::string report = out.str();
        std::ofstream result(results_root / "candidate01_component_diagnostic.json", std::ios::binary | std::ios::trunc);
        result << report << '\n';
        result.close();
        if (!result) throw std::runtime_error("failed writing candidate_01 component diagnostic");
        std::cout << report << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "candidate_01 component diagnostic error: " << exception.what() << '\n';
        return 3;
    }
}

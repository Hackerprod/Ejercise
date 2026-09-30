#include "v2_1.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace omega_v2_1 {

std::string q4_scalar_json_test(WorkerPool& pool) {
    std::vector<float> source(64 * 32);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(static_cast<int>((i * 17) % 101) - 50) * 0.00390625f;
    Q4Matrix weights;
    weights.pack(source, 64, 32);
    std::vector<float> input(4 * 32);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>((i * 7) % 37) - 18) * 0.015625f;
    std::vector<float> vectorized(4 * 64), scalar(4 * 64);
    q4_linear(weights, input.data(), vectorized.data(), 4, pool);
    q4_linear_single_thread(weights, input.data(), scalar.data(), 4);
    double max_abs = 0.0, max_rel = 0.0;
    for (std::size_t i = 0; i < vectorized.size(); ++i) {
        const double difference = std::fabs(static_cast<double>(vectorized[i]) - scalar[i]);
        max_abs = (std::max)(max_abs, difference);
        max_rel = (std::max)(max_rel, difference / (std::max)(std::fabs(static_cast<double>(scalar[i])), 1e-12));
    }
    const bool pass = max_abs <= 1e-5 && max_rel <= 1e-4;
    std::ostringstream out;
    out << "{\"d\":32,\"rows\":64,\"m\":4,\"group_size\":32,\"same_q4_bytes_and_fp16_scales\":true,\"max_abs_error\":"
        << max_abs << ",\"max_rel_error\":" << max_rel << ",\"abs_tolerance\":1e-5,\"rel_tolerance\":1e-4,\"pass\":"
        << (pass ? "true" : "false") << '}';
    return out.str();
}

std::string full_block_abc_correctness(WorkerPool& pool, const CoreWeights& weights) {
    const int d = weights.d;
    constexpr int m = 4, rounds = 8;
    std::vector<CoreWeights> b_weights;
    b_weights.reserve(rounds);
    for (int round = 0; round < rounds; ++round) b_weights.push_back(clone_weights(weights));
    const Q4Matrix* original[] = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    bool b_values_equal = true, b_storage_disjoint = true;
    std::vector<const void*> addresses;
    for (const Q4Matrix* matrix : original) { addresses.push_back(matrix->packed.data); addresses.push_back(matrix->scales_fp16.data); }
    for (const CoreWeights& clone : b_weights) {
        const Q4Matrix* matrices[] = {&clone.W_Q,&clone.W_K,&clone.W_V,&clone.W_O,&clone.W_gate,&clone.W_up,&clone.W_down};
        for (std::size_t index = 0; index < 7; ++index) {
            b_values_equal = b_values_equal
                && std::memcmp(matrices[index]->packed.data, original[index]->packed.data, matrices[index]->packed.logical_bytes) == 0
                && std::memcmp(matrices[index]->scales_fp16.data, original[index]->scales_fp16.data, matrices[index]->scales_fp16.logical_bytes) == 0;
            for (const void* address : addresses) if (address == matrices[index]->packed.data || address == matrices[index]->scales_fp16.data) b_storage_disjoint = false;
            addresses.push_back(matrices[index]->packed.data);
            addresses.push_back(matrices[index]->scales_fp16.data);
        }
    }
    Scratch a,b,c;
    a.resize_for(d,m); b.resize_for(d,m); c.resize_for(d,m);
    for (std::size_t i = 0; i < a.state.size(); ++i) {
        const float value = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.03125);
        a.state[i] = value; b.state[i] = value; c.state[i] = value;
    }
    for (int round = 0; round < rounds; ++round) {
        v2_full_block_round(weights,a,pool);
        v2_full_block_round(b_weights[static_cast<std::size_t>(round)],b,pool);
        v2_full_block_round(weights,c,pool);
    }
    const bool a_finite = std::all_of(a.state.begin(),a.state.end(),[](float value){return std::isfinite(value);});
    const bool b_finite = std::all_of(b.state.begin(),b.state.end(),[](float value){return std::isfinite(value);});
    const bool c_finite = std::all_of(c.state.begin(),c.state.end(),[](float value){return std::isfinite(value);});
    const bool equal = a.state == b.state && a.state == c.state;
    return std::string("{\"d\":") + std::to_string(d) + ",\"m\":4,\"K\":8,\"A_finite\":" + (a_finite?"true":"false")
        + ",\"B_finite\":" + (b_finite?"true":"false") + ",\"C_finite\":" + (c_finite?"true":"false")
        + ",\"A_B_C_bitwise_equal\":" + (equal?"true":"false") + ",\"A_C_same_storage\":true,\"B_values_equal_A\":"
        + (b_values_equal?"true":"false") + ",\"B_round_storages_disjoint\":" + (b_storage_disjoint?"true":"false") + '}';
}

} // namespace omega_v2_1

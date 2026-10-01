#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace omega_v2_1d {

struct MetricSummary {
    std::size_t element_count = 0;
    double candidate_l2 = 0.0;
    double reference_l2 = 0.0;
    double candidate_inf = 0.0;
    double reference_inf = 0.0;
    double candidate_rms = 0.0;
    double reference_rms = 0.0;
    double max_abs = 0.0;
    double e_l2 = 0.0;
    double e_inf = 0.0;
    double scale_aware_max_abs_ratio = 0.0;
    double old_max_rel = 0.0;
    std::size_t max_abs_index = 0;
    std::size_t scale_aware_index = 0;
    std::size_t old_max_rel_index = 0;
    bool both_zero_e_l2 = false;
    bool both_zero_e_inf = false;
    bool zero_scale = false;
    bool both_tensors_zero = false;
    bool all_finite = false;
};

MetricSummary summarize(const std::vector<float>& candidate,
                       const std::vector<float>& reference);

std::string metric_json(const MetricSummary& metrics,
                        const std::vector<float>& candidate,
                        const std::vector<float>& reference,
                        int rows,
                        int cols);

std::uint64_t checksum_floats(const std::vector<float>& values) noexcept;

} // namespace omega_v2_1d

#include "stage_a_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace omega_v2_1d {
namespace {

void write_location(std::ostringstream& out, std::size_t flat, int cols) {
    if (cols > 0) out << "[" << flat / static_cast<std::size_t>(cols) << "," << flat % static_cast<std::size_t>(cols) << "]";
    else out << "[" << flat << "]";
}

} // namespace

MetricSummary summarize(const std::vector<float>& candidate, const std::vector<float>& reference) {
    if (candidate.size() != reference.size()) throw std::invalid_argument("candidate/reference tensor sizes differ");
    MetricSummary result;
    result.element_count = candidate.size();
    result.all_finite = true;
    double delta_sq = 0.0;
    double candidate_sq = 0.0;
    double reference_sq = 0.0;
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const double c = candidate[i];
        const double r = reference[i];
        if (!std::isfinite(c) || !std::isfinite(r)) {
            result.all_finite = false;
            continue;
        }
        const double difference = std::fabs(c - r);
        delta_sq += difference * difference;
        candidate_sq += c * c;
        reference_sq += r * r;
        result.candidate_inf = (std::max)(result.candidate_inf, std::fabs(c));
        result.reference_inf = (std::max)(result.reference_inf, std::fabs(r));
        if (difference > result.max_abs) {
            result.max_abs = difference;
            result.max_abs_index = i;
        }
    }
    if (!result.all_finite) throw std::invalid_argument("metrics require finite candidate/reference tensors");

    const double delta_l2 = std::sqrt(delta_sq);
    result.candidate_l2 = std::sqrt(candidate_sq);
    result.reference_l2 = std::sqrt(reference_sq);
    result.candidate_rms = candidate.empty() ? 0.0 : std::sqrt(candidate_sq / static_cast<double>(candidate.size()));
    result.reference_rms = candidate.empty() ? 0.0 : std::sqrt(reference_sq / static_cast<double>(candidate.size()));
    const double l2_scale = (std::max)(result.candidate_l2, result.reference_l2);
    const double inf_scale = (std::max)(result.candidate_inf, result.reference_inf);
    result.both_zero_e_l2 = l2_scale == 0.0;
    result.both_zero_e_inf = inf_scale == 0.0;
    result.e_l2 = result.both_zero_e_l2 ? 0.0 : delta_l2 / l2_scale;
    result.e_inf = result.both_zero_e_inf ? 0.0 : result.max_abs / inf_scale;
    result.both_tensors_zero = std::all_of(candidate.begin(), candidate.end(), [](float x) { return x == 0.0f; })
                            && std::all_of(reference.begin(), reference.end(), [](float x) { return x == 0.0f; });
    result.zero_scale = result.both_tensors_zero;

    const double scale_floor = std::sqrt(std::ldexp(1.0, -23)) * (std::max)(result.candidate_rms, result.reference_rms);
    double scaled_max = 0.0;
    double old_max_rel = 0.0;
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const double c = candidate[i];
        const double r = reference[i];
        const double difference = std::fabs(c - r);
        const double scaled_denominator = (std::max)((std::max)(std::fabs(c), std::fabs(r)), scale_floor);
        const double scaled = result.both_tensors_zero ? 0.0 : difference / scaled_denominator;
        if (scaled > scaled_max) {
            scaled_max = scaled;
            result.scale_aware_index = i;
        }
        const double legacy = difference / (std::max)(std::fabs(r), 1e-12);
        if (legacy > old_max_rel) {
            old_max_rel = legacy;
            result.old_max_rel_index = i;
        }
    }
    result.scale_aware_max_abs_ratio = scaled_max;
    result.old_max_rel = old_max_rel;
    return result;
}

std::string metric_json(const MetricSummary& metrics,
                        const std::vector<float>& candidate,
                        const std::vector<float>& reference,
                        int rows,
                        int cols) {
    const std::size_t old_i = metrics.old_max_rel_index;
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"element_count\":" << metrics.element_count
        << ",\"all_finite\":" << (metrics.all_finite ? "true" : "false")
        << ",\"candidate_l2\":" << metrics.candidate_l2
        << ",\"reference_l2\":" << metrics.reference_l2
        << ",\"candidate_inf\":" << metrics.candidate_inf
        << ",\"reference_inf\":" << metrics.reference_inf
        << ",\"candidate_rms\":" << metrics.candidate_rms
        << ",\"reference_rms\":" << metrics.reference_rms
        << ",\"E_L2\":" << metrics.e_l2
        << ",\"E_inf\":" << metrics.e_inf
        << ",\"both_zero_E_L2\":" << (metrics.both_zero_e_l2 ? "true" : "false")
        << ",\"both_zero_E_inf\":" << (metrics.both_zero_e_inf ? "true" : "false")
        << ",\"max_abs\":" << metrics.max_abs
        << ",\"scale_aware_max_abs_ratio\":" << metrics.scale_aware_max_abs_ratio
        << ",\"zero_scale\":" << (metrics.zero_scale ? "true" : "false")
        << ",\"old_max_rel\":" << metrics.old_max_rel
        << ",\"old_max_rel_role\":\"HISTORICAL_COMPATIBILITY_DIAGNOSTIC\""
        << ",\"max_abs_index\":" << metrics.max_abs_index << ",\"max_abs_coordinate\":";
    write_location(out, metrics.max_abs_index, cols);
    out << ",\"max_abs_candidate\":" << candidate.at(metrics.max_abs_index)
        << ",\"max_abs_reference\":" << reference.at(metrics.max_abs_index)
        << ",\"scale_aware_index\":" << metrics.scale_aware_index << ",\"scale_aware_coordinate\":";
    write_location(out, metrics.scale_aware_index, cols);
    out << ",\"scale_aware_candidate\":" << candidate.at(metrics.scale_aware_index)
        << ",\"scale_aware_reference\":" << reference.at(metrics.scale_aware_index)
        << ",\"old_max_rel_index\":" << old_i << ",\"old_max_rel_coordinate\":";
    write_location(out, old_i, cols);
    out << ",\"old_max_rel_candidate\":" << candidate.at(old_i)
        << ",\"old_max_rel_reference\":" << reference.at(old_i)
        << ",\"old_max_rel_abs_error\":" << std::fabs(static_cast<double>(candidate.at(old_i)) - reference.at(old_i))
        << ",\"old_max_rel_denominator\":" << (std::max)(std::fabs(static_cast<double>(reference.at(old_i))), 1e-12)
        << ",\"candidate_checksum\":" << checksum_floats(candidate)
        << ",\"reference_checksum\":" << checksum_floats(reference)
        << ",\"shape\":[" << rows << ',' << cols << "]}";
    return out.str();
}

std::uint64_t checksum_floats(const std::vector<float>& values) noexcept {
    std::uint64_t hash = 1469598103934665603ull;
    for (float value : values) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<std::uint8_t>(bits >> (8 * byte));
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

} // namespace omega_v2_1d

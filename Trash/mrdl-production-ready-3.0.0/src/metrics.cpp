#include "mrdl/metrics.hpp"

namespace mrdl {
namespace {

std::string escape_json(std::string_view value) {
    std::string result;
    for (const char c : value) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

}  // namespace

void MetricsRegistry::observe_prediction(Lane lane, const LaneMetrics& metrics) {
    const std::string prefix = lane == Lane::Full ? "mrdl_full_" : "mrdl_clean_";
    std::lock_guard lock(mutex_);
    ++counters_[prefix + "predictions_total"];
    counters_[prefix + "candidate_retrievals_total"] += metrics.candidate_retrievals;
    counters_[prefix + "operator_evaluations_total"] += metrics.operator_evaluations;
    counters_[prefix + "gate_evaluations_total"] += metrics.gate_evaluations;
    counters_[prefix + "branches_created_total"] += metrics.branches_created;
    counters_[prefix + "branches_surviving_total"] += metrics.branches_surviving;
    counters_[prefix + "runtime_ns_total"] += metrics.runtime_ns;
    counters_[prefix + "empty_total"] += metrics.empty ? 1U : 0U;
}

void MetricsRegistry::increment(std::string_view name, std::uint64_t amount) {
    std::lock_guard lock(mutex_);
    counters_[std::string(name)] += amount;
}

std::string MetricsRegistry::prometheus() const {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<std::string, std::uint64_t>> ordered(counters_.begin(), counters_.end());
    std::sort(ordered.begin(), ordered.end());
    std::ostringstream output;
    for (const auto& [name, value] : ordered) output << name << ' ' << value << '\n';
    return output.str();
}

std::string MetricsRegistry::json() const {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<std::string, std::uint64_t>> ordered(counters_.begin(), counters_.end());
    std::sort(ordered.begin(), ordered.end());
    std::ostringstream output;
    output << '{';
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        if (i != 0U) output << ',';
        output << '"' << escape_json(ordered[i].first) << "\":" << ordered[i].second;
    }
    output << '}';
    return output.str();
}

}  // namespace mrdl

#pragma once

#include "mrdl/common.hpp"

namespace mrdl {

struct LaneMetrics {
    std::uint64_t candidate_retrievals{0};
    std::uint64_t operator_evaluations{0};
    std::uint64_t gate_evaluations{0};
    std::uint64_t branches_created{0};
    std::uint64_t branches_surviving{0};
    std::uint64_t port_assignments{0};
    std::uint64_t active_state_peak{0};
    std::uint64_t replay_steps{0};
    std::uint64_t runtime_ns{0};
    float clean_health_ratio{1.0F};
    bool empty{false};
};

struct DualMetrics {
    LaneMetrics full;
    LaneMetrics clean;
    float runtime_ratio{0.0F};
    float operator_ratio{0.0F};
    std::uint64_t total_runtime_ns{0};
};

class MetricsRegistry final {
public:
    void observe_prediction(Lane lane, const LaneMetrics& metrics);
    void increment(std::string_view name, std::uint64_t amount = 1U);
    [[nodiscard]] std::string prometheus() const;
    [[nodiscard]] std::string json() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::uint64_t> counters_;
};

}  // namespace mrdl

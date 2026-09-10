#include "rlm/metrics.hpp"

#include <sstream>

namespace rlm {

void EngineMetrics::record_inference(bool success, bool both_lanes, std::uint64_t latency_ns) noexcept {
  inference_requests_.fetch_add(1, std::memory_order_relaxed);
  if (!success) inference_errors_.fetch_add(1, std::memory_order_relaxed);
  if (both_lanes) dual_lane_requests_.fetch_add(1, std::memory_order_relaxed);
  inference_latency_ns_.fetch_add(latency_ns, std::memory_order_relaxed);
}

std::string EngineMetrics::prometheus(std::size_t m1_edges,
                                      std::size_t m2_edges,
                                      std::size_t replay_records) const {
  std::ostringstream output;
  output << "# TYPE rlm_inference_requests_total counter\n"
         << "rlm_inference_requests_total " << inference_requests_.load(std::memory_order_relaxed) << '\n'
         << "# TYPE rlm_inference_errors_total counter\n"
         << "rlm_inference_errors_total " << inference_errors_.load(std::memory_order_relaxed) << '\n'
         << "# TYPE rlm_dual_lane_requests_total counter\n"
         << "rlm_dual_lane_requests_total " << dual_lane_requests_.load(std::memory_order_relaxed) << '\n'
         << "# TYPE rlm_inference_latency_seconds_total counter\n"
         << "rlm_inference_latency_seconds_total "
         << static_cast<double>(inference_latency_ns_.load(std::memory_order_relaxed)) / 1.0e9 << '\n'
         << "# TYPE rlm_training_batches_total counter\n"
         << "rlm_training_batches_total " << training_batches_.load(std::memory_order_relaxed) << '\n'
         << "# TYPE rlm_promotions_total counter\n"
         << "rlm_promotions_total " << promotions_.load(std::memory_order_relaxed) << '\n'
         << "# TYPE rlm_promotion_failures_total counter\n"
         << "rlm_promotion_failures_total " << promotion_failures_.load(std::memory_order_relaxed) << '\n'
         << "# TYPE rlm_m1_edges gauge\nrlm_m1_edges " << m1_edges << '\n'
         << "# TYPE rlm_m2_edges gauge\nrlm_m2_edges " << m2_edges << '\n'
         << "# TYPE rlm_replay_records gauge\nrlm_replay_records " << replay_records << '\n';
  return output.str();
}

}  // namespace rlm

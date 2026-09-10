#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace rlm {

class EngineMetrics final {
 public:
  void record_inference(bool success, bool both_lanes, std::uint64_t latency_ns) noexcept;
  void record_training_batch() noexcept { training_batches_.fetch_add(1, std::memory_order_relaxed); }
  void record_promotion(bool committed) noexcept {
    (committed ? promotions_ : promotion_failures_).fetch_add(1, std::memory_order_relaxed);
  }
  [[nodiscard]] std::string prometheus(std::size_t m1_edges,
                                       std::size_t m2_edges,
                                       std::size_t replay_records) const;

 private:
  std::atomic<std::uint64_t> inference_requests_{0};
  std::atomic<std::uint64_t> inference_errors_{0};
  std::atomic<std::uint64_t> dual_lane_requests_{0};
  std::atomic<std::uint64_t> inference_latency_ns_{0};
  std::atomic<std::uint64_t> training_batches_{0};
  std::atomic<std::uint64_t> promotions_{0};
  std::atomic<std::uint64_t> promotion_failures_{0};
};

}  // namespace rlm

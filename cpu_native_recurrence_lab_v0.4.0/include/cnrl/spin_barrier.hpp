#pragma once
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <immintrin.h>
namespace cnrl {
class SpinBarrier {
 public:
  explicit SpinBarrier(std::uint32_t participants)
      : participants_(participants), remaining_(participants) {
    if (participants == 0) throw std::invalid_argument("barrier needs participants");
  }
  void arrive_and_wait() noexcept {
    const auto generation = generation_.load(std::memory_order_acquire);
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      remaining_.store(participants_, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
      return;
    }
    std::uint32_t spins = 0;
    while (generation_.load(std::memory_order_acquire) == generation) {
      _mm_pause();
      if ((++spins & 0x3FFFU) == 0) std::this_thread::yield();
    }
  }
 private:
  const std::uint32_t participants_;
  std::atomic<std::uint32_t> remaining_;
  std::atomic<std::uint32_t> generation_{0};
};
}  // namespace cnrl

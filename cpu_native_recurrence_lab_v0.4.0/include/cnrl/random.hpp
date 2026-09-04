#pragma once
#include <cstdint>
namespace cnrl {
class XorShift32 {
 public:
  explicit XorShift32(std::uint32_t seed) : state_(seed == 0 ? 0x6D2B79F5U : seed) {}
  [[nodiscard]] std::uint32_t next() noexcept {
    state_ ^= state_ << 13U;
    state_ ^= state_ >> 17U;
    state_ ^= state_ << 5U;
    return state_;
  }
  [[nodiscard]] std::int8_t symmetric_i8(std::int32_t magnitude) noexcept {
    const std::uint32_t span = static_cast<std::uint32_t>(2 * magnitude + 1);
    return static_cast<std::int8_t>(static_cast<std::int32_t>(next() % span) - magnitude);
  }
 private:
  std::uint32_t state_;
};
}  // namespace cnrl

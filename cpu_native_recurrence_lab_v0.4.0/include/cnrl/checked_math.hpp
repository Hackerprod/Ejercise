#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cnrl {

[[nodiscard]] inline std::uint64_t checked_add_u64(std::uint64_t a, std::uint64_t b) {
  if (a > (std::numeric_limits<std::uint64_t>::max)() - b) {
    throw std::overflow_error("uint64 addition overflow");
  }
  return a + b;
}

[[nodiscard]] inline std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b) {
  if (b != 0 && a > (std::numeric_limits<std::uint64_t>::max)() / b) {
    throw std::overflow_error("uint64 multiplication overflow");
  }
  return a * b;
}

[[nodiscard]] inline std::uint32_t ceil_div_u32(std::uint32_t value,
                                                std::uint32_t divisor) {
  if (divisor == 0) throw std::invalid_argument("division by zero");
  return value / divisor + (value % divisor == 0 ? 0U : 1U);
}

[[nodiscard]] inline std::int64_t abs_i64_no_ub(std::int64_t value) {
  if (value == (std::numeric_limits<std::int64_t>::min)()) {
    throw std::overflow_error("cannot take abs(INT64_MIN)");
  }
  return value < 0 ? -value : value;
}

[[nodiscard]] inline std::int32_t round_shift_i64(std::int64_t value,
                                                  std::uint32_t shift) {
  if (shift >= 63) throw std::invalid_argument("shift must be below 63");
  const std::int64_t magnitude = abs_i64_no_ub(value);
  const std::int64_t bias = shift == 0 ? 0 : (std::int64_t{1} << (shift - 1U));
  const std::int64_t rounded = shift == 0 ? magnitude : (magnitude + bias) >> shift;
  const std::int64_t signed_result = value < 0 ? -rounded : rounded;
  if (signed_result < (std::numeric_limits<std::int32_t>::min)() ||
      signed_result > (std::numeric_limits<std::int32_t>::max)()) {
    throw std::overflow_error("rounded value does not fit int32");
  }
  return static_cast<std::int32_t>(signed_result);
}

}  // namespace cnrl

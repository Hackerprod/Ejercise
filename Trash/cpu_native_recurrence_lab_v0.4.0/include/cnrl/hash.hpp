#pragma once
#include <cstddef>
#include <cstdint>
namespace cnrl {
inline constexpr std::uint64_t kFnv1a64Offset = 1469598103934665603ULL;
inline constexpr std::uint64_t kFnv1a64Prime = 1099511628211ULL;
[[nodiscard]] inline std::uint64_t fnv1a64_update(std::uint64_t hash,
                                                  const void* data,
                                                  std::size_t bytes) noexcept {
  const auto* input = static_cast<const std::uint8_t*>(data);
  for (std::size_t i=0;i<bytes;++i) { hash ^= input[i]; hash *= kFnv1a64Prime; }
  return hash;
}
[[nodiscard]] inline std::uint64_t fnv1a64(const void* data,std::size_t bytes) noexcept {
  return fnv1a64_update(kFnv1a64Offset,data,bytes);
}
[[nodiscard]] inline std::uint64_t hash_combine(std::uint64_t seed,std::uint64_t value) noexcept {
  return seed ^ (value + 0x9E3779B97F4A7C15ULL + (seed<<6U) + (seed>>2U));
}
}  // namespace cnrl

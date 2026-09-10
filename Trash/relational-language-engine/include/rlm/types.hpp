#pragma once

#include "rlm/status.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace rlm {

using TokenId = std::uint32_t;
using EdgeId = std::uint64_t;
using TraceId = std::uint64_t;
using Epoch = std::uint64_t;
using BatchId = std::uint64_t;

constexpr TokenId kInvalidToken = std::numeric_limits<TokenId>::max();

enum class Lane : std::uint8_t { full = 0, clean = 1 };
enum class EvidenceTier : std::uint8_t { m1 = 1, m2 = 2 };
enum class AuditVerdict : std::uint8_t { promote = 0, reject = 1, unknown = 2 };
enum class Durability : std::uint8_t { none = 0, data = 1, full = 2 };

[[nodiscard]] std::uint64_t stable_hash64(std::span<const std::byte> bytes,
                                          std::uint64_t seed = 1469598103934665603ULL) noexcept;
[[nodiscard]] std::uint64_t stable_hash64(std::string_view text,
                                          std::uint64_t seed = 1469598103934665603ULL) noexcept;
[[nodiscard]] std::uint64_t hash_combine64(std::uint64_t a, std::uint64_t b) noexcept;
[[nodiscard]] std::uint64_t unix_time_ms() noexcept;
[[nodiscard]] EdgeId deterministic_edge_id(TokenId source, TokenId target) noexcept;

struct QuantizedVector final {
  float scale{1.0F};
  std::vector<std::int8_t> values;

  [[nodiscard]] std::size_t size() const noexcept { return values.size(); }
  [[nodiscard]] bool empty() const noexcept { return values.empty(); }
  [[nodiscard]] Status validate(std::size_t expected_dim = 0) const;
  [[nodiscard]] std::vector<float> dequantize() const;
  [[nodiscard]] static Result<QuantizedVector> quantize(std::span<const float> input);
};

[[nodiscard]] float dot(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] float norm(std::span<const float> a) noexcept;
[[nodiscard]] float cosine(std::span<const float> a, std::span<const float> b) noexcept;
[[nodiscard]] std::vector<float> normalized(std::span<const float> a);
[[nodiscard]] float l2_distance(std::span<const float> a, std::span<const float> b) noexcept;

}  // namespace rlm

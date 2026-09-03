#include "rlm/types.hpp"

#include <cstring>

namespace rlm {

std::uint64_t stable_hash64(std::span<const std::byte> bytes, std::uint64_t seed) noexcept {
  std::uint64_t hash = seed;
  for (const std::byte value : bytes) {
    hash ^= static_cast<std::uint64_t>(std::to_integer<unsigned char>(value));
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t stable_hash64(std::string_view text, std::uint64_t seed) noexcept {
  return stable_hash64(std::as_bytes(std::span{text.data(), text.size()}), seed);
}

std::uint64_t hash_combine64(std::uint64_t a, std::uint64_t b) noexcept {
  a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U);
  return a;
}

std::uint64_t unix_time_ms() noexcept {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

EdgeId deterministic_edge_id(TokenId source, TokenId target) noexcept {
  const std::uint64_t packed = (static_cast<std::uint64_t>(source) << 32U) |
                               static_cast<std::uint64_t>(target);
  return hash_combine64(0x524c4d4544474501ULL, packed);
}

Status QuantizedVector::validate(std::size_t expected_dim) const {
  if (!std::isfinite(scale) || scale <= 0.0F) {
    return Status(ErrorCode::data_loss, "quantized vector has invalid scale");
  }
  if (values.empty()) return Status(ErrorCode::data_loss, "quantized vector is empty");
  if (expected_dim != 0 && values.size() != expected_dim) {
    return Status(ErrorCode::data_loss, "quantized vector dimension mismatch");
  }
  return Status::Ok();
}

std::vector<float> QuantizedVector::dequantize() const {
  std::vector<float> output(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = static_cast<float>(values[i]) * scale;
  }
  return output;
}

Result<QuantizedVector> QuantizedVector::quantize(std::span<const float> input) {
  if (input.empty()) return Status(ErrorCode::invalid_argument, "cannot quantize an empty vector");
  float max_abs = 0.0F;
  for (const float value : input) {
    if (!std::isfinite(value)) {
      return Status(ErrorCode::invalid_argument, "vector contains a non-finite value");
    }
    max_abs = std::max(max_abs, std::abs(value));
  }
  QuantizedVector output;
  output.scale = max_abs > 0.0F ? max_abs / 127.0F : 1.0F / 127.0F;
  output.values.resize(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float q = std::round(input[i] / output.scale);
    output.values[i] = static_cast<std::int8_t>(std::clamp(q, -127.0F, 127.0F));
  }
  return output;
}

float dot(std::span<const float> a, std::span<const float> b) noexcept {
  if (a.size() != b.size()) return 0.0F;
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
  return static_cast<float>(sum);
}

float norm(std::span<const float> a) noexcept {
  return std::sqrt(std::max(0.0F, dot(a, a)));
}

float cosine(std::span<const float> a, std::span<const float> b) noexcept {
  if (a.size() != b.size() || a.empty()) return 0.0F;
  const float denominator = norm(a) * norm(b);
  if (denominator <= std::numeric_limits<float>::epsilon()) return 0.0F;
  return std::clamp(dot(a, b) / denominator, -1.0F, 1.0F);
}

std::vector<float> normalized(std::span<const float> a) {
  std::vector<float> output(a.begin(), a.end());
  const float length = norm(a);
  if (length > std::numeric_limits<float>::epsilon()) {
    for (float& value : output) value /= length;
  }
  return output;
}

float l2_distance(std::span<const float> a, std::span<const float> b) noexcept {
  if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = static_cast<double>(a[i]) - b[i];
    sum += diff * diff;
  }
  return static_cast<float>(std::sqrt(sum));
}

}  // namespace rlm

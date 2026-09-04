#include "cnrl/transitions.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <immintrin.h>
#include "cnrl/hash.hpp"

namespace cnrl {
namespace {
inline __m256i round_shift_epi32(__m256i value, std::uint32_t shift) noexcept {
  if (shift == 0) return value;
  const __m256i sign = _mm256_srai_epi32(value, 31);
  __m256i magnitude = _mm256_sub_epi32(_mm256_xor_si256(value, sign), sign);
  magnitude = _mm256_add_epi32(magnitude, _mm256_set1_epi32(1 << (shift - 1U)));
  const __m256i count = _mm256_set1_epi32(static_cast<int>(shift));
  const __m256i rounded = _mm256_srlv_epi32(magnitude, count);
  return _mm256_sub_epi32(_mm256_xor_si256(rounded, sign), sign);
}

inline __m256i residual8(const std::int8_t* state, const std::int32_t* output,
                         const TransitionConfig& config) noexcept {
  const __m128i state8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(state));
  __m256i x = _mm256_cvtepi8_epi32(state8);
  __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(output));
  if (config.state_multiplier != 1) x = _mm256_mullo_epi32(x, _mm256_set1_epi32(config.state_multiplier));
  if (config.output_multiplier != 1) y = _mm256_mullo_epi32(y, _mm256_set1_epi32(config.output_multiplier));
  y = round_shift_epi32(y, config.projection_shift);
  return _mm256_add_epi32(x, y);
}

inline std::int32_t residual1(std::int8_t state, std::int32_t output,
                              const TransitionConfig& config) noexcept {
  const std::int64_t projected_product = static_cast<std::int64_t>(output) * config.output_multiplier;
  const std::int64_t magnitude = projected_product < 0 ? -projected_product : projected_product;
  const std::int64_t bias = config.projection_shift == 0 ? 0 :
      (std::int64_t{1} << (config.projection_shift - 1U));
  const std::int64_t projected_mag = config.projection_shift == 0 ? magnitude :
      (magnitude + bias) >> config.projection_shift;
  const std::int64_t projected = projected_product < 0 ? -projected_mag : projected_mag;
  return static_cast<std::int32_t>(static_cast<std::int64_t>(state) * config.state_multiplier + projected);
}

inline std::int32_t shift_scalar(std::int32_t value, std::uint32_t shift) noexcept {
  if (shift == 0) return value;
  const std::int64_t v = value;
  const std::int64_t mag = v < 0 ? -v : v;
  const std::int64_t q = (mag + (std::int64_t{1} << (shift - 1U))) >> shift;
  return static_cast<std::int32_t>(v < 0 ? -q : q);
}

inline std::int8_t clamp_i8(std::int32_t value, TransitionStats& stats) noexcept {
  ++stats.cells;
  if (value > 127) { ++stats.clipped_cells; return 127; }
  if (value < -127) { ++stats.clipped_cells; return -127; }
  return static_cast<std::int8_t>(value);
}

inline void store_clamped_i8x8(__m256i value, std::int8_t* destination,
                               TransitionStats& stats) noexcept {
  const __m256i maximum = _mm256_set1_epi32(127);
  const __m256i minimum = _mm256_set1_epi32(-127);
  const __m256i outside = _mm256_or_si256(
      _mm256_cmpgt_epi32(value, maximum), _mm256_cmpgt_epi32(minimum, value));
  stats.cells += 8U;
  stats.clipped_cells += static_cast<std::uint64_t>(std::popcount(
      static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(outside)))));
  value = _mm256_min_epi32(maximum, _mm256_max_epi32(minimum, value));
  const __m128i packed16 = _mm_packs_epi32(
      _mm256_castsi256_si128(value), _mm256_extracti128_si256(value, 1));
  const __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
  _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), packed8);
}

inline void store_clamped_i8x4(__m128i value, std::int8_t* destination,
                               TransitionStats& stats) noexcept {
  const __m128i maximum = _mm_set1_epi32(127);
  const __m128i minimum = _mm_set1_epi32(-127);
  const __m128i outside = _mm_or_si128(
      _mm_cmpgt_epi32(value, maximum), _mm_cmpgt_epi32(minimum, value));
  stats.cells += 4U;
  stats.clipped_cells += static_cast<std::uint64_t>(std::popcount(
      static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(outside)))));
  value = _mm_min_epi32(maximum, _mm_max_epi32(minimum, value));
  const __m128i packed16 = _mm_packs_epi32(value, _mm_setzero_si128());
  const __m128i packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
  const std::uint32_t packed = static_cast<std::uint32_t>(_mm_cvtsi128_si32(packed8));
  std::memcpy(destination, &packed, sizeof(packed));
}

inline double sum_squares8(__m256i value) noexcept {
  const __m128i low = _mm256_castsi256_si128(value);
  const __m128i high = _mm256_extracti128_si256(value, 1);
  __m256d a = _mm256_cvtepi32_pd(low);
  __m256d b = _mm256_cvtepi32_pd(high);
  a = _mm256_mul_pd(a, a);
  b = _mm256_mul_pd(b, b);
  alignas(32) double lanes[4];
  _mm256_store_pd(lanes, _mm256_add_pd(a, b));
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

void apply_rms_range(const std::int32_t* residual, std::int8_t* next,
                     std::size_t begin, std::size_t count, double factor,
                     TransitionStats& stats) noexcept {
  std::size_t i = 0;
  const __m256d vf = _mm256_set1_pd(factor);
  for (; i + 4 <= count; i += 4) {
    const __m128i vi = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(residual + begin + i));
    __m256d vd = _mm256_cvtepi32_pd(vi);
    vd = _mm256_round_pd(_mm256_mul_pd(vd, vf),
                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    store_clamped_i8x4(_mm256_cvtpd_epi32(vd), next + begin + i, stats);
  }
  for (; i < count; ++i) {
    next[begin + i] = clamp_i8(static_cast<std::int32_t>(std::nearbyint(residual[begin + i] * factor)), stats);
  }
}
}  // namespace


void validate_transition_config(const Shape& shape,
                                const TransitionConfig& config) {
  if (shape.dimension == 0U || shape.slots == 0U || shape.slots > 32U) {
    throw std::invalid_argument("transition requires D>0 and 1<=S<=32");
  }
  if (config.kind == TransitionKind::frozen) return;
  if (config.projection_shift > 30U || config.final_shift > 30U) {
    throw std::invalid_argument("transition shifts must be <=30");
  }
  if (std::llabs(static_cast<long long>(config.state_multiplier)) > 16LL ||
      std::llabs(static_cast<long long>(config.output_multiplier)) > 1LL) {
    throw std::invalid_argument(
        "AVX2 transition requires |state_multiplier|<=16 and |output_multiplier|<=1");
  }
  if (config.kind != TransitionKind::fixed_point && config.final_shift != 0U) {
    throw std::invalid_argument("final_shift is defined only for fixed-point transition");
  }
  if (!(config.target_rms > 0.0) || !std::isfinite(config.target_rms) ||
      !(config.epsilon > 0.0) || !std::isfinite(config.epsilon)) {
    throw std::invalid_argument("RMS parameters must be finite and positive");
  }
  const std::int64_t max_dot = static_cast<std::int64_t>(shape.dimension) * 127LL * 127LL;
  const std::int64_t output_scale =
      std::llabs(static_cast<long long>(config.output_multiplier));
  const std::int64_t scaled_dot = max_dot * output_scale;
  const std::int64_t max_projected = config.projection_shift == 0U
      ? scaled_dot
      : (scaled_dot + (std::int64_t{1} << (config.projection_shift - 1U))) >>
            config.projection_shift;
  const std::int64_t max_state =
      127LL * std::llabs(static_cast<long long>(config.state_multiplier));
  if (max_projected + max_state >
      static_cast<std::int64_t>((std::numeric_limits<std::int32_t>::max)())) {
    throw std::invalid_argument("transition residual can overflow int32");
  }
}

void prepare_transition_workspace(TransitionWorkspace& workspace,
                                  std::uint32_t workers,
                                  const Shape& shape) {
  if (shape.slots == 0 || shape.slots > 32) throw std::invalid_argument("transition supports 1..32 slots");
  workspace.residual.resize(static_cast<std::size_t>(shape.slots) * shape.dimension);
  workspace.partial_sums.assign(workers, PartialSums{});
  workspace.inverse_rms.resize(shape.slots);
}

void transition_fixed_point_local(const std::int8_t* current_state,
                                  const std::int32_t* output,
                                  std::int8_t* next_state,
                                  const Shape& shape,
                                  const ShardSpec& shard,
                                  const TransitionConfig& config,
                                  TransitionStats& stats) noexcept {
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * shape.dimension + shard.row_offset;
    std::uint32_t row = 0;
    for (; row + 8 <= shard.rows; row += 8) {
      __m256i mixed = residual8(current_state + base + row,
                                output + base + row, config);
      mixed = round_shift_epi32(mixed, config.final_shift);
      store_clamped_i8x8(mixed, next_state + base + row, stats);
    }
    for (; row < shard.rows; ++row) {
      next_state[base + row] = clamp_i8(
          shift_scalar(residual1(current_state[base + row], output[base + row], config), config.final_shift), stats);
    }
  }
}

void transition_group_rms_local(const std::int8_t* current_state,
                                const std::int32_t* output,
                                std::int8_t* next_state,
                                const Shape& shape,
                                const ShardSpec& shard,
                                const TransitionConfig& config,
                                TransitionWorkspace& workspace,
                                TransitionStats& stats) noexcept {
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * shape.dimension + shard.row_offset;
    double sum = 0.0;
    std::uint32_t row = 0;
    for (; row + 8 <= shard.rows; row += 8) {
      const __m256i residual = residual8(current_state + base + row, output + base + row, config);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(workspace.residual.data() + base + row), residual);
      sum += sum_squares8(residual);
    }
    for (; row < shard.rows; ++row) {
      const auto value = residual1(current_state[base + row], output[base + row], config);
      workspace.residual[base + row] = value;
      sum += static_cast<double>(value) * value;
    }
    const double rms = std::sqrt(sum / static_cast<double>(shard.rows) + config.epsilon);
    apply_rms_range(workspace.residual.data(), next_state, base, shard.rows,
                    config.target_rms / rms, stats);
  }
}

void transition_global_rms_prepare(const std::int8_t* current_state,
                                   const std::int32_t* output,
                                   const Shape& shape,
                                   const ShardSpec& shard,
                                   const TransitionConfig& config,
                                   TransitionWorkspace& workspace) noexcept {
  PartialSums& partial = workspace.partial_sums[shard.worker_index];
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * shape.dimension + shard.row_offset;
    double sum = 0.0;
    std::uint32_t row = 0;
    for (; row + 8 <= shard.rows; row += 8) {
      const __m256i residual = residual8(current_state + base + row, output + base + row, config);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(workspace.residual.data() + base + row), residual);
      sum += sum_squares8(residual);
    }
    for (; row < shard.rows; ++row) {
      const auto value = residual1(current_state[base + row], output[base + row], config);
      workspace.residual[base + row] = value;
      sum += static_cast<double>(value) * value;
    }
    partial.values[slot] = sum;
  }
}

void transition_global_rms_reduce(std::uint32_t workers,
                                  const Shape& shape,
                                  const TransitionConfig& config,
                                  TransitionWorkspace& workspace) noexcept {
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    double sum = 0.0;
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
      sum += workspace.partial_sums[worker].values[slot];
    }
    const double rms = std::sqrt(sum / static_cast<double>(shape.dimension) + config.epsilon);
    workspace.inverse_rms[slot] = config.target_rms / rms;
  }
}

void transition_global_rms_apply(std::int8_t* next_state,
                                 const Shape& shape,
                                 const ShardSpec& shard,
                                 const TransitionConfig&,
                                 const TransitionWorkspace& workspace,
                                 TransitionStats& stats) noexcept {
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * shape.dimension + shard.row_offset;
    apply_rms_range(workspace.residual.data(), next_state, base, shard.rows,
                    workspace.inverse_rms[slot], stats);
  }
}

std::uint64_t checksum_i8(const std::int8_t* data, std::size_t count) noexcept {
  return fnv1a64(data, count * sizeof(std::int8_t));
}
std::uint64_t checksum_i32(const std::int32_t* data, std::size_t count) noexcept {
  return fnv1a64(data, count * sizeof(std::int32_t));
}
}  // namespace cnrl

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "cnrl/aligned_buffer.hpp"
#include "cnrl/types.hpp"
namespace cnrl {
struct alignas(64) PartialSums {
  double values[32]{};
};
struct TransitionWorkspace {
  AlignedBuffer<std::int32_t> residual;
  std::vector<PartialSums> partial_sums;
  AlignedBuffer<double> inverse_rms;
};
struct TransitionStats {
  std::uint64_t clipped_cells = 0;
  std::uint64_t cells = 0;
};
void validate_transition_config(const Shape& shape,
                                const TransitionConfig& config);
void prepare_transition_workspace(TransitionWorkspace& workspace,
                                  std::uint32_t workers,
                                  const Shape& shape);
void transition_fixed_point_local(const std::int8_t* current_state,
                                  const std::int32_t* output,
                                  std::int8_t* next_state,
                                  const Shape& shape,
                                  const ShardSpec& shard,
                                  const TransitionConfig& config,
                                  TransitionStats& stats) noexcept;
void transition_group_rms_local(const std::int8_t* current_state,
                                const std::int32_t* output,
                                std::int8_t* next_state,
                                const Shape& shape,
                                const ShardSpec& shard,
                                const TransitionConfig& config,
                                TransitionWorkspace& workspace,
                                TransitionStats& stats) noexcept;
void transition_global_rms_prepare(const std::int8_t* current_state,
                                   const std::int32_t* output,
                                   const Shape& shape,
                                   const ShardSpec& shard,
                                   const TransitionConfig& config,
                                   TransitionWorkspace& workspace) noexcept;
void transition_global_rms_reduce(std::uint32_t workers,
                                  const Shape& shape,
                                  const TransitionConfig& config,
                                  TransitionWorkspace& workspace) noexcept;
void transition_global_rms_apply(std::int8_t* next_state,
                                 const Shape& shape,
                                 const ShardSpec& shard,
                                 const TransitionConfig& config,
                                 const TransitionWorkspace& workspace,
                                 TransitionStats& stats) noexcept;
[[nodiscard]] std::uint64_t checksum_i8(const std::int8_t* data,
                                        std::size_t count) noexcept;
[[nodiscard]] std::uint64_t checksum_i32(const std::int32_t* data,
                                         std::size_t count) noexcept;
}  // namespace cnrl

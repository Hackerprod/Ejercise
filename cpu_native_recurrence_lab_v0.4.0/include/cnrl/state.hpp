#pragma once

#include <cstdint>

#include "cnrl/aligned_buffer.hpp"
#include "cnrl/types.hpp"

namespace cnrl {

void initialize_state(AlignedBuffer<std::int8_t>& state,
                      const Shape& shape,
                      std::uint32_t seed);

void copy_state_shard(const std::int8_t* source,
                      std::int8_t* destination,
                      const Shape& shape,
                      const ShardSpec& shard) noexcept;

void clear_state_shard(std::int8_t* state,
                       const Shape& shape,
                       const ShardSpec& shard) noexcept;

}  // namespace cnrl

#include "cnrl/state.hpp"

#include <cstring>
#include <stdexcept>

#include "cnrl/checked_math.hpp"
#include "cnrl/random.hpp"

namespace cnrl {

void initialize_state(AlignedBuffer<std::int8_t>& state,
                      const Shape& shape,
                      std::uint32_t seed) {
  if (shape.dimension == 0U || shape.slots == 0U) {
    throw std::invalid_argument("state dimensions must be positive");
  }
  const std::uint64_t cells = checked_mul_u64(shape.dimension, shape.slots);
  state.resize(static_cast<std::size_t>(cells));
  XorShift32 random(seed ^ 0xA5A5F00DU);
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    for (std::uint32_t dimension = 0; dimension < shape.dimension; ++dimension) {
      state[static_cast<std::size_t>(slot) * shape.dimension + dimension] =
          random.symmetric_i8(31);
    }
  }
}

void copy_state_shard(const std::int8_t* source,
                      std::int8_t* destination,
                      const Shape& shape,
                      const ShardSpec& shard) noexcept {
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    const std::size_t offset = static_cast<std::size_t>(slot) * shape.dimension +
                               shard.row_offset;
    std::memcpy(destination + offset, source + offset, shard.rows);
  }
}

void clear_state_shard(std::int8_t* state,
                       const Shape& shape,
                       const ShardSpec& shard) noexcept {
  for (std::uint32_t slot = 0; slot < shape.slots; ++slot) {
    const std::size_t offset = static_cast<std::size_t>(slot) * shape.dimension +
                               shard.row_offset;
    std::memset(state + offset, 0, shard.rows);
  }
}

}  // namespace cnrl

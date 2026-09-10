#pragma once
#include <cstdint>
#include <vector>
#include "cnrl/aligned_buffer.hpp"
#include "cnrl/types.hpp"
namespace cnrl {
struct ShardWeights {
  ShardSpec shard{};
  std::uint32_t block_count = 0;
  std::uint64_t block_bytes = 0;
  std::uint64_t block_stride_bytes = 0;
  AlignedBuffer<std::int8_t> storage;
  std::vector<std::uint64_t> block_hashes;
  [[nodiscard]] const std::int8_t* block(std::uint32_t round) const noexcept;
  [[nodiscard]] std::int8_t* block(std::uint32_t round) noexcept;
};
struct WeightBank {
  Shape shape{};
  WeightVariant variant = WeightVariant::shared;
  std::vector<ShardWeights> shards;
  std::uint64_t base_weight_bytes = 0;
  std::uint64_t allocated_weight_bytes = 0;
  std::uint64_t hash_signature = 0;
  bool clone_hashes_equal = false;
  bool clone_addresses_distinct = false;
};
[[nodiscard]] WeightBank make_weight_bank(const RunConfig& config);
void validate_weight_bank(const WeightBank& bank);
[[nodiscard]] std::uint64_t calculate_logical_weight_load_bytes(
    const RunConfig& config);
}  // namespace cnrl

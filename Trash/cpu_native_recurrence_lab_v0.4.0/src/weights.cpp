#include "cnrl/weights.hpp"
#include <cstring>
#include <stdexcept>
#include "cnrl/checked_math.hpp"
#include "cnrl/hash.hpp"
#include "cnrl/sharding.hpp"

namespace cnrl {
namespace {
std::uint32_t block_index(const ShardWeights& shard, std::uint32_t round) noexcept {
  return shard.block_count == 1 ? 0U : round % shard.block_count;
}
std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

std::int8_t deterministic_weight(std::uint32_t seed, std::uint32_t global_row,
                                 std::uint32_t column, std::uint32_t round_key) noexcept {
  std::uint64_t key = static_cast<std::uint64_t>(seed) << 32U;
  key ^= static_cast<std::uint64_t>(global_row) * 0xD6E8FEB86659FD93ULL;
  key ^= static_cast<std::uint64_t>(column) * 0xA0761D6478BD642FULL;
  key ^= static_cast<std::uint64_t>(round_key) * 0xE7037ED1A0B428DBULL;
  return static_cast<std::int8_t>(static_cast<std::int32_t>(splitmix64(key) % 255ULL) - 127);
}

void fill_block(std::int8_t* destination, const ShardSpec& shard,
                std::uint32_t dimension, std::uint32_t seed,
                std::uint32_t round_key) noexcept {
  for (std::uint32_t local_row = 0; local_row < shard.rows; ++local_row) {
    const std::uint32_t global_row = shard.row_offset + local_row;
    for (std::uint32_t column = 0; column < dimension; ++column) {
      destination[static_cast<std::size_t>(local_row) * dimension + column] =
          deterministic_weight(seed, global_row, column, round_key);
    }
  }
}
std::uint32_t fused_weight_passes(std::uint32_t slots, std::uint32_t tile) {
  std::uint32_t passes = 0;
  while (slots > 0) {
    if (tile >= 8 && slots >= 8) slots -= 8;
    else if (tile >= 4 && slots >= 4) slots -= 4;
    else if (tile >= 2 && slots >= 2) slots -= 2;
    else --slots;
    ++passes;
  }
  return passes;
}
}  // namespace

const std::int8_t* ShardWeights::block(std::uint32_t round) const noexcept {
  return storage.data() + static_cast<std::size_t>(block_index(*this, round)) *
                              static_cast<std::size_t>(block_stride_bytes);
}
std::int8_t* ShardWeights::block(std::uint32_t round) noexcept {
  return storage.data() + static_cast<std::size_t>(block_index(*this, round)) *
                              static_cast<std::size_t>(block_stride_bytes);
}

WeightBank make_weight_bank(const RunConfig& config) {
  validate_shards(config.shape, config.shards,
                  config.transition.kind != TransitionKind::frozen);
  WeightBank bank;
  bank.shape = config.shape;
  bank.variant = config.variant;
  bank.shards.reserve(config.shards.size());
  const bool many_blocks = config.variant == WeightVariant::clone ||
                           config.variant == WeightVariant::untied;
  const std::uint32_t block_count = many_blocks ? config.shape.depth : 1U;
  bool hash_equal = config.variant == WeightVariant::clone;
  bool addresses_distinct = config.variant == WeightVariant::clone;
  std::uint64_t signature = kFnv1a64Offset;

  for (const auto& spec : config.shards) {
    ShardWeights shard;
    shard.shard = spec;
    shard.block_count = block_count;
    shard.block_bytes = checked_mul_u64(spec.rows, config.shape.dimension);
    shard.block_stride_bytes = checked_add_u64(shard.block_bytes, 63U) &
                               ~std::uint64_t{63U};
    const auto storage_bytes = checked_mul_u64(shard.block_stride_bytes, block_count);
    shard.storage.resize(static_cast<std::size_t>(storage_bytes));
    shard.storage.fill_zero();
    shard.block_hashes.resize(block_count);
    fill_block(shard.block(0), spec, config.shape.dimension, config.seed, 0U);
    for (std::uint32_t round = 1; round < block_count; ++round) {
      if (config.variant == WeightVariant::clone) {
        std::memcpy(shard.block(round), shard.block(0), static_cast<std::size_t>(shard.block_bytes));
      } else {
        fill_block(shard.block(round), spec, config.shape.dimension,
                   config.seed, round);
      }
    }
    for (std::uint32_t round = 0; round < block_count; ++round) {
      const auto hash = fnv1a64(shard.block(round), static_cast<std::size_t>(shard.block_bytes));
      shard.block_hashes[round] = hash;
      if (round == 0) signature = fnv1a64_update(signature, shard.block(round), static_cast<std::size_t>(shard.block_bytes));
      if (config.variant == WeightVariant::clone && round > 0) {
        hash_equal = hash_equal && hash == shard.block_hashes[0];
        addresses_distinct = addresses_distinct && shard.block(round) != shard.block(0);
      }
    }
    bank.base_weight_bytes = checked_add_u64(bank.base_weight_bytes, shard.block_bytes);
    bank.allocated_weight_bytes = checked_add_u64(bank.allocated_weight_bytes, storage_bytes);
    bank.shards.push_back(std::move(shard));
  }
  bank.hash_signature = signature;
  bank.clone_hashes_equal = hash_equal;
  bank.clone_addresses_distinct = addresses_distinct;
  validate_weight_bank(bank);
  return bank;
}

void validate_weight_bank(const WeightBank& bank) {
  if (bank.shards.empty()) throw std::invalid_argument("weight bank has no shards");
  for (const auto& shard : bank.shards) {
    if (shard.block_count == 0 || shard.block_bytes == 0 || shard.block_stride_bytes < shard.block_bytes) {
      throw std::invalid_argument("invalid weight block layout");
    }
    if (shard.storage.bytes() != shard.block_stride_bytes * shard.block_count) {
      throw std::invalid_argument("weight storage size mismatch");
    }
    if (shard.block_hashes.size() != shard.block_count) {
      throw std::invalid_argument("weight hash count mismatch");
    }
  }
  if (bank.variant == WeightVariant::clone && bank.shape.depth > 1) {
    if (!bank.clone_hashes_equal) throw std::invalid_argument("Bclone bytes are not identical");
    if (!bank.clone_addresses_distinct) throw std::invalid_argument("Bclone addresses are not distinct");
  }
}

std::uint64_t calculate_logical_weight_load_bytes(const RunConfig& config) {
  std::uint64_t base = 0;
  for (const auto& shard : config.shards) {
    base = checked_add_u64(base, checked_mul_u64(shard.rows, config.shape.dimension));
  }
  const std::uint32_t passes = config.kernel == KernelKind::avx2_fused
      ? fused_weight_passes(config.shape.slots, config.slot_tile)
      : config.shape.slots;
  std::uint64_t total = checked_mul_u64(base, passes);
  total = checked_mul_u64(total, config.shape.depth);
  total = checked_mul_u64(total, config.sequences_per_repetition);
  total = checked_mul_u64(total, config.timed_repetitions);
  return total;
}
}  // namespace cnrl

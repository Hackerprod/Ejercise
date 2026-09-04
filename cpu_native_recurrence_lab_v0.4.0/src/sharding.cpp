#include "cnrl/sharding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cnrl {
namespace {
struct Fraction {
  std::size_t index = 0;
  double value = 0.0;
};

std::vector<std::uint32_t> distribute_units(std::uint32_t units,
                                            const std::vector<double>& rates) {
  if (units < rates.size()) {
    throw std::invalid_argument("not enough aligned row units for non-empty shards");
  }
  const double rate_sum = std::accumulate(rates.begin(), rates.end(), 0.0);
  if (!(rate_sum > 0.0) || !std::isfinite(rate_sum)) {
    throw std::invalid_argument("rate sum must be positive and finite");
  }
  std::vector<std::uint32_t> result(rates.size(), 1U);
  const std::uint32_t unassigned = units - static_cast<std::uint32_t>(rates.size());
  std::uint32_t assigned = 0;
  std::vector<Fraction> fractions;
  fractions.reserve(rates.size());
  for (std::size_t index = 0; index < rates.size(); ++index) {
    const double exact = static_cast<double>(unassigned) * rates[index] / rate_sum;
    const auto base = static_cast<std::uint32_t>(std::floor(exact));
    result[index] += base;
    assigned += base;
    fractions.push_back({index, exact - static_cast<double>(base)});
  }
  std::sort(fractions.begin(), fractions.end(), [](const Fraction& left,
                                                   const Fraction& right) {
    if (left.value != right.value) return left.value > right.value;
    return left.index < right.index;
  });
  for (std::uint32_t extra = assigned; extra < unassigned; ++extra) {
    result[fractions[extra - assigned].index] += 1U;
  }
  return result;
}
}  // namespace

std::vector<std::uint32_t> proportional_rows(std::uint32_t total_rows,
                                             const std::vector<double>& rates,
                                             std::uint32_t alignment) {
  if (rates.empty()) throw std::invalid_argument("rates must not be empty");
  if (alignment == 0) throw std::invalid_argument("row alignment must be positive");
  for (double rate : rates) {
    if (!(rate > 0.0) || !std::isfinite(rate)) {
      throw std::invalid_argument("rates must be positive and finite");
    }
  }
  if (alignment > 1U && total_rows % alignment != 0U) {
    throw std::invalid_argument(
        "total rows must be divisible by row alignment; use alignment 1 for exact heterogeneous sharding");
  }
  const std::uint32_t units = total_rows / alignment;
  auto unit_rows = distribute_units(units, rates);
  for (auto& value : unit_rows) value *= alignment;
  return unit_rows;
}

std::vector<ShardSpec> make_shards(const std::vector<std::uint32_t>& cpus,
                                   const std::vector<std::uint32_t>& rows) {
  if (cpus.empty() || cpus.size() != rows.size()) {
    throw std::invalid_argument("CPU/row count mismatch");
  }
  std::vector<ShardSpec> result;
  result.reserve(rows.size());
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    if (rows[index] == 0) throw std::invalid_argument("zero-row shard");
    if (offset > UINT32_MAX || rows[index] > UINT32_MAX - offset) {
      throw std::overflow_error("shard row offsets exceed uint32");
    }
    result.push_back({static_cast<std::uint32_t>(index), cpus[index],
                      static_cast<std::uint32_t>(offset), rows[index]});
    offset += rows[index];
  }
  return result;
}

std::uint32_t total_output_rows(const std::vector<ShardSpec>& shards) {
  std::uint64_t total = 0;
  for (const auto& shard : shards) total += shard.rows;
  if (total > UINT32_MAX) throw std::overflow_error("total rows exceed uint32");
  return static_cast<std::uint32_t>(total);
}

void validate_shards(const Shape& shape, const std::vector<ShardSpec>& shards,
                     bool require_square_output) {
  if (shards.empty()) throw std::invalid_argument("at least one shard is required");
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < shards.size(); ++index) {
    const auto& shard = shards[index];
    if (shard.worker_index != index || shard.rows == 0 ||
        shard.row_offset != offset) {
      throw std::invalid_argument(
          "shards must be dense, non-empty, and contiguous");
    }
    offset += shard.rows;
  }
  if (require_square_output && offset != shape.dimension) {
    throw std::invalid_argument("real recurrence requires sum(rows)==D");
  }
}
}  // namespace cnrl

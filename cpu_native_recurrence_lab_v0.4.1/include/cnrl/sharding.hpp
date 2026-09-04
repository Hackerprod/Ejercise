#pragma once
#include <cstdint>
#include <vector>
#include "cnrl/types.hpp"
namespace cnrl {
[[nodiscard]] std::vector<std::uint32_t> proportional_rows(
    std::uint32_t total_rows, const std::vector<double>& rates,
    std::uint32_t alignment = 1);
[[nodiscard]] std::vector<ShardSpec> make_shards(
    const std::vector<std::uint32_t>& logical_cpus,
    const std::vector<std::uint32_t>& rows);
void validate_shards(const Shape& shape, const std::vector<ShardSpec>& shards,
                     bool require_square_output);
[[nodiscard]] std::uint32_t total_output_rows(const std::vector<ShardSpec>& shards);
}  // namespace cnrl

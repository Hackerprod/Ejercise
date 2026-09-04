#pragma once
#include "cnrl/types.hpp"
namespace cnrl {
void validate_run_config(const RunConfig& config);
[[nodiscard]] RunResult run_benchmark(const RunConfig& config);
}  // namespace cnrl

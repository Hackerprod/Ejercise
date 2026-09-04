#pragma once
#include <iosfwd>
#include <string>
#include "cnrl/types.hpp"
namespace cnrl {
[[nodiscard]] std::string shard_rows_string(const RunConfig& config);
[[nodiscard]] std::string shard_cpus_string(const RunConfig& config);
void write_run_csv_header(std::ostream& output);
void write_run_csv_row(std::ostream& output, const RunConfig& config,
                       const RunResult& result);
}  // namespace cnrl

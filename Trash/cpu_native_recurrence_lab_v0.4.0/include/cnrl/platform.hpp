#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cnrl {

struct CpuFeatures {
  bool avx2 = false;
  bool fma = false;
  bool rdtscp = false;
};

struct LogicalProcessor {
  std::uint32_t global_index = 0;
  std::uint16_t group = 0;
  std::uint16_t group_index = 0;
};

struct PhysicalCore {
  std::uint32_t core_index = 0;
  std::uint8_t efficiency_class = 0;
  std::vector<LogicalProcessor> logical_processors;
};

struct CpuTopology {
  std::vector<LogicalProcessor> logical_processors;
  std::vector<PhysicalCore> physical_cores;
};

[[nodiscard]] CpuFeatures detect_cpu_features() noexcept;
[[nodiscard]] CpuTopology discover_cpu_topology();
[[nodiscard]] std::vector<std::uint32_t> choose_one_logical_per_physical_core(
    const CpuTopology& topology);
[[nodiscard]] const LogicalProcessor& find_logical_processor(
    const CpuTopology& topology, std::uint32_t global_index);
[[nodiscard]] std::uint32_t find_physical_core_index(
    const CpuTopology& topology, std::uint32_t global_logical_index);
[[nodiscard]] std::string topology_as_json(const CpuTopology& topology);
[[nodiscard]] std::string topology_as_csv(const CpuTopology& topology);

class AffinityGuard {
 public:
  explicit AffinityGuard(const LogicalProcessor& processor) noexcept;
  ~AffinityGuard();
  AffinityGuard(const AffinityGuard&) = delete;
  AffinityGuard& operator=(const AffinityGuard&) = delete;
  [[nodiscard]] bool succeeded() const noexcept { return succeeded_; }
  [[nodiscard]] std::uint32_t error() const noexcept { return error_; }
 private:
  struct State;
  State* state_ = nullptr;
  bool succeeded_ = false;
  std::uint32_t error_ = 0;
};

class MonotonicClock {
 public:
  MonotonicClock();
  [[nodiscard]] std::uint64_t now() const noexcept;
  [[nodiscard]] double seconds_between(std::uint64_t begin,
                                       std::uint64_t end) const noexcept;
 private:
  std::uint64_t frequency_ = 1;
};

[[nodiscard]] std::uint64_t read_tsc() noexcept;
void flush_cache_range(const void* data, std::size_t bytes) noexcept;

}  // namespace cnrl

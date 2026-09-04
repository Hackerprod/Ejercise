#include "cnrl/platform.hpp"

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <map>
#include <new>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <immintrin.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <chrono>
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>
#endif

namespace cnrl {

CpuFeatures detect_cpu_features() noexcept {
  CpuFeatures result{};
#if defined(_MSC_VER)
  int regs[4]{};
  __cpuid(regs, 0);
  const int max_basic = regs[0];
  if (max_basic >= 1) {
    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    result.fma = (regs[2] & (1 << 12)) != 0;
    if (osxsave && avx && (_xgetbv(0) & 0x6U) == 0x6U && max_basic >= 7) {
      __cpuidex(regs, 7, 0);
      result.avx2 = (regs[1] & (1 << 5)) != 0;
    }
  }
  __cpuid(regs, static_cast<int>(0x80000000U));
  if (static_cast<std::uint32_t>(regs[0]) >= 0x80000001U) {
    __cpuid(regs, static_cast<int>(0x80000001U));
    result.rdtscp = (regs[3] & (1 << 27)) != 0;
  }
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_cpu_init();
  result.avx2 = __builtin_cpu_supports("avx2");
  result.fma = __builtin_cpu_supports("fma");
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid_max(0x80000000U, nullptr) >= 0x80000001U &&
      __get_cpuid(0x80000001U, &eax, &ebx, &ecx, &edx)) {
    result.rdtscp = (edx & (1U << 27U)) != 0;
  }
#endif
  return result;
}

namespace {
#if !defined(_WIN32)
int read_int_file(const std::string& path, int fallback) {
  std::ifstream input(path);
  int value = fallback;
  if (input) input >> value;
  return value;
}
#endif
}  // namespace

#if defined(_WIN32)

struct AffinityGuard::State {
  GROUP_AFFINITY previous{};
  bool saved = false;
  bool active = false;
};

CpuTopology discover_cpu_topology() {
  CpuTopology topology;
  std::map<std::pair<WORD, BYTE>, std::uint32_t> globals;
  std::uint32_t global = 0;
  const WORD group_count = GetActiveProcessorGroupCount();
  for (WORD group = 0; group < group_count; ++group) {
    const DWORD count = GetActiveProcessorCount(group);
    for (DWORD index = 0; index < count; ++index) {
      LogicalProcessor logical{global, group, static_cast<std::uint16_t>(index)};
      topology.logical_processors.push_back(logical);
      globals[{group, static_cast<BYTE>(index)}] = global++;
    }
  }

  DWORD bytes = 0;
  (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (bytes == 0) throw std::runtime_error("GetLogicalProcessorInformationEx returned no bytes");
  std::vector<std::uint8_t> storage(bytes);
  auto* first = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &bytes)) {
    throw std::runtime_error("GetLogicalProcessorInformationEx failed: " +
                             std::to_string(GetLastError()));
  }

  std::size_t offset = 0;
  while (offset < bytes) {
    auto* record = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data() + offset);
    if (record->Size == 0) throw std::runtime_error("zero-sized topology record");
    if (record->Relationship == RelationProcessorCore) {
      PhysicalCore core;
      core.core_index = static_cast<std::uint32_t>(topology.physical_cores.size());
      core.efficiency_class = record->Processor.EfficiencyClass;
      for (WORD gi = 0; gi < record->Processor.GroupCount; ++gi) {
        const GROUP_AFFINITY& group_mask = record->Processor.GroupMask[gi];
        for (BYTE bit = 0; bit < sizeof(KAFFINITY) * 8U; ++bit) {
          if ((group_mask.Mask & (static_cast<KAFFINITY>(1) << bit)) == 0) continue;
          const auto found = globals.find({group_mask.Group, bit});
          if (found != globals.end()) {
            core.logical_processors.push_back(
                {found->second, group_mask.Group, static_cast<std::uint16_t>(bit)});
          }
        }
      }
      std::sort(core.logical_processors.begin(), core.logical_processors.end(),
                [](const auto& a, const auto& b) { return a.global_index < b.global_index; });
      topology.physical_cores.push_back(std::move(core));
    }
    offset += record->Size;
  }
  return topology;
}

AffinityGuard::AffinityGuard(const LogicalProcessor& processor) noexcept {
  state_ = new (std::nothrow) State();
  if (state_ == nullptr) { error_ = ERROR_NOT_ENOUGH_MEMORY; return; }
  if (!GetThreadGroupAffinity(GetCurrentThread(), &state_->previous)) {
    error_ = GetLastError(); return;
  }
  state_->saved = true;
  GROUP_AFFINITY requested{};
  requested.Group = processor.group;
  requested.Mask = static_cast<KAFFINITY>(1) << processor.group_index;
  if (!SetThreadGroupAffinity(GetCurrentThread(), &requested, nullptr)) {
    error_ = GetLastError(); return;
  }
  state_->active = true;
  succeeded_ = true;
}

AffinityGuard::~AffinityGuard() {
  if (state_ != nullptr) {
    if (state_->active && state_->saved) {
      (void)SetThreadGroupAffinity(GetCurrentThread(), &state_->previous, nullptr);
    }
    delete state_;
  }
}

MonotonicClock::MonotonicClock() {
  LARGE_INTEGER frequency{};
  if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
    throw std::runtime_error("QueryPerformanceFrequency failed");
  }
  frequency_ = static_cast<std::uint64_t>(frequency.QuadPart);
}
std::uint64_t MonotonicClock::now() const noexcept {
  LARGE_INTEGER value{};
  QueryPerformanceCounter(&value);
  return static_cast<std::uint64_t>(value.QuadPart);
}
std::uint64_t read_tsc() noexcept { unsigned aux = 0; return __rdtscp(&aux); }

#else

struct AffinityGuard::State {
  cpu_set_t previous{};
  bool saved = false;
  bool active = false;
};

CpuTopology discover_cpu_topology() {
  CpuTopology topology;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  std::map<std::pair<int, int>, std::size_t> core_map;
  std::uint32_t global = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;
    LogicalProcessor logical{global++, 0, static_cast<std::uint16_t>(cpu)};
    topology.logical_processors.push_back(logical);
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const int package = read_int_file(base + "physical_package_id", 0);
    const int core_id = read_int_file(base + "core_id", cpu);
    const auto key = std::make_pair(package, core_id);
    auto found = core_map.find(key);
    if (found == core_map.end()) {
      const std::size_t index = topology.physical_cores.size();
      core_map[key] = index;
      topology.physical_cores.push_back({static_cast<std::uint32_t>(index), 0, {}});
      found = core_map.find(key);
    }
    topology.physical_cores[found->second].logical_processors.push_back(logical);
  }
  return topology;
}

AffinityGuard::AffinityGuard(const LogicalProcessor& processor) noexcept {
  state_ = new (std::nothrow) State();
  if (state_ == nullptr) { error_ = ENOMEM; return; }
  int status = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &state_->previous);
  if (status != 0) { error_ = static_cast<std::uint32_t>(status); return; }
  state_->saved = true;
  cpu_set_t requested;
  CPU_ZERO(&requested);
  CPU_SET(processor.group_index, &requested);
  status = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &requested);
  if (status != 0) { error_ = static_cast<std::uint32_t>(status); return; }
  state_->active = true;
  succeeded_ = true;
}

AffinityGuard::~AffinityGuard() {
  if (state_ != nullptr) {
    if (state_->active && state_->saved) {
      (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &state_->previous);
    }
    delete state_;
  }
}

MonotonicClock::MonotonicClock() : frequency_(1'000'000'000ULL) {}
std::uint64_t MonotonicClock::now() const noexcept {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::uint64_t read_tsc() noexcept { unsigned aux = 0; return __rdtscp(&aux); }

#endif

double MonotonicClock::seconds_between(std::uint64_t begin,
                                       std::uint64_t end) const noexcept {
  return static_cast<double>(end - begin) / static_cast<double>(frequency_);
}

void flush_cache_range(const void* data, std::size_t bytes) noexcept {
  const auto* source = static_cast<const char*>(data);
  for (std::size_t offset = 0; offset < bytes; offset += 64) _mm_clflush(source + offset);
  _mm_mfence();
}

std::vector<std::uint32_t> choose_one_logical_per_physical_core(const CpuTopology& topology) {
  std::vector<std::uint32_t> result;
  for (const auto& core : topology.physical_cores) {
    if (!core.logical_processors.empty()) result.push_back(core.logical_processors.front().global_index);
  }
  return result;
}

const LogicalProcessor& find_logical_processor(const CpuTopology& topology,
                                               std::uint32_t global_index) {
  for (const auto& processor : topology.logical_processors) {
    if (processor.global_index == global_index) return processor;
  }
  throw std::out_of_range("logical CPU index not found");
}

std::uint32_t find_physical_core_index(
    const CpuTopology& topology, std::uint32_t global_logical_index) {
  for (const auto& core : topology.physical_cores) {
    for (const auto& processor : core.logical_processors) {
      if (processor.global_index == global_logical_index) return core.core_index;
    }
  }
  throw std::out_of_range("logical CPU does not belong to a discovered physical core");
}

std::string topology_as_json(const CpuTopology& topology) {
  std::ostringstream out;
  out << "{\n  \"logical_processor_count\": " << topology.logical_processors.size()
      << ",\n  \"physical_core_count\": " << topology.physical_cores.size()
      << ",\n  \"physical_cores\": [\n";
  for (std::size_t i = 0; i < topology.physical_cores.size(); ++i) {
    const auto& core = topology.physical_cores[i];
    out << "    {\"core_index\": " << core.core_index
        << ", \"efficiency_class\": " << static_cast<unsigned>(core.efficiency_class)
        << ", \"logical_cpus\": [";
    for (std::size_t j = 0; j < core.logical_processors.size(); ++j) {
      if (j) out << ", ";
      out << core.logical_processors[j].global_index;
    }
    out << "]}" << (i + 1 == topology.physical_cores.size() ? "" : ",") << "\n";
  }
  out << "  ]\n}\n";
  return out.str();
}

std::string topology_as_csv(const CpuTopology& topology) {
  std::ostringstream out;
  out << "core_index,efficiency_class,global_logical_cpu,group,group_index\n";
  for (const auto& core : topology.physical_cores) {
    for (const auto& logical : core.logical_processors) {
      out << core.core_index << ',' << static_cast<unsigned>(core.efficiency_class) << ','
          << logical.global_index << ',' << logical.group << ',' << logical.group_index << '\n';
    }
  }
  return out.str();
}

}  // namespace cnrl

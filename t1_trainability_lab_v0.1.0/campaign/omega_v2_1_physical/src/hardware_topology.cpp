#include "v2_1.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace omega_v2_1 {
namespace {

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) out << "?";
            else out << static_cast<char>(c);
        }
    }
    return out.str();
}

void cpuid(int leaf, int subleaf, int out[4]) {
    __cpuidex(out, leaf, subleaf);
}

std::string cpu_brand_string() {
    int r[4]{};
    cpuid(0x80000000, 0, r);
    if (static_cast<unsigned>(r[0]) < 0x80000004u) return "UNKNOWN";
    char brand[49]{};
    for (int leaf = 0; leaf < 3; ++leaf) {
        cpuid(0x80000002 + leaf, 0, r);
        std::memcpy(brand + leaf * 16, r, 16);
    }
    std::string result(brand);
    while (!result.empty() && result.front() == ' ') result.erase(result.begin());
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

bool collect_cpu_sets(std::vector<CpuSetRecord>& sets, std::string& error) {
    ULONG bytes = 0;
    GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    if (bytes == 0) {
        error = "GetSystemCpuSetInformation size query returned zero";
        return false;
    }
    std::vector<std::uint8_t> buffer(bytes);
    if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), bytes, &bytes, GetCurrentProcess(), 0)) {
        error = "GetSystemCpuSetInformation failed, win32=" + std::to_string(GetLastError());
        return false;
    }
    std::size_t offset = 0;
    while (offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= bytes) {
        const auto* record = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (record->Size == 0 || offset + record->Size > bytes) {
            error = "malformed CPU-set record size";
            return false;
        }
        if (record->Type == CpuSetInformation) {
            CpuSetRecord row;
            row.id = record->CpuSet.Id;
            row.group = record->CpuSet.Group;
            row.logical_index = record->CpuSet.LogicalProcessorIndex;
            row.core_index = record->CpuSet.CoreIndex;
            row.last_level_cache_index = record->CpuSet.LastLevelCacheIndex;
            row.numa_node_index = record->CpuSet.NumaNodeIndex;
            row.efficiency_class = record->CpuSet.EfficiencyClass;
            row.parked = record->CpuSet.Parked != 0;
            row.allocated = record->CpuSet.Allocated != 0;
            row.realtime = record->CpuSet.RealTime != 0;
            sets.push_back(row);
        }
        offset += record->Size;
    }
    if (sets.empty()) {
        error = "no SYSTEM_CPU_SET_INFORMATION CpuSet records returned";
        return false;
    }
    return true;
}

bool collect_processor_cores(const std::vector<CpuSetRecord>& sets, std::vector<CoreRecord>& cores, std::string& error) {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    if (bytes == 0) {
        error = "GetLogicalProcessorInformationEx(RelationProcessorCore) size query returned zero";
        return false;
    }
    std::vector<std::uint8_t> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
        error = "GetLogicalProcessorInformationEx(RelationProcessorCore) failed, win32=" + std::to_string(GetLastError());
        return false;
    }
    std::size_t offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bytes) {
        const auto* item = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (item->Size == 0 || offset + item->Size > bytes) {
            error = "malformed processor-core relationship size";
            return false;
        }
        if (item->Relationship == RelationProcessorCore) {
            const auto& processor = item->Processor;
            for (WORD group_slot = 0; group_slot < processor.GroupCount; ++group_slot) {
                const GROUP_AFFINITY& mask = processor.GroupMask[group_slot];
                CoreRecord core;
                core.group = mask.Group;
                core.core_index = processor.GroupCount == 1 ? static_cast<BYTE>(cores.size()) : static_cast<BYTE>(group_slot);
                core.efficiency_class = processor.EfficiencyClass;
                for (const CpuSetRecord& set : sets) {
                    if (set.group == mask.Group && set.core_index == core.core_index) {
                        const KAFFINITY bit = static_cast<KAFFINITY>(1) << set.logical_index;
                        if (mask.Mask & bit) core.cpu_sets.push_back(set);
                    }
                }
                if (core.cpu_sets.empty()) {
                    for (const CpuSetRecord& set : sets) {
                        if (set.group == mask.Group && (mask.Mask & (static_cast<KAFFINITY>(1) << set.logical_index))) {
                            core.cpu_sets.push_back(set);
                            core.core_index = set.core_index;
                            core.efficiency_class = set.efficiency_class;
                        }
                    }
                }
                if (core.cpu_sets.empty()) {
                    error = "processor-core relation could not be mapped to a CPU-set ID";
                    return false;
                }
                std::sort(core.cpu_sets.begin(), core.cpu_sets.end(), [](const auto& a, const auto& b) { return a.logical_index < b.logical_index; });
                core.efficiency_class = core.cpu_sets.front().efficiency_class;
                cores.push_back(std::move(core));
            }
        }
        offset += item->Size;
    }
    if (cores.empty()) {
        error = "no physical processor cores enumerated";
        return false;
    }
    return true;
}

bool collect_caches(HardwareInfo& info, std::string& error) {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes);
    if (bytes == 0) {
        error = "GetLogicalProcessorInformationEx(RelationCache) size query returned zero";
        return false;
    }
    std::vector<std::uint8_t> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
        error = "GetLogicalProcessorInformationEx(RelationCache) failed, win32=" + std::to_string(GetLastError());
        return false;
    }
    std::size_t offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bytes) {
        const auto* item = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (item->Size == 0 || offset + item->Size > bytes) {
            error = "malformed cache relationship size";
            return false;
        }
        if (item->Relationship == RelationCache) {
            const auto& c = item->Cache;
            CacheRecord row;
            row.level = c.Level;
            row.type = static_cast<BYTE>(c.Type);
            row.size_bytes = c.CacheSize;
            row.line_bytes = c.LineSize;
            row.associativity = c.Associativity;
            row.group_masks.reserve(c.GroupCount);
            for (WORD i = 0; i < c.GroupCount; ++i) row.group_masks.emplace_back(c.GroupMasks[i].Group, c.GroupMasks[i].Mask);
            info.caches.push_back(row);
        }
        offset += item->Size;
    }
    for (const CacheRecord& cache : info.caches) {
        if (cache.line_bytes > info.cache_line_bytes) info.cache_line_bytes = cache.line_bytes;
        if (cache.level == 1 && cache.type == CacheData) info.l1d_bytes_total += cache.size_bytes;
        if (cache.level == 1 && cache.type == CacheInstruction) info.l1i_bytes_total += cache.size_bytes;
        if (cache.level == 2) info.l2_bytes_total += cache.size_bytes;
        if (cache.level >= 3) info.llc_bytes = (std::max)(info.llc_bytes, static_cast<std::uint64_t>(cache.size_bytes));
    }
    return !info.caches.empty();
}

void collect_isa(HardwareInfo& info) {
    int r[4]{};
    __cpuidex(r, 1, 0);
    const unsigned eax = static_cast<unsigned>(r[0]);
    const unsigned ecx = static_cast<unsigned>(r[2]);
    const unsigned edx = static_cast<unsigned>(r[3]);
    const unsigned base_family = (eax >> 8) & 0xfu;
    const unsigned ext_family = (eax >> 20) & 0xffu;
    const unsigned base_model = (eax >> 4) & 0xfu;
    const unsigned ext_model = (eax >> 16) & 0xfu;
    info.family = base_family == 0xf ? base_family + ext_family : base_family;
    info.model = (base_family == 0x6 || base_family == 0xf) ? (ext_model << 4) + base_model : base_model;
    info.stepping = eax & 0xfu;
    info.fma = (ecx & (1u << 12)) != 0;
    info.clflush_supported = (edx & (1u << 19)) != 0;
    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx = (ecx & (1u << 28)) != 0;
    const std::uint64_t xcr0 = (osxsave && avx) ? _xgetbv(0) : 0;
    __cpuidex(r, 7, 0);
    const unsigned ebx = static_cast<unsigned>(r[1]);
    info.avx2 = (ebx & (1u << 5)) != 0 && (xcr0 & 0x6u) == 0x6u;
    info.avx512f_os_enabled = (ebx & (1u << 16)) != 0 && (xcr0 & 0xe6u) == 0xe6u;
    __cpuidex(r, 7, 1);
    info.avx_vnni = (static_cast<unsigned>(r[0]) & (1u << 4)) != 0 && (xcr0 & 0x6u) == 0x6u;
}

} // namespace

bool query_hardware(HardwareInfo& info, std::string& error) {
    info.cpu_model = cpu_brand_string();
    collect_isa(info);
    info.processor_group_count = GetActiveProcessorGroupCount();
    info.logical_processor_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) info.ram_bytes = memory.ullTotalPhys;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll) {
        using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
        auto rtl = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"));
        if (rtl) {
            RTL_OSVERSIONINFOW version{};
            version.dwOSVersionInfoSize = sizeof(version);
            if (rtl(&version) == 0) {
                info.windows_version = std::to_string(version.dwMajorVersion) + "." + std::to_string(version.dwMinorVersion);
                info.windows_build = version.dwBuildNumber;
            }
        }
    }
    LARGE_INTEGER frequency{};
    if (QueryPerformanceFrequency(&frequency)) info.qpc_frequency = frequency.QuadPart;

    std::vector<CpuSetRecord> cpu_sets;
    if (!collect_cpu_sets(cpu_sets, error)) return false;
    if (!collect_processor_cores(cpu_sets, info.cores, error)) return false;
    if (!classify_intel_hybrid_cores(info, error)) return false;
    info.physical_core_count = static_cast<DWORD>(info.cores.size());
    if (!collect_caches(info, error)) return false;
    if (info.logical_processor_count == 0 || info.processor_group_count == 0 || info.llc_bytes == 0 || info.cache_line_bytes == 0) {
        error = "hardware preflight lacks logical processors, groups, LLC, or cache-line size";
        return false;
    }
    return true;
}

bool classify_intel_hybrid_cores(HardwareInfo& info, std::string& error) {
    if (info.cores.empty()) {
        error = "Intel hybrid core classification requires enumerated physical cores";
        return false;
    }
    for (CoreRecord& core : info.cores) {
        if (core.cpu_sets.empty()) {
            error = "Intel hybrid core classification found a physical core without CPU sets";
            return false;
        }
        BYTE core_type = 0;
        std::string probe_error;
        const CpuSetRecord cpu = core.cpu_sets.front();
        std::thread probe([&] {
            std::string affinity_error;
            if (!set_current_cpu_set(cpu.id, cpu.group, cpu.logical_index, affinity_error)) {
                probe_error = "CPUID probe could not pin to CPU-set " + std::to_string(cpu.id) + ": " + affinity_error;
                return;
            }
            PROCESSOR_NUMBER current{};
            GetCurrentProcessorNumberEx(&current);
            if (current.Group != cpu.group || current.Number != cpu.logical_index) {
                probe_error = "CPUID probe did not land on its requested Windows CPU-set";
                return;
            }
            int registers[4]{};
            __cpuidex(registers, 0, 0);
            if (static_cast<unsigned>(registers[0]) < 0x1au) {
                probe_error = "CPUID leaf 0x1A is unavailable; cannot classify Intel P/E cores without timing inference";
                return;
            }
            __cpuidex(registers, 0x1a, 0);
            core_type = static_cast<BYTE>(static_cast<unsigned>(registers[0]) >> 24);
        });
        probe.join();
        if (!probe_error.empty()) {
            error = probe_error;
            return false;
        }
        if (core_type != 0x40 && core_type != 0x20) {
            error = "CPUID leaf 0x1A returned unsupported Intel core type " + std::to_string(core_type)
                + " on CPU-set " + std::to_string(cpu.id);
            return false;
        }
        core.intel_core_type = core_type;
        core.classified_p_core = core_type == 0x40;
    }
    info.p_core_count = static_cast<DWORD>(std::count_if(info.cores.begin(), info.cores.end(), [](const CoreRecord& row) { return row.classified_p_core; }));
    info.e_core_count = static_cast<DWORD>(info.cores.size()) - info.p_core_count;
    if (info.p_core_count < 4 || info.e_core_count == 0) {
        error = "CPUID leaf 0x1A did not identify the required Intel hybrid topology with at least four P-cores and one E-core";
        return false;
    }
    return true;
}

std::string hardware_json(const HardwareInfo& info) {
    std::ostringstream out;
    out << "{\"cpu_model\":\"" << json_escape(info.cpu_model) << "\",\"family\":" << info.family
        << ",\"model\":" << info.model << ",\"stepping\":" << info.stepping
        << ",\"windows_version\":\"" << json_escape(info.windows_version) << "\",\"windows_build\":" << info.windows_build
        << ",\"physical_core_count\":" << info.physical_core_count << ",\"logical_processor_count\":" << info.logical_processor_count
        << ",\"processor_group_count\":" << info.processor_group_count << ",\"p_core_count\":" << info.p_core_count
        << ",\"e_core_count\":" << info.e_core_count << ",\"cache_line_bytes\":" << info.cache_line_bytes
        << ",\"coordinator_cpu_set_id\":" << info.coordinator_cpu_set_id << ",\"coordinator_group\":" << info.coordinator_group
        << ",\"coordinator_logical_index\":" << static_cast<unsigned>(info.coordinator_logical_index)
        << ",\"l1d_bytes_total\":" << info.l1d_bytes_total << ",\"l1i_bytes_total\":" << info.l1i_bytes_total
        << ",\"l2_bytes_total\":" << info.l2_bytes_total << ",\"llc_bytes\":" << info.llc_bytes
        << ",\"ram_bytes\":" << info.ram_bytes << ",\"ram_channels\":\"NOT_DETECTED\",\"ram_effective_speed\":\"NOT_DETECTED\""
        << ",\"qpc_overhead_ns\":" << info.qpc_overhead_ns << ",\"isa\":{\"avx2\":" << (info.avx2 ? "true" : "false")
        << ",\"fma\":" << (info.fma ? "true" : "false") << ",\"avx_vnni\":" << (info.avx_vnni ? "true" : "false")
        << ",\"avx512f_os_enabled\":" << (info.avx512f_os_enabled ? "true" : "false")
        << ",\"clflush_supported\":" << (info.clflush_supported ? "true" : "false") << "},\"qpc\":{\"frequency\":" << info.qpc_frequency
        << ",\"monotonic\":" << (info.qpc_monotonic ? "true" : "false") << "},\"cores\":[";
    for (std::size_t i = 0; i < info.cores.size(); ++i) {
        if (i) out << ',';
        const auto& core = info.cores[i];
        out << "{\"group\":" << core.group << ",\"core_index\":" << static_cast<unsigned>(core.core_index)
            << ",\"efficiency_class\":" << static_cast<unsigned>(core.efficiency_class)
            << ",\"intel_cpuid_core_type\":" << static_cast<unsigned>(core.intel_core_type)
            << ",\"v_i\":" << core.v_i
            << ",\"classified_p_core\":" << (core.classified_p_core ? "true" : "false") << ",\"cpu_sets\":[";
        for (std::size_t j = 0; j < core.cpu_sets.size(); ++j) {
            if (j) out << ',';
            const auto& set = core.cpu_sets[j];
            out << "{\"id\":" << set.id << ",\"group\":" << set.group << ",\"logical_index\":" << static_cast<unsigned>(set.logical_index)
                << ",\"core_index\":" << static_cast<unsigned>(set.core_index) << ",\"efficiency_class\":" << static_cast<unsigned>(set.efficiency_class)
                << ",\"shard_weight_vi\":" << set.shard_weight << "}";
        }
        out << "]}";
    }
    out << "],\"caches\":[";
    for (std::size_t i = 0; i < info.caches.size(); ++i) {
        if (i) out << ',';
        const auto& cache = info.caches[i];
        out << "{\"level\":" << static_cast<unsigned>(cache.level) << ",\"type\":" << static_cast<unsigned>(cache.type)
            << ",\"size_bytes\":" << cache.size_bytes << ",\"line_bytes\":" << cache.line_bytes
            << ",\"associativity\":" << static_cast<unsigned>(cache.associativity) << ",\"group_masks\":[";
        for (std::size_t j = 0; j < cache.group_masks.size(); ++j) {
            if (j) out << ',';
            out << "{\"group\":" << cache.group_masks[j].first << ",\"mask\":" << cache.group_masks[j].second << "}";
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

std::vector<H0Row> measure_h0_cores(const HardwareInfo& hardware, const CoreWeights& d512_weights, std::string& error) {
    std::vector<H0Row> rows;
    if (d512_weights.d != kD512) {
        error = "H0 requires the d512 Q4 reference microkernel";
        return rows;
    }
    constexpr int warmup_count = 10;
    constexpr int sample_count = 31;
    for (const CoreRecord& core : hardware.cores) {
        if (!core.classified_p_core) continue;
        if (core.cpu_sets.empty()) continue;
        const CpuSetRecord cpu = core.cpu_sets.front();
        WorkerPool pool({cpu});
        if (!pool.ready()) {
            error = "H0 could not start a CPU-set-pinned persistent single-core worker";
            if (!pool.affinity_results().empty()) error += ": " + pool.affinity_results().front();
            return {};
        }
        for (int m : {1, 4, 8, 16}) {
            std::vector<float> input(static_cast<std::size_t>(m) * kD512);
            std::vector<float> output(static_cast<std::size_t>(m) * kD512);
            for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>(i % 61) - 30) * 0.015625f;
            for (int warmup = 0; warmup < warmup_count; ++warmup) q4_linear(d512_weights.W_Q, input.data(), output.data(), m, pool);
            std::vector<double> mac_rates;
            mac_rates.reserve(sample_count);
            const std::uint64_t macs = static_cast<std::uint64_t>(m) * kD512 * kD512;
            for (int sample = 0; sample < sample_count; ++sample) {
                const std::int64_t start = qpc_ticks();
                q4_linear(d512_weights.W_Q, input.data(), output.data(), m, pool);
                const std::int64_t stop = qpc_ticks();
                const double seconds = static_cast<double>(stop - start) / static_cast<double>(qpc_frequency());
                if (!pool.affinity_intact()) {
                    error = "H0 worker lost its fixed single-CPU-set affinity during measurement";
                    return {};
                }
                if (stop <= start || seconds <= 0.0) {
                    error = "H0 QPC sample was non-positive";
                    return {};
                }
                mac_rates.push_back(static_cast<double>(macs) / seconds);
            }
            std::sort(mac_rates.begin(), mac_rates.end());
            H0Row row;
            row.group = core.group;
            row.core_index = core.core_index;
            row.efficiency_class = core.efficiency_class;
            row.intel_core_type = core.intel_core_type;
            row.logical_index = cpu.logical_index;
            row.cpu_set_id = cpu.id;
            row.m = m;
            row.median_macs_per_second = mac_rates[mac_rates.size() / 2];
            row.sample_macs_per_second = std::move(mac_rates);
            row.warmup_count = warmup_count;
            row.measured_repetitions = sample_count;
            row.affinity_ok = pool.affinity_intact();
            rows.push_back(row);
        }
    }
    if (rows.size() != hardware.p_core_count * 4) {
        error = "H0 did not produce all four m points for each CPUID-identified P-core candidate";
        return {};
    }
    return rows;
}

std::vector<CoreRecord> select_p_cores_by_h0(HardwareInfo& hardware, const std::vector<H0Row>& h0,
                                             std::string& p_class_method, std::string& error) {
    for (CoreRecord& core : hardware.cores) {
        if (!core.classified_p_core || core.intel_core_type != 0x40) continue;
        auto median_for_m = [&](int m) -> double {
            auto found = std::find_if(h0.begin(), h0.end(), [&](const H0Row& row) {
                return row.group == core.group && row.core_index == core.core_index && row.m == m;
            });
            return found == h0.end() ? 0.0 : found->median_macs_per_second;
        };
        const double v4 = median_for_m(4);
        const double v8 = median_for_m(8);
        const double v16 = median_for_m(16);
        if (!(v4 > 0.0 && v8 > 0.0 && v16 > 0.0)) {
            error = "H0 is missing positive m4/m8/m16 medians for a physical core";
            return {};
        }
        core.v_i = std::cbrt(v4 * v8 * v16);
    }
    if (hardware.p_core_count < 4) {
        error = "fewer than four CPUID-identified Intel Core (P-core) candidates";
        return {};
    }
    p_class_method = "Intel CPUID leaf 0x1A core type 0x40 identifies P-cores; H0 v_i ranks only those candidates; Windows EfficiencyClass is recorded";
    std::vector<CoreRecord> p_cores;
    for (CoreRecord& core : hardware.cores) {
        if (core.classified_p_core) p_cores.push_back(core);
    }
    hardware.p_core_count = static_cast<DWORD>(p_cores.size());
    hardware.e_core_count = static_cast<DWORD>(hardware.cores.size()) - hardware.p_core_count;
    if (p_cores.size() < 4) {
        error = "fewer than four P-core candidates after H0 EfficiencyClass classification";
        return {};
    }
    for (CoreRecord& core : p_cores) {
        std::sort(core.cpu_sets.begin(), core.cpu_sets.end(), [](const CpuSetRecord& a, const CpuSetRecord& b) {
            return a.id < b.id;
        });
    }
    std::sort(p_cores.begin(), p_cores.end(), [](const CoreRecord& a, const CoreRecord& b) {
        if (a.v_i != b.v_i) return a.v_i > b.v_i;
        const DWORD a_id = a.cpu_sets.empty() ? MAXDWORD : a.cpu_sets.front().id;
        const DWORD b_id = b.cpu_sets.empty() ? MAXDWORD : b.cpu_sets.front().id;
        return a_id < b_id;
    });
    p_cores.resize(4);
    for (CoreRecord& core : p_cores) {
        CpuSetRecord selected = core.cpu_sets.front();
        selected.shard_weight = core.v_i;
        core.cpu_sets.assign(1, selected); // one hardware thread; SMT sibling excluded
    }
    return p_cores;
}

} // namespace omega_v2_1

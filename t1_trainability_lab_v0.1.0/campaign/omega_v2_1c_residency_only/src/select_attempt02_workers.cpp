#include "v2_1.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace omega_v2_1 {
namespace {

std::vector<DWORD> parse_ids(const std::wstring& text) {
    std::vector<DWORD> values;
    std::wstringstream input(text);
    std::wstring item;
    while (std::getline(input,item,L',')) if (!item.empty()) values.push_back(static_cast<DWORD>(std::stoul(item)));
    return values;
}

std::vector<double> parse_weights(const std::wstring& text) {
    std::vector<double> values;
    std::wstringstream input(text);
    std::wstring item;
    while (std::getline(input,item,L',')) if (!item.empty()) values.push_back(std::stod(item));
    return values;
}

std::wstring environment(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768,L'\0');
    const DWORD count=GetEnvironmentVariableW(name,buffer.data(),static_cast<DWORD>(buffer.size()));
    if(count==0||count>=buffer.size())return{};
    return std::wstring(buffer.data(),count);
}

} // namespace

std::vector<CoreRecord> select_p_cores_by_h0_attempt02(HardwareInfo& hardware,
                                                       const std::vector<H0Row>& h0,
                                                       std::string& p_class_method,
                                                       std::string& error) {
    (void)h0; // attempt_02 H0 v_i and worker IDs are frozen by MD/310 for this replication.
    const auto ids=parse_ids(environment(L"OMEGA_V2_1C_ATTEMPT02_CPU_SET_IDS"));
    const auto weights=parse_weights(environment(L"OMEGA_V2_1C_ATTEMPT02_V_I"));
    if(ids!=std::vector<DWORD>{266,264,258,270}||weights.size()!=4) {
        error="V2-1c selection is missing the immutable attempt_02 CPU-set IDs/v_i";
        return{};
    }
    std::vector<CoreRecord> selected;
    selected.reserve(4);
    for(std::size_t index=0;index<ids.size();++index) {
        if(!(weights[index]>0.0)||!std::isfinite(weights[index])) {
            error="V2-1c attempt_02 v_i must be finite and positive";
            return{};
        }
        auto core=std::find_if(hardware.cores.begin(),hardware.cores.end(),[&](const CoreRecord& candidate){
            return candidate.intel_core_type==0x40&&std::any_of(candidate.cpu_sets.begin(),candidate.cpu_sets.end(),[&](const CpuSetRecord& cpu){return cpu.id==ids[index];});
        });
        if(core==hardware.cores.end()) {
            error="V2-1c frozen attempt_02 CPU-set is not a measured physical P-core: "+std::to_string(ids[index]);
            return{};
        }
        auto cpu=std::find_if(core->cpu_sets.begin(),core->cpu_sets.end(),[&](const CpuSetRecord& item){return item.id==ids[index];});
        cpu->shard_weight=weights[index];
        core->v_i=weights[index];
        CoreRecord worker=*core;
        CpuSetRecord selected_cpu=*cpu;
        worker.cpu_sets.assign(1,selected_cpu);
        worker.v_i=weights[index];
        selected.push_back(std::move(worker));
    }
    p_class_method="frozen attempt_02 CPU-set IDs and v_i from immutable benchmark_config.json; no V2-1c reselection";
    return selected;
}

} // namespace omega_v2_1

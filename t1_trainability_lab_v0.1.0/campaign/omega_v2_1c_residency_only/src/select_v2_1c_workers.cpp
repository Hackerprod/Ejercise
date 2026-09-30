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

std::vector<H0Row> measure_h0_cores_v2_1c_frozen(const HardwareInfo& hardware,const CoreWeights& weights,std::string& error) {
    (void)hardware;
    (void)weights;
    error.clear();
    // H0 v_i is measured exactly once by the companion's --core-selection-preflight mode.
    // The 72-cell --run path consumes the sealed V2-1c selection; it does not retime/reselect.
    return {};
}

std::vector<CoreRecord> select_p_cores_by_h0_v2_1c_frozen(HardwareInfo& hardware,const std::vector<H0Row>& h0,
                                                          std::string& method,std::string& error) {
    (void)h0;
    const std::vector<DWORD> ids=parse_ids(environment(L"OMEGA_V2_1C_SELECTED_CPU_SET_IDS"));
    const std::vector<double> weights=parse_weights(environment(L"OMEGA_V2_1C_SELECTED_V_I"));
    if(ids.size()!=4||weights.size()!=4){error="V2-1c frozen core-selection manifest did not provide four CPU-set IDs/v_i";return{};}
    std::vector<CoreRecord> selected;selected.reserve(4);
    for(std::size_t i=0;i<4;++i){
        if(!(weights[i]>0.0)||!std::isfinite(weights[i])){error="V2-1c frozen v_i must be finite and positive";return{};}
        auto core=std::find_if(hardware.cores.begin(),hardware.cores.end(),[&](const CoreRecord& candidate){return candidate.intel_core_type==0x40&&std::any_of(candidate.cpu_sets.begin(),candidate.cpu_sets.end(),[&](const CpuSetRecord& cpu){return cpu.id==ids[i];});});
        if(core==hardware.cores.end()){error="V2-1c selected CPU-set is not a physical P-core: "+std::to_string(ids[i]);return{};}
        auto cpu=std::find_if(core->cpu_sets.begin(),core->cpu_sets.end(),[&](const CpuSetRecord& item){return item.id==ids[i];});
        cpu->shard_weight=weights[i];core->v_i=weights[i];
        CoreRecord worker=*core;CpuSetRecord chosen=*cpu;worker.cpu_sets.assign(1,chosen);worker.v_i=weights[i];selected.push_back(std::move(worker));
    }
    method="frozen V2-1c CORE_SELECTION_PREFLIGHT v_i; no H0 remeasurement/reselection inside the 72-cell sweep";
    return selected;
}

} // namespace omega_v2_1

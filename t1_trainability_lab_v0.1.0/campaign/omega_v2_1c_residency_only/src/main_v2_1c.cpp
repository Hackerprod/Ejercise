#include "v2_1.hpp"

#define main omega_v2_1c_original_physical_main
#include "../../omega_v2_1_physical/src/main.cpp"
#undef main
#undef measure_h0_cores
#undef select_p_cores_by_h0

#define main omega_v2_1c_candidate02_equivalence_main
#include "equivalence_candidate02.cpp"
#undef main

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace omega_v2_1 {
std::vector<H0Row> measure_h0_cores(const HardwareInfo& hardware,const CoreWeights& d512_weights,std::string& error);
std::vector<CoreRecord> select_p_cores_by_h0(HardwareInfo& hardware,const std::vector<H0Row>& h0,
                                              std::string& p_classification_method,std::string& error);
}

namespace {

void write_selection_json(const std::filesystem::path& path,const std::string& json) {
    std::ofstream output(path,std::ios::binary|std::ios::trunc);
    if(!output)throw std::runtime_error("cannot create V2-1c core-selection preflight report");
    output<<json<<'\n';
    if(!output)throw std::runtime_error("failed writing V2-1c core-selection preflight report");
}

std::string h0_rows_json(const std::vector<omega_v2_1::H0Row>& rows) {
    std::ostringstream out;out.precision(17);out<<'[';
    for(std::size_t index=0;index<rows.size();++index){if(index)out<<',';const auto& row=rows[index];
        out<<"{\"cpu_set_id\":"<<row.cpu_set_id<<",\"logical_processor_id\":"<<row.logical_index<<",\"physical_core_id\":"<<static_cast<unsigned>(row.core_index)<<",\"group\":"<<row.group<<",\"m\":"<<row.m<<",\"warmups\":"<<row.warmup_count<<",\"samples\":"<<row.measured_repetitions<<",\"median_macs_per_second\":"<<row.median_macs_per_second<<",\"affinity_ok\":"<<(row.affinity_ok?"true":"false")<<",\"sample_macs_per_second\":[";
        for(std::size_t sample=0;sample<row.sample_macs_per_second.size();++sample){if(sample)out<<',';out<<row.sample_macs_per_second[sample];}
        out<<"]}";
    }
    out<<']';return out.str();
}

std::string ranking_json(const omega_v2_1::HardwareInfo& hardware) {
    std::vector<omega_v2_1::CoreRecord> candidates;
    for(const auto& core:hardware.cores)if(core.intel_core_type==0x40&&core.classified_p_core&&!core.cpu_sets.empty())candidates.push_back(core);
    const auto lowest_cpu_set_id=[](const omega_v2_1::CoreRecord& core){return std::min_element(core.cpu_sets.begin(),core.cpu_sets.end(),[](const auto& left,const auto& right){return left.id<right.id;})->id;};
    std::sort(candidates.begin(),candidates.end(),[&](const auto& left,const auto& right){if(left.v_i!=right.v_i)return left.v_i>right.v_i;return lowest_cpu_set_id(left)<lowest_cpu_set_id(right);});
    std::ostringstream out;out.precision(17);out<<'[';
    for(std::size_t i=0;i<candidates.size();++i){if(i)out<<',';const auto& core=candidates[i];const auto cpu=std::min_element(core.cpu_sets.begin(),core.cpu_sets.end(),[](const auto&a,const auto&b){return a.id<b.id;});out<<"{\"rank\":"<<(i+1)<<",\"cpu_set_id\":"<<cpu->id<<",\"physical_core_id\":"<<static_cast<unsigned>(core.core_index)<<",\"logical_processor_id\":"<<static_cast<unsigned>(cpu->logical_index)<<",\"v_i\":"<<core.v_i<<'}';}
    out<<']';return out.str();
}

std::string workers_json(const std::vector<omega_v2_1::CoreRecord>& workers) {
    std::ostringstream out;out.precision(17);out<<'[';
    for(std::size_t i=0;i<workers.size();++i){if(i)out<<',';const auto& core=workers[i];const auto& cpu=core.cpu_sets.front();out<<"{\"worker_id\":"<<i<<",\"cpu_set_id\":"<<cpu.id<<",\"group\":"<<cpu.group<<",\"logical_processor_id\":"<<static_cast<unsigned>(cpu.logical_index)<<",\"physical_core_id\":"<<static_cast<unsigned>(core.core_index)<<",\"v_i\":"<<core.v_i<<",\"shard_weight\":"<<cpu.shard_weight<<",\"smt_sibling_worker_selected\":false}";}
    out<<']';return out.str();
}

std::string row_shards_json(const std::vector<omega_v2_1::CoreRecord>& workers) {
    struct Matrix {const char* name;int d;int rows;int cols;};
    std::vector<Matrix> matrices;
    for(int d:{512,640})matrices.insert(matrices.end(),{{"W_Q",d,d,d},{"W_K",d,d,d},{"W_V",d,d,d},{"W_O",d,d,d},{"W_gate",d,4*d,d},{"W_up",d,4*d,d},{"W_down",d,d,4*d}});
    std::vector<omega_v2_1::CpuSetRecord> cpus;for(const auto& worker:workers)cpus.push_back(worker.cpu_sets.front());
    omega_v2_1::WorkerPool pool(cpus);
    std::ostringstream out;out<<'[';bool first=true;
    for(const auto& matrix:matrices){if(!first)out<<',';first=false;const std::size_t tiles=(static_cast<std::size_t>(matrix.rows)+3)/4;const auto ranges=pool.row_shards(tiles);out<<"{\"d\":"<<matrix.d<<",\"matrix\":\""<<matrix.name<<"\",\"rows\":"<<matrix.rows<<",\"cols\":"<<matrix.cols<<",\"row_tile\":4,\"shards\":[";
        for(std::size_t i=0;i<ranges.size();++i){if(i)out<<',';const auto begin=(std::min)(static_cast<std::size_t>(matrix.rows),ranges[i].first*4);const auto end=(std::min)(static_cast<std::size_t>(matrix.rows),ranges[i].second*4);const auto rows=end-begin;const auto packed=rows*static_cast<std::size_t>(matrix.cols)/2;const auto scales=rows*static_cast<std::size_t>(matrix.cols/omega_v2_1::kGroup)*sizeof(std::uint16_t);out<<"{\"worker_id\":"<<i<<",\"cpu_set_id\":"<<workers[i].cpu_sets.front().id<<",\"v_i\":"<<workers[i].v_i<<",\"first_output_row\":"<<begin<<",\"last_output_row_exclusive\":"<<end<<",\"rows\":"<<rows<<",\"packed_weight_bytes\":"<<packed<<",\"scale_bytes\":"<<scales<<'}';}
        out<<"],\"rows_covered\":"<<matrix.rows<<",\"tile_count\":"<<tiles<<'}';
    }
    pool.stop();out<<']';return out.str();
}

int run_core_selection_preflight(const std::filesystem::path& output_path) {
    using namespace omega_v2_1;
    if(std::filesystem::exists(output_path))throw std::runtime_error("core-selection preflight result is immutable and already exists");
    HardwareInfo hardware;std::string error;
    if(!monotonic_qpc_test(error))throw std::runtime_error("inherited QPC monotonicity preflight failed: "+error);
    hardware.qpc_monotonic=true;
    if(!query_hardware(hardware,error))throw std::runtime_error("core-selection topology query failed: "+error);
    hardware.qpc_overhead_ns=measure_qpc_overhead_ns();
    if(!(hardware.qpc_overhead_ns>0.0))throw std::runtime_error("inherited QPC-overhead calibration returned a non-positive result");
    if(hardware.cpu_model.find("i7-13700F")==std::string::npos||!hardware.avx2||!hardware.fma)throw std::runtime_error("core-selection requires the sealed i7-13700F AVX2/FMA host");
    const CpuSetRecord* coordinator=nullptr;
    for(const auto& core:hardware.cores)if(core.intel_core_type==0x20&&core.cpu_sets.size()==1&&(!coordinator||core.cpu_sets.front().id<coordinator->id))coordinator=&core.cpu_sets.front();
    if(!coordinator||!set_current_cpu_set(coordinator->id,coordinator->group,coordinator->logical_index,error))throw std::runtime_error("core-selection coordinator E-core pin failed: "+error);
    hardware.coordinator_cpu_set_id=coordinator->id;hardware.coordinator_group=coordinator->group;hardware.coordinator_logical_index=coordinator->logical_index;
    CoreWeights weights=make_seeded_weights(kD512,kSeed);
    const auto h0=measure_h0_cores(hardware,weights,error);
    if(!error.empty())throw std::runtime_error("candidate_02 H0 v_i measurement failed: "+error);
    std::string classification;
    const auto selected=select_p_cores_by_h0(hardware,h0,classification,error);
    if(!error.empty()||selected.size()!=4)throw std::runtime_error("candidate_02 H0 core selection failed: "+error);
    const std::string topology_method=classification;
    const std::string raw_h0=h0_rows_json(h0);
    const std::string ranking=ranking_json(hardware);
    const std::string workers=workers_json(selected);
    const std::string shards=row_shards_json(selected);
    const bool pcore_pass=std::all_of(selected.begin(),selected.end(),[](const CoreRecord& core){return core.intel_core_type==0x40&&core.cpu_sets.size()==1&&core.cpu_sets.front().shard_weight>0;});
    std::ostringstream out;out.precision(17);
    out<<"{\"schema\":\"omega-v2-1c-core-selection-preflight-v1\",\"status\":\""<<(pcore_pass?"PASS":"FAIL")<<"\",\"cpu_model\":\""<<hardware.cpu_model<<"\",\"qpc_frequency\":"<<qpc_frequency()<<",\"qpc_monotonic\":"<<(hardware.qpc_monotonic?"true":"false")<<",\"qpc_overhead_ns\":"<<hardware.qpc_overhead_ns<<",\"source_weight_seed\":"<<kSeed<<",\"kernel_candidate_id\":\"KQ2_ROW_TILE4_SLOT2_FUSED\",\"selection_method\":\""<<topology_method<<"\",\"v_i_formula\":\"cbrt(m4_median_MACps*m8_median_MACps*m16_median_MACps);m1 excluded\",\"m1_role\":\"measured/report only\",\"h0_measurement\":{\"d\":512,\"matrix\":\"W_Q full output shard\",\"threads_per_core\":1,\"smt_sibling_active\":false,\"warmups\":10,\"samples\":31,\"rows\":"<<raw_h0<<"},\"p_core_ranking\":"<<ranking<<",\"selected_workers\":"<<workers<<",\"worker_row_shards\":"<<shards<<",\"pass\":"<<(pcore_pass?"true":"false")<<'}';
    write_selection_json(output_path,out.str());
    std::cout<<out.str()<<'\n';return pcore_pass?0:1;
}

} // namespace

int main(int argc,char**argv) {
    const std::string mode=argc>1?argv[1]:"";
    if(mode=="--run")return omega_v2_1c_original_physical_main(argc,argv);
    if(mode=="--core-selection-preflight"&&argc==3){try{return run_core_selection_preflight(std::filesystem::absolute(argv[2]));}catch(const std::exception& exception){std::cerr<<"V2-1c core-selection preflight error: "<<exception.what()<<'\n';return 3;}}
    if((mode=="--binding-preflight"||mode=="--correctness-preflight")&&argc==5)return omega_v2_1c_candidate02_equivalence_main(argc,argv);
    std::cerr<<"Usage: omega_v2_1c_bench.exe --run | --binding-preflight <weights> <golden-dir> <report> | --core-selection-preflight <report> | --correctness-preflight <weights> <golden-dir> <report>\n";
    return 2;
}

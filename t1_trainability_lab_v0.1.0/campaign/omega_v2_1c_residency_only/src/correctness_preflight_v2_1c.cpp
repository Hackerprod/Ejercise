#include "kq_candidate2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace omega_v2_1;

std::vector<double> parse_csv(const std::string& text) {
    std::vector<double> values;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input,item,',')) if (!item.empty()) values.push_back(std::stod(item));
    return values;
}

std::string env_utf8(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768,L'\0');
    const DWORD count=GetEnvironmentVariableW(name,buffer.data(),static_cast<DWORD>(buffer.size()));
    if(count==0||count>=buffer.size())return{};
    const int needed=WideCharToMultiByte(CP_UTF8,0,buffer.data(),-1,nullptr,0,nullptr,nullptr);
    if(needed<=1)return{};
    std::string result(static_cast<std::size_t>(needed),'\0');
    WideCharToMultiByte(CP_UTF8,0,buffer.data(),-1,result.data(),needed,nullptr,nullptr);
    result.resize(static_cast<std::size_t>(needed-1));
    return result;
}

std::vector<float> initial_state(int d,int m) {
    std::vector<float> state(static_cast<std::size_t>(d)*m);
    for(std::size_t i=0;i<state.size();++i){const int v=static_cast<int>((i*37+static_cast<std::size_t>(d+m))%127)-63;state[i]=static_cast<float>(v)*(1.0f/256.0f);}
    return state;
}

std::vector<float> toy_state(int d,int m) {
    std::vector<float> state(static_cast<std::size_t>(d)*m);
    for(std::size_t i=0;i<state.size();++i)state[i]=static_cast<float>(static_cast<int>((i*19)%83)-41)*(1.0f/256.0f);
    return state;
}

struct Error {double max_abs=0,max_rel=0;};

Error compare(const std::vector<float>& candidate,const std::vector<float>& oracle) {
    if(candidate.size()!=oracle.size())throw std::runtime_error("correctness output dimensions differ");
    Error result;
    for(std::size_t i=0;i<candidate.size();++i){const double diff=std::fabs(static_cast<double>(candidate[i])-oracle[i]);result.max_abs=(std::max)(result.max_abs,diff);result.max_rel=(std::max)(result.max_rel,diff/(std::max)(std::fabs(static_cast<double>(oracle[i])),1e-12));}
    return result;
}

std::uint64_t checksum(const std::vector<float>& values) {
    std::uint64_t hash=1469598103934665603ull;
    for(float value:values){std::uint32_t bits=0;std::memcpy(&bits,&value,sizeof(bits));for(int byte=0;byte<4;++byte){hash^=static_cast<std::uint8_t>(bits>>(8*byte));hash*=1099511628211ull;}}
    return hash;
}

bool bit_equal(const std::vector<float>& left,const std::vector<float>& right) {
    return left.size()==right.size()&&std::memcmp(left.data(),right.data(),left.size()*sizeof(float))==0;
}

bool same_q4_values(const CoreWeights& left,const CoreWeights& right) {
    const Q4Matrix* a[]={&left.W_Q,&left.W_K,&left.W_V,&left.W_O,&left.W_gate,&left.W_up,&left.W_down};
    const Q4Matrix* b[]={&right.W_Q,&right.W_K,&right.W_V,&right.W_O,&right.W_gate,&right.W_up,&right.W_down};
    for(std::size_t i=0;i<7;++i)if(a[i]->rows!=b[i]->rows||a[i]->cols!=b[i]->cols||std::memcmp(a[i]->packed.data,b[i]->packed.data,a[i]->packed.logical_bytes)!=0||std::memcmp(a[i]->scales_fp16.data,b[i]->scales_fp16.data,a[i]->scales_fp16.logical_bytes)!=0)return false;
    return true;
}

std::uint64_t q4_checksum(const CoreWeights& weights) {
    std::uint64_t hash=1469598103934665603ull;
    const Q4Matrix* matrices[]={&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    for(const Q4Matrix* matrix:matrices){
        for(std::size_t i=0;i<matrix->packed.logical_bytes;++i){hash^=matrix->packed.data[i];hash*=1099511628211ull;}
        for(std::size_t i=0;i<matrix->scales_fp16.logical_bytes;++i){hash^=matrix->scales_fp16.data[i];hash*=1099511628211ull;}
    }
    return hash;
}

bool storage_disjoint(const CoreWeights& base,const std::vector<CoreWeights>& copies) {
    std::vector<const void*> seen;
    auto add=[&](const CoreWeights& weights){const Q4Matrix* m[]={&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};for(const Q4Matrix* matrix:m){if(std::find(seen.begin(),seen.end(),matrix->packed.data)!=seen.end()||std::find(seen.begin(),seen.end(),matrix->scales_fp16.data)!=seen.end())return false;seen.push_back(matrix->packed.data);seen.push_back(matrix->scales_fp16.data);}return true;};
    if(!add(base))return false;
    for(const auto& copy:copies)if(!add(copy))return false;
    return true;
}

std::vector<float> run_variant(const CoreWeights& base,const std::vector<CoreWeights>& copies,const std::vector<float>& input,int d,int m,int K,char variant,WorkerPool& pool) {
    Scratch scratch;scratch.resize_for(d,m);scratch.state=input;
    for(int round=0;round<K;++round){
        const CoreWeights* selected=&base;
        if(variant=='B')selected=&copies[static_cast<std::size_t>(round)];
        v2_full_block_round(*selected,scratch,pool);
    }
    return scratch.state;
}

std::string run_toy_recurrence(WorkerPool& pool) {
    constexpr int d=32,m=4;
    const CoreWeights weights=make_seeded_weights(d,kSeed);
    std::vector<std::string> rows;
    bool all_pass=true;
    for(int K:{1,4,8}){
        Scratch candidate,oracle;candidate.resize_for(d,m);oracle.resize_for(d,m);candidate.state=toy_state(d,m);oracle.state=candidate.state;
        for(int k=0;k<K;++k){v2_full_block_round(weights,candidate,pool);v2_full_block_round_single_thread(weights,oracle);}
        const Error error=compare(candidate.state,oracle.state);
        const bool pass=error.max_abs<=1e-5&&error.max_rel<=1e-4;all_pass=all_pass&&pass;
        std::ostringstream out;out<<std::setprecision(17)<<"{\"d\":32,\"m\":4,\"K\":"<<K<<",\"max_abs_error\":"<<error.max_abs<<",\"max_rel_error\":"<<error.max_rel<<",\"pass\":"<<(pass?"true":"false")<<'}';rows.push_back(out.str());
    }
    std::ostringstream result;result<<"{\"all_pass\":"<<(all_pass?"true":"false")<<",\"cells\":[";for(std::size_t i=0;i<rows.size();++i){if(i)result<<',';result<<rows[i];}result<<"]}";return result.str();
}

} // namespace

int main() {
    using namespace omega_v2_1;
    try {
        HardwareInfo hardware;std::string error;
        if(!query_hardware(hardware,error))throw std::runtime_error("hardware query failed: "+error);
        const auto shard_weights=parse_csv(env_utf8(L"OMEGA_V2_1C_SELECTED_V_I"));
        const auto raw_ids=parse_csv(env_utf8(L"OMEGA_V2_1C_SELECTED_CPU_SET_IDS"));
        if(shard_weights.size()!=4||raw_ids.size()!=4)throw std::runtime_error("frozen V2-1c selected CPU-set IDs/v_i missing");
        std::vector<CpuSetRecord> workers;
        std::vector<DWORD> selected_ids;for(double value:raw_ids)selected_ids.push_back(static_cast<DWORD>(value));
        for(std::size_t index=0;index<4;++index){const CpuSetRecord* selected=nullptr;for(const auto& core:hardware.cores)if(core.intel_core_type==0x40)for(const auto& cpu:core.cpu_sets)if(cpu.id==selected_ids[index])selected=&cpu;if(!selected)throw std::runtime_error("frozen V2-1c selected CPU-set is not a P-core");CpuSetRecord cpu=*selected;cpu.shard_weight=shard_weights[index];workers.push_back(cpu);}
        const CpuSetRecord* coordinator=nullptr;for(const auto& core:hardware.cores)if(core.intel_core_type==0x20&&core.cpu_sets.size()==1&&(!coordinator||core.cpu_sets.front().id<coordinator->id))coordinator=&core.cpu_sets.front();
        if(!coordinator||!set_current_cpu_set(coordinator->id,coordinator->group,coordinator->logical_index,error))throw std::runtime_error("E-core coordinator pin failed: "+error);
        WorkerPool pool(workers);
        std::vector<std::string> scalar_cells,abc_cells;
        bool scalar_all=true,abc_all=true,deterministic_all=true,storage_all=true,values_all=true;
        for(int d:{512,640}){
            const CoreWeights base=make_seeded_weights(d,kSeed);
            for(int m:{1,4,8,16}){
                Scratch candidate,oracle;candidate.resize_for(d,m);oracle.resize_for(d,m);candidate.state=initial_state(d,m);oracle.state=candidate.state;
                v2_full_block_round(base,candidate,pool);v2_full_block_round_single_thread(base,oracle);
                const Error error_values=compare(candidate.state,oracle.state);
                const bool pass=error_values.max_abs<=1e-5&&error_values.max_rel<=1e-4;scalar_all=scalar_all&&pass;
                std::ostringstream row;row<<std::setprecision(17)<<"{\"d\":"<<d<<",\"m\":"<<m<<",\"K\":1,\"max_abs_error\":"<<error_values.max_abs<<",\"max_rel_error\":"<<error_values.max_rel<<",\"pass\":"<<(pass?"true":"false")<<'}';scalar_cells.push_back(row.str());
            }
            for(int m:{1,4,8,16})for(int K:{1,4,8}){
                const std::vector<float> input=initial_state(d,m);
                std::vector<CoreWeights> copies;copies.reserve(static_cast<std::size_t>(K));for(int round=0;round<K;++round)copies.push_back(clone_weights(base));
                const bool values_equal=std::all_of(copies.begin(),copies.end(),[&](const CoreWeights&copy){return same_q4_values(base,copy);});
                const bool disjoint=storage_disjoint(base,copies);values_all=values_all&&values_equal;storage_all=storage_all&&disjoint;
                std::vector<float> a1=run_variant(base,copies,input,d,m,K,'A',pool),b1=run_variant(base,copies,input,d,m,K,'B',pool),c1=run_variant(base,copies,input,d,m,K,'C',pool);
                std::vector<float> a2=run_variant(base,copies,input,d,m,K,'A',pool),b2=run_variant(base,copies,input,d,m,K,'B',pool),c2=run_variant(base,copies,input,d,m,K,'C',pool);
                const bool repeats=bit_equal(a1,a2)&&bit_equal(b1,b2)&&bit_equal(c1,c2);
                const bool variants=bit_equal(a1,b1)&&bit_equal(a1,c1);
                deterministic_all=deterministic_all&&repeats;abc_all=abc_all&&variants;
                std::ostringstream row;row<<"{\"d\":"<<d<<",\"m\":"<<m<<",\"K\":"<<K<<",\"input_checksum\":"<<checksum(input)<<",\"q4_weights_checksum\":"<<q4_checksum(base)<<",\"A_B_C_bit_exact\":"<<(variants?"true":"false")<<",\"repeat_bit_exact\":"<<(repeats?"true":"false")<<",\"B_values_equal_A\":"<<(values_equal?"true":"false")<<",\"B_storage_disjoint\":"<<(disjoint?"true":"false")<<",\"checksums\":{"<<"\"A\":"<<checksum(a1)<<",\"B\":"<<checksum(b1)<<",\"C\":"<<checksum(c1)<<"}}";abc_cells.push_back(row.str());
            }
        }
        const std::string toy=run_toy_recurrence(pool);
        const bool affinity=pool.affinity_intact();pool.stop();
        const bool pass=scalar_all&&abc_all&&deterministic_all&&storage_all&&values_all&&affinity&&toy.find("\"all_pass\":true")!=std::string::npos;
        std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\"omega-v2-1c-correctness-preflight-v1\",\"d512_d640_full_size_scalar_K1\":[";
        for(std::size_t i=0;i<scalar_cells.size();++i){if(i)out<<',';out<<scalar_cells[i];}
        out<<"],\"full_sweep_cell_abc_and_determinism\":[";for(std::size_t i=0;i<abc_cells.size();++i){if(i)out<<',';out<<abc_cells[i];}
        out<<"],\"selected_cpu_set_ids\":[";for(std::size_t i=0;i<selected_ids.size();++i){if(i)out<<',';out<<selected_ids[i];}
        out<<"],\"selected_v_i\":[";for(std::size_t i=0;i<shard_weights.size();++i){if(i)out<<',';out<<shard_weights[i];}
        out<<"],\"toy_recurrence_d32_m4\":"<<toy<<",\"all_d512_d640_scalar_pass\":"<<(scalar_all?"true":"false")<<",\"all_72_cell_abc_bit_exact\":"<<(abc_all?"true":"false")<<",\"all_72_cell_repeats_bit_exact\":"<<(deterministic_all?"true":"false")<<",\"all_B_values_equal_A\":"<<(values_all?"true":"false")<<",\"all_B_storage_disjoint\":"<<(storage_all?"true":"false")<<",\"worker_affinity_ok\":"<<(affinity?"true":"false")<<",\"pass\":"<<(pass?"true":"false")<<'}';
        std::cout<<out.str()<<'\n';
        return pass?0:1;
    }catch(const std::exception&exception){std::cerr<<"V2-1c correctness preflight failed: "<<exception.what()<<'\n';return 3;}
}

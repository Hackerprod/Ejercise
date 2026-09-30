#include "kq_candidate2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace omega_v2_1;
constexpr int kD = 512;
constexpr DWORD kCpuSetIds[4] = {266,264,258,270};

std::vector<double> parse_csv(const std::string& text) {
    std::vector<double> values;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input,item,',')) if (!item.empty()) values.push_back(std::stod(item));
    return values;
}

std::string env_utf8(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768,L'\0');
    const DWORD count = GetEnvironmentVariableW(name,buffer.data(),static_cast<DWORD>(buffer.size()));
    if (count == 0 || count >= buffer.size()) return {};
    const int needed = WideCharToMultiByte(CP_UTF8,0,buffer.data(),-1,nullptr,0,nullptr,nullptr);
    if (needed <= 1) return {};
    std::string result(static_cast<std::size_t>(needed),'\0');
    WideCharToMultiByte(CP_UTF8,0,buffer.data(),-1,result.data(),needed,nullptr,nullptr);
    result.resize(static_cast<std::size_t>(needed-1));
    return result;
}

CoreWeights load_v2_0_q4_weights(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("V2-0 FP32 pre-Q4 equivalence weights missing");
    constexpr std::array<std::pair<int,int>,7> shapes = {{{kD,kD},{kD,kD},{kD,kD},{kD,kD},{4*kD,kD},{4*kD,kD},{kD,4*kD}}};
    CoreWeights weights; weights.d=kD;
    Q4Matrix* matrices[] = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    for (std::size_t index=0; index<shapes.size(); ++index) {
        const auto [rows,cols] = shapes[index];
        std::vector<float> values(static_cast<std::size_t>(rows)*cols);
        input.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(float)));
        if (!input) throw std::runtime_error("truncated V2-0 FP32 equivalence weight stream");
        matrices[index]->pack(values,rows,cols);
        weights.logical_weight_bytes += matrices[index]->packed.logical_bytes;
        weights.logical_scale_bytes += matrices[index]->scales_fp16.logical_bytes;
        weights.alignment_padding_bytes += matrices[index]->alignment_padding_bytes;
        weights.physical_buffer_bytes += matrices[index]->packed.allocated_bytes + matrices[index]->scales_fp16.allocated_bytes;
    }
    if (input.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("V2-0 equivalence stream has trailing bytes");
    return weights;
}

std::vector<float> fixed_state(int m) {
    std::vector<float> state(static_cast<std::size_t>(m)*kD);
    for (std::size_t i=0;i<state.size();++i) {
        const int value=static_cast<int>((i*37+static_cast<std::size_t>(kD+m))%127)-63;
        state[i]=static_cast<float>(value)*(1.0f/256.0f);
    }
    return state;
}

std::uint64_t checksum(const std::vector<float>& values) {
    std::uint64_t hash=1469598103934665603ull;
    for (float value:values) {
        std::uint32_t bits=0; std::memcpy(&bits,&value,sizeof(bits));
        for (int byte=0;byte<4;++byte) { hash^=static_cast<std::uint8_t>(bits>>(8*byte)); hash*=1099511628211ull; }
    }
    return hash;
}

struct Error { double max_abs=0,max_rel=0; };

Error compare_scalar(const std::vector<float>& candidate,const std::vector<float>& reference) {
    Error error;
    if (candidate.size()!=reference.size()) throw std::runtime_error("equivalence output vector sizes differ");
    for (std::size_t i=0;i<candidate.size();++i) {
        const double delta=std::fabs(static_cast<double>(candidate[i])-reference[i]);
        error.max_abs=(std::max)(error.max_abs,delta);
        error.max_rel=(std::max)(error.max_rel,delta/(std::max)(std::fabs(static_cast<double>(reference[i])),1e-12));
    }
    return error;
}

bool bit_equal(const std::vector<float>& left,const std::vector<float>& right) {
    return left.size()==right.size() && std::memcmp(left.data(),right.data(),left.size()*sizeof(float))==0;
}

std::vector<float> run_full(const CoreWeights& weights,int m,int rounds,WorkerPool& pool,bool scalar) {
    Scratch scratch; scratch.resize_for(kD,m); scratch.state=fixed_state(m);
    for (int round=0;round<rounds;++round) {
        if (scalar) v2_full_block_round_single_thread(weights,scratch);
        else v2_full_block_round(weights,scratch,pool);
    }
    return scratch.state;
}

std::vector<float> load_floats(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("sealed candidate_02 KQ output missing: "+path.string());
    input.seekg(0,std::ios::end); const auto bytes=input.tellg(); input.seekg(0,std::ios::beg);
    if (bytes<0 || bytes%static_cast<std::streamoff>(sizeof(float))!=0) throw std::runtime_error("sealed KQ output size is not FP32-aligned");
    std::vector<float> values(static_cast<std::size_t>(bytes)/sizeof(float));
    input.read(reinterpret_cast<char*>(values.data()),bytes);
    if (!input) throw std::runtime_error("failed to read sealed KQ output");
    return values;
}

double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    const auto mid=values.size()/2;
    return (values.size()&1u)?values[mid]:(values[mid-1]+values[mid])*0.5;
}

struct TimedSeries { double median_seconds=0,median_macs_per_second=0; int warmups=10,repetitions=31; std::vector<std::uint64_t> allocations; bool no_timed_allocations=false; };

TimedSeries time_full(const CoreWeights& weights,int m,int rounds,WorkerPool& pool,DWORD line_bytes) {
    const auto initial=fixed_state(m); Scratch scratch; scratch.resize_for(kD,m);
    const std::uint64_t macs=static_cast<std::uint64_t>(rounds)*(16ull*m*kD*kD+2ull*m*m*kD);
    auto reset=[&]{std::copy(initial.begin(),initial.end(),scratch.state.begin());};
    auto compute=[&]{for(int round=0;round<rounds;++round)v2_full_block_round(weights,scratch,pool);};
    for(int warmup=0;warmup<10;++warmup){reset();touch_weights(weights,line_bytes);compute();}
    std::vector<double> samples;samples.reserve(31);TimedSeries series;series.allocations.reserve(31);
    for(int sample=0;sample<31;++sample){reset();touch_weights(weights,line_bytes);begin_timed_allocation_count();const auto start=qpc_ticks();compute();const auto stop=qpc_ticks();const auto allocations=end_timed_allocation_count();if(stop<=start||!pool.affinity_intact())throw std::runtime_error("FULL timing/affinity sanity check failed");samples.push_back(static_cast<double>(stop-start)/qpc_frequency());series.allocations.push_back(allocations);}
    series.median_seconds=median(samples);series.median_macs_per_second=static_cast<double>(macs)/series.median_seconds;series.no_timed_allocations=std::all_of(series.allocations.begin(),series.allocations.end(),[](std::uint64_t value){return value==0;});return series;
}

void write_json(const std::filesystem::path& path,const std::string& text) {
    std::ofstream output(path,std::ios::binary|std::ios::trunc);
    if(!output)throw std::runtime_error("cannot create equivalence report");
    output<<text<<'\n';
    if(!output)throw std::runtime_error("failed writing equivalence report");
}

} // namespace

int main(int argc,char**argv) {
    using namespace omega_v2_1;
    const std::string mode=argc>1?argv[1]:"";
    const bool binding_mode=mode=="--binding-preflight";
    const bool correctness_mode=mode=="--correctness-preflight";
    if(argc!=5||(!binding_mode&&!correctness_mode))return 2;
    try {
        const auto golden_dir=std::filesystem::path(argv[3]);
        const auto output_path=std::filesystem::absolute(argv[4]);
        if(std::filesystem::exists(output_path))throw std::runtime_error("equivalence report already exists");
        HardwareInfo hardware;std::string error;
        if(!query_hardware(hardware,error))throw std::runtime_error("equivalence topology query failed: "+error);
        const auto shard_weights=parse_csv(env_utf8(L"OMEGA_V2_1C_SHARD_WEIGHTS"));
        auto selected_ids_raw=parse_csv(env_utf8(L"OMEGA_V2_1C_WORKER_CPU_SET_IDS"));
        std::vector<DWORD> selected_ids;
        for(double value:selected_ids_raw)selected_ids.push_back(static_cast<DWORD>(value));
        if(selected_ids.empty())selected_ids.assign(std::begin(kCpuSetIds),std::end(kCpuSetIds));
        if(shard_weights.size()!=4)throw std::runtime_error("V2-1c equivalence needs four frozen attempt_02 shard weights");
        if(selected_ids.size()!=4)throw std::runtime_error("V2-1c equivalence needs exactly four selected CPU-set IDs");
        std::vector<CpuSetRecord> workers;
        for(std::size_t index=0;index<4;++index){
            const CpuSetRecord* selected=nullptr;
            for(const CoreRecord& core:hardware.cores)if(core.classified_p_core)for(const CpuSetRecord& cpu:core.cpu_sets)if(cpu.id==selected_ids[index])selected=&cpu;
            if(!selected)throw std::runtime_error("frozen attempt_02 CPU-set missing in equivalence preflight");
            CpuSetRecord cpu=*selected;cpu.shard_weight=shard_weights[index];workers.push_back(cpu);
        }
        const CpuSetRecord* coordinator=nullptr;
        for(const CoreRecord& core:hardware.cores)if(core.intel_core_type==0x20&&core.cpu_sets.size()==1&&(!coordinator||core.cpu_sets.front().id<coordinator->id))coordinator=&core.cpu_sets.front();
        if(!coordinator||!set_current_cpu_set(coordinator->id,coordinator->group,coordinator->logical_index,error))throw std::runtime_error("cannot pin equivalence coordinator to E-core: "+error);
        hardware.coordinator_cpu_set_id=coordinator->id;hardware.coordinator_group=coordinator->group;hardware.coordinator_logical_index=coordinator->logical_index;

        const CoreWeights weights=load_v2_0_q4_weights(argv[2]);
        WorkerPool pool(workers);
        std::vector<std::string> cell_reports;
        bool equivalence_pass=true;
        if(correctness_mode){
            struct Cell {int m;int k;};
            constexpr Cell cells[]={{1,1},{1,4},{4,1},{4,4},{16,1},{16,4}};
            for(const Cell cell:cells){
                const auto first=run_full(weights,cell.m,cell.k,pool,false);
                const auto repeated=run_full(weights,cell.m,cell.k,pool,false);
                const auto scalar=run_full(weights,cell.m,cell.k,pool,true);
                const Error error_values=compare_scalar(first,scalar);
                const bool deterministic=bit_equal(first,repeated);
                const bool scalar_pass=error_values.max_abs<=1e-5&&error_values.max_rel<=1e-4;
                equivalence_pass=equivalence_pass&&deterministic&&scalar_pass;
                std::ostringstream row;row<<std::setprecision(17)<<"{\"d\":512,\"m\":"<<cell.m<<",\"K\":"<<cell.k<<",\"max_abs_error_vs_scalar\":"<<error_values.max_abs<<",\"max_rel_error_vs_scalar\":"<<error_values.max_rel<<",\"abs_tolerance\":1e-5,\"rel_tolerance\":1e-4,\"scalar_oracle_pass\":"<<(scalar_pass?"true":"false")<<",\"deterministic_bit_exact\":"<<(deterministic?"true":"false")<<",\"checksum\":"<<checksum(first)<<'}';
                cell_reports.push_back(row.str());
            }
        }
        const auto sealed_m4=load_floats(golden_dir/"candidate2_full_m4_k1.bin");
        const auto sealed_m16=load_floats(golden_dir/"candidate2_full_m16_k1.bin");
        const auto sealed_m8k4=load_floats(golden_dir/"candidate2_full_m8_k4.bin");
        const bool frozen_m4=bit_equal(run_full(weights,4,1,pool,false),sealed_m4);
        const bool frozen_m16=bit_equal(run_full(weights,16,1,pool,false),sealed_m16);
        const bool frozen_m8k4=bit_equal(run_full(weights,8,4,pool,false),sealed_m8k4);

        std::vector<TimedSeries> h0;TimedSeries f4,f16,f8k4;
        if(binding_mode){
            f4=time_full(weights,4,1,pool,hardware.cache_line_bytes);
            f16=time_full(weights,16,1,pool,hardware.cache_line_bytes);
            f8k4=time_full(weights,8,4,pool,hardware.cache_line_bytes);
        }
        const bool affinity_ok=pool.affinity_intact();pool.stop();

        std::ostringstream out;out.precision(17);
        if(binding_mode){
            const bool no_timed_allocations=f4.no_timed_allocations&&f16.no_timed_allocations&&f8k4.no_timed_allocations;
            out<<"{\"schema\":\"omega-v2-1c-candidate02-binding-preflight-v1\",\"frozen_candidate02_exe_hash\":\"be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f\",\"cpu_model\":\""<<hardware.cpu_model<<"\",\"qpc_frequency\":"<<qpc_frequency()<<",\"worker_cpu_set_ids\":["<<selected_ids[0]<<','<<selected_ids[1]<<','<<selected_ids[2]<<','<<selected_ids[3]<<"],\"worker_affinity_ok\":"<<(affinity_ok?"true":"false")
                <<",\"sealed_output_comparisons\":{\"d512_m4_k1_bit_exact\":"<<(frozen_m4?"true":"false")<<",\"d512_m16_k1_bit_exact\":"<<(frozen_m16?"true":"false")<<",\"d512_m8_k4_bit_exact\":"<<(frozen_m8k4?"true":"false")<<"}"
                <<",\"timing_protocol\":{\"warmups\":10,\"samples\":31,\"outside_timer_weight_touch\":true,\"no_timed_allocations\":"<<(no_timed_allocations?"true":"false")<<"}"
                <<",\"cells\":[{\"cell\":\"d512_m4_K1\",\"median_seconds\":"<<f4.median_seconds<<"},{\"cell\":\"d512_m16_K1\",\"median_seconds\":"<<f16.median_seconds<<"},{\"cell\":\"d512_m8_K4\",\"median_seconds\":"<<f8k4.median_seconds<<"}]"
                <<",\"pass\":"<<((frozen_m4&&frozen_m16&&frozen_m8k4&&affinity_ok&&no_timed_allocations)?"true":"false")<<'}';
        }else{
            out<<"{\"schema\":\"omega-v2-1c-candidate02-correctness-preflight-v1\",\"frozen_candidate02_exe_hash\":\"be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f\",\"cpu_model\":\""<<hardware.cpu_model<<"\",\"qpc_frequency\":"<<qpc_frequency()<<",\"worker_cpu_set_ids\":["<<selected_ids[0]<<','<<selected_ids[1]<<','<<selected_ids[2]<<','<<selected_ids[3]<<"],\"worker_affinity_ok\":"<<(affinity_ok?"true":"false")
                <<",\"kq_sealed_output_comparisons\":{\"d512_m4_k1_bit_exact\":"<<(frozen_m4?"true":"false")<<",\"d512_m16_k1_bit_exact\":"<<(frozen_m16?"true":"false")<<",\"d512_m8_k4_bit_exact\":"<<(frozen_m8k4?"true":"false")<<",\"all_available_sealed_outputs_bit_exact\":"<<((frozen_m4&&frozen_m16&&frozen_m8k4)?"true":"false")<<"}"
                <<",\"unavailable_sealed_outputs\":[\"d512_m1_k1\",\"d512_m1_k4\",\"d512_m4_k4\",\"d512_m16_k4\"]"
                <<",\"six_cell_scalar_equivalence\":[";
            for(std::size_t index=0;index<cell_reports.size();++index){if(index)out<<',';out<<cell_reports[index];}
            out<<"],\"six_cell_scalar_and_determinism_pass\":"<<(equivalence_pass?"true":"false")<<",\"timing_sanity_performed\":false,\"pass\":"<<((equivalence_pass&&frozen_m4&&frozen_m16&&frozen_m8k4&&affinity_ok)?"true":"false")<<'}';
        }
        write_json(output_path,out.str());
        std::cout<<out.str()<<'\n';
        return binding_mode?((frozen_m4&&frozen_m16&&frozen_m8k4&&affinity_ok&&f4.no_timed_allocations&&f16.no_timed_allocations&&f8k4.no_timed_allocations)?0:1):((equivalence_pass&&frozen_m4&&frozen_m16&&frozen_m8k4&&affinity_ok)?0:1);
    } catch(const std::exception& exception) {
        std::cerr<<"V2-1c equivalence failure: "<<exception.what()<<'\n';return 3;
    }
}

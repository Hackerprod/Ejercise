#include "stage_a_execution.hpp"

#include "stage_a_metrics.hpp"
#include "stage_a_trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>

namespace omega_v2_1d {
namespace {
using namespace omega_v2_1;

constexpr std::uint32_t kWeightSeed = 20260929;
constexpr std::uint64_t kHistoricalD640Q4Checksum = 15453147065333665836ull;
constexpr const char* kMatrixNames[] = {"W_Q","W_K","W_V","W_O","W_gate","W_up","W_down"};
constexpr DWORD kWorkerCpuSetIds[] = {266,264,258,270};
constexpr BYTE kWorkerLogicalIds[] = {10,8,2,14};

std::array<std::pair<int,int>,7> matrix_shapes(int d) {
    return {{{d,d},{d,d},{d,d},{d,d},{4*d,d},{4*d,d},{d,4*d}}};
}

std::vector<CpuSetRecord> select_kq_workers() {
    std::vector<CpuSetRecord> workers;
    for(std::size_t i=0;i<std::size(kWorkerCpuSetIds);++i){
        CpuSetRecord cpu{};cpu.id=kWorkerCpuSetIds[i];cpu.group=0;cpu.logical_index=kWorkerLogicalIds[i];
        cpu.core_index=static_cast<BYTE>(i);cpu.efficiency_class=1;cpu.shard_weight=1.0;
        workers.push_back(cpu);
    }
    return workers;
}

bool finite_vector(const std::vector<float>& values) {
    return std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);});
}

bool finite_trace(const Trace& t) {
    return finite_vector(t.normalized)&&finite_vector(t.query)&&finite_vector(t.key)&&finite_vector(t.value)
        &&finite_vector(t.attention_logits)&&finite_vector(t.attention_probabilities)&&finite_vector(t.attention_context)
        &&finite_vector(t.projected)&&finite_vector(t.hidden)&&finite_vector(t.mlp_normalized)&&finite_vector(t.gate)
        &&finite_vector(t.up)&&finite_vector(t.gated)&&finite_vector(t.down)&&finite_vector(t.state);
}

bool trace_shapes(const Trace& t,int d,int m) {
    const std::size_t md=static_cast<std::size_t>(m)*d;
    const std::size_t m4d=static_cast<std::size_t>(m)*4*d;
    return t.normalized.size()==md&&t.query.size()==md&&t.key.size()==md&&t.value.size()==md
        &&t.attention_logits.size()==static_cast<std::size_t>(m)*m&&t.attention_probabilities.size()==static_cast<std::size_t>(m)*m
        &&t.attention_context.size()==md&&t.projected.size()==md&&t.hidden.size()==md&&t.mlp_normalized.size()==md
        &&t.gate.size()==m4d&&t.up.size()==m4d&&t.gated.size()==m4d&&t.down.size()==md&&t.state.size()==md;
}

CoreWeights load_d512_weights(const std::filesystem::path& path, Fp32Weights& original) {
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("d512 FP32 source stream missing");
    CoreWeights weights; weights.d=512; original.d=512;
    const auto shapes=matrix_shapes(512);
    Q4Matrix* matrices[]={&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    for(std::size_t i=0;i<shapes.size();++i){
        const auto [rows,cols]=shapes[i];
        std::vector<float> values(static_cast<std::size_t>(rows)*cols);
        input.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(float)));
        if(!input) throw std::runtime_error("truncated d512 FP32 source stream");
        matrices[i]->pack(values,rows,cols);
        original.matrices[i]=std::move(values);
    }
    if(input.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("d512 FP32 source stream has trailing bytes");
    return weights;
}

Fp32Weights regenerate_fp32_source(int d,std::uint32_t seed) {
    Fp32Weights result; result.d=d;
    std::mt19937 generator(seed);
    const auto shapes=matrix_shapes(d);
    for(std::size_t matrix=0;matrix<shapes.size();++matrix){
        const auto [rows,cols]=shapes[matrix];
        const float bound=std::sqrt(6.0f/static_cast<float>(rows+cols));
        std::uniform_real_distribution<float> distribution(-bound,bound);
        auto& values=result.matrices[matrix];
        values.resize(static_cast<std::size_t>(rows)*cols);
        for(float& value:values) value=distribution(generator);
    }
    return result;
}

CoreWeights pack_fp32(const Fp32Weights& original) {
    CoreWeights weights; weights.d=original.d;
    const auto shapes=matrix_shapes(original.d);
    Q4Matrix* matrices[]={&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    for(std::size_t i=0;i<shapes.size();++i){
        matrices[i]->pack(original.matrices[i],shapes[i].first,shapes[i].second);
        weights.logical_weight_bytes+=matrices[i]->packed.logical_bytes;
        weights.logical_scale_bytes+=matrices[i]->scales_fp16.logical_bytes;
        weights.alignment_padding_bytes+=matrices[i]->alignment_padding_bytes;
        weights.physical_buffer_bytes+=matrices[i]->packed.allocated_bytes+matrices[i]->scales_fp16.allocated_bytes;
    }
    return weights;
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

bool q4_equal(const CoreWeights& a,const CoreWeights& b) {
    const Q4Matrix* left[]={&a.W_Q,&a.W_K,&a.W_V,&a.W_O,&a.W_gate,&a.W_up,&a.W_down};
    const Q4Matrix* right[]={&b.W_Q,&b.W_K,&b.W_V,&b.W_O,&b.W_gate,&b.W_up,&b.W_down};
    for(std::size_t i=0;i<7;++i){
        if(left[i]->rows!=right[i]->rows||left[i]->cols!=right[i]->cols) return false;
        if(std::memcmp(left[i]->packed.data,right[i]->packed.data,left[i]->packed.logical_bytes)!=0) return false;
        if(std::memcmp(left[i]->scales_fp16.data,right[i]->scales_fp16.data,left[i]->scales_fp16.logical_bytes)!=0) return false;
    }
    return true;
}

std::vector<char> canonical_q4_bytes(const CoreWeights& weights) {
    std::ostringstream out(std::ios::out|std::ios::binary);
    const std::string tag="OMEGA-V2-1D-D640-Q4-CANONICAL-V1";
    out.write(tag.data(),static_cast<std::streamsize>(tag.size())); out.put('\0');
    auto u32=[&](std::uint32_t value){for(unsigned shift=0;shift<32;shift+=8)out.put(static_cast<char>((value>>shift)&0xffu));};
    auto u64=[&](std::uint64_t value){for(unsigned shift=0;shift<64;shift+=8)out.put(static_cast<char>((value>>shift)&0xffull));};
    const Q4Matrix* matrices[]={&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    for(std::size_t i=0;i<7;++i){
        const auto& matrix=*matrices[i]; const std::string name=kMatrixNames[i];
        u32(static_cast<std::uint32_t>(name.size())); out.write(name.data(),static_cast<std::streamsize>(name.size()));
        u32(2);u32(static_cast<std::uint32_t>(matrix.rows));u32(static_cast<std::uint32_t>(matrix.cols));
        u64(static_cast<std::uint64_t>(matrix.packed.logical_bytes));
        out.write(reinterpret_cast<const char*>(matrix.packed.data),static_cast<std::streamsize>(matrix.packed.logical_bytes));
        u64(static_cast<std::uint64_t>(matrix.scales_fp16.logical_bytes));
        out.write(reinterpret_cast<const char*>(matrix.scales_fp16.data),static_cast<std::streamsize>(matrix.scales_fp16.logical_bytes));
    }
    const std::string bytes=out.str();
    return {bytes.begin(),bytes.end()};
}

std::vector<float> load_state(const std::filesystem::path& root,int d,int m,const std::string& family) {
    const auto path=root/("d"+std::to_string(d)+"_m"+std::to_string(m)+"_"+family+".f32");
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("pre-generated state bytes missing: "+path.string());
    const std::size_t count=static_cast<std::size_t>(d)*m;
    std::vector<float> state(count);
    input.read(reinterpret_cast<char*>(state.data()),static_cast<std::streamsize>(count*sizeof(float)));
    if(!input||input.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("state input byte count mismatch: "+path.string());
    return state;
}

std::string trace_checkpoints_json(const Trace& c,const Trace& r,int d,int m) {
    struct Field {const char* name; const std::vector<float>* c; const std::vector<float>* r; int rows; int cols; const char* candidate_provenance; const char* scalar_provenance;};
    const Field fields[]={
        {"input_rmsnorm",&c.normalized,&r.normalized,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"Q",&c.query,&r.query,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"K",&c.key,&r.key,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"V",&c.value,&r.value,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"attention_logits",&c.attention_logits,&r.attention_logits,m,m,"RECONSTRUCTED_DIAGNOSTIC","VALIDATED_MIRROR"},
        {"attention_probabilities",&c.attention_probabilities,&r.attention_probabilities,m,m,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"attention_context",&c.attention_context,&r.attention_context,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"output_projection",&c.projected,&r.projected,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"first_residual_hidden",&c.hidden,&r.hidden,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"post_attention_mlp_rmsnorm",&c.mlp_normalized,&r.mlp_normalized,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"gate",&c.gate,&r.gate,m,4*d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"up",&c.up,&r.up,m,4*d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"silu_gate_times_up",&c.gated,&r.gated,m,4*d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
        {"down_projection",&c.down,&r.down,m,d,"FROZEN_DIRECT","VALIDATED_MIRROR"},
    };
    std::ostringstream out;out<<'[';
    for(std::size_t i=0;i<std::size(fields);++i){
        if(i)out<<',';
        if(fields[i].c->size()!=fields[i].r->size()) throw std::runtime_error("checkpoint shape mismatch");
        out<<"{\"name\":\""<<fields[i].name<<"\",\"candidate_provenance\":\""<<fields[i].candidate_provenance
           <<"\",\"scalar_provenance\":\""<<fields[i].scalar_provenance<<"\",\"metrics\":"
           <<metric_json(summarize(*fields[i].c,*fields[i].r),*fields[i].c,*fields[i].r,fields[i].rows,fields[i].cols)<<'}';
    }
    out<<']';return out.str();
}

void write_primary(const std::filesystem::path& root,int d,int m,int K,const std::string& family,int round,
                   const std::vector<float>& candidate,const std::vector<float>& scalar) {
    const auto dir=root/"primary"/("d"+std::to_string(d)+"_m"+std::to_string(m)+"_K"+std::to_string(K)+"_"+family+"_round"+std::to_string(round));
    std::filesystem::create_directories(dir);
    auto write=[&](const std::filesystem::path& path,const std::vector<float>& values){
        std::ofstream out(path,std::ios::binary|std::ios::trunc);
        if(!out)throw std::runtime_error("cannot create primary residual output");
        out.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(float)));
        out.close();if(!out)throw std::runtime_error("failed writing primary residual output");
    };
    write(dir/"frozen_candidate_state.f32",candidate);
    write(dir/"frozen_scalar_state.f32",scalar);
}

std::vector<std::string> persist_d640_regenerated_fp32(const Fp32Weights& weights,const std::filesystem::path& root) {
    const auto directory=root/"representation_sources";
    std::filesystem::create_directories(directory);
    std::vector<std::string> paths;
    for(std::size_t i=0;i<7;++i){
        const auto path=directory/(std::string("d640_")+kMatrixNames[i]+"_regenerated_pre_q4.f32");
        std::ofstream output(path,std::ios::binary|std::ios::trunc);
        if(!output)throw std::runtime_error("cannot persist regenerated d640 FP32 source");
        const auto& values=weights.matrices[i];
        output.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(float)));
        output.close();if(!output)throw std::runtime_error("failed persisting regenerated d640 FP32 source");
        paths.push_back(path.generic_string());
    }
    return paths;
}

std::string run_cell(const CoreWeights& weights,const Fp32Weights& original,int d,int m,int K,
                     const std::string& family,const std::filesystem::path& state_root,
                     const std::filesystem::path& output_root,WorkerPool& pool) {
    const std::vector<float> initial=load_state(state_root,d,m,family);
    Scratch candidate,frozen_scalar,candidate_repeat,scalar_repeat;
    candidate.resize_for(d,m);frozen_scalar.resize_for(d,m);candidate_repeat.resize_for(d,m);scalar_repeat.resize_for(d,m);
    candidate.state=initial;frozen_scalar.state=initial;candidate_repeat.state=initial;scalar_repeat.state=initial;
    std::vector<float> scalar_mirror_state=initial;
    std::vector<float> fp32_representation_state=initial;
    std::vector<std::string> round_rows;
    bool mirror_valid=true,candidate_repeat_ok=true,scalar_repeat_ok=true,finite_ok=true,shapes_ok=true;
    for(int round=1;round<=K;++round){
        const Trace candidate_trace=capture_candidate_round(weights,candidate,pool);
        v2_full_block_round_single_thread(weights,frozen_scalar);
        const Trace scalar_mirror=scalar_q4_round_trace(weights,scalar_mirror_state,m);
        v2_full_block_round(weights,candidate_repeat,pool);
        v2_full_block_round_single_thread(weights,scalar_repeat);
        const bool round_mirror=bitwise_equal(scalar_mirror_state,frozen_scalar.state);
        mirror_valid=mirror_valid&&round_mirror;
        candidate_repeat_ok=candidate_repeat_ok&&bitwise_equal(candidate.state,candidate_repeat.state);
        scalar_repeat_ok=scalar_repeat_ok&&bitwise_equal(frozen_scalar.state,scalar_repeat.state);
        finite_ok=finite_ok&&finite_trace(candidate_trace)&&finite_trace(scalar_mirror)&&finite_vector(frozen_scalar.state);
        shapes_ok=shapes_ok&&trace_shapes(candidate_trace,d,m)&&trace_shapes(scalar_mirror,d,m);
        if(!finite_ok||!shapes_ok) throw std::runtime_error("INSTRUMENTATION_FAILURE: shape or finite check failed");
        if(!candidate_repeat_ok||!scalar_repeat_ok) throw std::runtime_error("INSTRUMENTATION_FAILURE: frozen implementation deterministic repeat failed");
        write_primary(output_root,d,m,K,family,round,candidate.state,frozen_scalar.state);
        const auto primary=summarize(candidate.state,frozen_scalar.state);
        std::ostringstream row;row<<"{\"round\":"<<round<<",\"candidate_primary_provenance\":\"FROZEN_DIRECT\",\"scalar_primary_provenance\":\"FROZEN_DIRECT\",\"primary_state_metrics\":"
            <<metric_json(primary,candidate.state,frozen_scalar.state,m,d)<<",\"scalar_mirror_status\":\""<<(mirror_valid?"VALID":"INVALID")<<"\"";
        if(mirror_valid){
            row<<",\"checkpoint_metrics\":"<<trace_checkpoints_json(candidate_trace,scalar_mirror,d,m);
            const Trace fp32_rep=scalar_fp32_round_trace(original,fp32_representation_state,m);
            const auto rep=summarize(frozen_scalar.state,fp32_rep.state);
            row<<",\"q4_representation_state_metrics\":{\"classification\":\""
               <<(d==512?"DIRECT_CALIBRATION_DIAGNOSTIC":"RECONSTRUCTED_CALIBRATION_DIAGNOSTIC")
               <<"\",\"comparison_method\":\"SCALAR_Q4_VS_SCALAR_FP32\",\"metrics\":"
               <<metric_json(rep,frozen_scalar.state,fp32_rep.state,m,d)<<'}';
        }else{
            row<<",\"checkpoint_metrics\":null,\"checkpoint_attribution\":\"NOT_USABLE\",\"checkpoint_provenance\":\"NOT_CAPTURED\",\"q4_representation_state_metrics\":null";
        }
        row<<'}';round_rows.push_back(row.str());
    }
    std::ostringstream out;
    out<<"{\"d\":"<<d<<",\"m\":"<<m<<",\"K\":"<<K<<",\"state_family\":\""<<family
       <<"\",\"functional_checks\":{\"round_count_exact\":true,\"shapes_correct\":"<<(shapes_ok?"true":"false")
       <<",\"all_finite\":"<<(finite_ok?"true":"false")<<",\"candidate_deterministic_repeat\":"<<(candidate_repeat_ok?"true":"false")
       <<",\"scalar_deterministic_repeat\":"<<(scalar_repeat_ok?"true":"false")<<",\"scalar_mirror\":\""<<(mirror_valid?"VALID":"INVALID")<<"\"}"
       <<",\"rounds\":[";
    for(std::size_t i=0;i<round_rows.size();++i){if(i)out<<',';out<<round_rows[i];}
    out<<"]}";
    return out.str();
}

} // namespace

std::vector<char> canonical_d640_q4_v1(const omega_v2_1::CoreWeights& weights) {
    return canonical_q4_bytes(weights);
}

void execute_stage_a_calibration(const std::filesystem::path& d512_fp32_stream,
                                 const std::filesystem::path& state_inputs_root,
                                 const std::filesystem::path& d640_canonical_stream,
                                 const std::string& d640_canonical_sha256,
                                 const std::filesystem::path& output_root) {
    if(std::filesystem::exists(output_root))throw std::runtime_error("Stage A output root already exists; no overwrite/retry");
    if(d640_canonical_sha256.size()!=64)throw std::runtime_error("d640 canonical SHA-256 is missing/invalid");
    if(!std::filesystem::is_regular_file(state_inputs_root/"state_inputs_manifest.json"))throw std::runtime_error("state input hash manifest missing");
    std::filesystem::create_directories(output_root);
    Fp32Weights d512_fp32;
    const CoreWeights d512_q4=load_d512_weights(d512_fp32_stream,d512_fp32);
    const CoreWeights d640_q4=make_seeded_weights(640,kWeightSeed);
    const std::uint64_t d640_checksum=q4_checksum(d640_q4);
    if(d640_checksum!=kHistoricalD640Q4Checksum)throw std::runtime_error("D640_WEIGHT_BINDING_HOLD");
    const Fp32Weights d640_fp32=regenerate_fp32_source(640,kWeightSeed);
    const CoreWeights d640_from_fp32=pack_fp32(d640_fp32);
    if(q4_checksum(d640_from_fp32)!=kHistoricalD640Q4Checksum||!q4_equal(d640_q4,d640_from_fp32))throw std::runtime_error("D640_WEIGHT_BINDING_HOLD: regenerated pre-Q4 source does not reproduce Q4 bytes");
    const auto d640_fp32_paths=persist_d640_regenerated_fp32(d640_fp32,output_root);
    const auto canonical=canonical_q4_bytes(d640_q4);
    std::ifstream canonical_input(d640_canonical_stream,std::ios::binary);
    if(!canonical_input)throw std::runtime_error("ratified d640 canonical stream missing");
    const std::vector<char> canonical_sealed((std::istreambuf_iterator<char>(canonical_input)),std::istreambuf_iterator<char>());
    if(canonical!=canonical_sealed)throw std::runtime_error("D640_WEIGHT_BINDING_HOLD: canonical stream differs from regenerated Q4 family");

    const auto workers=select_kq_workers();WorkerPool pool(workers);
    std::vector<std::string> cells;
    for(int d:{512,640})for(int m:{1,4,8,16})for(int K:{1,4})for(const char* family:{"HISTORICAL_FORMULA","CAL_RANDN"}){
        const CoreWeights& weights=d==512?d512_q4:d640_q4;
        const Fp32Weights& original=d==512?d512_fp32:d640_fp32;
        cells.push_back(run_cell(weights,original,d,m,K,family,state_inputs_root,output_root,pool));
    }
    const bool affinity=pool.affinity_intact();pool.stop();
    if(!affinity)throw std::runtime_error("EXECUTION_FAILURE: worker affinity was not intact");
    std::ofstream report(output_root/"stage_a_report.json",std::ios::binary|std::ios::trunc);
    if(!report)throw std::runtime_error("cannot create Stage A report");
    report<<"{\"schema\":\"omega-v2-1d-stage-a-results-v1\",\"terminal_status\":\"STAGE_A_DIAGNOSTIC_COMPLETE\",\"classification\":\"CALIBRATION_DIAGNOSTIC_ONLY\",\"architectural_verdict\":\"NONE\",\"d640_regenerated_q4_checksum\":"<<d640_checksum
          <<",\"d640_canonical_serialization_version\":\"OMEGA-V2-1D-D640-Q4-CANONICAL-V1\",\"d640_canonical_sha256\":\""<<d640_canonical_sha256
          <<"\",\"d640_fp32_source\":\"REGENERATED_FP32_SOURCE_BOUND_TO_HISTORICAL_Q4\",\"d640_regenerated_fp32_source_paths\":[";
    for(std::size_t i=0;i<d640_fp32_paths.size();++i){if(i)report<<',';report<<'\"'<<d640_fp32_paths[i]<<'\"';}
    report<<"],\"d512_q4_vs_fp32_classification\":\"DIRECT_CALIBRATION_DIAGNOSTIC\",\"d640_q4_vs_fp32_classification\":\"RECONSTRUCTED_CALIBRATION_DIAGNOSTIC\",\"historical_pre_Q4_FP32_bytes_available\":false,\"decomposition_status\":{\"input_rmsnorm\":\"CHECKPOINT_COMPARISON\",\"q4_projection_reduction\":\"CHECKPOINT_COMPARISON\",\"qk_reduction\":\"CHECKPOINT_COMPARISON\",\"softmax\":\"SOFTMAX_STANDALONE_ATTRIBUTION=NO_SEPARABLE\",\"attention_v_accumulation\":\"CHECKPOINT_COMPARISON\",\"output_projection\":\"CHECKPOINT_COMPARISON\",\"first_residual\":\"CHECKPOINT_COMPARISON\",\"post_attention_mlp_rmsnorm\":\"CHECKPOINT_COMPARISON\",\"gate_up_q4\":\"CHECKPOINT_COMPARISON\",\"silu_reciprocal\":\"NO_SEPARABLE_NO_IDENTICAL_GATE_UP_REPLAY\",\"down_projection\":\"CHECKPOINT_COMPARISON\",\"final_residual\":\"CHECKPOINT_COMPARISON\",\"recurrent_accumulation\":\"PER_ROUND_PRIMARY_METRICS\"},\"timing_performed\":false,\"held_out_data_used\":false,\"cells\":[";
    for(std::size_t i=0;i<cells.size();++i){if(i)report<<',';report<<cells[i];}
    report<<"]}\n";report.close();if(!report)throw std::runtime_error("failed writing Stage A report");
}

} // namespace omega_v2_1d

#include "kq_candidate2.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace omega_v2_1b_candidate_02 {
using namespace omega_v2_1;
namespace {

void rms_rows_avx2(const float* input,float* output,int rows,int width){
    for(int row=0;row<rows;++row){const float* x=input+static_cast<std::size_t>(row)*width;float* y=output+static_cast<std::size_t>(row)*width;__m256 sums=_mm256_setzero_ps();for(int col=0;col<width;col+=8){const __m256 v=_mm256_loadu_ps(x+col);sums=_mm256_fmadd_ps(v,v,sums);}alignas(32)float lanes[8];_mm256_store_ps(lanes,sums);const double square_sum=((static_cast<double>(lanes[0])+lanes[1])+(lanes[2]+lanes[3]))+((lanes[4]+lanes[5])+(lanes[6]+lanes[7]));const float inverse=1.0f/std::sqrt(static_cast<float>(square_sum/width)+1e-6f);const __m256 scale=_mm256_set1_ps(inverse);for(int col=0;col<width;col+=8)_mm256_storeu_ps(y+col,_mm256_mul_ps(_mm256_loadu_ps(x+col),scale));}
}

float dot_avx2(const float* left,const float* right,int width){__m256 sums=_mm256_setzero_ps();for(int col=0;col<width;col+=8)sums=_mm256_fmadd_ps(_mm256_loadu_ps(left+col),_mm256_loadu_ps(right+col),sums);alignas(32)float lanes[8];_mm256_store_ps(lanes,sums);return ((lanes[0]+lanes[1])+(lanes[2]+lanes[3]))+((lanes[4]+lanes[5])+(lanes[6]+lanes[7]));}

void attention_avx2(Scratch& scratch){const int m=scratch.m,d=scratch.d;const float inverse_sqrt=1.0f/std::sqrt(static_cast<float>(d));for(int qrow=0;qrow<m;++qrow){float* score=scratch.scores.data()+static_cast<std::size_t>(qrow)*m;const float* q=scratch.query.data()+static_cast<std::size_t>(qrow)*d;float peak=-(std::numeric_limits<float>::infinity)();for(int krow=0;krow<m;++krow){score[krow]=dot_avx2(q,scratch.key.data()+static_cast<std::size_t>(krow)*d,d)*inverse_sqrt;peak=(std::max)(peak,score[krow]);}float denominator=0;for(int krow=0;krow<m;++krow){score[krow]=std::exp(score[krow]-peak);denominator+=score[krow];}for(int krow=0;krow<m;++krow)score[krow]/=denominator;float* result=scratch.attention.data()+static_cast<std::size_t>(qrow)*d;for(int col=0;col<d;col+=8){__m256 mixed=_mm256_setzero_ps();for(int krow=0;krow<m;++krow)mixed=_mm256_fmadd_ps(_mm256_set1_ps(score[krow]),_mm256_loadu_ps(scratch.value.data()+static_cast<std::size_t>(krow)*d+col),mixed);_mm256_storeu_ps(result+col,mixed);}}}

__m256 exp_approx_avx2(__m256 input){const __m256 x=_mm256_max_ps(_mm256_set1_ps(-80.0f),_mm256_min_ps(_mm256_set1_ps(80.0f),input));const __m256 n_float=_mm256_round_ps(_mm256_mul_ps(x,_mm256_set1_ps(1.4426950408889634f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);const __m256 n_as_float=_mm256_cvtepi32_ps(_mm256_cvtps_epi32(n_float));__m256 r=_mm256_fnmadd_ps(n_as_float,_mm256_set1_ps(0.693359375f),x);r=_mm256_fnmadd_ps(n_as_float,_mm256_set1_ps(-0.00021219444005469058f),r);__m256 p=_mm256_set1_ps(1.0f/5040.0f);p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/720.0f));p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/120.0f));p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/24.0f));p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/6.0f));p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));const __m256i exponent=_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n_as_float),_mm256_set1_epi32(127)),23);return _mm256_mul_ps(p,_mm256_castsi256_ps(exponent));}

void silu_up_fused_avx2(const float* gate,const float* up,float* output,std::size_t count){const __m256 one=_mm256_set1_ps(1),two=_mm256_set1_ps(2);for(std::size_t i=0;i<count;i+=8){const __m256 g=_mm256_loadu_ps(gate+i);const __m256 expv=exp_approx_avx2(_mm256_sub_ps(_mm256_setzero_ps(),g));const __m256 denom=_mm256_add_ps(one,expv);__m256 reciprocal=_mm256_rcp_ps(denom);reciprocal=_mm256_mul_ps(reciprocal,_mm256_fnmadd_ps(denom,reciprocal,two));const __m256 silu=_mm256_mul_ps(g,reciprocal);_mm256_storeu_ps(output+i,_mm256_mul_ps(silu,_mm256_loadu_ps(up+i)));}}

void add_residual_avx2(float* output,const float* left,const float* right,std::size_t count){for(std::size_t i=0;i<count;i+=8)_mm256_storeu_ps(output+i,_mm256_add_ps(_mm256_loadu_ps(left+i),_mm256_loadu_ps(right+i)));}

std::vector<float> scalar_step(const CoreWeights& weights,const std::vector<float>& state,int m){const int d=weights.d;std::vector<float> norm(static_cast<std::size_t>(m)*d),q(norm.size()),k(norm.size()),v(norm.size()),scores(static_cast<std::size_t>(m)*m),att(norm.size()),proj(norm.size()),hidden(norm.size()),mlpnorm(norm.size()),gate(static_cast<std::size_t>(m)*4*d),up(gate.size()),gated(gate.size()),down(norm.size()),out(norm.size());auto rms=[&](const std::vector<float>& x,std::vector<float>& y,int rows,int width){for(int r=0;r<rows;++r){double sum=0;for(int c=0;c<width;++c){const double z=x[static_cast<std::size_t>(r)*width+c];sum+=z*z;}const float inv=1.0f/std::sqrt(static_cast<float>(sum/width)+1e-6f);for(int c=0;c<width;++c)y[static_cast<std::size_t>(r)*width+c]=x[static_cast<std::size_t>(r)*width+c]*inv;}};auto linear=[&](const Q4Matrix& w,const std::vector<float>& x,std::vector<float>& y){q4_linear_single_thread(w,x.data(),y.data(),m);};rms(state,norm,m,d);linear(weights.W_Q,norm,q);linear(weights.W_K,norm,k);linear(weights.W_V,norm,v);const float inv=1.0f/std::sqrt(static_cast<float>(d));for(int qr=0;qr<m;++qr){float* row=scores.data()+static_cast<std::size_t>(qr)*m;float peak=-(std::numeric_limits<float>::infinity)();for(int kr=0;kr<m;++kr){float dot=0;for(int c=0;c<d;++c)dot+=q[static_cast<std::size_t>(qr)*d+c]*k[static_cast<std::size_t>(kr)*d+c];row[kr]=dot*inv;peak=(std::max)(peak,row[kr]);}float denom=0;for(int kr=0;kr<m;++kr){row[kr]=std::exp(row[kr]-peak);denom+=row[kr];}for(int kr=0;kr<m;++kr)row[kr]/=denom;for(int c=0;c<d;++c){float sum=0;for(int kr=0;kr<m;++kr)sum+=row[kr]*v[static_cast<std::size_t>(kr)*d+c];att[static_cast<std::size_t>(qr)*d+c]=sum;}}linear(weights.W_O,att,proj);for(std::size_t i=0;i<hidden.size();++i)hidden[i]=state[i]+proj[i];rms(hidden,mlpnorm,m,d);linear(weights.W_gate,mlpnorm,gate);linear(weights.W_up,mlpnorm,up);for(std::size_t i=0;i<gated.size();++i){const float sigmoid=1.0f/(1.0f+std::exp(-gate[i]));gated[i]=(gate[i]*sigmoid)*up[i];}linear(weights.W_down,gated,down);for(std::size_t i=0;i<out.size();++i)out[i]=hidden[i]+down[i];return out;}

} // namespace

void full_block_round_candidate2(const CoreWeights& weights,Scratch& scratch,WorkerPool& pool){const int d=weights.d,m=scratch.m;const std::size_t md=static_cast<std::size_t>(m)*d,m4d=static_cast<std::size_t>(m)*4*d;rms_rows_avx2(scratch.state.data(),scratch.normalized.data(),m,d);omega_v2_1b_candidate_02::q4_linear_fused_qkv(weights,scratch.normalized.data(),scratch.query.data(),scratch.key.data(),scratch.value.data(),m,pool);attention_avx2(scratch);q4_linear(weights.W_O,scratch.attention.data(),scratch.projected.data(),m,pool);add_residual_avx2(scratch.hidden.data(),scratch.state.data(),scratch.projected.data(),md);rms_rows_avx2(scratch.hidden.data(),scratch.mlp_normalized.data(),m,d);omega_v2_1b_candidate_02::q4_linear_fused_gate_up(weights,scratch.mlp_normalized.data(),scratch.gate.data(),scratch.up.data(),m,pool);silu_up_fused_avx2(scratch.gate.data(),scratch.up.data(),scratch.gated.data(),m4d);q4_linear(weights.W_down,scratch.gated.data(),scratch.down.data(),m,pool);add_residual_avx2(scratch.output.data(),scratch.hidden.data(),scratch.down.data(),md);scratch.state.swap(scratch.output);}

void full_block_round_single_thread_candidate2(const CoreWeights& weights,Scratch& scratch){scratch.state=scalar_step(weights,scratch.state,scratch.m);}

std::string full_block_scalar_reference_test_candidate2(WorkerPool& pool) {
    constexpr int d = 32, m = 4, rounds = 2;
    const CoreWeights weights = make_seeded_weights(d, kSeed);
    Scratch vectorized, reference;
    vectorized.resize_for(d, m);
    reference.resize_for(d, m);
    for (std::size_t index = 0; index < vectorized.state.size(); ++index) {
        const float value = static_cast<float>(static_cast<int>((index * 19) % 83) - 41) * (1.0f / 256.0f);
        vectorized.state[index] = value;
        reference.state[index] = value;
    }
    for (int round = 0; round < rounds; ++round) {
        full_block_round_candidate2(weights, vectorized, pool);
        full_block_round_single_thread_candidate2(weights, reference);
    }
    double max_abs = 0.0, max_rel = 0.0;
    for (std::size_t index = 0; index < vectorized.state.size(); ++index) {
        const double difference = std::fabs(static_cast<double>(vectorized.state[index]) - reference.state[index]);
        max_abs = (std::max)(max_abs, difference);
        max_rel = (std::max)(max_rel, difference / (std::max)(std::fabs(static_cast<double>(reference.state[index])), 1e-12));
    }

    double exp_abs = 0.0, exp_rel = 0.0;
    alignas(32) float input[8], output[8];
    for (int base = -8000; base <= 7992; base += 8) {
        for (int lane = 0; lane < 8; ++lane) input[lane] = static_cast<float>(base + lane) * 0.01f;
        _mm256_store_ps(output, exp_approx_avx2(_mm256_load_ps(input)));
        for (int lane = 0; lane < 8; ++lane) {
            const double expected = std::exp(static_cast<double>(input[lane]));
            const double difference = std::fabs(static_cast<double>(output[lane]) - expected);
            exp_abs = (std::max)(exp_abs, difference);
            exp_rel = (std::max)(exp_rel, difference / (std::max)(expected, 1e-30));
        }
    }
    const float upper_endpoint = 80.0f;
    _mm256_store_ps(output, exp_approx_avx2(_mm256_set1_ps(upper_endpoint)));
    const double upper_expected = std::exp(static_cast<double>(upper_endpoint));
    const double upper_abs = std::fabs(static_cast<double>(output[0]) - upper_expected);
    exp_abs = (std::max)(exp_abs, upper_abs);
    exp_rel = (std::max)(exp_rel, upper_abs / upper_expected);

    const bool exp_pass = exp_rel <= 2e-6;
    const bool pass = max_abs <= 1e-5 && max_rel <= 1e-4 && exp_pass;
    std::ostringstream result;
    result << std::setprecision(17)
        << "{\"d\":32,\"m\":4,\"K\":2,\"max_abs_error\":" << max_abs
        << ",\"max_rel_error\":" << max_rel << ",\"abs_tolerance\":1e-5,\"rel_tolerance\":1e-4"
        << ",\"exp_approx_domain\":[-80,80],\"exp_approx_grid_points\":16001"
        << ",\"exp_approx_relative_error_bound\":2e-6,\"exp_approx_measured_max_abs\":" << exp_abs
        << ",\"exp_approx_measured_max_rel\":" << exp_rel
        << ",\"exp_approx_pass\":" << (exp_pass ? "true" : "false")
        << ",\"pass\":" << (pass ? "true" : "false") << '}';
    return result.str();
}

} // namespace omega_v2_1b_candidate_02

namespace omega_v2_1 {

void v2_full_block_round(const CoreWeights& weights, Scratch& scratch, WorkerPool& pool) {
    omega_v2_1b_candidate_02::full_block_round_candidate2(weights, scratch, pool);
}

void v2_full_block_round_single_thread(const CoreWeights& weights, Scratch& scratch) {
    omega_v2_1b_candidate_02::full_block_round_single_thread_candidate2(weights, scratch);
}

std::string full_block_scalar_reference_test(WorkerPool& pool) {
    return omega_v2_1b_candidate_02::full_block_scalar_reference_test_candidate2(pool);
}

} // namespace omega_v2_1

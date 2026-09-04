#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnrl/aligned_buffer.hpp"
#include "cnrl/benchmark.hpp"
#include "cnrl/csv.hpp"
#include "cnrl/kernels.hpp"
#include "cnrl/platform.hpp"
#include "cnrl/random.hpp"
#include "cnrl/sharding.hpp"
#include "cnrl/state.hpp"
#include "cnrl/transitions.hpp"
#include "cnrl/weights.hpp"

namespace {
int failures=0;
#define REQUIRE(expr) do { if(!(expr)) throw std::runtime_error(std::string("requirement failed: ")+ #expr); } while(false)

void run_test(const char* name,const std::function<void()>& fn) {
  try { fn(); std::cout<<"PASS "<<name<<'\n'; }
  catch(const std::exception& e) { ++failures; std::cerr<<"FAIL "<<name<<": "<<e.what()<<'\n'; }
}

void fill_i8(cnrl::AlignedBuffer<std::int8_t>& buffer,std::uint32_t seed,int magnitude=31) {
  cnrl::XorShift32 rng(seed);
  for(std::size_t i=0;i<buffer.size();++i) buffer[i]=rng.symmetric_i8(magnitude);
}

void test_kernels() {
  const std::vector<std::uint32_t> dimensions={13,16,31,64,512};
  const std::vector<std::uint32_t> slots={1,2,4,8,16};
  for(auto D:dimensions) for(auto S:slots) {
    const std::uint32_t rows=7;
    cnrl::AlignedBuffer<std::int8_t> w(static_cast<std::size_t>(rows)*D);
    cnrl::AlignedBuffer<std::int8_t> x(static_cast<std::size_t>(S)*D);
    fill_i8(w,0x1111U+D+S,127); fill_i8(x,0x2222U+D+S,127);
    std::vector<std::int32_t> ref(static_cast<std::size_t>(S)*rows),repeat(ref.size()),fused(ref.size());
    cnrl::KernelCall call{w.data(),x.data(),ref.data(),rows,D,S,0,rows,8};
    cnrl::matmul_scalar_reference(call);
    call.output=repeat.data(); cnrl::matmul_avx2_repeat(call);
    call.output=fused.data(); cnrl::matmul_avx2_fused(call);
    REQUIRE(ref==repeat); REQUIRE(ref==fused);
    std::fill(fused.begin(),fused.end(),0);
    cnrl::run_kernel_unchecked(cnrl::KernelKind::avx2_fused,call);
    REQUIRE(ref==fused);
    if(S>=4) { call.output=fused.data(); call.slot_tile=4; cnrl::matmul_avx2_fused(call); REQUIRE(ref==fused); }
  }
}

void test_sharding() {
  const auto rows=cnrl::proportional_rows(1472,{19.3,18.1,10.9,17.0},1);
  REQUIRE(rows==std::vector<std::uint32_t>({435,408,246,383}));
  const auto shards=cnrl::make_shards({0,1,2,3},rows);
  cnrl::validate_shards({1472,8,8},shards,true);

  const auto aligned=cnrl::proportional_rows(1472,{19.3,18.1,10.9,17.0},64);
  REQUIRE(aligned[0]+aligned[1]+aligned[2]+aligned[3]==1472);
  for(const auto value:aligned) REQUIRE(value%64U==0U);
  bool rejected=false;
  try { (void)cnrl::proportional_rows(1473,{1.0,1.0},64); }
  catch(const std::invalid_argument&) { rejected=true; }
  REQUIRE(rejected);
}

void test_weight_variants() {
  cnrl::RunConfig config;
  config.shape={64,4,4}; config.transition.kind=cnrl::TransitionKind::fixed_point;
  config.shards=cnrl::make_shards({0,1},{32,32}); config.require_affinity=false;
  config.variant=cnrl::WeightVariant::shared;
  const auto shared=cnrl::make_weight_bank(config);
  config.variant=cnrl::WeightVariant::clone;
  const auto clone=cnrl::make_weight_bank(config);
  REQUIRE(clone.clone_hashes_equal); REQUIRE(clone.clone_addresses_distinct);
  REQUIRE(shared.hash_signature==clone.hash_signature);
  for(std::size_t s=0;s<clone.shards.size();++s) {
    for(std::uint32_t r=0;r<config.shape.depth;++r) {
      REQUIRE(std::memcmp(clone.shards[s].block(0),clone.shards[s].block(r),
                          static_cast<std::size_t>(clone.shards[s].block_bytes))==0);
      if(r>0) REQUIRE(clone.shards[s].block(0)!=clone.shards[s].block(r));
    }
  }
  config.variant=cnrl::WeightVariant::untied;
  const auto untied=cnrl::make_weight_bank(config);
  REQUIRE(untied.shards[0].block_hashes[0]!=untied.shards[0].block_hashes[1]);
}


const std::int8_t* global_weight_row(const cnrl::WeightBank& bank,
                                     std::uint32_t global_row,
                                     std::uint32_t round) {
  for (const auto& shard : bank.shards) {
    const auto begin = shard.shard.row_offset;
    const auto end = begin + shard.shard.rows;
    if (global_row >= begin && global_row < end) {
      return shard.block(round) +
          static_cast<std::size_t>(global_row - begin) * bank.shape.dimension;
    }
  }
  throw std::out_of_range("global weight row not found");
}

void test_weight_sharding_invariance() {
  cnrl::RunConfig first;
  first.shape = {64, 1, 3};
  first.transition.kind = cnrl::TransitionKind::frozen;
  first.variant = cnrl::WeightVariant::shared;
  first.seed = 0x1234ABCDU;
  first.require_affinity = false;
  first.shards = cnrl::make_shards({0, 1}, {24, 40});

  auto second = first;
  second.shards = cnrl::make_shards({0, 1, 2}, {16, 16, 32});

  const auto bank_a = cnrl::make_weight_bank(first);
  const auto bank_b = cnrl::make_weight_bank(second);
  REQUIRE(bank_a.hash_signature == bank_b.hash_signature);
  REQUIRE(bank_a.base_weight_bytes == bank_b.base_weight_bytes);
  for (std::uint32_t row = 0; row < first.shape.dimension; ++row) {
    REQUIRE(std::memcmp(global_weight_row(bank_a, row, 0),
                        global_weight_row(bank_b, row, 0),
                        first.shape.dimension) == 0);
  }

  first.variant = cnrl::WeightVariant::untied;
  second.variant = cnrl::WeightVariant::untied;
  const auto untied_a = cnrl::make_weight_bank(first);
  const auto untied_b = cnrl::make_weight_bank(second);
  for (std::uint32_t round = 0; round < first.shape.depth; ++round) {
    for (std::uint32_t row = 0; row < first.shape.dimension; ++row) {
      REQUIRE(std::memcmp(global_weight_row(untied_a, row, round),
                          global_weight_row(untied_b, row, round),
                          first.shape.dimension) == 0);
    }
  }
}

void test_fixed_transition() {
  cnrl::Shape shape{64,4,3};
  auto shards=cnrl::make_shards({0,1},{31,33});
  cnrl::AlignedBuffer<std::int8_t> current(static_cast<std::size_t>(shape.slots)*shape.dimension);
  cnrl::AlignedBuffer<std::int8_t> next(current.size());
  cnrl::AlignedBuffer<std::int32_t> output(current.size());
  fill_i8(current,17,31);
  cnrl::XorShift32 rng(19);
  for(std::size_t i=0;i<output.size();++i) output[i]=static_cast<std::int32_t>(rng.symmetric_i8(127))*512;
  cnrl::TransitionConfig cfg; cfg.kind=cnrl::TransitionKind::fixed_point; cfg.projection_shift=8;
  cnrl::TransitionStats stats;
  for(const auto& shard:shards) cnrl::transition_fixed_point_local(current.data(),output.data(),next.data(),shape,shard,cfg,stats);
  REQUIRE(stats.cells==current.size());
  for(std::size_t i=0;i<current.size();++i) {
    const std::int32_t y=output[i];
    const std::int32_t projected=y>=0?(y+128)/256:-((-y+128)/256);
    const std::int32_t expected=std::clamp(static_cast<std::int32_t>(current[i])+projected,-127,127);
    REQUIRE(next[i]==expected);
  }
}


std::int32_t projected_scalar(std::int32_t value,std::uint32_t shift) {
  if(shift==0) return value;
  const std::int64_t wide=value;
  const std::int64_t magnitude=wide<0?-wide:wide;
  const std::int64_t rounded=(magnitude+(std::int64_t{1}<<(shift-1U)))>>shift;
  return static_cast<std::int32_t>(wide<0?-rounded:rounded);
}

void test_rms_transitions() {
  const cnrl::Shape shape{64,3,2};
  const auto shards=cnrl::make_shards({0,1},{31,33});
  cnrl::AlignedBuffer<std::int8_t> current(static_cast<std::size_t>(shape.slots)*shape.dimension);
  cnrl::AlignedBuffer<std::int32_t> output(current.size());
  fill_i8(current,0x51U,31);
  cnrl::XorShift32 rng(0x91U);
  for(std::size_t i=0;i<output.size();++i) output[i]=static_cast<std::int32_t>(rng.symmetric_i8(90))*256;
  cnrl::TransitionConfig cfg;
  cfg.projection_shift=8;
  cfg.target_rms=32.0;
  cfg.epsilon=1.0e-6;

  cnrl::TransitionWorkspace workspace;
  cnrl::prepare_transition_workspace(workspace,static_cast<std::uint32_t>(shards.size()),shape);
  cnrl::AlignedBuffer<std::int8_t> actual(current.size()),expected(current.size());
  cnrl::TransitionStats stats;

  cfg.kind=cnrl::TransitionKind::group_rms;
  for(const auto& shard:shards) {
    cnrl::transition_group_rms_local(current.data(),output.data(),actual.data(),shape,shard,cfg,workspace,stats);
    for(std::uint32_t slot=0;slot<shape.slots;++slot) {
      double sum=0.0;
      for(std::uint32_t row=0;row<shard.rows;++row) {
        const auto index=static_cast<std::size_t>(slot)*shape.dimension+shard.row_offset+row;
        const auto residual=static_cast<std::int32_t>(current[index])+projected_scalar(output[index],8);
        sum+=static_cast<double>(residual)*residual;
      }
      const double factor=cfg.target_rms/std::sqrt(sum/static_cast<double>(shard.rows)+cfg.epsilon);
      for(std::uint32_t row=0;row<shard.rows;++row) {
        const auto index=static_cast<std::size_t>(slot)*shape.dimension+shard.row_offset+row;
        const auto residual=static_cast<std::int32_t>(current[index])+projected_scalar(output[index],8);
        const auto quantized=std::clamp(static_cast<std::int32_t>(std::nearbyint(residual*factor)),-127,127);
        expected[index]=static_cast<std::int8_t>(quantized);
      }
    }
  }
  REQUIRE(std::memcmp(actual.data(),expected.data(),actual.bytes())==0);
  REQUIRE(stats.cells==actual.size());

  actual.fill_zero(); expected.fill_zero(); stats={};
  cnrl::prepare_transition_workspace(workspace,static_cast<std::uint32_t>(shards.size()),shape);
  cfg.kind=cnrl::TransitionKind::global_rms;
  for(const auto& shard:shards) {
    cnrl::transition_global_rms_prepare(current.data(),output.data(),shape,shard,cfg,workspace);
  }
  cnrl::transition_global_rms_reduce(static_cast<std::uint32_t>(shards.size()),shape,cfg,workspace);
  for(const auto& shard:shards) {
    cnrl::transition_global_rms_apply(actual.data(),shape,shard,cfg,workspace,stats);
  }
  for(std::uint32_t slot=0;slot<shape.slots;++slot) {
    double sum=0.0;
    for(std::uint32_t row=0;row<shape.dimension;++row) {
      const auto index=static_cast<std::size_t>(slot)*shape.dimension+row;
      const auto residual=static_cast<std::int32_t>(current[index])+projected_scalar(output[index],8);
      sum+=static_cast<double>(residual)*residual;
    }
    const double factor=cfg.target_rms/std::sqrt(sum/static_cast<double>(shape.dimension)+cfg.epsilon);
    for(std::uint32_t row=0;row<shape.dimension;++row) {
      const auto index=static_cast<std::size_t>(slot)*shape.dimension+row;
      const auto residual=static_cast<std::int32_t>(current[index])+projected_scalar(output[index],8);
      const auto quantized=std::clamp(static_cast<std::int32_t>(std::nearbyint(residual*factor)),-127,127);
      expected[index]=static_cast<std::int8_t>(quantized);
    }
  }
  REQUIRE(std::memcmp(actual.data(),expected.data(),actual.bytes())==0);
  REQUIRE(stats.cells==actual.size());
}

std::size_t csv_field_count(const std::string& line) {
  std::size_t fields=1;
  bool quoted=false;
  for(std::size_t i=0;i<line.size();++i) {
    if(line[i]=='"') {
      if(quoted && i+1<line.size() && line[i+1]=='"') { ++i; continue; }
      quoted=!quoted;
    } else if(line[i]==',' && !quoted) {
      ++fields;
    }
  }
  REQUIRE(!quoted);
  return fields;
}

void test_csv_schema_and_accounting() {
  cnrl::RunConfig config;
  config.gate=cnrl::GateKind::t0m;
  config.shape={64,8,4};
  config.kernel=cnrl::KernelKind::avx2_fused;
  config.slot_tile=4;
  config.transition.kind=cnrl::TransitionKind::frozen;
  config.timed_repetitions=3;
  config.sequences_per_repetition=2;
  config.require_affinity=false;
  config.shards=cnrl::make_shards({0,1},{32,32});
  REQUIRE(cnrl::calculate_logical_weight_load_bytes(config)==196608U);
  config.kernel=cnrl::KernelKind::avx2_repeat;
  REQUIRE(cnrl::calculate_logical_weight_load_bytes(config)==786432U);
  config.kernel=cnrl::KernelKind::avx2_fused;
  config.shape.slots=10;
  REQUIRE(cnrl::calculate_logical_weight_load_bytes(config)==294912U);
  config.shape.slots=8;

  cnrl::RunResult result;
  result.valid=true;
  result.workers.resize(2);
  result.workers[0].logical_cpu=0;
  result.workers[0].physical_core_index=0;
  result.workers[1].logical_cpu=1;
  result.workers[1].physical_core_index=1;
  std::ostringstream stream;
  cnrl::write_run_csv_header(stream);
  cnrl::write_run_csv_row(stream,config,result);
  std::istringstream input(stream.str());
  std::string header,row;
  std::getline(input,header);
  std::getline(input,row);
  REQUIRE(!header.empty()&&!row.empty());
  REQUIRE(csv_field_count(header)==csv_field_count(row));
  REQUIRE(header.find("physical_cores")!=std::string::npos);
  REQUIRE(header.find("one_pass_weight_gb_per_second")!=std::string::npos);
  REQUIRE(header.find("project_version")!=std::string::npos);
  REQUIRE(header.find("projection_shift")!=std::string::npos);
  REQUIRE(header.find("phase_profile")!=std::string::npos);
}

void test_transition_validation() {
  cnrl::TransitionConfig config;
  config.kind=cnrl::TransitionKind::global_rms;
  cnrl::validate_transition_config({512,8,4},config);
  bool rejected=false;
  config.final_shift=1;
  try { cnrl::validate_transition_config({512,8,4},config); }
  catch(const std::invalid_argument&) { rejected=true; }
  REQUIRE(rejected);
  rejected=false;
  config.final_shift=0;
  config.projection_shift=31;
  try { cnrl::validate_transition_config({512,8,4},config); }
  catch(const std::invalid_argument&) { rejected=true; }
  REQUIRE(rejected);
  rejected=false;
  config.projection_shift=12;
  config.output_multiplier=2;
  try { cnrl::validate_transition_config({512,8,4},config); }
  catch(const std::invalid_argument&) { rejected=true; }
  REQUIRE(rejected);
}

cnrl::RunConfig integration_config(cnrl::WeightVariant variant,cnrl::TransitionKind transition);


struct ReferenceRun {
  std::uint64_t output_checksum = 0;
  std::uint64_t state_checksum = 0;
};

ReferenceRun recurrent_reference(const cnrl::RunConfig& config) {
  const auto bank = cnrl::make_weight_bank(config);
  cnrl::AlignedBuffer<std::int8_t> initial;
  cnrl::AlignedBuffer<std::int8_t> state_a;
  cnrl::AlignedBuffer<std::int8_t> state_b;
  cnrl::initialize_state(initial, config.shape, config.seed);
  state_a = initial;
  state_b.resize(initial.size());
  state_b.fill_zero();
  cnrl::AlignedBuffer<std::int32_t> output(initial.size());
  output.fill_zero();
  cnrl::TransitionWorkspace workspace;
  cnrl::prepare_transition_workspace(
      workspace, static_cast<std::uint32_t>(config.shards.size()), config.shape);
  std::vector<cnrl::TransitionStats> stats(config.shards.size());

  std::int8_t* current = state_a.data();
  std::int8_t* next = state_b.data();
  for (std::uint32_t round = 0; round < config.shape.depth; ++round) {
    for (const auto& shard : config.shards) {
      cnrl::KernelCall call;
      call.weights = bank.shards[shard.worker_index].block(round);
      call.state = current;
      call.output = output.data();
      call.rows = shard.rows;
      call.dimension = config.shape.dimension;
      call.slots = config.shape.slots;
      call.row_offset = shard.row_offset;
      call.output_stride = config.shape.dimension;
      call.slot_tile = config.slot_tile;
      cnrl::matmul_scalar_reference(call);
    }

    switch (config.transition.kind) {
      case cnrl::TransitionKind::fixed_point:
        for (const auto& shard : config.shards) {
          cnrl::transition_fixed_point_local(
              current, output.data(), next, config.shape, shard,
              config.transition, stats[shard.worker_index]);
        }
        break;
      case cnrl::TransitionKind::group_rms:
        for (const auto& shard : config.shards) {
          cnrl::transition_group_rms_local(
              current, output.data(), next, config.shape, shard,
              config.transition, workspace, stats[shard.worker_index]);
        }
        break;
      case cnrl::TransitionKind::global_rms:
        for (const auto& shard : config.shards) {
          cnrl::transition_global_rms_prepare(
              current, output.data(), config.shape, shard,
              config.transition, workspace);
        }
        cnrl::transition_global_rms_reduce(
            static_cast<std::uint32_t>(config.shards.size()), config.shape,
            config.transition, workspace);
        for (const auto& shard : config.shards) {
          cnrl::transition_global_rms_apply(
              next, config.shape, shard, config.transition, workspace,
              stats[shard.worker_index]);
        }
        break;
      case cnrl::TransitionKind::frozen:
        throw std::runtime_error("recurrent reference requires a real transition");
    }
    std::swap(current, next);
  }

  return {
      cnrl::checksum_i32(output.data(), output.size()),
      cnrl::checksum_i8(current, initial.size()),
  };
}

void test_recurrent_runner_against_scalar_reference(cnrl::TransitionKind kind) {
  auto config = integration_config(cnrl::WeightVariant::shared, kind);
  config.warmup_repetitions = 0;
  config.timed_repetitions = 1;
  config.sequences_per_repetition = 1;
  const ReferenceRun expected = recurrent_reference(config);
  const auto actual = cnrl::run_benchmark(config);
  REQUIRE(actual.valid);
  REQUIRE(actual.output_checksum == expected.output_checksum);
  REQUIRE(actual.state_checksum == expected.state_checksum);
}

void test_gate_contracts_and_invalid_cpu() {
  auto config=integration_config(cnrl::WeightVariant::shared,cnrl::TransitionKind::frozen);
  config.gate=cnrl::GateKind::t0r;
  config.shape.slots=2;
  REQUIRE(!cnrl::run_benchmark(config).valid);
  config.shape.slots=1;
  config.gate=cnrl::GateKind::t0rm;
  REQUIRE(!cnrl::run_benchmark(config).valid);
  config.gate=cnrl::GateKind::t0r;
  config.shards[0].logical_cpu=UINT32_MAX;
  REQUIRE(!cnrl::run_benchmark(config).valid);
}

cnrl::RunConfig integration_config(cnrl::WeightVariant variant,cnrl::TransitionKind transition) {
  const auto topology=cnrl::discover_cpu_topology();
  const auto cpus=cnrl::choose_one_logical_per_physical_core(topology);
  REQUIRE(!cpus.empty());
  cnrl::RunConfig config;
  config.gate=transition==cnrl::TransitionKind::frozen?cnrl::GateKind::t0m:cnrl::GateKind::t0rm;
  config.shape={64,4,3}; config.variant=variant; config.kernel=cnrl::KernelKind::avx2_fused;
  config.slot_tile=4; config.transition.kind=transition; config.transition.projection_shift=8;
  config.warmup_repetitions=0; config.timed_repetitions=2; config.require_affinity=false;
  const std::size_t worker_count=std::min<std::size_t>(2,cpus.size());
  std::vector<std::uint32_t> selected(cpus.begin(),
      cpus.begin()+static_cast<std::vector<std::uint32_t>::difference_type>(worker_count));
  std::vector<double> rates(worker_count,1.0);
  config.shards=cnrl::make_shards(selected,cnrl::proportional_rows(64,rates,1));
  return config;
}

void test_integration_clone_frozen() {
  auto a=integration_config(cnrl::WeightVariant::shared,cnrl::TransitionKind::frozen);
  auto b=a; b.variant=cnrl::WeightVariant::clone;
  const auto ra=cnrl::run_benchmark(a), rb=cnrl::run_benchmark(b);
  REQUIRE(ra.valid&&rb.valid); REQUIRE(ra.output_checksum==rb.output_checksum);
  REQUIRE(ra.round_sink==rb.round_sink); REQUIRE(ra.weight_hash_signature==rb.weight_hash_signature);
  REQUIRE(rb.clone_hashes_equal&&rb.clone_addresses_distinct);
  REQUIRE(ra.mac_total==static_cast<std::uint64_t>(64)*64*4*3*2);
}

void test_integration_clone_recurrent(cnrl::TransitionKind kind) {
  auto a=integration_config(cnrl::WeightVariant::shared,kind);
  auto b=a; b.variant=cnrl::WeightVariant::clone;
  const auto ra=cnrl::run_benchmark(a), rb=cnrl::run_benchmark(b);
  REQUIRE(ra.valid&&rb.valid); REQUIRE(ra.output_checksum==rb.output_checksum);
  REQUIRE(ra.state_checksum==rb.state_checksum); REQUIRE(ra.round_sink==rb.round_sink);
  const std::uint64_t expected_cells=static_cast<std::uint64_t>(a.shape.dimension)*
      a.shape.slots*a.shape.depth*a.timed_repetitions*a.sequences_per_repetition;
  REQUIRE(ra.transition_cells==expected_cells);
  REQUIRE(rb.transition_cells==expected_cells);
}
}

int main() {
  const auto features=cnrl::detect_cpu_features();
  if(!features.avx2) { std::cerr<<"AVX2 required for this test suite\n"; return 77; }
  run_test("kernels exact scalar-repeat-fused",test_kernels);
  run_test("proportional sharding",test_sharding);
  run_test("weight variants and Bclone invariants",test_weight_variants);
  run_test("weight matrix is invariant to shard partition",test_weight_sharding_invariance);
  run_test("fixed-point transition",test_fixed_transition);
  run_test("group/global RMS transition oracles",test_rms_transitions);
  run_test("CSV schema and byte accounting",test_csv_schema_and_accounting);
  run_test("transition configuration validation",test_transition_validation);
  run_test("gate contracts and invalid CPU rejection",test_gate_contracts_and_invalid_cpu);
  run_test("runner fixed matches scalar recurrence",[]{test_recurrent_runner_against_scalar_reference(cnrl::TransitionKind::fixed_point);});
  run_test("runner group-rms matches scalar recurrence",[]{test_recurrent_runner_against_scalar_reference(cnrl::TransitionKind::group_rms);});
  run_test("runner global-rms matches scalar recurrence",[]{test_recurrent_runner_against_scalar_reference(cnrl::TransitionKind::global_rms);});
  run_test("integration frozen shared==clone",test_integration_clone_frozen);
  run_test("integration fixed shared==clone",[]{test_integration_clone_recurrent(cnrl::TransitionKind::fixed_point);});
  run_test("integration group-rms shared==clone",[]{test_integration_clone_recurrent(cnrl::TransitionKind::group_rms);});
  run_test("integration global-rms shared==clone",[]{test_integration_clone_recurrent(cnrl::TransitionKind::global_rms);});
  if(failures){std::cerr<<failures<<" test(s) failed\n";return 1;}
  std::cout<<"all tests passed\n";return 0;
}

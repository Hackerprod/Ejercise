#include "cnrl/csv.hpp"
#include <iomanip>
#include <ostream>
#include <sstream>

#ifndef CNRL_VERSION_STRING
#define CNRL_VERSION_STRING "unknown"
#endif

namespace cnrl {
namespace {
std::string csv_escape(const std::string& value) {
  std::string out = "\"";
  for (char c : value) {
    if (c == '\"') out += '\"';
    out += c;
  }
  out += '\"';
  return out;
}
template <typename F>
std::string worker_values(const RunResult& result, F selector) {
  std::ostringstream out;
  out << std::setprecision(17);
  for (std::size_t i = 0; i < result.workers.size(); ++i) {
    if (i) out << ';';
    out << selector(result.workers[i]);
  }
  return out.str();
}
}  // namespace

std::string shard_rows_string(const RunConfig& config) {
  std::ostringstream out;
  for (std::size_t i=0;i<config.shards.size();++i) { if (i) out << ';'; out << config.shards[i].rows; }
  return out.str();
}
std::string shard_cpus_string(const RunConfig& config) {
  std::ostringstream out;
  for (std::size_t i=0;i<config.shards.size();++i) { if (i) out << ';'; out << config.shards[i].logical_cpu; }
  return out.str();
}

void write_run_csv_header(std::ostream& out) {
  out << "project_version,gate,D,S,R,total_rows,kernel,variant,transition,timing_scope,slot_tile,"
         "seed,phase_profile,require_affinity,projection_shift,state_multiplier,output_multiplier,final_shift,target_rms,epsilon,"
         "warmup_repetitions,timed_repetitions,sequences_per_repetition,worker_count,allow_smt_siblings,cpus,physical_cores,rows,"
         "base_weight_bytes,allocated_weight_bytes,logical_weight_load_bytes,one_pass_weight_bytes,distinct_weight_storage_bytes,"
         "elapsed_seconds,mac_total,mac_per_second,logical_weight_load_gb_per_second,one_pass_weight_gb_per_second,"
         "output_checksum,state_checksum,round_sink,weight_hash_signature,clone_hashes_equal,"
         "clone_addresses_distinct,all_affinity_succeeded,clipped_cells,transition_cells,"
         "worker_compute_seconds,worker_transition_seconds,worker_sync_seconds,worker_cold_seconds,"
         "worker_affinity_errors,valid,error\n";
}

void write_run_csv_row(std::ostream& out, const RunConfig& config,
                       const RunResult& result) {
  std::uint64_t total_rows = 0;
  for (const auto& shard : config.shards) total_rows += shard.rows;
  out << std::setprecision(17)
      << CNRL_VERSION_STRING << ',' << to_string(config.gate) << ','
      << config.shape.dimension << ',' << config.shape.slots << ','
      << config.shape.depth << ',' << total_rows << ',' << to_string(config.kernel) << ','
      << to_string(config.variant) << ',' << to_string(config.transition.kind) << ','
      << to_string(config.timing_scope) << ',' << config.slot_tile << ','
      << config.seed << ',' << (config.phase_profile ? "true" : "false") << ','
      << (config.require_affinity ? "true" : "false") << ','
      << config.transition.projection_shift << ','
      << config.transition.state_multiplier << ','
      << config.transition.output_multiplier << ','
      << config.transition.final_shift << ','
      << config.transition.target_rms << ',' << config.transition.epsilon << ','
      << config.warmup_repetitions << ',' << config.timed_repetitions << ','
      << config.sequences_per_repetition << ',' << config.shards.size() << ','
      << (config.allow_smt_siblings ? "true" : "false") << ','
      << csv_escape(shard_cpus_string(config)) << ','
      << csv_escape(worker_values(result, [](const PerWorkerMetrics& m){ return m.physical_core_index; })) << ','
      << csv_escape(shard_rows_string(config)) << ','
      << result.base_weight_bytes << ',' << result.allocated_weight_bytes << ','
      << result.logical_weight_load_bytes << ',' << result.one_pass_weight_bytes << ','
      << result.distinct_weight_storage_bytes << ',' << result.elapsed_seconds << ','
      << result.mac_total << ',' << result.mac_per_second << ','
      << result.logical_weight_load_gb_per_second << ',' << result.one_pass_weight_gb_per_second << ','
      << result.output_checksum << ',' << result.state_checksum << ',' << result.round_sink << ','
      << result.weight_hash_signature << ','
      << (result.clone_hashes_equal ? "true" : "false") << ','
      << (result.clone_addresses_distinct ? "true" : "false") << ','
      << (result.all_affinity_succeeded ? "true" : "false") << ','
      << result.clipped_cells << ',' << result.transition_cells << ','
      << csv_escape(worker_values(result, [](const PerWorkerMetrics& m){ return m.compute_seconds; })) << ','
      << csv_escape(worker_values(result, [](const PerWorkerMetrics& m){ return m.transition_seconds; })) << ','
      << csv_escape(worker_values(result, [](const PerWorkerMetrics& m){ return m.synchronization_seconds; })) << ','
      << csv_escape(worker_values(result, [](const PerWorkerMetrics& m){ return m.cold_prepare_seconds; })) << ','
      << csv_escape(worker_values(result, [](const PerWorkerMetrics& m){ return m.affinity_error; })) << ','
      << (result.valid ? "true" : "false") << ',' << csv_escape(result.error) << '\n';
}
}  // namespace cnrl

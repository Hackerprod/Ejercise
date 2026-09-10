#include "rlm/config.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <string_view>
#include <unordered_set>

namespace rlm {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string strip_comment(const std::string& line) {
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '"' && (i == 0 || line[i - 1] != '\\')) quoted = !quoted;
    if (line[i] == '#' && !quoted) return line.substr(0, i);
  }
  return line;
}

Result<std::string> parse_string(std::string value) {
  value = trim(std::move(value));
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    return Status(ErrorCode::invalid_argument, "expected a quoted string");
  }
  std::string output;
  output.reserve(value.size() - 2);
  for (std::size_t i = 1; i + 1 < value.size(); ++i) {
    if (value[i] == '\\') {
      if (i + 2 >= value.size()) return Status(ErrorCode::invalid_argument, "invalid string escape");
      const char escaped = value[++i];
      switch (escaped) {
        case 'n': output.push_back('\n'); break;
        case 'r': output.push_back('\r'); break;
        case 't': output.push_back('\t'); break;
        case '\\': output.push_back('\\'); break;
        case '"': output.push_back('"'); break;
        default: return Status(ErrorCode::invalid_argument, "unsupported string escape");
      }
    } else {
      output.push_back(value[i]);
    }
  }
  return output;
}

template <typename T>
Result<T> parse_unsigned(std::string value) {
  value = trim(std::move(value));
  T output{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), output);
  if (error != std::errc{} || end != value.data() + value.size()) {
    return Status(ErrorCode::invalid_argument, "expected an unsigned integer");
  }
  return output;
}

Result<float> parse_float(std::string value) {
  value = trim(std::move(value));
  try {
    std::size_t consumed = 0;
    const float output = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(output)) {
      return Status(ErrorCode::invalid_argument, "expected a finite floating-point number");
    }
    return output;
  } catch (const std::exception&) {
    return Status(ErrorCode::invalid_argument, "expected a floating-point number");
  }
}

Result<bool> parse_bool(std::string value) {
  value = trim(std::move(value));
  if (value == "true") return true;
  if (value == "false") return false;
  return Status(ErrorCode::invalid_argument, "expected true or false");
}

Result<Durability> parse_durability(std::string value) {
  auto parsed = parse_string(std::move(value));
  if (!parsed) return parsed.status();
  if (parsed.value() == "none") return Durability::none;
  if (parsed.value() == "data") return Durability::data;
  if (parsed.value() == "full") return Durability::full;
  return Status(ErrorCode::invalid_argument, "durability must be none, data, or full");
}

Status assign_value(EngineConfig& config, const std::string& key, const std::string& value) {
#define ASSIGN_PARSED(target, parser)                    \
  do {                                                   \
    auto parsed_value = parser(value);                   \
    if (!parsed_value) return parsed_value.status();     \
    target = parsed_value.value();                       \
    return Status::Ok();                                 \
  } while (false)

  if (key == "search.beam_width") ASSIGN_PARSED(config.search.beam_width, parse_unsigned<std::size_t>);
  if (key == "search.candidate_k") ASSIGN_PARSED(config.search.candidate_k, parse_unsigned<std::size_t>);
  if (key == "search.max_depth") ASSIGN_PARSED(config.search.max_depth, parse_unsigned<std::size_t>);
  if (key == "search.replay_reopen_limit") ASSIGN_PARSED(config.search.replay_reopen_limit, parse_unsigned<std::size_t>);
  if (key == "search.state_decay") ASSIGN_PARSED(config.search.state_decay, parse_float);
  if (key == "search.relation_mix") ASSIGN_PARSED(config.search.relation_mix, parse_float);
  if (key == "search.target_mix") ASSIGN_PARSED(config.search.target_mix, parse_float);

  if (key == "scoring.confidence_weight") ASSIGN_PARSED(config.scoring.confidence_weight, parse_float);
  if (key == "scoring.support_weight") ASSIGN_PARSED(config.scoring.support_weight, parse_float);
  if (key == "scoring.relation_weight") ASSIGN_PARSED(config.scoring.relation_weight, parse_float);
  if (key == "scoring.target_weight") ASSIGN_PARSED(config.scoring.target_weight, parse_float);
  if (key == "scoring.context_weight") ASSIGN_PARSED(config.scoring.context_weight, parse_float);
  if (key == "scoring.repetition_penalty") ASSIGN_PARSED(config.scoring.repetition_penalty, parse_float);

  if (key == "audit.min_exact_cases") ASSIGN_PARSED(config.audit.min_exact_cases, parse_unsigned<std::size_t>);
  if (key == "audit.max_cases") ASSIGN_PARSED(config.audit.max_cases, parse_unsigned<std::size_t>);
  if (key == "audit.max_unknown_cases") ASSIGN_PARSED(config.audit.max_unknown_cases, parse_unsigned<std::size_t>);
  if (key == "audit.causal_margin") ASSIGN_PARSED(config.audit.causal_margin, parse_float);
  if (key == "audit.min_pass_fraction") ASSIGN_PARSED(config.audit.min_pass_fraction, parse_float);
  if (key == "audit.allow_empty_clean_bootstrap") ASSIGN_PARSED(config.audit.allow_empty_clean_bootstrap, parse_bool);

  if (key == "training.batch_tokens") ASSIGN_PARSED(config.training.batch_tokens, parse_unsigned<std::size_t>);
  if (key == "training.context_radius") ASSIGN_PARSED(config.training.context_radius, parse_unsigned<std::size_t>);
  if (key == "training.evidence_cases_per_edge") ASSIGN_PARSED(config.training.evidence_cases_per_edge, parse_unsigned<std::size_t>);
  if (key == "training.auto_promote_per_batch") ASSIGN_PARSED(config.training.auto_promote_per_batch, parse_unsigned<std::size_t>);
  if (key == "training.min_support_for_promotion") ASSIGN_PARSED(config.training.min_support_for_promotion, parse_unsigned<std::uint64_t>);
  if (key == "training.min_confidence_for_promotion") ASSIGN_PARSED(config.training.min_confidence_for_promotion, parse_float);
  if (key == "training.m1_ttl_seconds") ASSIGN_PARSED(config.training.m1_ttl_seconds, parse_unsigned<std::uint64_t>);
  if (key == "training.max_m1_edges") ASSIGN_PARSED(config.training.max_m1_edges, parse_unsigned<std::size_t>);
  if (key == "training.checkpoint_every_batches") ASSIGN_PARSED(config.training.checkpoint_every_batches, parse_unsigned<std::size_t>);
  if (key == "training.reject_unknown_tokens") ASSIGN_PARSED(config.training.reject_unknown_tokens, parse_bool);

  if (key == "storage.state_dir") {
    auto parsed = parse_string(value); if (!parsed) return parsed.status();
    config.storage.state_dir = parsed.value(); return Status::Ok();
  }
  if (key == "storage.embedding_file") {
    auto parsed = parse_string(value); if (!parsed) return parsed.status();
    config.storage.embedding_file = parsed.value(); return Status::Ok();
  }
  if (key == "storage.shards") ASSIGN_PARSED(config.storage.shards, parse_unsigned<std::size_t>);
  if (key == "storage.durability") ASSIGN_PARSED(config.storage.durability, parse_durability);

  if (key == "replay.max_records") ASSIGN_PARSED(config.replay.max_records, parse_unsigned<std::size_t>);

  if (key == "runtime.parallel_lanes") ASSIGN_PARSED(config.runtime.parallel_lanes, parse_bool);
  if (key == "runtime.worker_threads") ASSIGN_PARSED(config.runtime.worker_threads, parse_unsigned<std::size_t>);
  if (key == "runtime.queue_capacity") ASSIGN_PARSED(config.runtime.queue_capacity, parse_unsigned<std::size_t>);
  if (key == "runtime.service_port") ASSIGN_PARSED(config.runtime.service_port, parse_unsigned<std::uint16_t>);

#undef ASSIGN_PARSED
  return Status(ErrorCode::invalid_argument, "unknown configuration key: " + key);
}

bool finite_nonnegative(float value) { return std::isfinite(value) && value >= 0.0F; }

}  // namespace

Status EngineConfig::validate() const {
  if (search.beam_width == 0 || search.beam_width > 4096) return Status(ErrorCode::invalid_argument, "search.beam_width must be 1..4096");
  if (search.candidate_k == 0 || search.candidate_k > 65536) return Status(ErrorCode::invalid_argument, "search.candidate_k must be 1..65536");
  if (search.max_depth == 0 || search.max_depth > 256) return Status(ErrorCode::invalid_argument, "search.max_depth must be 1..256");
  if (search.replay_reopen_limit < search.candidate_k || search.replay_reopen_limit > 1'000'000) {
    return Status(ErrorCode::invalid_argument, "search.replay_reopen_limit must be >= candidate_k and <= 1000000");
  }
  if (!finite_nonnegative(search.state_decay) || !finite_nonnegative(search.relation_mix) ||
      !finite_nonnegative(search.target_mix) || search.relation_mix + search.target_mix <= 0.0F) {
    return Status(ErrorCode::invalid_argument, "search mixing coefficients are invalid");
  }
  const float weights[] = {scoring.confidence_weight, scoring.support_weight, scoring.relation_weight,
                           scoring.target_weight, scoring.context_weight, scoring.repetition_penalty};
  for (float weight : weights) if (!finite_nonnegative(weight)) return Status(ErrorCode::invalid_argument, "scoring weights must be finite and nonnegative");
  if (audit.min_exact_cases == 0 || audit.max_cases < audit.min_exact_cases || audit.max_cases > 1024) {
    return Status(ErrorCode::invalid_argument, "audit case limits are invalid");
  }
  if (!std::isfinite(audit.causal_margin) || audit.causal_margin < 0.0F ||
      !std::isfinite(audit.min_pass_fraction) || audit.min_pass_fraction <= 0.0F || audit.min_pass_fraction > 1.0F) {
    return Status(ErrorCode::invalid_argument, "audit thresholds are invalid");
  }
  if (training.batch_tokens == 0 || training.context_radius == 0 || training.context_radius > 128 ||
      training.checkpoint_every_batches == 0 || training.max_m1_edges == 0) {
    return Status(ErrorCode::invalid_argument, "training limits are invalid");
  }
  if (!std::isfinite(training.min_confidence_for_promotion) || training.min_confidence_for_promotion < 0.0F ||
      training.min_confidence_for_promotion > 1.0F) {
    return Status(ErrorCode::invalid_argument, "promotion confidence must be in [0,1]");
  }
  if (storage.state_dir.empty() || storage.embedding_file.empty()) return Status(ErrorCode::invalid_argument, "storage paths cannot be empty");
  if (storage.shards == 0 || storage.shards > 4096) return Status(ErrorCode::invalid_argument, "storage.shards must be 1..4096");
  if (replay.max_records < audit.max_cases) return Status(ErrorCode::invalid_argument, "replay.max_records is too small for auditing");
  if (runtime.worker_threads == 0 || runtime.worker_threads > 1024 || runtime.queue_capacity == 0) {
    return Status(ErrorCode::invalid_argument, "runtime worker settings are invalid");
  }
  return Status::Ok();
}

Result<EngineConfig> EngineConfig::load(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) return Status(ErrorCode::io_error, "cannot open configuration: '" + path.string() + "'");
  EngineConfig config;
  std::string section;
  std::string line;
  std::size_t line_number = 0;
  std::unordered_set<std::string> seen;
  while (std::getline(input, line)) {
    ++line_number;
    line = trim(strip_comment(line));
    if (line.empty()) continue;
    if (line.front() == '[' && line.back() == ']') {
      section = trim(line.substr(1, line.size() - 2));
      if (section.empty()) return Status(ErrorCode::invalid_argument, "empty section at line " + std::to_string(line_number));
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) return Status(ErrorCode::invalid_argument, "missing '=' at line " + std::to_string(line_number));
    const std::string name = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    const std::string key = section.empty() ? name : section + "." + name;
    if (!seen.insert(key).second) return Status(ErrorCode::invalid_argument, "duplicate key: " + key);
    const Status status = assign_value(config, key, value);
    if (!status) return Status(status.code(), status.message() + " at line " + std::to_string(line_number));
  }
  const std::filesystem::path base = std::filesystem::absolute(path).parent_path();
  if (config.storage.state_dir.is_relative()) config.storage.state_dir = base / config.storage.state_dir;
  if (config.storage.embedding_file.is_relative()) config.storage.embedding_file = base / config.storage.embedding_file;
  const Status status = config.validate();
  if (!status) return status;
  return config;
}

}  // namespace rlm

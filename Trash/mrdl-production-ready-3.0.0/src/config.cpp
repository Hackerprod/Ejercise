#include "mrdl/config.hpp"

#include <charconv>
#include <cctype>

namespace mrdl {
namespace {

std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string key_of(std::string_view section, std::string_view key) {
    std::string result;
    result.reserve(section.size() + key.size() + 1U);
    result.append(section);
    result.push_back('.');
    result.append(key);
    return result;
}

template <typename T>
T parse_integer(const std::string& text, std::string_view section, std::string_view key) {
    T value{};
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    const auto [ptr, error] = std::from_chars(begin, end, value);
    if (error != std::errc{} || ptr != end) {
        throw Error("invalid integer for [" + std::string(section) + "]." +
                    std::string(key) + ": " + text);
    }
    return value;
}

}  // namespace

IniDocument IniDocument::load(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw Error("cannot open config: " + path.string());

    IniDocument document;
    std::string section;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        const auto comment = line.find_first_of("#;");
        if (comment != std::string::npos) line.erase(comment);
        line = trim(std::move(line));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            require(!section.empty(), "empty section in config");
            continue;
        }
        const auto equals = line.find('=');
        if (equals == std::string::npos) {
            throw Error("invalid config line " + std::to_string(line_number));
        }
        const auto key = trim(line.substr(0, equals));
        auto value = trim(line.substr(equals + 1));
        require(!key.empty(), "empty key in config");
        if (value.size() >= 2U && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        const auto full_key = key_of(section, key);
        const auto [_, inserted] = document.values_.emplace(full_key, std::move(value));
        if (!inserted) {
            throw Error("duplicate config key at line " + std::to_string(line_number) + ": " + full_key);
        }
    }
    return document;
}

std::optional<std::string> IniDocument::get(std::string_view section, std::string_view key) const {
    const auto it = values_.find(key_of(section, key));
    if (it == values_.end()) return std::nullopt;
    return it->second;
}

std::string IniDocument::get_string(std::string_view section, std::string_view key, std::string fallback) const {
    const auto value = get(section, key);
    return value ? *value : std::move(fallback);
}

std::uint32_t IniDocument::get_u32(std::string_view section, std::string_view key, std::uint32_t fallback) const {
    const auto value = get(section, key);
    return value ? parse_integer<std::uint32_t>(*value, section, key) : fallback;
}

std::uint64_t IniDocument::get_u64(std::string_view section, std::string_view key, std::uint64_t fallback) const {
    const auto value = get(section, key);
    return value ? parse_integer<std::uint64_t>(*value, section, key) : fallback;
}

std::int64_t IniDocument::get_i64(std::string_view section, std::string_view key, std::int64_t fallback) const {
    const auto value = get(section, key);
    return value ? parse_integer<std::int64_t>(*value, section, key) : fallback;
}

float IniDocument::get_float(std::string_view section, std::string_view key, float fallback) const {
    const auto value = get(section, key);
    if (!value) return fallback;
    char* end = nullptr;
    const float parsed = std::strtof(value->c_str(), &end);
    if (end == nullptr || end != value->c_str() + value->size() || !std::isfinite(parsed)) {
        throw Error("invalid floating-point value for [" + std::string(section) + "]." +
                    std::string(key) + ": " + *value);
    }
    return parsed;
}

bool IniDocument::get_bool(std::string_view section, std::string_view key, bool fallback) const {
    const auto value = get(section, key);
    if (!value) return fallback;
    std::string normalized = *value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "true" || normalized == "yes" || normalized == "1" || normalized == "on") return true;
    if (normalized == "false" || normalized == "no" || normalized == "0" || normalized == "off") return false;
    throw Error("invalid boolean for [" + std::string(section) + "]." +
                std::string(key) + ": " + *value);
}

void IniDocument::validate_known_keys(std::span<const std::string_view> allowed) const {
    std::unordered_set<std::string_view> known;
    known.reserve(allowed.size());
    known.insert(allowed.begin(), allowed.end());
    for (const auto& [key, _] : values_) {
        if (!known.contains(key)) throw Error("unknown config key: " + key);
    }
}

AppConfig AppConfig::load(const std::filesystem::path& path) {
    const auto ini = IniDocument::load(path);
    static constexpr std::array allowed_keys{
        std::string_view{"model.embedding_dim"},
        std::string_view{"model.relation_dim"},
        std::string_view{"model.max_relation_prototypes"},
        std::string_view{"model.seed"},
        std::string_view{"tokenizer.vocab_size"},
        std::string_view{"tokenizer.heavy_hitter_multiplier"},
        std::string_view{"tokenizer.lowercase"},
        std::string_view{"engine.top_k_full"},
        std::string_view{"engine.top_k_clean"},
        std::string_view{"engine.beam_full"},
        std::string_view{"engine.beam_clean"},
        std::string_view{"engine.max_rounds"},
        std::string_view{"engine.max_ports_per_node"},
        std::string_view{"engine.port_capacity"},
        std::string_view{"engine.port_similarity_threshold"},
        std::string_view{"engine.port_pressure_threshold"},
        std::string_view{"engine.branch_energy_floor"},
        std::string_view{"engine.clean_margin"},
        std::string_view{"engine.clean_health_threshold"},
        std::string_view{"engine.repetition_penalty"},
        std::string_view{"engine.cycle_penalty"},
        std::string_view{"engine.saturation_penalty"},
        std::string_view{"engine.length_log_penalty"},
        std::string_view{"engine.confidence_epsilon"},
        std::string_view{"engine.exact_pure_reuse"},
        std::string_view{"engine.parallel_lanes"},
        std::string_view{"memory.m1_ttl_seconds"},
        std::string_view{"memory.m1_confidence_cap"},
        std::string_view{"memory.promotion_min_support"},
        std::string_view{"memory.promotion_min_contexts"},
        std::string_view{"memory.promotion_min_influence"},
        std::string_view{"memory.promotion_stability_ratio"},
        std::string_view{"memory.audit_top_m"},
        std::string_view{"training.mode"},
        std::string_view{"training.context_tokens"},
        std::string_view{"training.max_source_capsules"},
        std::string_view{"training.epochs"},
        std::string_view{"training.batch_tokens"},
        std::string_view{"training.checkpoint_every_tokens"},
        std::string_view{"training.fast_learning_rate"},
        std::string_view{"training.controller_learning_rate"},
        std::string_view{"training.relation_weight_decay"},
        std::string_view{"training.negative_samples"},
        std::string_view{"training.auto_audit"},
        std::string_view{"training.trusted_source"},
        std::string_view{"persistence.model_dir"},
        std::string_view{"persistence.database"},
        std::string_view{"persistence.tokenizer"},
        std::string_view{"persistence.embeddings"},
        std::string_view{"persistence.sqlite_busy_timeout_ms"},
        std::string_view{"persistence.synchronous_full"},
        std::string_view{"runtime.threads"},
        std::string_view{"runtime.max_generation_tokens"},
        std::string_view{"runtime.temperature"},
        std::string_view{"runtime.top_p_candidates"},
    };
    ini.validate_known_keys(allowed_keys);
    AppConfig config;

    config.model.embedding_dim = ini.get_u32("model", "embedding_dim", config.model.embedding_dim);
    config.model.relation_dim = ini.get_u32("model", "relation_dim", config.model.relation_dim);
    config.model.max_relation_prototypes = ini.get_u32("model", "max_relation_prototypes", config.model.max_relation_prototypes);
    config.model.seed = ini.get_u64("model", "seed", config.model.seed);

    config.tokenizer.vocab_size = ini.get_u32("tokenizer", "vocab_size", config.tokenizer.vocab_size);
    config.tokenizer.heavy_hitter_multiplier = ini.get_u32("tokenizer", "heavy_hitter_multiplier", config.tokenizer.heavy_hitter_multiplier);
    config.tokenizer.lowercase = ini.get_bool("tokenizer", "lowercase", config.tokenizer.lowercase);

    config.engine.top_k_full = ini.get_u32("engine", "top_k_full", config.engine.top_k_full);
    config.engine.top_k_clean = ini.get_u32("engine", "top_k_clean", config.engine.top_k_clean);
    config.engine.beam_full = ini.get_u32("engine", "beam_full", config.engine.beam_full);
    config.engine.beam_clean = ini.get_u32("engine", "beam_clean", config.engine.beam_clean);
    config.engine.max_rounds = ini.get_u32("engine", "max_rounds", config.engine.max_rounds);
    config.engine.max_ports_per_node = ini.get_u32("engine", "max_ports_per_node", config.engine.max_ports_per_node);
    config.engine.port_capacity = ini.get_u32("engine", "port_capacity", config.engine.port_capacity);
    config.engine.port_similarity_threshold = ini.get_float("engine", "port_similarity_threshold", config.engine.port_similarity_threshold);
    config.engine.port_pressure_threshold = ini.get_float("engine", "port_pressure_threshold", config.engine.port_pressure_threshold);
    config.engine.branch_energy_floor = ini.get_float("engine", "branch_energy_floor", config.engine.branch_energy_floor);
    config.engine.clean_margin = ini.get_float("engine", "clean_margin", config.engine.clean_margin);
    config.engine.clean_health_threshold = ini.get_float("engine", "clean_health_threshold", config.engine.clean_health_threshold);
    config.engine.repetition_penalty = ini.get_float("engine", "repetition_penalty", config.engine.repetition_penalty);
    config.engine.cycle_penalty = ini.get_float("engine", "cycle_penalty", config.engine.cycle_penalty);
    config.engine.saturation_penalty = ini.get_float("engine", "saturation_penalty", config.engine.saturation_penalty);
    config.engine.length_log_penalty = ini.get_float("engine", "length_log_penalty", config.engine.length_log_penalty);
    config.engine.confidence_epsilon = ini.get_float("engine", "confidence_epsilon", config.engine.confidence_epsilon);
    config.engine.exact_pure_reuse = ini.get_bool("engine", "exact_pure_reuse", config.engine.exact_pure_reuse);
    config.engine.parallel_lanes = ini.get_bool("engine", "parallel_lanes", config.engine.parallel_lanes);

    config.memory.m1_ttl_seconds = ini.get_i64("memory", "m1_ttl_seconds", config.memory.m1_ttl_seconds);
    config.memory.m1_confidence_cap = ini.get_float("memory", "m1_confidence_cap", config.memory.m1_confidence_cap);
    config.memory.promotion_min_support = ini.get_u32("memory", "promotion_min_support", config.memory.promotion_min_support);
    config.memory.promotion_min_contexts = ini.get_u32("memory", "promotion_min_contexts", config.memory.promotion_min_contexts);
    config.memory.promotion_min_influence = ini.get_float("memory", "promotion_min_influence", config.memory.promotion_min_influence);
    config.memory.promotion_stability_ratio = ini.get_float("memory", "promotion_stability_ratio", config.memory.promotion_stability_ratio);
    config.memory.audit_top_m = ini.get_u32("memory", "audit_top_m", config.memory.audit_top_m);

    config.training.mode = ini.get_string("training", "mode", config.training.mode);
    config.training.context_tokens = ini.get_u32("training", "context_tokens", config.training.context_tokens);
    config.training.max_source_capsules = ini.get_u32("training", "max_source_capsules", config.training.max_source_capsules);
    config.training.epochs = ini.get_u32("training", "epochs", config.training.epochs);
    config.training.batch_tokens = ini.get_u32("training", "batch_tokens", config.training.batch_tokens);
    config.training.checkpoint_every_tokens = ini.get_u32("training", "checkpoint_every_tokens", config.training.checkpoint_every_tokens);
    config.training.fast_learning_rate = ini.get_float("training", "fast_learning_rate", config.training.fast_learning_rate);
    config.training.controller_learning_rate = ini.get_float("training", "controller_learning_rate", config.training.controller_learning_rate);
    config.training.relation_weight_decay = ini.get_float("training", "relation_weight_decay", config.training.relation_weight_decay);
    config.training.negative_samples = ini.get_u32("training", "negative_samples", config.training.negative_samples);
    config.training.auto_audit = ini.get_bool("training", "auto_audit", config.training.auto_audit);
    config.training.trusted_source = ini.get_bool("training", "trusted_source", config.training.trusted_source);

    config.persistence.model_dir = ini.get_string("persistence", "model_dir", config.persistence.model_dir.string());
    config.persistence.database = ini.get_string("persistence", "database", config.persistence.database.string());
    config.persistence.tokenizer = ini.get_string("persistence", "tokenizer", config.persistence.tokenizer.string());
    config.persistence.embeddings = ini.get_string("persistence", "embeddings", config.persistence.embeddings.string());
    config.persistence.sqlite_busy_timeout_ms = ini.get_u32("persistence", "sqlite_busy_timeout_ms", config.persistence.sqlite_busy_timeout_ms);
    config.persistence.synchronous_full = ini.get_bool("persistence", "synchronous_full", config.persistence.synchronous_full);

    config.runtime.threads = ini.get_u32("runtime", "threads", config.runtime.threads);
    config.runtime.max_generation_tokens = ini.get_u32("runtime", "max_generation_tokens", config.runtime.max_generation_tokens);
    config.runtime.temperature = ini.get_float("runtime", "temperature", config.runtime.temperature);
    config.runtime.top_p_candidates = ini.get_u32("runtime", "top_p_candidates", config.runtime.top_p_candidates);

    const auto base = std::filesystem::absolute(path).parent_path();
    auto resolve = [&base](std::filesystem::path& candidate) {
        if (candidate.is_relative()) candidate = base / candidate;
        candidate = candidate.lexically_normal();
    };
    resolve(config.persistence.model_dir);
    resolve(config.persistence.database);
    resolve(config.persistence.tokenizer);
    resolve(config.persistence.embeddings);
    config.validate();
    return config;
}

void AppConfig::save(const std::filesystem::path& path) const {
    validate();
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp." + std::to_string(unix_millis());
    std::ofstream out(temporary, std::ios::trunc);
    if (!out) throw Error("cannot create config: " + temporary);
    out << "# Generated effective MRDL configuration. Paths are absolute.\n\n"
        << "[model]\n"
        << "embedding_dim = " << model.embedding_dim << '\n'
        << "relation_dim = " << model.relation_dim << '\n'
        << "max_relation_prototypes = " << model.max_relation_prototypes << '\n'
        << "seed = " << model.seed << "\n\n"
        << "[tokenizer]\n"
        << "vocab_size = " << tokenizer.vocab_size << '\n'
        << "heavy_hitter_multiplier = " << tokenizer.heavy_hitter_multiplier << '\n'
        << "lowercase = " << (tokenizer.lowercase ? "true" : "false") << "\n\n"
        << "[engine]\n"
        << "top_k_full = " << engine.top_k_full << '\n'
        << "top_k_clean = " << engine.top_k_clean << '\n'
        << "beam_full = " << engine.beam_full << '\n'
        << "beam_clean = " << engine.beam_clean << '\n'
        << "max_rounds = " << engine.max_rounds << '\n'
        << "max_ports_per_node = " << engine.max_ports_per_node << '\n'
        << "port_capacity = " << engine.port_capacity << '\n'
        << "port_similarity_threshold = " << engine.port_similarity_threshold << '\n'
        << "port_pressure_threshold = " << engine.port_pressure_threshold << '\n'
        << "branch_energy_floor = " << engine.branch_energy_floor << '\n'
        << "clean_margin = " << engine.clean_margin << '\n'
        << "clean_health_threshold = " << engine.clean_health_threshold << '\n'
        << "repetition_penalty = " << engine.repetition_penalty << '\n'
        << "cycle_penalty = " << engine.cycle_penalty << '\n'
        << "saturation_penalty = " << engine.saturation_penalty << '\n'
        << "length_log_penalty = " << engine.length_log_penalty << '\n'
        << "confidence_epsilon = " << engine.confidence_epsilon << '\n'
        << "exact_pure_reuse = " << (engine.exact_pure_reuse ? "true" : "false") << '\n'
        << "parallel_lanes = " << (engine.parallel_lanes ? "true" : "false") << "\n\n"
        << "[memory]\n"
        << "m1_ttl_seconds = " << memory.m1_ttl_seconds << '\n'
        << "m1_confidence_cap = " << memory.m1_confidence_cap << '\n'
        << "promotion_min_support = " << memory.promotion_min_support << '\n'
        << "promotion_min_contexts = " << memory.promotion_min_contexts << '\n'
        << "promotion_min_influence = " << memory.promotion_min_influence << '\n'
        << "promotion_stability_ratio = " << memory.promotion_stability_ratio << '\n'
        << "audit_top_m = " << memory.audit_top_m << "\n\n"
        << "[training]\n"
        << "mode = " << training.mode << '\n'
        << "context_tokens = " << training.context_tokens << '\n'
        << "max_source_capsules = " << training.max_source_capsules << '\n'
        << "epochs = " << training.epochs << '\n'
        << "batch_tokens = " << training.batch_tokens << '\n'
        << "checkpoint_every_tokens = " << training.checkpoint_every_tokens << '\n'
        << "fast_learning_rate = " << training.fast_learning_rate << '\n'
        << "controller_learning_rate = " << training.controller_learning_rate << '\n'
        << "relation_weight_decay = " << training.relation_weight_decay << '\n'
        << "negative_samples = " << training.negative_samples << '\n'
        << "auto_audit = " << (training.auto_audit ? "true" : "false") << '\n'
        << "trusted_source = " << (training.trusted_source ? "true" : "false") << "\n\n"
        << "[persistence]\n"
        << "model_dir = " << persistence.model_dir.string() << '\n'
        << "database = " << persistence.database.string() << '\n'
        << "tokenizer = " << persistence.tokenizer.string() << '\n'
        << "embeddings = " << persistence.embeddings.string() << '\n'
        << "sqlite_busy_timeout_ms = " << persistence.sqlite_busy_timeout_ms << '\n'
        << "synchronous_full = " << (persistence.synchronous_full ? "true" : "false") << "\n\n"
        << "[runtime]\n"
        << "threads = " << runtime.threads << '\n'
        << "max_generation_tokens = " << runtime.max_generation_tokens << '\n'
        << "temperature = " << runtime.temperature << '\n'
        << "top_p_candidates = " << runtime.top_p_candidates << '\n';
    out.flush();
    if (!out) throw Error("failed writing config: " + temporary);
    out.close();
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    if (error) {
        std::filesystem::remove(temporary);
        throw Error("cannot install config: " + error.message());
    }
}

void AppConfig::validate() const {
    require(model.embedding_dim >= 16U && model.embedding_dim <= 4096U, "embedding_dim out of range");
    require(model.relation_dim >= 8U && model.relation_dim <= 1024U, "relation_dim out of range");
    require(model.max_relation_prototypes >= 1U && model.max_relation_prototypes <= 8U, "max_relation_prototypes must be 1..8");
    require(tokenizer.vocab_size > kFirstLearnedToken, "vocab_size must leave room for learned tokens");
    require(engine.top_k_full > 0U && engine.top_k_clean > 0U, "top_k must be positive");
    require(engine.beam_full > 0U && engine.beam_clean > 0U, "beam must be positive");
    require(engine.max_rounds >= 1U && engine.max_rounds <= 32U, "max_rounds must be 1..32");
    require(engine.max_ports_per_node >= 1U && engine.port_capacity >= 1U, "port limits must be positive");
    require(engine.clean_health_threshold >= 0.0F && engine.clean_health_threshold <= 1.0F, "clean_health_threshold must be 0..1");
    require(memory.m1_ttl_seconds > 0, "m1_ttl_seconds must be positive");
    require(memory.m1_confidence_cap > 0.0F && memory.m1_confidence_cap < 1.0F, "m1_confidence_cap must be inside (0,1)");
    require(training.mode == "B", "this production build supports training.mode=B only; Mode A is intentionally excluded from the M1/M2 path");
    require(training.context_tokens >= 1U, "context_tokens must be positive");
    require(runtime.threads >= 1U, "runtime.threads must be positive");
}

}  // namespace mrdl

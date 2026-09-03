#include "mrdl/baselines.hpp"
#include "mrdl/config.hpp"
#include "mrdl/diagnostics.hpp"
#include "mrdl/process_lock.hpp"
#include "mrdl/training.hpp"
#include "mrdl/version.hpp"

#include <charconv>
#include <iomanip>
#include <initializer_list>
#include <iostream>

namespace {

using namespace mrdl;

class Arguments final {
public:
    Arguments(int argc, char** argv, int begin) {
        for (int i = begin; i < argc; ++i) {
            std::string value = argv[i];
            if (value == "--") {
                for (++i; i < argc; ++i) positional_.emplace_back(argv[i]);
                break;
            }
            if (!value.starts_with("--")) {
                positional_.push_back(std::move(value));
                continue;
            }
            value.erase(0, 2U);
            const auto equal = value.find('=');
            if (equal != std::string::npos) {
                options_[value.substr(0, equal)].push_back(value.substr(equal + 1U));
            } else if (boolean_options().contains(value)) {
                if (!flags_.insert(value).second) throw Error("flag --" + value + " was specified more than once");
            } else if (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("--")) {
                options_[value].emplace_back(argv[++i]);
            } else {
                flags_.insert(std::move(value));
            }
        }
    }

    [[nodiscard]] bool has(std::string_view key) const {
        return flags_.contains(std::string(key)) || options_.contains(std::string(key));
    }

    [[nodiscard]] std::optional<std::string> optional(std::string_view key) const {
        const auto it = options_.find(std::string(key));
        if (it == options_.end() || it->second.empty()) return std::nullopt;
        return it->second.back();
    }

    [[nodiscard]] std::string get(std::string_view key, std::string fallback = {}) const {
        const auto value = optional(key);
        return value ? *value : std::move(fallback);
    }

    [[nodiscard]] std::string require_value(std::string_view key) const {
        const auto value = optional(key);
        if (!value) throw Error("missing required option --" + std::string(key));
        return *value;
    }

    template <typename T>
    [[nodiscard]] T number(std::string_view key, T fallback) const {
        const auto value = optional(key);
        if (!value) return fallback;
        T result{};
        const auto [end, error] = std::from_chars(value->data(), value->data() + value->size(), result);
        if (error != std::errc{} || end != value->data() + value->size()) {
            throw Error("invalid numeric value for --" + std::string(key));
        }
        return result;
    }

    [[nodiscard]] float floating(std::string_view key, float fallback) const {
        const auto value = optional(key);
        if (!value) return fallback;
        std::size_t consumed = 0U;
        const float result = std::stof(*value, &consumed);
        if (consumed != value->size() || !std::isfinite(result)) {
            throw Error("invalid floating-point value for --" + std::string(key));
        }
        return result;
    }

    [[nodiscard]] const std::vector<std::string>& positional() const noexcept { return positional_; }

    void validate(std::initializer_list<std::string_view> value_options,
                  std::initializer_list<std::string_view> flag_options,
                  bool allow_positional = false) const {
        std::unordered_set<std::string_view> allowed_values{
            "config"
        };
        std::unordered_set<std::string_view> allowed_flags{
            "json", "wait-lock", "help"
        };
        allowed_values.insert(value_options.begin(), value_options.end());
        allowed_flags.insert(flag_options.begin(), flag_options.end());

        for (const auto& [key, values] : options_) {
            if (allowed_flags.contains(key)) {
                throw Error("flag --" + key + " does not take a value");
            }
            if (!allowed_values.contains(key)) throw Error("unknown option --" + key);
            if (values.size() != 1U) throw Error("option --" + key + " was specified more than once");
            if (values.front().empty()) throw Error("option --" + key + " requires a non-empty value");
        }
        for (const auto& key : flags_) {
            if (allowed_values.contains(key)) throw Error("option --" + key + " requires a value");
            if (!allowed_flags.contains(key)) throw Error("unknown flag --" + key);
        }
        if (!allow_positional && !positional_.empty()) {
            throw Error("unexpected positional argument: " + positional_.front());
        }
    }

private:
    [[nodiscard]] static const std::unordered_set<std::string>& boolean_options() {
        static const std::unordered_set<std::string> values{
            "help", "json", "wait-lock", "allow-unprepared", "force", "quiet", "bos", "eos"
        };
        return values;
    }

    std::unordered_map<std::string, std::vector<std::string>> options_;
    std::unordered_set<std::string> flags_;
    std::vector<std::string> positional_;
};

std::string escape_json(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 16U);
    for (const char value_byte : value) {
        const auto c = static_cast<unsigned char>(value_byte);
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    result += "\\u00";
                    result.push_back(digits[(c >> 4U) & 0xFU]);
                    result.push_back(digits[c & 0xFU]);
                } else result.push_back(static_cast<char>(c));
        }
    }
    return result;
}

std::string join_positional(const Arguments& arguments) {
    std::ostringstream output;
    for (std::size_t i = 0; i < arguments.positional().size(); ++i) {
        if (i != 0U) output << ' ';
        output << arguments.positional()[i];
    }
    return output.str();
}

std::filesystem::path config_path(const Arguments& arguments) {
    return std::filesystem::absolute(arguments.get("config", "config/vps.ini")).lexically_normal();
}

ProcessLock lock_model(const AppConfig& config, LockMode mode, const Arguments& arguments) {
    return ProcessLock(config.persistence.model_dir, mode, arguments.has("wait-lock"));
}

std::uint64_t file_hash(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw Error("cannot hash file: " + path.string());
    std::array<std::byte, 1U << 16U> buffer{};
    std::uint64_t hash = 1469598103934665603ULL;
    while (stream) {
        stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0) hash = hash_bytes(std::span<const std::byte>(buffer.data(), static_cast<std::size_t>(count)), hash);
    }
    return hash;
}

void remove_model_files(const AppConfig& config) {
    const std::array<std::filesystem::path, 7> files{
        config.persistence.database,
        std::filesystem::path(config.persistence.database.string() + "-wal"),
        std::filesystem::path(config.persistence.database.string() + "-shm"),
        config.persistence.tokenizer,
        config.persistence.embeddings,
        config.persistence.model_dir / "config.effective.ini",
        config.persistence.model_dir / "manifest.json"
    };
    for (const auto& file : files) {
        std::error_code error;
        std::filesystem::remove(file, error);
        if (error) throw Error("cannot remove " + file.string() + ": " + error.message());
    }
}

std::string train_json(const TrainStats& stats, std::string_view event = "train_complete") {
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"event\":\"" << event << "\",\"tokens\":" << stats.tokens
        << ",\"loss\":" << stats.average_loss()
        << ",\"perplexity\":" << stats.perplexity()
        << ",\"accuracy\":" << stats.accuracy()
        << ",\"tokens_per_second\":" << stats.tokens_per_second()
        << ",\"m1_writes\":" << stats.m1_writes
        << ",\"promotions\":" << stats.promotions
        << ",\"audits_deferred\":" << stats.audits_deferred
        << ",\"audits_rejected\":" << stats.audits_rejected
        << ",\"clean_empty\":" << stats.clean_empty
        << ",\"elapsed_seconds\":" << stats.elapsed_seconds << '}';
    return out.str();
}

std::string eval_json(const EvalStats& stats) {
    const auto divisor = static_cast<double>(std::max<std::uint64_t>(stats.tokens, 1U));
    std::ostringstream out;
    out << std::setprecision(10)
        << "{\"tokens\":" << stats.tokens
        << ",\"full_loss\":" << stats.full_nll / divisor
        << ",\"full_perplexity\":" << std::exp(std::min(stats.full_nll / divisor, 80.0))
        << ",\"full_accuracy\":" << static_cast<double>(stats.correct_full) / divisor
        << ",\"clean_loss\":" << stats.clean_nll / divisor
        << ",\"clean_perplexity\":" << std::exp(std::min(stats.clean_nll / divisor, 80.0))
        << ",\"clean_accuracy\":" << static_cast<double>(stats.correct_clean) / divisor
        << ",\"clean_empty\":" << stats.clean_empty
        << ",\"clean_empty_ratio\":" << static_cast<double>(stats.clean_empty) / divisor
        << ",\"tokens_per_second\":" << (stats.elapsed_seconds > 0.0 ? divisor / stats.elapsed_seconds : 0.0)
        << ",\"elapsed_seconds\":" << stats.elapsed_seconds << '}';
    return out.str();
}

void print_help() {
    std::cout << "MRDL " MRDL_VERSION_STRING " — sparse relational language runtime\n\n" << R"HELP(Usage:
  mrdl <command> [options]

Core lifecycle:
  doctor      Validate VPS resources and all persisted model artifacts.
  prepare     Build a reversible tokenizer and frozen Q8 embeddings.
  train       Train fast M1 relational memory (Mode B) and audit promotions.
  eval        Evaluate FULL and CLEAN lanes separately.
  generate    Autoregressive generation with per-token epistemic certification.
  audit       Audit eligible M1 records and promote stable causal relations.
  gc          Expire unpinned M1 records and collect rejected/unreplayable data.
  backup      Create a consistent SQLite + tokenizer + embedding backup.

Inspection and validation:
  inspect     Model, graph, controller and escrow statistics.
  relation    Decode one relation by --id.
  neighbors   Show physical FULL or CLEAN adjacency for --text.
  tokenize    Encode/decode text with the prepared tokenizer.
  baseline    Train/evaluate interpolated n-gram baselines on the same tokens.
  checkpoint  Flush controller/roles and truncate the WAL.
  version     Print the build version.

Common options:
  --config <path>       INI configuration (default: config/vps.ini)
  --json                Machine-readable output
  --wait-lock           Wait for the model lock instead of failing immediately

Run `mrdl <command> --help` for command-specific examples documented in README.md.
)HELP";
}

EmbeddingInit parse_embedding_mode(std::string_view mode) {
    if (mode == "random") return EmbeddingInit::Random;
    if (mode == "random-indexing") return EmbeddingInit::RandomIndexing;
    if (mode == "external") return EmbeddingInit::ExternalFloat32;
    throw Error("--embeddings must be random, random-indexing or external");
}

std::vector<std::uint32_t> parse_orders(std::string_view input) {
    std::vector<std::uint32_t> result;
    std::size_t begin = 0U;
    while (begin <= input.size()) {
        const auto comma = input.find(',', begin);
        const auto part = input.substr(begin, comma == std::string_view::npos ? input.size() - begin : comma - begin);
        std::uint32_t value{};
        const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), value);
        if (error != std::errc{} || end != part.data() + part.size() || value < 1U || value > 5U) {
            throw Error("--orders must contain comma-separated values in 1..5");
        }
        result.push_back(value);
        if (comma == std::string_view::npos) break;
        begin = comma + 1U;
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

void train_baselines(const HybridTokenizer& tokenizer,
                     const std::filesystem::path& corpus,
                     std::vector<NGramBaseline>& baselines) {
    std::ifstream stream(corpus);
    if (!stream) throw Error("cannot open baseline training corpus: " + corpus.string());
    std::string line;
    while (std::getline(stream, line)) {
        line.push_back('\n');
        const auto tokens = tokenizer.encode(line, true, true);
        std::deque<TokenId> context;
        context.push_back(kBosToken);
        for (std::size_t i = 1U; i < tokens.size(); ++i) {
            const std::vector<TokenId> view(context.begin(), context.end());
            for (auto& baseline : baselines) baseline.observe(view, tokens[i]);
            context.push_back(tokens[i]);
            while (context.size() > 4U) context.pop_front();
        }
    }
}

std::vector<BaselineMetrics> evaluate_baselines(const HybridTokenizer& tokenizer,
                                                const std::filesystem::path& corpus,
                                                const std::vector<NGramBaseline>& baselines,
                                                std::uint64_t max_tokens) {
    std::vector<BaselineMetrics> metrics(baselines.size());
    std::ifstream stream(corpus);
    if (!stream) throw Error("cannot open baseline evaluation corpus: " + corpus.string());
    const auto started = std::chrono::steady_clock::now();
    std::string line;
    bool stop = false;
    while (!stop && std::getline(stream, line)) {
        line.push_back('\n');
        const auto tokens = tokenizer.encode(line, true, true);
        std::deque<TokenId> context;
        context.push_back(kBosToken);
        for (std::size_t i = 1U; i < tokens.size(); ++i) {
            const std::vector<TokenId> view(context.begin(), context.end());
            for (std::size_t index = 0; index < baselines.size(); ++index) {
                metrics[index].negative_log_likelihood += baselines[index].negative_log_likelihood(view, tokens[i]);
                if (baselines[index].predict(view) == tokens[i]) ++metrics[index].correct;
                ++metrics[index].tokens;
            }
            context.push_back(tokens[i]);
            while (context.size() > 4U) context.pop_front();
            if (max_tokens != 0U && metrics.front().tokens >= max_tokens) { stop = true; break; }
        }
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    for (auto& metric : metrics) metric.elapsed_seconds = elapsed;
    return metrics;
}

void validate_command_arguments(std::string_view command, const Arguments& arguments) {
    if (command == "doctor") arguments.validate({}, {"allow-unprepared"});
    else if (command == "prepare") arguments.validate({"corpus", "embeddings", "external"}, {"force"});
    else if (command == "train") arguments.validate({"corpus"}, {"quiet"});
    else if (command == "eval") arguments.validate({"corpus", "max-tokens"}, {});
    else if (command == "generate") arguments.validate({"prompt", "max-tokens", "temperature", "seed"}, {}, true);
    else if (command == "audit") arguments.validate({"max"}, {});
    else if (command == "gc" || command == "checkpoint" || command == "inspect") arguments.validate({}, {});
    else if (command == "backup") arguments.validate({"output"}, {});
    else if (command == "relation") arguments.validate({"id"}, {});
    else if (command == "neighbors") arguments.validate({"text", "lane", "limit"}, {});
    else if (command == "tokenize") arguments.validate({"text"}, {"bos", "eos"}, true);
    else if (command == "baseline") {
        arguments.validate({"train-corpus", "eval-corpus", "orders", "max-tokens", "output-dir"}, {});
    } else if (command == "version") arguments.validate({}, {});
    else throw Error("unknown command: " + std::string(command));
}

int execute(std::string_view command, const Arguments& arguments) {
    const bool json = arguments.has("json");
    if (command == "help" || arguments.has("help")) {
        print_help();
        return 0;
    }
    validate_command_arguments(command, arguments);
    if (command == "version") {
        std::cout << (json ? "{\"version\":\"" MRDL_VERSION_STRING "\"}\n" : "MRDL " MRDL_VERSION_STRING "\n");
        return 0;
    }

    const auto path = config_path(arguments);
    const auto config = AppConfig::load(path);

    if (command == "doctor") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        const auto report = run_doctor(config, !arguments.has("allow-unprepared"));
        std::cout << (json ? report.json() + "\n" : report.text());
        return report.healthy() ? 0 : 1;
    }

    if (command == "prepare") {
        auto lock = lock_model(config, LockMode::Exclusive, arguments);
        if (arguments.has("force")) remove_model_files(config);
        const auto corpus = std::filesystem::absolute(arguments.require_value("corpus"));
        const auto mode = parse_embedding_mode(arguments.get("embeddings", "random-indexing"));
        std::optional<std::filesystem::path> external;
        if (mode == EmbeddingInit::ExternalFloat32) external = std::filesystem::absolute(arguments.require_value("external"));
        ModelRuntime::prepare(config, corpus, mode, external);
        const auto report = run_doctor(config, true);
        if (!report.healthy()) throw Error("prepared artifacts failed doctor validation");
        if (json) {
            std::cout << "{\"prepared\":true,\"model_dir\":\"" << escape_json(config.persistence.model_dir.string())
                      << "\",\"doctor\":" << report.json() << "}\n";
        } else {
            std::cout << "Prepared model at " << config.persistence.model_dir << '\n' << report.text();
        }
        return 0;
    }

    if (command == "train") {
        auto lock = lock_model(config, LockMode::Exclusive, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto corpus = std::filesystem::absolute(arguments.require_value("corpus"));
        const bool progress_enabled = !arguments.has("quiet");
        const auto progress = [&](const TrainStats& stats) {
            if (!progress_enabled) return;
            if (json) std::cerr << train_json(stats, "train_progress") << '\n';
            else std::cerr << "tokens=" << stats.tokens << " loss=" << stats.average_loss()
                           << " ppl=" << stats.perplexity() << " t/s=" << stats.tokens_per_second()
                           << " M1=" << stats.m1_writes << " M2+=" << stats.promotions << '\n';
        };
        const auto stats = runtime->train(corpus, progress);
        if (json) std::cout << train_json(stats) << '\n';
        else std::cout << "Training complete\n" << train_json(stats) << '\n';
        return 0;
    }

    if (command == "eval") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto corpus = std::filesystem::absolute(arguments.require_value("corpus"));
        const auto stats = runtime->evaluate(corpus, arguments.number<std::uint64_t>("max-tokens", 0U));
        std::cout << eval_json(stats) << '\n';
        return 0;
    }

    if (command == "generate") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        auto runtime = ModelRuntime::open(config);
        std::string prompt = arguments.get("prompt");
        if (prompt.empty()) prompt = join_positional(arguments);
        if (prompt.empty()) throw Error("generate requires --prompt or positional prompt text");
        const auto result = runtime->generate(prompt,
            arguments.number<std::uint32_t>("max-tokens", 0U),
            arguments.floating("temperature", -1.0F),
            arguments.number<std::uint64_t>("seed", 0U));
        std::array<std::uint64_t, 4> counts{};
        for (const auto certification : result.certifications) ++counts[static_cast<std::size_t>(certification)];
        if (json) {
            std::cout << "{\"text\":\"" << escape_json(result.text) << "\",\"tokens\":[";
            for (std::size_t i = 0; i < result.generated.size(); ++i) {
                if (i != 0U) std::cout << ',';
                std::cout << result.generated[i];
            }
            std::cout << "],\"certification_counts\":{\"clean\":" << counts[0]
                      << ",\"provisional\":" << counts[1] << ",\"fragile\":" << counts[2]
                      << ",\"empty\":" << counts[3] << "},\"tokens_per_second\":"
                      << result.tokens_per_second << "}\n";
        } else {
            std::cout << result.text << "\n\ncertification: clean=" << counts[0]
                      << " provisional=" << counts[1] << " fragile=" << counts[2]
                      << " empty=" << counts[3] << " t/s=" << result.tokens_per_second << '\n';
        }
        return 0;
    }

    if (command == "audit") {
        auto lock = lock_model(config, LockMode::Exclusive, arguments);
        auto runtime = ModelRuntime::open(config);
        TrainStats aggregate;
        const auto promoted = runtime->audit_pending(arguments.number<std::size_t>("max", 0U), &aggregate);
        runtime->checkpoint();
        std::cout << "{\"promoted\":" << promoted << ",\"deferred\":" << aggregate.audits_deferred
                  << ",\"rejected_or_unreplayable\":" << aggregate.audits_rejected << "}\n";
        return 0;
    }

    if (command == "gc") {
        auto lock = lock_model(config, LockMode::Exclusive, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto removed = runtime->garbage_collect();
        runtime->checkpoint();
        std::cout << "{\"removed\":" << removed << "}\n";
        return 0;
    }

    if (command == "checkpoint") {
        auto lock = lock_model(config, LockMode::Exclusive, arguments);
        auto runtime = ModelRuntime::open(config);
        runtime->checkpoint();
        std::cout << (json ? "{\"checkpointed\":true}\n" : "Checkpoint complete\n");
        return 0;
    }

    if (command == "backup") {
        auto lock = lock_model(config, LockMode::Exclusive, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto destination = std::filesystem::absolute(arguments.require_value("output"));
        runtime->backup(destination);
        const auto effective = config.persistence.model_dir / "config.effective.ini";
        if (std::filesystem::exists(effective)) {
            std::filesystem::copy_file(effective, destination / "config.effective.ini",
                                       std::filesystem::copy_options::overwrite_existing);
        }
        std::ofstream manifest(destination / "manifest.json", std::ios::trunc);
        if (!manifest) throw Error("cannot create backup manifest");
        manifest << "{\"format\":1,\"runtime_version\":\"" MRDL_VERSION_STRING
                 << "\",\"model_family\":\"MRDL-3-production-core\",\"created_at_ms\":"
                 << unix_millis() << ",\"files\":[";
        std::vector<std::filesystem::path> files{destination / "mrdl.db",
            destination / config.persistence.tokenizer.filename(),
            destination / config.persistence.embeddings.filename()};
        if (std::filesystem::exists(destination / "config.effective.ini")) {
            files.push_back(destination / "config.effective.ini");
        }
        for (std::size_t i = 0; i < files.size(); ++i) {
            if (i != 0U) manifest << ',';
            manifest << "{\"name\":\"" << escape_json(files[i].filename().string()) << "\",\"size\":"
                     << std::filesystem::file_size(files[i]) << ",\"hash64\":" << file_hash(files[i]) << '}';
        }
        manifest << "]}\n";
        manifest.flush();
        if (!manifest) throw Error("failed writing backup manifest");
        std::cout << "{\"backup\":\"" << escape_json(destination.string()) << "\"}\n";
        return 0;
    }

    if (command == "inspect") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto graph = runtime->graph().stats();
        const auto escrow = runtime->promotions().stats();
        const auto controller = runtime->controller().snapshot();
        std::string database_diagnostic;
        const bool database_ok = runtime->integrity_check(&database_diagnostic);
        std::cout << "{\"vocabulary\":" << runtime->tokenizer().size()
                  << ",\"embedding_dim\":" << runtime->embeddings().dimension()
                  << ",\"embedding_hash\":" << runtime->embeddings().content_hash()
                  << ",\"controller_version\":" << controller.version
                  << ",\"relations_total\":" << graph.relations_total
                  << ",\"relations_m1\":" << graph.relations_m1
                  << ",\"relations_m2\":" << graph.relations_m2
                  << ",\"full_index_entries\":" << graph.full_index_entries
                  << ",\"clean_index_entries\":" << graph.clean_index_entries
                  << ",\"escrow_total\":" << escrow.total
                  << ",\"escrow_pinned\":" << escrow.pinned
                  << ",\"escrow_observations\":" << escrow.observations
                  << ",\"database_ok\":" << (database_ok ? "true" : "false")
                  << ",\"database_diagnostic\":\"" << escape_json(database_diagnostic) << "\"}\n";
        return database_ok ? 0 : 1;
    }

    if (command == "relation") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto id = arguments.number<RelationId>("id", 0U);
        if (id == 0U) throw Error("relation requires --id");
        const auto relation = runtime->graph().get(id);
        if (!relation) throw Error("relation not found");
        std::cout << "{\"id\":" << relation->id << ",\"source\":" << relation->source
                  << ",\"source_piece\":\"" << escape_json(runtime->tokenizer().token(relation->source))
                  << "\",\"destination\":" << relation->destination << ",\"destination_piece\":\""
                  << escape_json(runtime->tokenizer().token(relation->destination)) << "\",\"prototype\":"
                  << static_cast<unsigned>(relation->prototype) << ",\"level\":\"" << to_string(relation->level)
                  << "\",\"support\":" << relation->support << ",\"confidence\":" << relation->confidence
                  << ",\"version\":" << relation->version << ",\"state\":\"" << to_string(relation->escrow_state)
                  << "\",\"transform_hash\":" << relation->transform.full_hash() << "}\n";
        return 0;
    }

    if (command == "neighbors") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        auto runtime = ModelRuntime::open(config);
        const auto text = arguments.require_value("text");
        const auto encoded = runtime->tokenizer().encode(text);
        if (encoded.empty()) throw Error("--text produced no tokens");
        const TokenId source = encoded.back();
        const auto lane_name = arguments.get("lane", "full");
        const Lane lane = lane_name == "clean" ? Lane::Clean : (lane_name == "full" ? Lane::Full :
            throw Error("--lane must be full or clean"));
        const auto edges = runtime->graph().outgoing(lane, source, arguments.number<std::size_t>("limit", 32U));
        std::cout << "{\"source\":" << source << ",\"piece\":\""
                  << escape_json(runtime->tokenizer().token(source)) << "\",\"lane\":\""
                  << to_string(lane) << "\",\"edges\":[";
        for (std::size_t i = 0; i < edges.size(); ++i) {
            if (i != 0U) std::cout << ',';
            std::cout << "{\"id\":" << edges[i]->id << ",\"to\":" << edges[i]->destination
                      << ",\"piece\":\"" << escape_json(runtime->tokenizer().token(edges[i]->destination))
                      << "\",\"confidence\":" << edges[i]->confidence << ",\"support\":"
                      << edges[i]->support << ",\"level\":\"" << to_string(edges[i]->level) << "\"}";
        }
        std::cout << "]}\n";
        return 0;
    }

    if (command == "tokenize") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        const auto tokenizer = HybridTokenizer::load(config.persistence.tokenizer);
        std::string text = arguments.get("text");
        if (text.empty()) text = join_positional(arguments);
        const auto ids = tokenizer.encode(text, arguments.has("bos"), arguments.has("eos"));
        std::cout << "{\"ids\":[";
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i != 0U) std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << "],\"roundtrip\":\"" << escape_json(tokenizer.decode(ids)) << "\"}\n";
        return 0;
    }

    if (command == "baseline") {
        auto lock = lock_model(config, LockMode::Shared, arguments);
        const auto tokenizer = HybridTokenizer::load(config.persistence.tokenizer);
        const auto orders = parse_orders(arguments.get("orders", "1,2,3"));
        std::vector<NGramBaseline> baselines;
        baselines.reserve(orders.size());
        for (const auto order : orders) baselines.emplace_back(order);
        const auto train_corpus = std::filesystem::absolute(arguments.require_value("train-corpus"));
        const auto eval_corpus = std::filesystem::absolute(arguments.get("eval-corpus", train_corpus.string()));
        train_baselines(tokenizer, train_corpus, baselines);
        const auto metrics = evaluate_baselines(tokenizer, eval_corpus, baselines,
                                                arguments.number<std::uint64_t>("max-tokens", 0U));
        if (const auto output = arguments.optional("output-dir")) {
            const auto directory = std::filesystem::absolute(*output);
            std::filesystem::create_directories(directory);
            for (std::size_t i = 0; i < baselines.size(); ++i) {
                baselines[i].save(directory / ("ngram-" + std::to_string(orders[i]) + ".mrdlngr"));
            }
        }
        std::cout << "{\"baselines\":[";
        for (std::size_t i = 0; i < baselines.size(); ++i) {
            if (i != 0U) std::cout << ',';
            std::cout << "{\"order\":" << orders[i] << ",\"tokens\":" << metrics[i].tokens
                      << ",\"loss\":" << metrics[i].average_loss() << ",\"perplexity\":"
                      << metrics[i].perplexity() << ",\"accuracy\":" << metrics[i].accuracy()
                      << ",\"tokens_per_second\":" << metrics[i].tokens_per_second()
                      << ",\"contexts\":" << baselines[i].context_count() << '}';
        }
        std::cout << "]}\n";
        return 0;
    }

    throw Error("unknown command: " + std::string(command));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_help();
            return 2;
        }
        const std::string command = argv[1];
        const Arguments arguments(argc, argv, 2);
        return execute(command, arguments);
    } catch (const mrdl::Error& error) {
        std::cerr << "mrdl: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "mrdl: unexpected failure: " << error.what() << '\n';
        return 3;
    }
}

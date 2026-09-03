#include "mrdl/controller.hpp"
#include "mrdl/engine.hpp"
#include "mrdl/graph.hpp"
#include "mrdl/relation.hpp"
#include "mrdl/replay.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {

using namespace mrdl;

class DenseEmbeddingStore final : public IEmbeddingStore {
public:
    DenseEmbeddingStore(std::size_t token_count, std::size_t dimension)
        : token_count_(token_count), dimension_(dimension), values_(token_count * dimension) {
        for (std::size_t token = 0; token < token_count_; ++token) {
            auto row = std::span<float>(values_).subspan(token * dimension_, dimension_);
            std::uint64_t state = mix64(token + 0x454d424544ULL);
            for (std::size_t column = 0; column < dimension_; ++column) {
                state = mix64(state + column);
                row[column] = static_cast<float>(static_cast<std::int32_t>(state >> 32U)) /
                              static_cast<float>(std::numeric_limits<std::int32_t>::max());
            }
            normalize_in_place(row);
        }
    }
    std::size_t token_count() const noexcept override { return token_count_; }
    std::size_t dimension() const noexcept override { return dimension_; }
    void dequantize(TokenId id, std::span<float> output) const override {
        require(id < token_count_ && output.size() == dimension_, "embedding access out of range");
        const auto source = std::span<const float>(values_).subspan(static_cast<std::size_t>(id) * dimension_, dimension_);
        std::copy(source.begin(), source.end(), output.begin());
    }
    float dot_row(TokenId id, std::span<const float> vector) const override {
        return dot(std::span<const float>(values_).subspan(static_cast<std::size_t>(id) * dimension_, dimension_), vector);
    }
    float cosine_row(TokenId id, std::span<const float> vector) const override {
        return cosine(std::span<const float>(values_).subspan(static_cast<std::size_t>(id) * dimension_, dimension_), vector);
    }
private:
    std::size_t token_count_;
    std::size_t dimension_;
    std::vector<float> values_;
};

struct Options {
    bool quick{false};
    bool json{false};
    std::uint32_t iterations{0};
    std::string suite{"all"};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view value = argv[i];
        if (value == "--quick") options.quick = true;
        else if (value == "--json") options.json = true;
        else if (value == "--help" || value == "-h") {
            std::cout << "Usage: mrdl_bench [--quick] [--json] [--iterations N] [--suite all|dual|monomial]\n";
            std::exit(0);
        } else if (value == "--iterations" && i + 1 < argc) {
            const std::string_view number = argv[++i];
            const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), options.iterations);
            require(error == std::errc{} && end == number.data() + number.size(), "invalid --iterations");
        } else if (value.starts_with("--iterations=")) {
            const auto number = value.substr(13U);
            const auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), options.iterations);
            require(error == std::errc{} && end == number.data() + number.size(), "invalid --iterations");
        } else if (value == "--suite" && i + 1 < argc) options.suite = argv[++i];
        else if (value.starts_with("--suite=")) options.suite = value.substr(8U);
        else throw Error("unknown benchmark option: " + std::string(value));
    }
    require(options.suite == "all" || options.suite == "dual" || options.suite == "monomial",
            "--suite must be all, dual, or monomial");
    return options;
}

RelationRecord make_relation(RelationId id,
                             TokenId source,
                             TokenId destination,
                             MemoryLevel level,
                             std::uint8_t prototype,
                             std::size_t embedding_dimension,
                             std::size_t relation_dimension,
                             std::uint64_t seed) {
    RelationRecord relation;
    relation.id = id;
    relation.source = source;
    relation.destination = destination;
    relation.prototype = prototype;
    relation.level = level;
    relation.lanes = LaneMask::from_level(level);
    relation.support = 8U + id % 64U;
    relation.confidence = level == MemoryLevel::M2 ? 0.78F : 0.42F;
    relation.version = 1U;
    relation.created_at_ms = unix_millis();
    relation.updated_at_ms = relation.created_at_ms;
    relation.expires_at_ms = level == MemoryLevel::M1 ? relation.created_at_ms + 3600000 : 0;
    relation.escrow_state = level == MemoryLevel::M2 ? EscrowState::Promoted : EscrowState::Active;
    relation.transform = MonomialOperator::seeded(embedding_dimension, hash_combine(seed, id));
    relation.relation = RelationVector(relation_dimension);
    auto& values = relation.relation.values();
    std::uint64_t state = hash_combine(seed, id);
    for (float& value : values) {
        state = mix64(state);
        value = (state & 1ULL) != 0ULL ? 0.25F : -0.25F;
    }
    return relation;
}

struct SyntheticGraph {
    std::unique_ptr<GraphStore> graph;
    TokenId root{0};
    std::uint64_t m1_edges{0};
    std::uint64_t total_edges{0};
};

SyntheticGraph build_layered_graph(std::uint32_t branching,
                                   std::uint32_t beam,
                                   std::uint32_t depth,
                                   float m1_fraction,
                                   std::size_t embedding_dimension,
                                   std::size_t relation_dimension,
                                   std::uint64_t seed) {
    SyntheticGraph result;
    result.graph = std::make_unique<GraphStore>();
    const std::uint32_t width = std::max<std::uint32_t>({branching, beam * 2U, 8U});
    result.root = 300U;
    RelationId relation_id = 1U;
    for (std::uint32_t layer = 0; layer < depth; ++layer) {
        const std::uint32_t source_count = layer == 0U ? 1U : width;
        for (std::uint32_t source_index = 0; source_index < source_count; ++source_index) {
            const TokenId source = layer == 0U ? result.root :
                result.root + layer * width + source_index;
            for (std::uint32_t edge = 0; edge < branching; ++edge) {
                const TokenId destination = result.root + (layer + 1U) * width +
                    ((source_index * 5U + edge * 7U + layer) % width);
                const std::uint64_t draw = mix64(hash_combine(seed, relation_id)) % 1000000ULL;
                const bool is_m1 = draw < static_cast<std::uint64_t>(m1_fraction * 1000000.0F);
                const auto level = is_m1 ? MemoryLevel::M1 : MemoryLevel::M2;
                result.graph->load_relation(make_relation(relation_id, source, destination, level,
                    static_cast<std::uint8_t>(edge), embedding_dimension, relation_dimension, seed));
                ++result.total_edges;
                if (is_m1) ++result.m1_edges;
                ++relation_id;
            }
        }
    }
    return result;
}

struct LatencySummary {
    double mean_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
};

LatencySummary summarize(std::vector<double> values) {
    require(!values.empty(), "cannot summarize empty latency set");
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double p) {
        const auto index = static_cast<std::size_t>(std::floor(p * static_cast<double>(values.size() - 1U)));
        return values[index];
    };
    return LatencySummary{
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size()),
        percentile(0.50), percentile(0.95)
    };
}

template <typename Function>
std::pair<LatencySummary, LanePrediction> time_lane(Function&& function,
                                                    std::uint32_t iterations) {
    LanePrediction last;
    for (std::uint32_t warmup = 0; warmup < 2U; ++warmup) last = function(0x1000U + warmup);
    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        last = function(0x2000U + iteration);
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        latencies.push_back(elapsed);
    }
    return {summarize(std::move(latencies)), std::move(last)};
}

template <typename Function>
std::pair<LatencySummary, DualPrediction> time_dual(Function&& function,
                                                    std::uint32_t iterations) {
    DualPrediction last;
    for (std::uint32_t warmup = 0; warmup < 2U; ++warmup) last = function(0x3000U + warmup);
    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        last = function(0x4000U + iteration);
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        latencies.push_back(elapsed);
    }
    return {summarize(std::move(latencies)), std::move(last)};
}

struct DualRow {
    std::uint32_t branching{0};
    std::uint32_t beam{0};
    std::uint32_t depth{0};
    float requested_m1_fraction{0.0F};
    float actual_m1_fraction{0.0F};
    std::uint32_t iterations{0};
    LatencySummary full;
    LatencySummary clean;
    LatencySummary dual_isolated;
    LatencySummary dual_parallel_reuse;
    float isolated_runtime_ratio{0.0F};
    float parallel_runtime_ratio{0.0F};
    float operator_ratio{0.0F};
    std::uint64_t full_ops{0};
    std::uint64_t clean_ops{0};
    std::uint64_t active_state_full{0};
    std::uint64_t active_state_clean{0};
    std::uint64_t replay_storage_entries{0};
    float clean_health{0.0F};
    bool clean_degenerate{false};
    bool clean_empty{false};
};

DualRow benchmark_dual(std::uint32_t branching,
                       std::uint32_t beam,
                       std::uint32_t depth,
                       float m1_fraction,
                       std::uint32_t iterations) {
    constexpr std::size_t embedding_dimension = 64U;
    constexpr std::size_t relation_dimension = 32U;
    const std::uint64_t seed = hash_combine(hash_combine(branching, beam),
                                             static_cast<std::uint64_t>(m1_fraction * 1000.0F));
    const std::uint32_t width = std::max<std::uint32_t>({branching, beam * 2U, 8U});
    const std::size_t token_count = 301U + static_cast<std::size_t>(depth + 1U) * width + 16U;
    DenseEmbeddingStore embeddings(token_count, embedding_dimension);
    auto synthetic = build_layered_graph(branching, beam, depth, m1_fraction,
                                         embedding_dimension, relation_dimension, seed);
    Controller controller;
    RoleInducer roles;
    EngineConfig config;
    config.top_k_full = branching;
    config.top_k_clean = branching;
    config.beam_full = beam;
    config.beam_clean = beam;
    config.max_rounds = depth;
    config.max_ports_per_node = std::max<std::uint32_t>(8U, branching);
    config.port_capacity = std::max<std::uint32_t>(8U, beam);
    config.port_similarity_threshold = 0.65F;
    config.port_pressure_threshold = 0.0F;
    config.branch_energy_floor = 1.0e-8F;
    config.parallel_lanes = false;
    config.exact_pure_reuse = false;
    const std::array<TokenId, 1> context{synthetic.root};

    LaneEngine full_engine(Lane::Full, *synthetic.graph, embeddings, controller, roles, config);
    LaneEngine clean_engine(Lane::Clean, *synthetic.graph, embeddings, controller, roles, config);
    auto [full_latency, full_prediction] = time_lane(
        [&](std::uint64_t run_seed) { return full_engine.predict(context, nullptr, nullptr, run_seed, run_seed); }, iterations);
    auto [clean_latency, clean_prediction] = time_lane(
        [&](std::uint64_t run_seed) { return clean_engine.predict(context, nullptr, nullptr, run_seed, run_seed); }, iterations);

    auto recorder = std::make_shared<ReplayRecorder>();
    DualLaneEngine dual_isolated(*synthetic.graph, embeddings, controller, roles, config, recorder);
    auto [isolated_latency, isolated_prediction] = time_dual(
        [&](std::uint64_t run_seed) { return dual_isolated.predict(context, false, run_seed); }, iterations);

    config.parallel_lanes = true;
    config.exact_pure_reuse = true;
    DualLaneEngine dual_parallel(*synthetic.graph, embeddings, controller, roles, config, recorder);
    auto [parallel_latency, parallel_prediction] = time_dual(
        [&](std::uint64_t run_seed) { return dual_parallel.predict(context, false, run_seed); }, iterations);
    const auto replay_prediction = dual_isolated.predict(context, true, 0xabcdefULL);
    std::uint64_t replay_entries = 0U;
    for (const ReplayId replay_id : replay_prediction.replay_ids) {
        if (const auto step = recorder->get(replay_id)) {
            replay_entries += step->relation_versions.size();
            replay_entries += step->parent_branch_ids.size();
            replay_entries += step->lanes[0].gate_decisions.size() + step->lanes[1].gate_decisions.size();
            replay_entries += step->lanes[0].survivor_ids.size() + step->lanes[1].survivor_ids.size();
        }
    }

    DualRow row;
    row.branching = branching;
    row.beam = beam;
    row.depth = depth;
    row.requested_m1_fraction = m1_fraction;
    row.actual_m1_fraction = synthetic.total_edges == 0U ? 0.0F :
        static_cast<float>(synthetic.m1_edges) / static_cast<float>(synthetic.total_edges);
    row.iterations = iterations;
    row.full = full_latency;
    row.clean = clean_latency;
    row.dual_isolated = isolated_latency;
    row.dual_parallel_reuse = parallel_latency;
    row.isolated_runtime_ratio = static_cast<float>(isolated_latency.mean_ms / std::max(full_latency.mean_ms, 1.0e-12));
    row.parallel_runtime_ratio = static_cast<float>(parallel_latency.mean_ms / std::max(full_latency.mean_ms, 1.0e-12));
    row.full_ops = full_prediction.metrics.operator_evaluations;
    row.clean_ops = clean_prediction.metrics.operator_evaluations;
    row.operator_ratio = row.full_ops == 0U ? 0.0F :
        static_cast<float>(row.full_ops + row.clean_ops) / static_cast<float>(row.full_ops);
    row.active_state_full = isolated_prediction.full.metrics.active_state_peak;
    row.active_state_clean = isolated_prediction.clean.metrics.active_state_peak;
    row.replay_storage_entries = replay_entries;
    row.clean_health = isolated_prediction.clean.metrics.clean_health_ratio;
    row.clean_degenerate = row.clean_health < 0.50F;
    row.clean_empty = isolated_prediction.clean.metrics.empty;
    (void)parallel_prediction;
    return row;
}

void print_dual_header(bool json) {
    if (json) return;
    std::cout << "suite,k,beam,depth,m1_requested,m1_actual,iterations,full_mean_ms,full_p50_ms,full_p95_ms,"
                 "clean_mean_ms,clean_p50_ms,clean_p95_ms,dual_isolated_mean_ms,dual_parallel_reuse_mean_ms,"
                 "R_runtime_isolated,R_runtime_parallel_reuse,R_ops,full_ops,clean_ops,active_full,active_clean,"
                 "replay_entries,clean_health,clean_degenerate,clean_empty\n";
}

void print_dual(const DualRow& row, bool json) {
    std::cout << std::fixed << std::setprecision(6);
    if (json) {
        std::cout << "{\"suite\":\"dual\",\"k\":" << row.branching
                  << ",\"beam\":" << row.beam << ",\"depth\":" << row.depth
                  << ",\"m1_requested\":" << row.requested_m1_fraction
                  << ",\"m1_actual\":" << row.actual_m1_fraction
                  << ",\"iterations\":" << row.iterations
                  << ",\"full_mean_ms\":" << row.full.mean_ms
                  << ",\"full_p50_ms\":" << row.full.p50_ms
                  << ",\"full_p95_ms\":" << row.full.p95_ms
                  << ",\"clean_mean_ms\":" << row.clean.mean_ms
                  << ",\"clean_p50_ms\":" << row.clean.p50_ms
                  << ",\"clean_p95_ms\":" << row.clean.p95_ms
                  << ",\"dual_isolated_mean_ms\":" << row.dual_isolated.mean_ms
                  << ",\"dual_parallel_reuse_mean_ms\":" << row.dual_parallel_reuse.mean_ms
                  << ",\"runtime_ratio_isolated\":" << row.isolated_runtime_ratio
                  << ",\"runtime_ratio_parallel_reuse\":" << row.parallel_runtime_ratio
                  << ",\"operator_ratio\":" << row.operator_ratio
                  << ",\"full_ops\":" << row.full_ops << ",\"clean_ops\":" << row.clean_ops
                  << ",\"active_full\":" << row.active_state_full
                  << ",\"active_clean\":" << row.active_state_clean
                  << ",\"replay_entries\":" << row.replay_storage_entries
                  << ",\"clean_health\":" << row.clean_health
                  << ",\"clean_degenerate\":" << (row.clean_degenerate ? "true" : "false")
                  << ",\"clean_empty\":" << (row.clean_empty ? "true" : "false") << "}\n";
    } else {
        std::cout << "dual," << row.branching << ',' << row.beam << ',' << row.depth << ','
                  << row.requested_m1_fraction << ',' << row.actual_m1_fraction << ',' << row.iterations << ','
                  << row.full.mean_ms << ',' << row.full.p50_ms << ',' << row.full.p95_ms << ','
                  << row.clean.mean_ms << ',' << row.clean.p50_ms << ',' << row.clean.p95_ms << ','
                  << row.dual_isolated.mean_ms << ',' << row.dual_parallel_reuse.mean_ms << ','
                  << row.isolated_runtime_ratio << ',' << row.parallel_runtime_ratio << ',' << row.operator_ratio << ','
                  << row.full_ops << ',' << row.clean_ops << ',' << row.active_state_full << ',' << row.active_state_clean << ','
                  << row.replay_storage_entries << ',' << row.clean_health << ','
                  << (row.clean_degenerate ? 1 : 0) << ',' << (row.clean_empty ? 1 : 0) << '\n';
    }
}

void benchmark_monomial(bool quick, bool json) {
    constexpr std::size_t dimension = 96U;
    const std::uint64_t iterations = quick ? 100000U : 1000000U;
    MonomialOperator first = MonomialOperator::seeded(dimension, 1U);
    MonomialOperator second = MonomialOperator::seeded(dimension, 2U);
    std::vector<float> input(dimension, 0.1F);
    std::vector<float> output(dimension);
    std::uint64_t checksum = 0U;
    auto started = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        first.apply(input, output);
        checksum = hash_combine(checksum, std::bit_cast<std::uint32_t>(output[i % dimension]));
        std::swap(input, output);
    }
    const double apply_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    started = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < iterations / 10U; ++i) {
        const auto composed = MonomialOperator::compose(second, first);
        checksum = hash_combine(checksum, composed.full_hash());
    }
    const double compose_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const double apply_ops = static_cast<double>(iterations) / apply_seconds;
    const double compose_ops = static_cast<double>(iterations / 10U) / compose_seconds;
    if (json) {
        std::cout << std::fixed << std::setprecision(3)
                  << "{\"suite\":\"monomial\",\"dimension\":" << dimension
                  << ",\"apply_ops_per_second\":" << apply_ops
                  << ",\"compose_ops_per_second\":" << compose_ops
                  << ",\"checksum\":" << checksum << "}\n";
    } else {
        std::cout << "monomial_dimension=" << dimension
                  << " apply_ops_per_second=" << std::fixed << std::setprecision(3) << apply_ops
                  << " compose_ops_per_second=" << compose_ops
                  << " checksum=" << checksum << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.suite == "all" || options.suite == "dual") {
            const std::uint32_t iterations = options.iterations != 0U ? options.iterations : (options.quick ? 3U : 15U);
            print_dual_header(options.json);
            const std::array<float, 4> fractions{0.0F, 0.01F, 0.10F, 0.50F};
            for (const auto& [branching, beam] : std::array<std::pair<std::uint32_t, std::uint32_t>, 2>{
                     std::pair{2U, 4U}, std::pair{8U, 32U}}) {
                for (const float fraction : fractions) {
                    print_dual(benchmark_dual(branching, beam, 16U, fraction, iterations), options.json);
                }
            }
        }
        if (options.suite == "all" || options.suite == "monomial") benchmark_monomial(options.quick, options.json);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return 1;
    }
}

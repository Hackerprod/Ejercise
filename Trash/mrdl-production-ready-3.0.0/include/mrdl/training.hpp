#pragma once

#include "mrdl/baselines.hpp"
#include "mrdl/common.hpp"
#include "mrdl/config.hpp"
#include "mrdl/controller.hpp"
#include "mrdl/embeddings.hpp"
#include "mrdl/engine.hpp"
#include "mrdl/graph.hpp"
#include "mrdl/metrics.hpp"
#include "mrdl/persistence.hpp"
#include "mrdl/promotion.hpp"
#include "mrdl/replay.hpp"
#include "mrdl/tokenizer.hpp"

#include <deque>

namespace mrdl {

struct TrainStats {
    std::uint64_t tokens{0};
    std::uint64_t correct{0};
    std::uint64_t m1_writes{0};
    std::uint64_t promotions{0};
    std::uint64_t audits_deferred{0};
    std::uint64_t audits_rejected{0};
    std::uint64_t clean_empty{0};
    double negative_log_likelihood{0.0};
    double elapsed_seconds{0.0};

    [[nodiscard]] double average_loss() const noexcept;
    [[nodiscard]] double perplexity() const noexcept;
    [[nodiscard]] double accuracy() const noexcept;
    [[nodiscard]] double tokens_per_second() const noexcept;
};

struct EvalStats {
    std::uint64_t tokens{0};
    std::uint64_t correct_full{0};
    std::uint64_t correct_clean{0};
    std::uint64_t clean_empty{0};
    double full_nll{0.0};
    double clean_nll{0.0};
    double elapsed_seconds{0.0};
};

struct GenerationResult {
    std::vector<TokenId> generated;
    std::vector<Certification> certifications;
    std::string text;
    double tokens_per_second{0.0};
};

using TrainProgressCallback = std::function<void(const TrainStats&)>;

class ModelRuntime final {
public:
    static void prepare(const AppConfig& config,
                        const std::filesystem::path& corpus,
                        EmbeddingInit embedding_mode,
                        const std::optional<std::filesystem::path>& external_embeddings = std::nullopt);
    static std::unique_ptr<ModelRuntime> open(AppConfig config);

    TrainStats train(const std::filesystem::path& corpus,
                     TrainProgressCallback progress = {});
    EvalStats evaluate(const std::filesystem::path& corpus, std::uint64_t max_tokens = 0);
    GenerationResult generate(std::string_view prompt,
                              std::uint32_t max_tokens = 0,
                              float temperature = -1.0F,
                              std::uint64_t seed = 0);

    std::size_t audit_pending(std::size_t max_relations = 0, TrainStats* aggregate = nullptr);
    std::size_t garbage_collect();
    void checkpoint();
    void backup(const std::filesystem::path& destination_directory);
    [[nodiscard]] bool integrity_check(std::string* diagnostic = nullptr) const;

    [[nodiscard]] const AppConfig& config() const noexcept { return config_; }
    [[nodiscard]] const HybridTokenizer& tokenizer() const noexcept { return tokenizer_; }
    [[nodiscard]] const FrozenEmbeddingStore& embeddings() const noexcept { return embeddings_; }
    [[nodiscard]] GraphStore& graph() noexcept { return graph_; }
    [[nodiscard]] const GraphStore& graph() const noexcept { return graph_; }
    [[nodiscard]] Controller& controller() noexcept { return controller_; }
    [[nodiscard]] RoleInducer& roles() noexcept { return roles_; }
    [[nodiscard]] PromotionManager& promotions() noexcept { return *promotion_; }
    [[nodiscard]] DualLaneEngine& engine() noexcept { return *engine_; }
    [[nodiscard]] const MetricsRegistry& metrics() const noexcept { return *metrics_; }

private:
    explicit ModelRuntime(AppConfig config);

    AppConfig config_;
    HybridTokenizer tokenizer_;
    FrozenEmbeddingStore embeddings_;
    std::shared_ptr<SqliteModelStore> persistence_;
    GraphStore graph_;
    Controller controller_;
    RoleInducer roles_;
    std::shared_ptr<ReplayRecorder> replay_;
    std::shared_ptr<MetricsRegistry> metrics_;
    std::unique_ptr<PromotionManager> promotion_;
    std::unique_ptr<DualLaneEngine> engine_;

    [[nodiscard]] double sparse_nll(const LanePrediction& prediction, TokenId target) const;
    [[nodiscard]] std::optional<RelationId> learn_mode_b(std::span<const TokenId> context,
                                                         TokenId target,
                                                         const DualPrediction& prediction,
                                                         std::string_view source_name);
    [[nodiscard]] ReplayClosure build_closure(const RelationRecord& root,
                                              const DualPrediction& prediction,
                                              std::span<const RelationSnapshot> prediction_snapshots = {});
    [[nodiscard]] AuditOutcome audit_record(const EscrowRecord& record) const;
    [[nodiscard]] std::vector<float> contextual_source_vector(std::span<const TokenId> context,
                                                              std::size_t source_position) const;
};

}  // namespace mrdl

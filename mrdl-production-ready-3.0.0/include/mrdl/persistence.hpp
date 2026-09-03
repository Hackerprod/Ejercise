#pragma once

#include "mrdl/common.hpp"
#include "mrdl/graph.hpp"
#include "mrdl/replay.hpp"

struct sqlite3;

namespace mrdl {

struct EscrowRow {
    std::uint64_t id{0};
    RelationId relation_id{0};
    EscrowState state{EscrowState::Active};
    std::int32_t pin_count{0};
    bool expiry_pending{false};
    std::int64_t expires_at_ms{0};
    std::vector<std::byte> payload;
    std::vector<std::byte> closure;
};

class SqliteModelStore final : public IGraphJournal,
                               public IReplayRepository,
                               public std::enable_shared_from_this<SqliteModelStore> {
public:
    class WriteTransaction final {
    public:
        ~WriteTransaction();
        WriteTransaction(const WriteTransaction&) = delete;
        WriteTransaction& operator=(const WriteTransaction&) = delete;
        WriteTransaction(WriteTransaction&& other) noexcept;
        WriteTransaction& operator=(WriteTransaction&&) = delete;

        void commit();
        void rollback() noexcept;
        [[nodiscard]] bool active() const noexcept { return active_; }

    private:
        friend class SqliteModelStore;
        explicit WriteTransaction(SqliteModelStore& store);

        SqliteModelStore* store_{nullptr};
        std::unique_lock<std::recursive_mutex> lock_;
        bool active_{false};
    };

    SqliteModelStore(const std::filesystem::path& path,
                     std::uint32_t busy_timeout_ms,
                     bool synchronous_full);
    ~SqliteModelStore() override;
    SqliteModelStore(const SqliteModelStore&) = delete;
    SqliteModelStore& operator=(const SqliteModelStore&) = delete;

    void persist_relation(const RelationRecord& relation) override;
    void delete_relation(RelationId id) override;
    [[nodiscard]] std::vector<RelationRecord> load_relations() const;

    void save_step(const ReplayStep& step) override;
    [[nodiscard]] std::optional<ReplayStep> load_step(ReplayId id) const override;
    [[nodiscard]] ReplayId max_step_id() const override;
    void delete_step(ReplayId id) override;

    void save_escrow(const EscrowRow& row);
    [[nodiscard]] std::optional<EscrowRow> load_escrow(std::uint64_t id) const;
    [[nodiscard]] std::vector<EscrowRow> load_escrows() const;
    bool compare_exchange_escrow_state(std::uint64_t id,
                                       EscrowState expected,
                                       EscrowState desired,
                                       std::int32_t pin_delta,
                                       std::optional<bool> expiry_pending = std::nullopt);
    bool set_escrow_expiry_pending(std::uint64_t id, bool pending);
    void delete_escrow(std::uint64_t id);

    void save_controller(std::span<const std::byte> payload, std::uint64_t version);
    [[nodiscard]] std::optional<std::vector<std::byte>> load_controller() const;
    void save_role_inducer(std::span<const std::byte> payload);
    [[nodiscard]] std::optional<std::vector<std::byte>> load_role_inducer() const;

    void promote_atomic(const RelationRecord& promoted, std::uint64_t escrow_id);
    void set_meta(std::string_view key, std::span<const std::byte> value);
    [[nodiscard]] std::optional<std::vector<std::byte>> get_meta(std::string_view key) const;
    void checkpoint_wal();
    void backup_to(const std::filesystem::path& destination) const;
    [[nodiscard]] bool integrity_check(std::string* diagnostic = nullptr) const;

    // Holds the SQLite connection mutex for the complete write unit. Every journal
    // operation invoked by the owning thread re-enters the recursive mutex, while
    // unrelated threads cannot accidentally join the same SQLite transaction.
    [[nodiscard]] WriteTransaction begin_write_transaction();

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    sqlite3* db_{nullptr};
    mutable std::recursive_mutex mutex_;

    void exec(std::string_view sql) const;
    void migrate();
    void begin_immediate();
    void commit();
    void rollback() noexcept;
};

}  // namespace mrdl

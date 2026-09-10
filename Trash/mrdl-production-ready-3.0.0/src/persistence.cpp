#include "mrdl/persistence.hpp"

#include <sqlite3.h>

namespace mrdl {
namespace {

class Statement final {
public:
    Statement(sqlite3* db, std::string_view sql) : db_(db) {
        if (sqlite3_prepare_v2(db, sql.data(), static_cast<int>(sql.size()), &statement_, nullptr) != SQLITE_OK) {
            throw Error("sqlite prepare failed: " + std::string(sqlite3_errmsg(db)));
        }
    }
    ~Statement() { if (statement_) sqlite3_finalize(statement_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    sqlite3_stmt* get() noexcept { return statement_; }
    void bind_int64(int index, std::int64_t value) {
        if (sqlite3_bind_int64(statement_, index, value) != SQLITE_OK) fail();
    }
    void bind_int(int index, int value) {
        if (sqlite3_bind_int(statement_, index, value) != SQLITE_OK) fail();
    }
    void bind_text(int index, std::string_view value) {
        if (sqlite3_bind_text(statement_, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT) != SQLITE_OK) fail();
    }
    void bind_blob(int index, std::span<const std::byte> value) {
        const void* data = value.empty() ? nullptr : value.data();
        if (sqlite3_bind_blob64(statement_, index, data, value.size(), SQLITE_TRANSIENT) != SQLITE_OK) fail();
    }
    int step() {
        const int rc = sqlite3_step(statement_);
        if (rc != SQLITE_ROW && rc != SQLITE_DONE) fail();
        return rc;
    }

private:
    [[noreturn]] void fail() { throw Error("sqlite statement failed: " + std::string(sqlite3_errmsg(db_))); }
    sqlite3* db_;
    sqlite3_stmt* statement_{nullptr};
};

std::vector<std::byte> column_blob(sqlite3_stmt* statement, int column) {
    const auto size = sqlite3_column_bytes(statement, column);
    const auto* data = static_cast<const std::byte*>(sqlite3_column_blob(statement, column));
    if (size <= 0 || data == nullptr) return {};
    return std::vector<std::byte>(data, data + size);
}

}  // namespace

SqliteModelStore::SqliteModelStore(const std::filesystem::path& path,
                                   std::uint32_t busy_timeout_ms,
                                   bool synchronous_full)
    : path_(path) {
    std::filesystem::create_directories(path.parent_path());
    const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
    if (sqlite3_open_v2(path.c_str(), &db_, flags, nullptr) != SQLITE_OK) {
        const std::string message = db_ ? sqlite3_errmsg(db_) : "unknown sqlite error";
        if (db_) sqlite3_close(db_);
        db_ = nullptr;
        throw Error("cannot open model database: " + message);
    }
    sqlite3_extended_result_codes(db_, 1);
    sqlite3_busy_timeout(db_, static_cast<int>(busy_timeout_ms));
    exec("PRAGMA journal_mode=WAL;");
    exec(synchronous_full ? "PRAGMA synchronous=FULL;" : "PRAGMA synchronous=NORMAL;");
    exec("PRAGMA foreign_keys=ON;");
    exec("PRAGMA temp_store=MEMORY;");
    exec("PRAGMA mmap_size=268435456;");
    migrate();
}

SqliteModelStore::~SqliteModelStore() {
    if (db_) sqlite3_close(db_);
}

SqliteModelStore::WriteTransaction::WriteTransaction(SqliteModelStore& store)
    : store_(&store), lock_(store.mutex_) {
    store_->begin_immediate();
    active_ = true;
}

SqliteModelStore::WriteTransaction::~WriteTransaction() { rollback(); }

SqliteModelStore::WriteTransaction::WriteTransaction(WriteTransaction&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)),
      lock_(std::move(other.lock_)),
      active_(std::exchange(other.active_, false)) {}

void SqliteModelStore::WriteTransaction::commit() {
    require(active_ && store_ != nullptr, "SQLite write transaction is not active");
    store_->commit();
    active_ = false;
    lock_.unlock();
}

void SqliteModelStore::WriteTransaction::rollback() noexcept {
    if (!active_ || store_ == nullptr) return;
    store_->rollback();
    active_ = false;
    if (lock_.owns_lock()) lock_.unlock();
}

SqliteModelStore::WriteTransaction SqliteModelStore::begin_write_transaction() {
    return WriteTransaction(*this);
}

void SqliteModelStore::exec(std::string_view sql) const {
    std::lock_guard lock(mutex_);
    char* error = nullptr;
    const int rc = sqlite3_exec(db_, std::string(sql).c_str(), nullptr, nullptr, &error);
    if (rc != SQLITE_OK) {
        const std::string message = error ? error : sqlite3_errmsg(db_);
        sqlite3_free(error);
        throw Error("sqlite exec failed: " + message);
    }
}

void SqliteModelStore::migrate() {
    int current_version = 0;
    {
        std::lock_guard lock(mutex_);
        Statement version_statement(db_, "PRAGMA user_version;");
        require(version_statement.step() == SQLITE_ROW, "cannot read database schema version");
        current_version = sqlite3_column_int(version_statement.get(), 0);
    }
    require(current_version == 0 || current_version == 3,
            "unsupported MRDL database schema version; use the documented migration/export path");

    exec(R"SQL(
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS relations(
    id INTEGER PRIMARY KEY,
    source INTEGER NOT NULL,
    destination INTEGER NOT NULL,
    prototype INTEGER NOT NULL,
    level INTEGER NOT NULL,
    version INTEGER NOT NULL,
    payload BLOB NOT NULL,
    UNIQUE(source, destination, prototype)
);
CREATE INDEX IF NOT EXISTS idx_relations_source_level ON relations(source, level);
CREATE TABLE IF NOT EXISTS replay_steps(
    id INTEGER PRIMARY KEY,
    operation_id INTEGER NOT NULL,
    payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS escrow(
    id INTEGER PRIMARY KEY,
    relation_id INTEGER NOT NULL UNIQUE,
    state INTEGER NOT NULL,
    pin_count INTEGER NOT NULL DEFAULT 0 CHECK(pin_count >= 0),
    expiry_pending INTEGER NOT NULL DEFAULT 0,
    expires_at_ms INTEGER NOT NULL,
    payload BLOB NOT NULL,
    closure BLOB NOT NULL,
    FOREIGN KEY(relation_id) REFERENCES relations(id) ON DELETE CASCADE
);
PRAGMA user_version=3;
)SQL");

    static constexpr std::string_view format = "mrdl-production-v3";
    const auto expected = std::as_bytes(std::span(format.data(), format.size()));
    if (const auto existing = get_meta("storage_format")) {
        require(existing->size() == expected.size() &&
                std::equal(existing->begin(), existing->end(), expected.begin()),
                "database storage_format does not belong to this MRDL implementation");
    } else {
        set_meta("storage_format", expected);
    }
}

void SqliteModelStore::begin_immediate() { exec("BEGIN IMMEDIATE;"); }
void SqliteModelStore::commit() { exec("COMMIT;"); }
void SqliteModelStore::rollback() noexcept {
    try { exec("ROLLBACK;"); } catch (...) {}
}

void SqliteModelStore::persist_relation(const RelationRecord& relation) {
    const auto payload = relation.serialize();
    std::lock_guard lock(mutex_);
    Statement statement(db_, R"SQL(
INSERT INTO relations(id, source, destination, prototype, level, version, payload)
VALUES(?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
 source=excluded.source,
 destination=excluded.destination,
 prototype=excluded.prototype,
 level=excluded.level,
 version=excluded.version,
 payload=excluded.payload;
)SQL");
    statement.bind_int64(1, static_cast<std::int64_t>(relation.id));
    statement.bind_int64(2, relation.source);
    statement.bind_int64(3, relation.destination);
    statement.bind_int(4, relation.prototype);
    statement.bind_int(5, static_cast<int>(relation.level));
    statement.bind_int64(6, static_cast<std::int64_t>(relation.version));
    statement.bind_blob(7, payload);
    (void)statement.step();
}

void SqliteModelStore::delete_relation(RelationId id) {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "DELETE FROM relations WHERE id=?;");
    statement.bind_int64(1, static_cast<std::int64_t>(id));
    (void)statement.step();
}

std::vector<RelationRecord> SqliteModelStore::load_relations() const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "SELECT payload FROM relations ORDER BY id;");
    std::vector<RelationRecord> result;
    while (statement.step() == SQLITE_ROW) {
        result.push_back(RelationRecord::deserialize(column_blob(statement.get(), 0)));
    }
    return result;
}

void SqliteModelStore::save_step(const ReplayStep& step) {
    const auto payload = step.serialize();
    std::lock_guard lock(mutex_);
    Statement statement(db_, R"SQL(
INSERT INTO replay_steps(id, operation_id, payload) VALUES(?, ?, ?)
ON CONFLICT(id) DO UPDATE SET operation_id=excluded.operation_id, payload=excluded.payload;
)SQL");
    statement.bind_int64(1, static_cast<std::int64_t>(step.id));
    statement.bind_int64(2, static_cast<std::int64_t>(step.operation_id));
    statement.bind_blob(3, payload);
    (void)statement.step();
}

std::optional<ReplayStep> SqliteModelStore::load_step(ReplayId id) const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "SELECT payload FROM replay_steps WHERE id=?;");
    statement.bind_int64(1, static_cast<std::int64_t>(id));
    if (statement.step() != SQLITE_ROW) return std::nullopt;
    return ReplayStep::deserialize(column_blob(statement.get(), 0));
}

ReplayId SqliteModelStore::max_step_id() const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "SELECT COALESCE(MAX(id), 0) FROM replay_steps;");
    require(statement.step() == SQLITE_ROW, "failed reading replay id high-water mark");
    return static_cast<ReplayId>(sqlite3_column_int64(statement.get(), 0));
}

void SqliteModelStore::delete_step(ReplayId id) {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "DELETE FROM replay_steps WHERE id=?;");
    statement.bind_int64(1, static_cast<std::int64_t>(id));
    (void)statement.step();
}

void SqliteModelStore::save_escrow(const EscrowRow& row) {
    std::lock_guard lock(mutex_);
    Statement statement(db_, R"SQL(
INSERT INTO escrow(id, relation_id, state, pin_count, expiry_pending, expires_at_ms, payload, closure)
VALUES(?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
 relation_id=excluded.relation_id,
 state=excluded.state,
 pin_count=excluded.pin_count,
 expiry_pending=excluded.expiry_pending,
 expires_at_ms=excluded.expires_at_ms,
 payload=excluded.payload,
 closure=excluded.closure;
)SQL");
    statement.bind_int64(1, static_cast<std::int64_t>(row.id));
    statement.bind_int64(2, static_cast<std::int64_t>(row.relation_id));
    statement.bind_int(3, static_cast<int>(row.state));
    statement.bind_int(4, row.pin_count);
    statement.bind_int(5, row.expiry_pending ? 1 : 0);
    statement.bind_int64(6, row.expires_at_ms);
    statement.bind_blob(7, row.payload);
    statement.bind_blob(8, row.closure);
    (void)statement.step();
}

std::optional<EscrowRow> SqliteModelStore::load_escrow(std::uint64_t id) const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "SELECT id, relation_id, state, pin_count, expiry_pending, expires_at_ms, payload, closure FROM escrow WHERE id=?;");
    statement.bind_int64(1, static_cast<std::int64_t>(id));
    if (statement.step() != SQLITE_ROW) return std::nullopt;
    EscrowRow row;
    row.id = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 0));
    row.relation_id = static_cast<RelationId>(sqlite3_column_int64(statement.get(), 1));
    row.state = static_cast<EscrowState>(sqlite3_column_int(statement.get(), 2));
    row.pin_count = sqlite3_column_int(statement.get(), 3);
    row.expiry_pending = sqlite3_column_int(statement.get(), 4) != 0;
    row.expires_at_ms = sqlite3_column_int64(statement.get(), 5);
    row.payload = column_blob(statement.get(), 6);
    row.closure = column_blob(statement.get(), 7);
    return row;
}

std::vector<EscrowRow> SqliteModelStore::load_escrows() const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "SELECT id, relation_id, state, pin_count, expiry_pending, expires_at_ms, payload, closure FROM escrow ORDER BY id;");
    std::vector<EscrowRow> result;
    while (statement.step() == SQLITE_ROW) {
        EscrowRow row;
        row.id = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 0));
        row.relation_id = static_cast<RelationId>(sqlite3_column_int64(statement.get(), 1));
        row.state = static_cast<EscrowState>(sqlite3_column_int(statement.get(), 2));
        row.pin_count = sqlite3_column_int(statement.get(), 3);
        row.expiry_pending = sqlite3_column_int(statement.get(), 4) != 0;
        row.expires_at_ms = sqlite3_column_int64(statement.get(), 5);
        row.payload = column_blob(statement.get(), 6);
        row.closure = column_blob(statement.get(), 7);
        result.push_back(std::move(row));
    }
    return result;
}

bool SqliteModelStore::compare_exchange_escrow_state(std::uint64_t id,
                                                      EscrowState expected,
                                                      EscrowState desired,
                                                      std::int32_t pin_delta,
                                                      std::optional<bool> expiry_pending) {
    std::lock_guard lock(mutex_);
    const char* sql_with_expiry = R"SQL(
UPDATE escrow SET state=?, pin_count=pin_count+?, expiry_pending=?
WHERE id=? AND state=? AND pin_count+? >= 0;
)SQL";
    const char* sql_without_expiry = R"SQL(
UPDATE escrow SET state=?, pin_count=pin_count+?
WHERE id=? AND state=? AND pin_count+? >= 0;
)SQL";
    Statement statement(db_, expiry_pending ? sql_with_expiry : sql_without_expiry);
    int index = 1;
    statement.bind_int(index++, static_cast<int>(desired));
    statement.bind_int(index++, pin_delta);
    if (expiry_pending) statement.bind_int(index++, *expiry_pending ? 1 : 0);
    statement.bind_int64(index++, static_cast<std::int64_t>(id));
    statement.bind_int(index++, static_cast<int>(expected));
    statement.bind_int(index, pin_delta);
    (void)statement.step();
    return sqlite3_changes(db_) == 1;
}

bool SqliteModelStore::set_escrow_expiry_pending(std::uint64_t id, bool pending) {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "UPDATE escrow SET expiry_pending=? WHERE id=?;");
    statement.bind_int(1, pending ? 1 : 0);
    statement.bind_int64(2, static_cast<std::int64_t>(id));
    (void)statement.step();
    return sqlite3_changes(db_) == 1;
}

void SqliteModelStore::delete_escrow(std::uint64_t id) {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "DELETE FROM escrow WHERE id=?;");
    statement.bind_int64(1, static_cast<std::int64_t>(id));
    (void)statement.step();
}

void SqliteModelStore::save_controller(std::span<const std::byte> payload, std::uint64_t version) {
    BinaryWriter writer;
    writer.pod(version);
    writer.vector<std::byte>(payload);
    const auto data = writer.take();
    set_meta("controller", data);
}

std::optional<std::vector<std::byte>> SqliteModelStore::load_controller() const {
    const auto data = get_meta("controller");
    if (!data) return std::nullopt;
    BinaryReader reader(*data);
    (void)reader.pod<std::uint64_t>();
    auto payload = reader.vector<std::byte>();
    require(reader.empty(), "corrupt controller metadata");
    return payload;
}

void SqliteModelStore::save_role_inducer(std::span<const std::byte> payload) { set_meta("role_inducer", payload); }
std::optional<std::vector<std::byte>> SqliteModelStore::load_role_inducer() const { return get_meta("role_inducer"); }

void SqliteModelStore::promote_atomic(const RelationRecord& promoted, std::uint64_t escrow_id) {
    auto transaction = begin_write_transaction();
    persist_relation(promoted);
    {
        Statement statement(db_, "UPDATE escrow SET state=?, pin_count=CASE WHEN pin_count>0 THEN pin_count-1 ELSE 0 END, expiry_pending=0 WHERE id=? AND state IN (?,?);");
        statement.bind_int(1, static_cast<int>(EscrowState::Promoted));
        statement.bind_int64(2, static_cast<std::int64_t>(escrow_id));
        statement.bind_int(3, static_cast<int>(EscrowState::AuditReserved));
        statement.bind_int(4, static_cast<int>(EscrowState::Auditing));
        (void)statement.step();
        require(sqlite3_changes(db_) == 1, "escrow promotion state changed concurrently");
    }
    transaction.commit();
}

void SqliteModelStore::set_meta(std::string_view key, std::span<const std::byte> value) {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;");
    statement.bind_text(1, key);
    statement.bind_blob(2, value);
    (void)statement.step();
}

std::optional<std::vector<std::byte>> SqliteModelStore::get_meta(std::string_view key) const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "SELECT value FROM meta WHERE key=?;");
    statement.bind_text(1, key);
    if (statement.step() != SQLITE_ROW) return std::nullopt;
    return column_blob(statement.get(), 0);
}

void SqliteModelStore::checkpoint_wal() {
    std::lock_guard lock(mutex_);
    if (sqlite3_wal_checkpoint_v2(db_, nullptr, SQLITE_CHECKPOINT_TRUNCATE, nullptr, nullptr) != SQLITE_OK) {
        throw Error("sqlite WAL checkpoint failed: " + std::string(sqlite3_errmsg(db_)));
    }
}

void SqliteModelStore::backup_to(const std::filesystem::path& destination) const {
    std::lock_guard lock(mutex_);
    std::filesystem::create_directories(destination.parent_path());
    sqlite3* target = nullptr;
    if (sqlite3_open_v2(destination.c_str(), &target, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) != SQLITE_OK) {
        const std::string message = target ? sqlite3_errmsg(target) : "unknown sqlite error";
        if (target) sqlite3_close(target);
        throw Error("cannot open backup database: " + message);
    }
    ScopeExit close_target([&] { sqlite3_close(target); });
    sqlite3_backup* backup = sqlite3_backup_init(target, "main", db_, "main");
    if (!backup) throw Error("sqlite backup init failed: " + std::string(sqlite3_errmsg(target)));
    ScopeExit finish([&] { sqlite3_backup_finish(backup); });
    int rc = SQLITE_OK;
    do {
        rc = sqlite3_backup_step(backup, 1024);
        if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED) sqlite3_sleep(10);
    } while (rc == SQLITE_OK || rc == SQLITE_BUSY || rc == SQLITE_LOCKED);
    require(rc == SQLITE_DONE, "sqlite backup failed");
}

bool SqliteModelStore::integrity_check(std::string* diagnostic) const {
    std::lock_guard lock(mutex_);
    Statement statement(db_, "PRAGMA integrity_check;");
    std::string result;
    bool ok = true;
    while (statement.step() == SQLITE_ROW) {
        const auto* text = reinterpret_cast<const char*>(sqlite3_column_text(statement.get(), 0));
        const std::string row = text ? text : "";
        if (!result.empty()) result.push_back('\n');
        result += row;
        if (row != "ok") ok = false;
    }
    if (diagnostic) *diagnostic = std::move(result);
    return ok;
}

}  // namespace mrdl

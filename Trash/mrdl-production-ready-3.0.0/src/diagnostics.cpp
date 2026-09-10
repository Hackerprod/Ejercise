#include "mrdl/diagnostics.hpp"

#include "mrdl/embeddings.hpp"
#include "mrdl/persistence.hpp"
#include "mrdl/tokenizer.hpp"
#include "mrdl/version.hpp"

#include <sys/sysinfo.h>

namespace mrdl {
namespace {

std::string escape_json(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8U);
    for (const char value_byte : value) {
        const auto c = static_cast<unsigned char>(value_byte);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20U) {
                    constexpr char digits[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(digits[(c >> 4U) & 0xFU]);
                    out.push_back(digits[c & 0xFU]);
                } else out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::string_view status_name(DiagnosticStatus status) {
    switch (status) {
        case DiagnosticStatus::Pass: return "pass";
        case DiagnosticStatus::Warn: return "warn";
        case DiagnosticStatus::Fail: return "fail";
    }
    return "unknown";
}

void add(DoctorReport& report, std::string name, DiagnosticStatus status, std::string detail) {
    report.checks.push_back(DiagnosticCheck{std::move(name), status, std::move(detail)});
}

std::string human_bytes(std::uint64_t bytes) {
    static constexpr std::array<std::string_view, 5> suffix{"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1U < suffix.size()) { value /= 1024.0; ++unit; }
    std::ostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(unit == 0U ? 0 : 2);
    stream << value << ' ' << suffix[unit];
    return stream.str();
}

}  // namespace

bool DoctorReport::healthy() const noexcept {
    return std::none_of(checks.begin(), checks.end(), [](const DiagnosticCheck& check) {
        return check.status == DiagnosticStatus::Fail;
    });
}

std::string DoctorReport::json() const {
    std::ostringstream out;
    out << "{\"healthy\":" << (healthy() ? "true" : "false") << ",\"checks\":[";
    for (std::size_t i = 0; i < checks.size(); ++i) {
        if (i != 0U) out << ',';
        out << "{\"name\":\"" << escape_json(checks[i].name) << "\",\"status\":\""
            << status_name(checks[i].status) << "\",\"detail\":\""
            << escape_json(checks[i].detail) << "\"}";
    }
    out << "]}";
    return out.str();
}

std::string DoctorReport::text() const {
    std::ostringstream out;
    for (const auto& check : checks) {
        out << '[' << status_name(check.status) << "] " << check.name << ": " << check.detail << '\n';
    }
    out << (healthy() ? "MRDL doctor: healthy" : "MRDL doctor: failures detected") << '\n';
    return out.str();
}

DoctorReport run_doctor(const AppConfig& config, bool require_prepared_model) {
    DoctorReport report;
    try {
        config.validate();
        add(report, "configuration", DiagnosticStatus::Pass, "validated");
    } catch (const std::exception& error) {
        add(report, "configuration", DiagnosticStatus::Fail, error.what());
        return report;
    }

    const auto cpu_count = std::max(1U, std::thread::hardware_concurrency());
    add(report, "cpu", cpu_count >= 2U ? DiagnosticStatus::Pass : DiagnosticStatus::Warn,
        std::to_string(cpu_count) + " hardware threads visible");

    struct sysinfo info {};
    if (::sysinfo(&info) == 0) {
        const auto total = static_cast<std::uint64_t>(info.totalram) * info.mem_unit;
        const auto free = static_cast<std::uint64_t>(info.freeram + info.bufferram) * info.mem_unit;
        add(report, "memory", total >= 4ULL * 1024ULL * 1024ULL * 1024ULL ? DiagnosticStatus::Pass : DiagnosticStatus::Warn,
            human_bytes(total) + " total, " + human_bytes(free) + " immediately available");
    } else {
        add(report, "memory", DiagnosticStatus::Warn, "sysinfo unavailable");
    }

    try {
        std::filesystem::create_directories(config.persistence.model_dir);
        const auto space = std::filesystem::space(config.persistence.model_dir);
        const auto status = space.available >= 2ULL * 1024ULL * 1024ULL * 1024ULL ?
            DiagnosticStatus::Pass : DiagnosticStatus::Warn;
        add(report, "disk", status, human_bytes(space.available) + " free at " + config.persistence.model_dir.string());
        const auto probe = config.persistence.model_dir / (".write-test-" + std::to_string(unix_millis()));
        {
            std::ofstream stream(probe, std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(stream), "cannot create write probe");
            stream << "ok";
        }
        std::filesystem::remove(probe);
        add(report, "model_directory", DiagnosticStatus::Pass, "writable");
    } catch (const std::exception& error) {
        add(report, "model_directory", DiagnosticStatus::Fail, error.what());
    }

    const bool tokenizer_exists = std::filesystem::exists(config.persistence.tokenizer);
    const bool embeddings_exist = std::filesystem::exists(config.persistence.embeddings);
    const bool database_exists = std::filesystem::exists(config.persistence.database);
    if (require_prepared_model && (!tokenizer_exists || !embeddings_exist || !database_exists)) {
        add(report, "model_files", DiagnosticStatus::Fail, "model is not prepared; run `mrdl prepare`");
        return report;
    }
    if (!tokenizer_exists && !embeddings_exist && !database_exists) {
        add(report, "model_files", DiagnosticStatus::Warn, "no prepared model yet");
        return report;
    }
    if (!(tokenizer_exists && embeddings_exist && database_exists)) {
        add(report, "model_files", DiagnosticStatus::Fail, "partial model detected; tokenizer, embeddings and database must coexist");
        return report;
    }

    try {
        const auto tokenizer = HybridTokenizer::load(config.persistence.tokenizer);
        const auto embeddings = FrozenEmbeddingStore::load(config.persistence.embeddings);
        require(tokenizer.size() == embeddings.token_count(), "vocabulary mismatch");
        require(embeddings.dimension() == config.model.embedding_dim, "embedding dimension mismatch");
        add(report, "frozen_knowledge", DiagnosticStatus::Pass,
            std::to_string(tokenizer.size()) + " tokens, d=" + std::to_string(embeddings.dimension()) +
            ", checksum=" + std::to_string(embeddings.content_hash()));
    } catch (const std::exception& error) {
        add(report, "frozen_knowledge", DiagnosticStatus::Fail, error.what());
    }

    try {
        SqliteModelStore database(config.persistence.database,
                                  config.persistence.sqlite_busy_timeout_ms,
                                  config.persistence.synchronous_full);
        std::string diagnostic;
        const bool ok = database.integrity_check(&diagnostic);
        add(report, "sqlite_integrity", ok ? DiagnosticStatus::Pass : DiagnosticStatus::Fail, diagnostic);

        static constexpr std::string_view expected_family = "MRDL-3-production-core";
        const auto family = database.get_meta("model_family");
        const auto expected_family_bytes = std::as_bytes(std::span(expected_family.data(), expected_family.size()));
        const bool family_ok = family && family->size() == expected_family_bytes.size() &&
                               std::equal(family->begin(), family->end(), expected_family_bytes.begin());

        bool config_ok = false;
        std::string config_detail = "missing model_config metadata";
        if (const auto metadata = database.get_meta("model_config")) {
            BinaryReader reader(*metadata);
            const auto embedding_dim = reader.pod<std::uint32_t>();
            const auto relation_dim = reader.pod<std::uint32_t>();
            const auto prototypes = reader.pod<std::uint32_t>();
            const auto seed = reader.pod<std::uint64_t>();
            require(reader.empty(), "trailing model_config metadata");
            config_ok = embedding_dim == config.model.embedding_dim &&
                        relation_dim == config.model.relation_dim &&
                        prototypes == config.model.max_relation_prototypes &&
                        seed == config.model.seed;
            std::ostringstream detail;
            detail << "stored d_e=" << embedding_dim << ", d_r=" << relation_dim
                   << ", prototypes=" << prototypes << ", seed=" << seed;
            config_detail = detail.str();
        }
        add(report, "model_metadata", family_ok && config_ok ? DiagnosticStatus::Pass : DiagnosticStatus::Fail,
            family_ok ? config_detail : "incompatible or missing model_family metadata");

        if (const auto created = database.get_meta("created_by_version")) {
            const std::string created_version(reinterpret_cast<const char*>(created->data()), created->size());
            add(report, "runtime_version", created_version == MRDL_VERSION_STRING ? DiagnosticStatus::Pass : DiagnosticStatus::Warn,
                "model created by " + created_version + ", running " + std::string(MRDL_VERSION_STRING));
        } else {
            add(report, "runtime_version", DiagnosticStatus::Warn, "created_by_version metadata missing");
        }

        const auto effective = config.persistence.model_dir / "config.effective.ini";
        add(report, "effective_config", std::filesystem::exists(effective) ? DiagnosticStatus::Pass : DiagnosticStatus::Warn,
            std::filesystem::exists(effective) ? effective.string() : "config.effective.ini missing");
    } catch (const std::exception& error) {
        add(report, "database", DiagnosticStatus::Fail, error.what());
    }
    return report;
}

}  // namespace mrdl

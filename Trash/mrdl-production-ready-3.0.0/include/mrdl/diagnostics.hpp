#pragma once

#include "mrdl/common.hpp"
#include "mrdl/config.hpp"

namespace mrdl {

enum class DiagnosticStatus : std::uint8_t { Pass = 0, Warn = 1, Fail = 2 };

struct DiagnosticCheck {
    std::string name;
    DiagnosticStatus status{DiagnosticStatus::Pass};
    std::string detail;
};

struct DoctorReport {
    std::vector<DiagnosticCheck> checks;

    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] std::string json() const;
    [[nodiscard]] std::string text() const;
};

[[nodiscard]] DoctorReport run_doctor(const AppConfig& config, bool require_prepared_model = true);

}  // namespace mrdl

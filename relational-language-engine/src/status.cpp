#include "rlm/status.hpp"

namespace rlm {

std::string Status::to_string() const {
  if (ok()) return "OK";
  std::string_view name = "UNKNOWN";
  switch (code_) {
    case ErrorCode::ok: name = "OK"; break;
    case ErrorCode::invalid_argument: name = "INVALID_ARGUMENT"; break;
    case ErrorCode::not_found: name = "NOT_FOUND"; break;
    case ErrorCode::already_exists: name = "ALREADY_EXISTS"; break;
    case ErrorCode::failed_precondition: name = "FAILED_PRECONDITION"; break;
    case ErrorCode::resource_exhausted: name = "RESOURCE_EXHAUSTED"; break;
    case ErrorCode::data_loss: name = "DATA_LOSS"; break;
    case ErrorCode::unavailable: name = "UNAVAILABLE"; break;
    case ErrorCode::io_error: name = "IO_ERROR"; break;
    case ErrorCode::internal: name = "INTERNAL"; break;
    case ErrorCode::unknown: name = "UNKNOWN"; break;
  }
  return std::string(name) + ": " + message_;
}

}  // namespace rlm

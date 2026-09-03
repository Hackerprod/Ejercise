#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace rlm {

enum class ErrorCode {
  ok = 0,
  invalid_argument,
  not_found,
  already_exists,
  failed_precondition,
  resource_exhausted,
  data_loss,
  unavailable,
  io_error,
  internal,
  unknown,
};

class Status final {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

  [[nodiscard]] static Status Ok() { return {}; }
  [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }
  [[nodiscard]] std::string to_string() const;

 private:
  ErrorCode code_{ErrorCode::ok};
  std::string message_;
};

template <typename T>
class Result final {
 public:
  Result(T value) : value_(std::move(value)), status_(Status::Ok()) {}
  Result(Status status) : status_(std::move(status)) {
    if (status_.ok()) {
      status_ = Status(ErrorCode::internal, "Result constructed with OK status and no value");
    }
  }

  [[nodiscard]] bool ok() const noexcept { return status_.ok(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] const Status& status() const noexcept { return status_; }

  [[nodiscard]] const T& value() const& {
    if (!ok() || !value_.has_value()) throw std::logic_error(status_.to_string());
    return *value_;
  }
  [[nodiscard]] T& value() & {
    if (!ok() || !value_.has_value()) throw std::logic_error(status_.to_string());
    return *value_;
  }
  [[nodiscard]] T&& value() && {
    if (!ok() || !value_.has_value()) throw std::logic_error(status_.to_string());
    return std::move(*value_);
  }

 private:
  std::optional<T> value_;
  Status status_;
};

#define RLM_RETURN_IF_ERROR(expr)          \
  do {                                     \
    const ::rlm::Status _status = (expr);  \
    if (!_status.ok()) return _status;     \
  } while (false)

#define RLM_ASSIGN_OR_RETURN(lhs, expr)                  \
  do {                                                   \
    auto _result = (expr);                               \
    if (!_result.ok()) return _result.status();          \
    lhs = std::move(_result).value();                    \
  } while (false)

}  // namespace rlm

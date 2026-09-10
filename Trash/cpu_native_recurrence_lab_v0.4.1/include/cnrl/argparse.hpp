#pragma once
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace cnrl {
class ArgParser {
 public:
  ArgParser(int argc, char** argv) : argc_(argc), argv_(argv) {}
  [[nodiscard]] bool done() const noexcept { return index_ >= argc_; }
  [[nodiscard]] std::string_view next() {
    if (done()) throw std::runtime_error("unexpected end of arguments");
    return argv_[index_++];
  }
  [[nodiscard]] std::string_view value(std::string_view option) {
    if (done()) throw std::runtime_error("missing value for " + std::string(option));
    return argv_[index_++];
  }
 private:
  int argc_;
  char** argv_;
  int index_ = 1;
};
inline std::uint32_t parse_u32(std::string_view text, std::string_view name,
                               bool allow_zero=false) {
  std::uint32_t value=0;
  auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
  if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || (!allow_zero && value==0)) {
    throw std::runtime_error("invalid value for " + std::string(name));
  }
  return value;
}
inline std::int32_t parse_i32(std::string_view text, std::string_view name) {
  std::int32_t value=0;
  auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
  if (parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size()) {
    throw std::runtime_error("invalid value for " + std::string(name));
  }
  return value;
}
inline double parse_double(std::string_view text, std::string_view name) {
  std::string copy(text);
  std::size_t used=0;
  double value=0.0;
  try { value=std::stod(copy,&used); } catch (...) {
    throw std::runtime_error("invalid value for " + std::string(name));
  }
  if (used!=copy.size()) throw std::runtime_error("invalid value for " + std::string(name));
  return value;
}
inline std::vector<std::uint32_t> parse_u32_list(std::string_view text,
                                                 std::string_view name,
                                                 bool allow_zero=false) {
  std::vector<std::uint32_t> out;
  std::size_t start=0;
  while (start<text.size()) {
    const auto comma=text.find(',',start);
    const auto end=comma==std::string_view::npos?text.size():comma;
    if (end==start) throw std::runtime_error("empty item in " + std::string(name));
    out.push_back(parse_u32(text.substr(start,end-start),name,allow_zero));
    start=comma==std::string_view::npos?text.size():comma+1;
  }
  if (out.empty()) throw std::runtime_error(std::string(name)+" requires values");
  return out;
}
inline std::vector<double> parse_double_list(std::string_view text,
                                             std::string_view name) {
  std::vector<double> out;
  std::size_t start=0;
  while (start<text.size()) {
    const auto comma=text.find(',',start);
    const auto end=comma==std::string_view::npos?text.size():comma;
    if (end==start) throw std::runtime_error("empty item in " + std::string(name));
    out.push_back(parse_double(text.substr(start,end-start),name));
    start=comma==std::string_view::npos?text.size():comma+1;
  }
  if (out.empty()) throw std::runtime_error(std::string(name)+" requires values");
  return out;
}
}  // namespace cnrl

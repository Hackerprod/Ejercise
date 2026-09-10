#include "rlm/baseline.hpp"
#include "rlm/config.hpp"
#include "rlm/embedding_store.hpp"
#include "rlm/engine.hpp"
#include "rlm/status.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <fcntl.h>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <poll.h>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
using namespace rlm;

std::string json_escape(std::string_view input) {
  std::ostringstream output;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (ch < 0x20) {
          constexpr char hex[] = "0123456789abcdef";
          output << "\\u00" << hex[(ch >> 4U) & 0x0fU] << hex[ch & 0x0fU];
        } else {
          output << static_cast<char>(ch);
        }
    }
  }
  return output.str();
}

class Arguments final {
 public:
  Arguments(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) values_.emplace_back(argv[i]);
  }
  [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
  [[nodiscard]] std::string_view command(std::size_t index = 0) const {
    return index < values_.size() ? std::string_view(values_[index]) : std::string_view{};
  }
  [[nodiscard]] bool has(std::string_view flag) const {
    return std::find(values_.begin(), values_.end(), flag) != values_.end();
  }
  [[nodiscard]] Result<std::string> required(std::string_view flag) const {
    for (std::size_t i = 0; i < values_.size(); ++i) {
      if (values_[i] == flag) {
        if (i + 1 >= values_.size() || values_[i + 1].starts_with("--")) {
          return Status(ErrorCode::invalid_argument, "missing value for " + std::string(flag));
        }
        return values_[i + 1];
      }
    }
    return Status(ErrorCode::invalid_argument, "missing required argument " + std::string(flag));
  }
  [[nodiscard]] std::optional<std::string> optional(std::string_view flag) const {
    for (std::size_t i = 0; i + 1 < values_.size(); ++i) if (values_[i] == flag) return values_[i + 1];
    return std::nullopt;
  }
  [[nodiscard]] Result<std::size_t> size_value(std::string_view flag, std::size_t fallback) const {
    auto raw = optional(flag);
    if (!raw) return fallback;
    try {
      std::size_t consumed = 0;
      const unsigned long long value = std::stoull(*raw, &consumed);
      if (consumed != raw->size()) throw std::invalid_argument("trailing");
      return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
      return Status(ErrorCode::invalid_argument, "invalid integer for " + std::string(flag));
    }
  }
 private:
  std::vector<std::string> values_;
};

void print_usage() {
  std::cout <<
      "Relational Language Engine control tool\n\n"
      "Commands:\n"
      "  rlmctl embeddings build --input embeddings.txt --output embeddings.rle\n"
      "  rlmctl doctor --config production.toml\n"
      "  rlmctl train --config production.toml --corpus corpus.txt [--no-resume] [--no-promote]\n"
      "  rlmctl infer --config production.toml --text \"known tokens\" [--lane both|full|clean] [--depth N]\n"
      "  rlmctl inspect --config production.toml\n"
      "  rlmctl checkpoint --config production.toml\n"
      "  rlmctl expire --config production.toml\n"
      "  rlmctl metrics --config production.toml\n"
      "  rlmctl benchmark --config production.toml --corpus corpus.txt [--samples N]\n"
      "  rlmctl serve --config production.toml [--port N]\n";
}

Result<EngineConfig> load_config(const Arguments& arguments) {
  auto path = arguments.required("--config");
  if (!path) return path.status();
  return EngineConfig::load(path.value());
}

Result<std::unique_ptr<RelationalLanguageEngine>> open_engine(const Arguments& arguments) {
  auto config = load_config(arguments);
  if (!config) return config.status();
  auto engine = std::make_unique<RelationalLanguageEngine>();
  const Status status = engine->open(std::move(config).value());
  if (!status) return status;
  return std::move(engine);
}

std::string search_json(const SearchResult& result, const IEmbeddingStore& embeddings) {
  std::ostringstream output;
  output << "{\"has_prediction\":" << (result.has_prediction ? "true" : "false")
         << ",\"exact_within_beam\":" << (result.exact_within_beam ? "true" : "false")
         << ",\"stable_epoch\":" << (result.stable_epoch ? "true" : "false")
         << ",\"repository_epoch\":" << result.repository_epoch;
  if (result.has_prediction) {
    auto token = embeddings.token(result.best_token);
    output << ",\"token_id\":" << result.best_token
           << ",\"token\":\"" << json_escape(token ? token.value() : std::string_view{"<unknown>"}) << "\""
           << ",\"score\":" << result.score
           << ",\"path\":[";
    for (std::size_t i = 0; i < result.path_tokens.size(); ++i) {
      if (i != 0) output << ',';
      auto path_token = embeddings.token(result.path_tokens[i]);
      output << "{\"edge_id\":" << result.path_edges[i]
             << ",\"token_id\":" << result.path_tokens[i]
             << ",\"token\":\"" << json_escape(path_token ? path_token.value() : std::string_view{"<unknown>"}) << "\"}";
    }
    output << ']';
  }
  output << '}';
  return output.str();
}

std::string inference_json(const DualLaneResult& result, const IEmbeddingStore& embeddings) {
  std::ostringstream output;
  output << "{\"unknown_tokens\":" << result.unknown_tokens;
  if (result.full) output << ",\"full\":" << search_json(*result.full, embeddings);
  if (result.clean) output << ",\"clean\":" << search_json(*result.clean, embeddings);
  output << '}';
  return output.str();
}

std::string health_json(const EngineHealth& health) {
  std::ostringstream output;
  output << "{\"ready\":" << (health.ready ? "true" : "false")
         << ",\"embedding_checksum\":" << health.embedding_checksum
         << ",\"embedding_dimension\":" << health.embedding_dimension
         << ",\"vocabulary_size\":" << health.vocabulary_size
         << ",\"m1_edges\":" << health.m1_edges
         << ",\"m2_edges\":" << health.m2_edges
         << ",\"replay_records\":" << health.replay_records
         << ",\"repository_epoch\":" << health.repository_epoch << '}';
  return output.str();
}

std::string stats_json(const TrainingStats& stats) {
  std::ostringstream output;
  output << "{\"raw_tokens_seen\":" << stats.raw_tokens_seen
         << ",\"known_tokens_seen\":" << stats.known_tokens_seen
         << ",\"unknown_tokens_seen\":" << stats.unknown_tokens_seen
         << ",\"batches_committed\":" << stats.batches_committed
         << ",\"unique_relations_observed\":" << stats.unique_relations_observed
         << ",\"replay_traces_written\":" << stats.replay_traces_written
         << ",\"promotions_committed\":" << stats.promotions_committed
         << ",\"promotions_rejected\":" << stats.promotions_rejected
         << ",\"promotions_unknown\":" << stats.promotions_unknown
         << ",\"m1_expired\":" << stats.m1_expired << '}';
  return output.str();
}

Result<std::vector<TokenId>> read_corpus_tokens(const std::filesystem::path& path,
                                                const IEmbeddingStore& embeddings) {
  std::ifstream input(path);
  if (!input) return Status(ErrorCode::io_error, "cannot open benchmark corpus");
  std::vector<TokenId> output;
  std::string word;
  while (input >> word) {
    auto tokenized = tokenize_whitespace(word, embeddings, false);
    if (tokenized) output.insert(output.end(), tokenized.value().tokens.begin(), tokenized.value().tokens.end());
  }
  if (output.size() < 20) return Status(ErrorCode::invalid_argument, "benchmark corpus has fewer than 20 known tokens");
  return output;
}

std::atomic<bool> stop_server{false};
void signal_handler(int) { stop_server.store(true, std::memory_order_relaxed); }

struct HttpRequest final {
  std::string method;
  std::string target;
  std::map<std::string, std::string> headers;
  std::string body;
};

std::string lower(std::string value) {
  for (char& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

Result<HttpRequest> read_http_request(int fd) {
  constexpr std::size_t kMaxHeader = 64U * 1024U;
  constexpr std::size_t kMaxBody = 4U * 1024U * 1024U;
  std::string buffer;
  buffer.reserve(8192);
  char chunk[4096];
  std::size_t header_end = std::string::npos;
  while ((header_end = buffer.find("\r\n\r\n")) == std::string::npos) {
    const ssize_t count = ::recv(fd, chunk, sizeof(chunk), 0);
    if (count <= 0) return Status(ErrorCode::io_error, "client closed before HTTP header completed");
    buffer.append(chunk, static_cast<std::size_t>(count));
    if (buffer.size() > kMaxHeader) return Status(ErrorCode::resource_exhausted, "HTTP header exceeds 64 KiB");
  }
  HttpRequest request;
  std::istringstream headers(buffer.substr(0, header_end));
  std::string line;
  if (!std::getline(headers, line)) return Status(ErrorCode::invalid_argument, "missing HTTP request line");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  std::istringstream request_line(line);
  std::string version;
  if (!(request_line >> request.method >> request.target >> version) || !version.starts_with("HTTP/1.")) {
    return Status(ErrorCode::invalid_argument, "invalid HTTP request line");
  }
  while (std::getline(headers, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t separator = line.find(':');
    if (separator == std::string::npos) continue;
    std::string name = lower(line.substr(0, separator));
    std::string value = line.substr(separator + 1);
    const std::size_t first = value.find_first_not_of(" \t");
    value = first == std::string::npos ? std::string{} : value.substr(first);
    request.headers[std::move(name)] = std::move(value);
  }
  std::size_t content_length = 0;
  if (const auto found = request.headers.find("content-length"); found != request.headers.end()) {
    try { content_length = static_cast<std::size_t>(std::stoull(found->second)); }
    catch (...) { return Status(ErrorCode::invalid_argument, "invalid Content-Length"); }
  }
  if (content_length > kMaxBody) return Status(ErrorCode::resource_exhausted, "HTTP body exceeds 4 MiB");
  request.body = buffer.substr(header_end + 4);
  while (request.body.size() < content_length) {
    const ssize_t count = ::recv(fd, chunk, std::min(sizeof(chunk), content_length - request.body.size()), 0);
    if (count <= 0) return Status(ErrorCode::io_error, "client closed before HTTP body completed");
    request.body.append(chunk, static_cast<std::size_t>(count));
  }
  if (request.body.size() > content_length) request.body.resize(content_length);
  return request;
}

void send_http(int fd, int status, std::string_view content_type, std::string_view body) {
  const char* reason = status == 200 ? "OK" : status == 400 ? "Bad Request" :
                       status == 404 ? "Not Found" : status == 405 ? "Method Not Allowed" :
                       status == 413 ? "Payload Too Large" : "Internal Server Error";
  std::ostringstream header;
  header << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n"
         << "X-Content-Type-Options: nosniff\r\n\r\n";
  const std::string prefix = header.str();
  auto write_all = [&](std::string_view data) {
    std::size_t offset = 0;
    while (offset < data.size()) {
      const ssize_t count = ::send(fd, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
      if (count <= 0) break;
      offset += static_cast<std::size_t>(count);
    }
  };
  write_all(prefix); write_all(body);
}

std::string query_value(std::string_view target, std::string_view key, std::string_view fallback) {
  const std::size_t question = target.find('?');
  if (question == std::string_view::npos) return std::string(fallback);
  std::string_view query = target.substr(question + 1);
  while (!query.empty()) {
    const std::size_t amp = query.find('&');
    const std::string_view item = query.substr(0, amp);
    const std::size_t equal = item.find('=');
    if (equal != std::string_view::npos && item.substr(0, equal) == key) return std::string(item.substr(equal + 1));
    if (amp == std::string_view::npos) break;
    query.remove_prefix(amp + 1);
  }
  return std::string(fallback);
}

void handle_http(int fd, RelationalLanguageEngine& engine) {
  struct timeval timeout {10, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  auto request = read_http_request(fd);
  if (!request) {
    const int code = request.status().code() == ErrorCode::resource_exhausted ? 413 : 400;
    send_http(fd, code, "application/json", "{\"error\":\"" + json_escape(request.status().to_string()) + "\"}");
    return;
  }
  const std::string_view target_view(request.value().target);
  const std::string_view path = target_view.substr(0, target_view.find('?'));
  if (request.value().method == "GET" && path == "/healthz") {
    send_http(fd, 200, "application/json", health_json(engine.health()));
    return;
  }
  if (request.value().method == "GET" && path == "/metrics") {
    send_http(fd, 200, "text/plain; version=0.0.4", engine.metrics_text());
    return;
  }
  if (request.value().method == "POST" && path == "/v1/infer") {
    const std::string lane = query_value(request.value().target, "lane", "both");
    std::size_t depth = 0;
    try { depth = static_cast<std::size_t>(std::stoull(query_value(request.value().target, "depth", "0"))); }
    catch (...) { send_http(fd, 400, "application/json", "{\"error\":\"invalid depth\"}"); return; }
    auto inference = engine.infer_text(request.value().body, lane, depth);
    if (!inference) {
      send_http(fd, 400, "application/json", "{\"error\":\"" + json_escape(inference.status().to_string()) + "\"}");
      return;
    }
    send_http(fd, 200, "application/json", inference_json(inference.value(), engine.embeddings()));
    return;
  }
  send_http(fd, 404, "application/json", "{\"error\":\"endpoint not found\"}");
}

Status serve(RelationalLanguageEngine& engine, std::uint16_t port, std::size_t worker_count) {
  const int listener = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener < 0) return Status(ErrorCode::io_error, std::string("socket: ") + std::strerror(errno));
  int reuse = 1;
  ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    const Status status(ErrorCode::io_error, std::string("bind: ") + std::strerror(errno)); ::close(listener); return status;
  }
  if (::listen(listener, 256) != 0) {
    const Status status(ErrorCode::io_error, std::string("listen: ") + std::strerror(errno)); ::close(listener); return status;
  }
  stop_server.store(false, std::memory_order_relaxed);
  std::signal(SIGINT, signal_handler); std::signal(SIGTERM, signal_handler);
  std::cerr << "{\"event\":\"server_started\",\"port\":" << port
            << ",\"workers\":" << worker_count << "}\n";
  std::vector<std::jthread> workers;
  workers.reserve(worker_count);
  for (std::size_t i = 0; i < worker_count; ++i) {
    workers.emplace_back([listener, &engine](std::stop_token stop) {
      while (!stop.stop_requested() && !stop_server.load(std::memory_order_relaxed)) {
        pollfd descriptor{listener, POLLIN, 0};
        const int ready = ::poll(&descriptor, 1, 500);
        if (ready <= 0 || (descriptor.revents & POLLIN) == 0) continue;
        const int client = ::accept(listener, nullptr, nullptr);
        if (client < 0) continue;
        const int flags = ::fcntl(client, F_GETFD);
        if (flags >= 0) ::fcntl(client, F_SETFD, flags | FD_CLOEXEC);
        handle_http(client, engine);
        ::close(client);
      }
    });
  }
  while (!stop_server.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (auto& worker : workers) worker.request_stop();
  ::close(listener);
  workers.clear();
  return Status::Ok();
}

int fail(const Status& status) {
  std::cerr << "{\"ok\":false,\"error\":\"" << json_escape(status.to_string()) << "\"}\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments(argc, argv);
  if (arguments.empty() || arguments.has("--help") || arguments.command() == "help") { print_usage(); return 0; }

  if (arguments.command() == "embeddings" && arguments.command(1) == "build") {
    auto input = arguments.required("--input"); auto output = arguments.required("--output");
    if (!input) return fail(input.status()); if (!output) return fail(output.status());
    const Status status = EmbeddingFileBuilder::from_text(input.value(), output.value());
    if (!status) return fail(status);
    std::cout << "{\"ok\":true,\"output\":\"" << json_escape(output.value()) << "\"}\n";
    return 0;
  }

  auto engine_result = open_engine(arguments);
  if (!engine_result) return fail(engine_result.status());
  std::unique_ptr<RelationalLanguageEngine> engine = std::move(engine_result).value();

  if (arguments.command() == "doctor" || arguments.command() == "inspect") {
    std::cout << health_json(engine->health()) << '\n'; return 0;
  }
  if (arguments.command() == "infer") {
    auto text = arguments.required("--text"); if (!text) return fail(text.status());
    const std::string lane = arguments.optional("--lane").value_or("both");
    auto depth = arguments.size_value("--depth", 0); if (!depth) return fail(depth.status());
    auto result = engine->infer_text(text.value(), lane, depth.value());
    if (!result) return fail(result.status());
    std::cout << inference_json(result.value(), engine->embeddings()) << '\n'; return 0;
  }
  if (arguments.command() == "train") {
    auto corpus = arguments.required("--corpus"); if (!corpus) return fail(corpus.status());
    TrainerOptions options; options.resume = !arguments.has("--no-resume"); options.auto_promote = !arguments.has("--no-promote");
    auto result = engine->train(corpus.value(), options); if (!result) return fail(result.status());
    std::cout << stats_json(result.value()) << '\n'; return 0;
  }
  if (arguments.command() == "checkpoint") {
    const Status status = engine->checkpoint(); if (!status) return fail(status);
    std::cout << "{\"ok\":true}\n"; return 0;
  }
  if (arguments.command() == "expire") {
    auto result = engine->expire_now(); if (!result) return fail(result.status());
    std::cout << "{\"expired\":" << result.value() << "}\n"; return 0;
  }
  if (arguments.command() == "metrics") {
    std::cout << engine->metrics_text(); return 0;
  }
  if (arguments.command() == "benchmark") {
    auto corpus = arguments.required("--corpus"); if (!corpus) return fail(corpus.status());
    auto samples_arg = arguments.size_value("--samples", 1000); if (!samples_arg) return fail(samples_arg.status());
    auto tokens = read_corpus_tokens(corpus.value(), engine->embeddings()); if (!tokens) return fail(tokens.status());
    const std::size_t split = std::max<std::size_t>(3, tokens.value().size() * 8 / 10);
    TrigramBaseline trigram; trigram.train(std::span<const TokenId>(tokens.value()).first(split));
    const std::size_t end = std::min(tokens.value().size(), split + samples_arg.value());
    std::size_t relational_correct = 0, trigram_correct = 0, evaluated = 0;
    auto relational_start = std::chrono::steady_clock::now();
    for (std::size_t i = split; i < end; ++i) {
      const std::size_t context_begin = i > 31 ? i - 31 : 0;
      const std::span<const TokenId> context(tokens.value().data() + context_begin, i - context_begin);
      auto relational = engine->infer_tokens(context, "full", 1);
      if (!relational) return fail(relational.status());
      if (relational.value().full && relational.value().full->has_prediction &&
          relational.value().full->best_token == tokens.value()[i]) ++relational_correct;
      auto trigram_prediction = trigram.predict(context);
      if (trigram_prediction && *trigram_prediction == tokens.value()[i]) ++trigram_correct;
      ++evaluated;
    }
    const double relational_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - relational_start).count();
    const std::size_t runtime_samples = std::min<std::size_t>(evaluated, 200);
    auto time_lane = [&](std::string_view lane) {
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t n = 0; n < runtime_samples; ++n) {
        const std::size_t i = split + n;
        const std::size_t context_begin = i > 31 ? i - 31 : 0;
        auto ignored = engine->infer_tokens(std::span<const TokenId>(tokens.value().data() + context_begin, i - context_begin), lane, 1);
        if (!ignored) return -1.0;
      }
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    const double full_seconds = time_lane("full");
    const double both_seconds = time_lane("both");
    std::cout << "{\"evaluated\":" << evaluated
              << ",\"relational_accuracy\":" << (evaluated ? static_cast<double>(relational_correct) / evaluated : 0.0)
              << ",\"trigram_accuracy\":" << (evaluated ? static_cast<double>(trigram_correct) / evaluated : 0.0)
              << ",\"relational_eval_seconds\":" << relational_seconds
              << ",\"full_only_seconds\":" << full_seconds
              << ",\"full_clean_seconds\":" << both_seconds
              << ",\"dual_lane_runtime_ratio\":" << (full_seconds > 0.0 ? both_seconds / full_seconds : 0.0)
              << ",\"trigram_states\":" << trigram.state_count() << "}\n";
    return 0;
  }
  if (arguments.command() == "serve") {
    auto port_arg = arguments.size_value("--port", engine->config().runtime.service_port); if (!port_arg) return fail(port_arg.status());
    if (port_arg.value() > 65535) return fail(Status(ErrorCode::invalid_argument, "port exceeds 65535"));
    const Status status = serve(*engine, static_cast<std::uint16_t>(port_arg.value()), engine->config().runtime.worker_threads);
    return status ? 0 : fail(status);
  }

  print_usage();
  return fail(Status(ErrorCode::invalid_argument, "unknown command"));
}

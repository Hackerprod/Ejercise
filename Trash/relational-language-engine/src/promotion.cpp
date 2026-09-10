#include "rlm/promotion.hpp"

#include "rlm/binary.hpp"

#include <algorithm>

namespace rlm {
namespace {
constexpr std::uint16_t kPrepare = 1;
constexpr std::uint16_t kCommit = 2;

std::vector<std::byte> prepare_payload(std::uint64_t transaction, const RelationEdge& edge) {
  ByteWriter writer;
  writer.u64(transaction);
  const auto serialized = serialize_relation_edge(edge);
  writer.u32(static_cast<std::uint32_t>(serialized.size()));
  writer.bytes(serialized);
  return writer.take();
}
}  // namespace

Status PromotionManager::open(const std::filesystem::path& root, Durability durability) {
  if (opened_) return Status(ErrorCode::failed_precondition, "promotion manager already open");
  RLM_RETURN_IF_ERROR(ensure_directory(root));
  RLM_RETURN_IF_ERROR(journal_.open(root / "promotion.wal", durability));
  RLM_RETURN_IF_ERROR(journal_.replay([this](const WalRecord& record) { return apply_journal(record); }, true));
  RLM_RETURN_IF_ERROR(recover_pending());
  opened_ = true;
  return Status::Ok();
}

Result<PromotionResult> PromotionManager::promote(EdgeId edge_id) {
  if (!opened_) return Status(ErrorCode::failed_precondition, "promotion manager is not open");
  auto pin = relations_.pin_m1(edge_id);
  if (!pin) return pin.status();
  auto edge = relations_.get(edge_id, EvidenceTier::m1);
  if (!edge) return edge.status();
  auto audit = auditor_.audit(edge.value(), edge.value().evidence);
  if (!audit) return audit.status();
  PromotionResult result;
  result.verdict = audit.value().verdict;
  result.audit = audit.value();
  if (result.verdict != AuditVerdict::promote) {
    result.message = result.verdict == AuditVerdict::reject
        ? "audit rejected promotion; edge remains physically in M1"
        : "audit is inconclusive; edge remains pinned in M1 until this attempt completes";
    return result;
  }
  std::lock_guard commit_lock(commit_mutex_);
  auto current = relations_.get(edge_id, EvidenceTier::m1);
  if (!current) {
    auto existing = relations_.get(edge_id, EvidenceTier::m2);
    if (existing) {
      result.committed = true;
      result.message = "edge was already promoted by another committed path";
      return result;
    }
    return current.status();
  }
  const std::uint64_t transaction = next_transaction_.fetch_add(1, std::memory_order_relaxed);
  const std::vector<std::byte> prepare = prepare_payload(transaction, current.value());
  auto prepared = journal_.append(kPrepare, prepare);
  if (!prepared) return prepared.status();
  pending_[transaction] = Pending{transaction, current.value()};
  const Status commit_status = commit_locked(transaction, current.value());
  if (!commit_status) return commit_status;
  result.committed = true;
  result.message = "M1 edge promoted transactionally into the physically separate M2 store";
  return result;
}

Status PromotionManager::flush() { return journal_.flush(); }

Status PromotionManager::apply_journal(const WalRecord& record) {
  ByteReader reader(record.payload);
  auto transaction = reader.u64();
  if (!transaction) return transaction.status();
  next_transaction_.store(std::max(next_transaction_.load(std::memory_order_relaxed), transaction.value() + 1),
                          std::memory_order_relaxed);
  if (record.kind == kPrepare) {
    auto size = reader.u32();
    if (!size || size.value() > 256U * 1024U * 1024U) return Status(ErrorCode::data_loss, "invalid promotion prepare size");
    auto payload = reader.bytes(size.value());
    if (!payload || reader.remaining() != 0) return Status(ErrorCode::data_loss, "truncated promotion prepare");
    auto edge = deserialize_relation_edge(payload.value(), 0);
    if (!edge) return edge.status();
    pending_[transaction.value()] = Pending{transaction.value(), std::move(edge).value()};
    return Status::Ok();
  }
  if (record.kind == kCommit) {
    if (reader.remaining() != 0) return Status(ErrorCode::data_loss, "trailing bytes in promotion commit");
    pending_.erase(transaction.value());
    return Status::Ok();
  }
  return Status(ErrorCode::data_loss, "unknown promotion journal record kind");
}

Status PromotionManager::recover_pending() {
  std::lock_guard commit_lock(commit_mutex_);
  std::vector<Pending> pending;
  pending.reserve(pending_.size());
  for (const auto& [transaction, item] : pending_) { (void)transaction; pending.push_back(item); }
  std::sort(pending.begin(), pending.end(), [](const Pending& a, const Pending& b) { return a.transaction < b.transaction; });
  for (const Pending& item : pending) RLM_RETURN_IF_ERROR(commit_locked(item.transaction, item.edge));
  return Status::Ok();
}

Status PromotionManager::commit_locked(std::uint64_t transaction, const RelationEdge& edge) {
  auto existing = relations_.get(edge.id, EvidenceTier::m2);
  if (!existing) {
    if (existing.status().code() != ErrorCode::not_found) return existing.status();
    RLM_RETURN_IF_ERROR(relations_.upsert_m2(edge));
  } else if (existing.value().source != edge.source || existing.value().target != edge.target ||
             existing.value().relation.values != edge.relation.values ||
             existing.value().relation.scale != edge.relation.scale) {
    return Status(ErrorCode::data_loss, "promotion recovery found conflicting M2 content");
  }
  RLM_RETURN_IF_ERROR(relations_.erase_m1(edge.id));
  ByteWriter commit;
  commit.u64(transaction);
  auto committed = journal_.append(kCommit, commit.data());
  if (!committed) return committed.status();
  pending_.erase(transaction);
  return Status::Ok();
}

}  // namespace rlm

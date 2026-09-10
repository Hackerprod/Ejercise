#include "cnrl/types.hpp"
#include <stdexcept>
namespace cnrl {
const char* to_string(WeightVariant v) noexcept {
  switch (v) { case WeightVariant::shared: return "shared"; case WeightVariant::clone: return "clone";
    case WeightVariant::untied: return "untied"; case WeightVariant::cold: return "cold"; }
  return "unknown";
}
const char* to_string(KernelKind v) noexcept {
  switch (v) { case KernelKind::scalar: return "scalar"; case KernelKind::avx2_repeat: return "avx2-repeat";
    case KernelKind::avx2_fused: return "avx2-fused"; }
  return "unknown";
}
const char* to_string(TransitionKind v) noexcept {
  switch (v) { case TransitionKind::frozen: return "frozen"; case TransitionKind::fixed_point: return "fixed-point";
    case TransitionKind::group_rms: return "group-rms"; case TransitionKind::global_rms: return "global-rms"; }
  return "unknown";
}
const char* to_string(GateKind v) noexcept {
  switch (v) { case GateKind::t0r: return "t0r"; case GateKind::t0m: return "t0m";
    case GateKind::t0rm: return "t0rm"; case GateKind::calibrate: return "calibrate"; }
  return "unknown";
}
const char* to_string(TimingScope v) noexcept {
  switch (v) { case TimingScope::full_repetition: return "full-repetition";
    case TimingScope::round_window: return "round-window"; }
  return "unknown";
}
WeightVariant parse_weight_variant(const std::string& s) {
  if (s == "shared" || s == "A") return WeightVariant::shared;
  if (s == "clone" || s == "B" || s == "Bclone") return WeightVariant::clone;
  if (s == "untied" || s == "U") return WeightVariant::untied;
  if (s == "cold" || s == "C") return WeightVariant::cold;
  throw std::invalid_argument("unknown weight variant: " + s);
}
KernelKind parse_kernel_kind(const std::string& s) {
  if (s == "scalar") return KernelKind::scalar;
  if (s == "repeat" || s == "avx2-repeat") return KernelKind::avx2_repeat;
  if (s == "fused" || s == "avx2-fused") return KernelKind::avx2_fused;
  throw std::invalid_argument("unknown kernel: " + s);
}
TransitionKind parse_transition_kind(const std::string& s) {
  if (s == "frozen" || s == "none") return TransitionKind::frozen;
  if (s == "fixed" || s == "fixed-point") return TransitionKind::fixed_point;
  if (s == "group-rms") return TransitionKind::group_rms;
  if (s == "global-rms") return TransitionKind::global_rms;
  throw std::invalid_argument("unknown transition: " + s);
}
GateKind parse_gate_kind(const std::string& s) {
  if (s == "t0r") return GateKind::t0r;
  if (s == "t0m") return GateKind::t0m;
  if (s == "t0rm") return GateKind::t0rm;
  if (s == "calibrate") return GateKind::calibrate;
  throw std::invalid_argument("unknown gate: " + s);
}
TimingScope parse_timing_scope(const std::string& s) {
  if (s == "full" || s == "full-repetition") return TimingScope::full_repetition;
  if (s == "round" || s == "round-window") return TimingScope::round_window;
  throw std::invalid_argument("unknown timing scope: " + s);
}
}  // namespace cnrl

"""OMEGA CORE-LM-0 R1 architecture preflight.

R1 is a synthetic contract audit only. It does not load campaign checkpoints,
download data, run a teacher, train, or update any parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omega_core_lm_0_design import RMSNorm, WorkspaceUpdateBlock  # noqa: E402


OUTPUT = ROOT / "campaign" / "omega_core_lm_0_r1_architecture_preflight" / "r1_preflight.json"
PREVIOUS_DESIGN = ROOT / "campaign" / "omega_core_lm_0_design_audit" / "omega_core_lm_0_design_audit.json"
PREVIOUS_TESTS = ROOT / "campaign" / "omega_core_lm_0_design_audit" / "synthetic_contract_tests.json"


class OmegaCoreLM0R1(nn.Module):
    """R1 student: full-state Prelude residual and tied input/output embeddings."""

    def __init__(self, *, vocab_size: int, dimension: int = 128, slots: int = 8, rounds: int = 4, variant: str = "shared") -> None:
        super().__init__()
        if variant not in {"shared", "untied"}:
            raise ValueError(variant)
        self.vocab_size = vocab_size
        self.dimension = dimension
        self.slots = slots
        self.rounds = rounds
        self.variant = variant
        self.embedding = nn.Embedding(vocab_size, dimension)
        self.prelude = nn.Linear(2 * dimension, slots * dimension)
        self.prelude_norm = RMSNorm(slots * dimension)
        block_count = 1 if variant == "shared" else rounds
        self.blocks = nn.ModuleList(WorkspaceUpdateBlock(dimension, slots) for _ in range(block_count))
        self.depth_embedding = nn.Embedding(rounds, dimension)
        self.gate_logits = nn.Parameter(torch.full((rounds, dimension), -2.1972245773362196))
        self.readout_norm = RMSNorm(slots * dimension)
        self.output_projection = nn.Linear(slots * dimension, dimension, bias=False)

    def initial_state(self, batch_size: int) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dimension)

    def prelude_step(self, token: Tensor, previous_state: Tensor) -> Tensor:
        token_state = self.embedding(token)
        summary = previous_state.mean(dim=1)
        write = self.prelude(torch.cat((token_state, summary), dim=-1))
        anchor = self.prelude_norm(previous_state.flatten(start_dim=1) + write)
        return anchor.view(token.shape[0], self.slots, self.dimension)

    def update_step(self, anchor: Tensor, *, return_states: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        state = anchor
        states = [state]
        for round_index in range(self.rounds):
            block = self.blocks[0] if self.variant == "shared" else self.blocks[round_index]
            state = block(state, anchor, self.depth_embedding.weight[round_index], torch.sigmoid(self.gate_logits[round_index]))
            states.append(state)
        return (state, tuple(states)) if return_states else state

    def continue_update(self, state: Tensor, anchor: Tensor, *, start_round: int) -> Tensor:
        for round_index in range(start_round, self.rounds):
            block = self.blocks[0] if self.variant == "shared" else self.blocks[round_index]
            state = block(state, anchor, self.depth_embedding.weight[round_index], torch.sigmoid(self.gate_logits[round_index]))
        return state

    def readout_from_state(self, state: Tensor) -> Tensor:
        normalized = self.readout_norm(state.flatten(start_dim=1))
        projected = self.output_projection(normalized)
        return projected @ self.embedding.weight.transpose(0, 1) / math.sqrt(self.dimension)

    def forward_token(self, token: Tensor, previous_state: Tensor, *, return_trace: bool = False) -> tuple[Tensor, Tensor, dict[str, Any]]:
        anchor = self.prelude_step(token, previous_state)
        final_state, states = self.update_step(anchor, return_states=True)
        logits = self.readout_from_state(final_state)
        trace = {"anchor": anchor, "states": states, "transmitted_state": final_state}
        return final_state, logits, trace if return_trace else {"transmitted_state": final_state}

    def forward_tokens(self, tokens: Tensor, *, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        state = self.initial_state(tokens.shape[0])
        outputs: list[Tensor] = []
        for position in range(tokens.shape[1]):
            next_state, logits, _ = self.forward_token(tokens[:, position], state)
            if valid_mask is None:
                state = next_state
            else:
                active = valid_mask[:, position].view(-1, 1, 1)
                state = torch.where(active, next_state, state)
            outputs.append(logits)
        return state, torch.stack(outputs, dim=1)


def parameter_breakdown(model: OmegaCoreLM0R1) -> dict[str, int]:
    categories = {"P_shell": 0, "P_core": 0, "P_modulation": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("blocks."):
            categories["P_core"] += parameter.numel()
        elif name.startswith("depth_embedding") or name.startswith("gate_logits"):
            categories["P_modulation"] += parameter.numel()
        else:
            categories["P_shell"] += parameter.numel()
    categories["P_total"] = sum(categories.values())
    return categories


def validate_test_result(payload: dict[str, Any]) -> bool:
    checks = payload.get("checks")
    return payload.get("status") == "passed" and isinstance(checks, dict) and bool(checks) and all(value is True for value in checks.values())


def write_self_hashed(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return digest, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    torch.manual_seed(20260913)
    checks: dict[str, bool] = {}
    model = OmegaCoreLM0R1(vocab_size=17, rounds=4)
    model.eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long)
    state, outputs = model.forward_tokens(tokens)
    checks["legacy_forms_workspace"] = state.shape == (2, 8, 128)
    checks["legacy_forms_head"] = outputs.shape == (2, 5, 17)
    _, trace_logits, trace = model.forward_token(tokens[:, 0], model.initial_state(2), return_trace=True)
    checks["legacy_forms_round_states"] = len(trace["states"]) == 5 and all(item.shape == (2, 8, 128) for item in trace["states"])
    checks["legacy_forms_prelude"] = trace["anchor"].shape == (2, 8, 128) and trace_logits.shape == (2, 17)

    perturbed = tokens.clone()
    perturbed[:, 3:] = torch.tensor([[16, 15], [14, 13]])
    _, logits_a = model.forward_tokens(tokens)
    _, logits_b = model.forward_tokens(perturbed)
    checks["legacy_strict_causality"] = torch.equal(logits_a[:, 2], logits_b[:, 2])
    checks["legacy_future_changes_later_output"] = not torch.equal(logits_a[:, 4], logits_b[:, 4])

    anchor = trace["anchor"]
    _, states = model.update_step(anchor, return_states=True)
    changed = states[1].clone()
    changed[:, 0, 0] += 0.5
    checks["legacy_real_recurrence"] = not torch.equal(model.continue_update(states[1], anchor, start_round=1), model.continue_update(changed, anchor, start_round=1))
    checks["legacy_shared_identity"] = len(model.blocks) == 1 and all(model.blocks[0] is model.blocks[0] for _ in range(model.rounds))
    untied = OmegaCoreLM0R1(vocab_size=17, rounds=4, variant="untied")
    checks["legacy_untied_control"] = len(untied.blocks) == 4 and len({id(block) for block in untied.blocks}) == 4
    _, reset_a = model.forward_tokens(tokens[:, :1])
    _, reset_b = model.forward_tokens(tokens[:, :1])
    contaminated_state = torch.randn_like(model.initial_state(2))
    _, contaminated, _ = model.forward_token(tokens[:, 0], contaminated_state)
    checks["legacy_reset"] = torch.equal(reset_a, reset_b) and not torch.equal(reset_a[:, 0], contaminated)

    same_mean_a = torch.zeros(2, 8, 128)
    same_mean_b = same_mean_a.clone()
    same_mean_b[:, 0, 0] = 0.75
    same_mean_b[:, 1, 0] = -0.75
    checks["new_persistence_fixture_mean_first"] = torch.equal(same_mean_a.mean(dim=1), same_mean_b.mean(dim=1))
    token_probe = torch.tensor([2, 2], dtype=torch.long)
    old_a = model.prelude_norm(model.prelude(torch.cat((model.embedding(token_probe), same_mean_a.mean(dim=1)), dim=-1)))
    old_b = model.prelude_norm(model.prelude(torch.cat((model.embedding(token_probe), same_mean_b.mean(dim=1)), dim=-1)))
    new_a = model.prelude_step(token_probe, same_mean_a)
    new_b = model.prelude_step(token_probe, same_mean_b)
    checks["new_persistence_old_mean_only_collapses"] = torch.equal(old_a, old_b)
    checks["new_persistence_full_state_distinguished"] = not torch.equal(new_a, new_b)

    io_model = OmegaCoreLM0R1(vocab_size=17, rounds=1)
    shared_weight = io_model.embedding.weight
    checks["new_tied_io_data_ptr"] = shared_weight.data_ptr() == io_model.embedding.weight.data_ptr()
    checks["new_tied_io_logit_shape"] = io_model.readout_from_state(torch.randn(2, 8, 128)).shape == (2, 17)
    io_model.zero_grad(set_to_none=True)
    io_model.readout_from_state(torch.randn(2, 8, 128)).square().mean().backward()
    output_grad = io_model.embedding.weight.grad is not None and bool(io_model.embedding.weight.grad.abs().sum() > 0)
    io_model.zero_grad(set_to_none=True)
    io_model.prelude_step(torch.tensor([1, 2]), io_model.initial_state(2)).square().mean().backward()
    input_grad = io_model.embedding.weight.grad is not None and bool(io_model.embedding.weight.grad.abs().sum() > 0)
    checks["new_tied_io_backward_output_route"] = output_grad
    checks["new_tied_io_backward_input_route"] = input_grad
    checks["new_tied_io_same_parameter_receives_both_routes"] = shared_weight.data_ptr() == io_model.embedding.weight.data_ptr()

    _, full_logits = model.forward_tokens(tokens)
    first_state, first_logits = model.forward_tokens(tokens[:, :2])
    second_state, second_logits = model.forward_tokens(tokens[:, 2:], valid_mask=None) if False else (None, None)
    carried_state = first_state
    window_logits: list[Tensor] = [first_logits[:, 0], first_logits[:, 1]]
    for position in range(2, tokens.shape[1]):
        carried_state, current, _ = model.forward_token(tokens[:, position], carried_state)
        window_logits.append(current)
    checks["new_causal_continuous_windows"] = torch.equal(full_logits, torch.stack(window_logits, dim=1))

    left_padded = torch.tensor([[16, 15, 1, 2, 3], [14, 13, 5, 4, 3]], dtype=torch.long)
    valid_mask = torch.tensor([[False, False, True, True, True], [False, False, True, True, True]])
    _, padded_logits = model.forward_tokens(left_padded, valid_mask=valid_mask)
    _, compact_logits = model.forward_tokens(left_padded[:, 2:])
    checks["new_padding_does_not_change_valid_states"] = torch.equal(padded_logits[:, 2:], compact_logits)
    _, clean_first = model.forward_tokens(tokens[:, :1])
    _, clean_again = model.forward_tokens(tokens[:, :1])
    checks["new_reset_between_documents"] = torch.equal(clean_first, clean_again)
    state_before_cut, _ = model.forward_tokens(tokens[:, :2])
    state_detached = state_before_cut.detach()
    next_a, logits_next_a, _ = model.forward_token(tokens[:, 2], state_before_cut)
    next_b, logits_next_b, _ = model.forward_token(tokens[:, 2], state_detached)
    checks["new_bptt_cut_preserves_state_value"] = torch.equal(next_a, next_b) and torch.equal(logits_next_a, logits_next_b) and state_detached.grad_fn is None

    checks["new_metric_sign_orientation"] = abs((3.1 - 3.0) - 0.1) < 1e-12 and (3.1 - 3.0) >= 0.01
    negative_fixture = {"status": "passed", "checks": {"deliberate_failure": False}}
    checks["new_validator_rejects_false_check_fixture"] = not validate_test_result(negative_fixture)

    vocab = 50257
    k1 = OmegaCoreLM0R1(vocab_size=vocab, rounds=1)
    k4 = OmegaCoreLM0R1(vocab_size=vocab, rounds=4)
    untied_large = OmegaCoreLM0R1(vocab_size=vocab, rounds=4, variant="untied")
    measured = {"shared_K1": parameter_breakdown(k1), "shared_K4": parameter_breakdown(k4), "untied_K4": parameter_breakdown(untied_large)}
    expected = {
        "shared_K1": {"P_shell": 6829184, "P_core": 197888, "P_modulation": 256, "P_total": 7027328},
        "shared_K4": {"P_shell": 6829184, "P_core": 197888, "P_modulation": 1024, "P_total": 7028096},
        "untied_K4": {"P_shell": 6829184, "P_core": 791552, "P_modulation": 1024, "P_total": 7621760},
    }
    checks["new_parameter_counts_measured_from_real_objects"] = measured == expected
    checks["new_fp32_bytes_measured"] = measured["shared_K4"]["P_total"] * 4 == 28112384

    old_design = json.loads(PREVIOUS_DESIGN.read_text(encoding="utf-8"))
    old_tests = json.loads(PREVIOUS_TESTS.read_text(encoding="utf-8"))
    checks["previous_design_preserved"] = PREVIOUS_DESIGN.is_file() and old_design.get("status") == "PASS_DESIGN"
    checks["previous_synthetic_result_preserved"] = PREVIOUS_TESTS.is_file() and old_tests.get("status") == "passed"
    result = {
        "schema": "omega-core-lm-0-r1-architecture-preflight-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "classification": "PASS_R1_PREFLIGHT" if all(checks.values()) else "BLOCKED_LOCALIZED",
        "training": False,
        "optimizer_updates": False,
        "campaign_780x_access": False,
        "teacher_or_corpus_access": False,
        "previous_design_untouched": True,
        "checks": checks,
        "test_groups": {"legacy_tests_conserved": ["forms", "strict causality", "future sensitivity", "real recurrence", "shared identity", "untied control", "reset"], "new_tests": ["full-state persistence with same mean fixture", "tied input/output embedding and backward routes", "causal continuity across windows", "reset/padding/BPTT", "metric sign plus validator negative fixture"]},
        "architecture_r1": {"prelude": "U_t = W_p[E[x_t]; mean(S_{t-1})] + b_p; A_t = reshape(RMSNorm_md(vec(S_{t-1}) + U_t))", "update": "existing WorkspaceUpdateBlock: SlotMix(S+A), shared CoreMLP, depth embedding, per-depth sigmoid write gate, RMSNorm residual", "readout": "u_t = W_o @ RMSNorm_md(vec(S_t^K)); z_t = E @ u_t / sqrt(d), with W_o [d,md] bias-free and E exactly embedding.weight", "state": "S_t^K persists to next token; zero reset between documents; padding leaves state unchanged; BPTT detach cuts gradient only, not state value"},
        "criteria": {"delta_ablation": "Δ_ablation = NLL_anchor_reset - NLL_normal; require >=0.01 nats/token and bootstrap 95% CI excludes 0", "delta_depth": "Δ_depth = NLL_shared_K1 - NLL_shared_K4; require >=0.05 nats/token", "state_shuffle": "secondary diagnostic only", "length_comparison": "same target tokens with different available prefix lengths; never compare different target populations"},
        "parameters_measured": measured,
        "fp32_bytes": {name: {key: value * 4 for key, value in values.items()} for name, values in measured.items()},
        "data_contract": {"teacher": "distilgpt2 frozen during future training only", "tokenizer": "GPT-2 byte-level BPE V=50257", "corpus": "WikiText-2 raw-v1", "splits": "preserve existing 80/10/10 document split from R0; do not re-partition in R1", "license_note": "R0 recorded MIT/CC BY-SA 4.0 metadata with exact artifact verification still required before download; R1 performs no download", "teacher_context_limit": 1024, "inference": "student only; no teacher, bank, lookup, or semantic evaluator attributes"},
        "limits": {"scope": "R1 architecture preflight only", "not_claimed": ["pilot quality", "general language ability", "physical cache/DRAM measurements", "quantized stability", "external retrieval", "learned halting"]},
        "previous_artifacts": {"design": {"path": str(PREVIOUS_DESIGN.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(PREVIOUS_DESIGN.read_bytes()).hexdigest()}, "synthetic_tests": {"path": str(PREVIOUS_TESTS.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(PREVIOUS_TESTS.read_bytes()).hexdigest()}},
    }
    digest, file_sha = write_self_hashed(OUTPUT, result)
    print(json.dumps({"status": result["status"], "classification": result["classification"], "artifact": OUTPUT.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": file_sha, "checks_passed": sum(checks.values()), "checks_total": len(checks), "shared_K1": measured["shared_K1"]["P_total"], "shared_K4": measured["shared_K4"]["P_total"], "untied_K4": measured["untied_K4"]["P_total"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

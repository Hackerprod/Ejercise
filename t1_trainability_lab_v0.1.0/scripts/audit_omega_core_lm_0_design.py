"""Synthetic executable contract tests for the OMEGA CORE-LM-0 design.

This file performs no training, corpus access, checkpoint loading, or model
forward over campaign artifacts. It exercises only a fresh synthetic student
module implementing the proposed causal workspace interface.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability.model import CoreMLP, RMSNorm, SlotMix


OUTPUT = ROOT / "campaign" / "omega_core_lm_0_design_audit" / "synthetic_contract_tests.json"


class WorkspaceUpdateBlock(nn.Module):
    """One update block; shared variant reuses one instance across all rounds."""

    def __init__(self, dimension: int, slots: int) -> None:
        super().__init__()
        self.slot_mix = SlotMix(dimension)
        self.core = CoreMLP(dimension)
        self.rms_norm = RMSNorm(dimension)
        self.slots = slots

    def forward(self, state: Tensor, anchor: Tensor, depth: Tensor, gate: Tensor) -> Tensor:
        mixed = self.slot_mix(state + anchor)
        update = self.core(mixed + depth.view(1, 1, -1))
        return self.rms_norm(state + gate.view(1, 1, -1) * update)


class OmegaCoreLM0(nn.Module):
    """Minimal causal student used only for synthetic shape/contract tests."""

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
        self.head = nn.Linear(slots * dimension, vocab_size)

    @property
    def shared_core_parameter_ids(self) -> tuple[int, ...]:
        return tuple(id(parameter) for parameter in self.blocks[0].parameters())

    def initial_state(self, batch_size: int, *, device: torch.device | None = None) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dimension, device=device)

    def prelude_step(self, token: Tensor, previous_state: Tensor) -> Tensor:
        token_state = self.embedding(token)
        summary = previous_state.mean(dim=1)
        anchor = self.prelude_norm(self.prelude(torch.cat((token_state, summary), dim=-1)))
        return anchor.view(token.shape[0], self.slots, self.dimension)

    def update_step(self, anchor: Tensor, *, return_states: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
        state = anchor
        states = [state]
        for round_index in range(self.rounds):
            block = self.blocks[0] if self.variant == "shared" else self.blocks[round_index]
            depth = self.depth_embedding.weight[round_index]
            gate = torch.sigmoid(self.gate_logits[round_index])
            state = block(state, anchor, depth, gate)
            states.append(state)
        return (state, tuple(states)) if return_states else state

    def continue_update(self, state: Tensor, anchor: Tensor, *, start_round: int) -> Tensor:
        for round_index in range(start_round, self.rounds):
            block = self.blocks[0] if self.variant == "shared" else self.blocks[round_index]
            state = block(state, anchor, self.depth_embedding.weight[round_index], torch.sigmoid(self.gate_logits[round_index]))
        return state

    def forward_token(self, token: Tensor, previous_state: Tensor, *, return_trace: bool = False) -> tuple[Tensor, Tensor, dict[str, Any]]:
        anchor = self.prelude_step(token, previous_state)
        final_state, states = self.update_step(anchor, return_states=True)
        flat = final_state.flatten(start_dim=1)
        logits = self.head(self.readout_norm(flat))
        trace = {"anchor": anchor, "states": states, "transmitted_state": final_state}
        return final_state, logits, trace if return_trace else {"transmitted_state": final_state}


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


def assert_true(checks: dict[str, bool], name: str, value: bool) -> None:
    checks[name] = bool(value)
    if not value:
        raise AssertionError(name)


def parameter_breakdown(model: OmegaCoreLM0) -> dict[str, int]:
    core_names = {"blocks"}
    categories = {"P_shell": 0, "P_core": 0, "P_modulation": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("blocks."):
            categories["P_core"] += parameter.numel()
        elif name.startswith("depth_embedding") or name.startswith("gate_logits"):
            categories["P_modulation"] += parameter.numel()
        else:
            categories["P_shell"] += parameter.numel()
    del core_names
    categories["P_total"] = sum(categories.values())
    return categories


def main() -> int:
    torch.manual_seed(20260913)
    checks: dict[str, bool] = {}
    small = OmegaCoreLM0(vocab_size=17, rounds=4)
    small.eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long)
    state = small.initial_state(tokens.shape[0])
    traces: list[dict[str, Any]] = []
    logits: list[Tensor] = []
    for position in range(tokens.shape[1]):
        state, current_logits, trace = small.forward_token(tokens[:, position], state, return_trace=True)
        logits.append(current_logits)
        traces.append(trace)
    assert_true(checks, "forms_workspace", all(item["transmitted_state"].shape == (2, 8, 128) for item in traces))
    assert_true(checks, "forms_prelude", all(item["anchor"].shape == (2, 8, 128) for item in traces))
    assert_true(checks, "forms_round_states", all(len(item["states"]) == 5 and all(round_state.shape == (2, 8, 128) for round_state in item["states"]) for item in traces))
    assert_true(checks, "forms_head", all(item.shape == (2, 17) for item in logits))

    perturbed = tokens.clone()
    perturbed[:, 3:] = torch.tensor([[16, 15], [14, 13]])
    state_a = small.initial_state(2)
    state_b = small.initial_state(2)
    prefix_logits_a: list[Tensor] = []
    prefix_logits_b: list[Tensor] = []
    for position in range(tokens.shape[1]):
        state_a, output_a, _ = small.forward_token(tokens[:, position], state_a)
        state_b, output_b, _ = small.forward_token(perturbed[:, position], state_b)
        prefix_logits_a.append(output_a)
        prefix_logits_b.append(output_b)
    assert_true(checks, "strict_causality_before_perturbation", torch.equal(prefix_logits_a[2], prefix_logits_b[2]))
    assert_true(checks, "future_can_change_later_output", not torch.equal(prefix_logits_a[4], prefix_logits_b[4]))

    anchor = traces[0]["anchor"]
    _, round_states = small.update_step(anchor, return_states=True)
    changed_round_one = round_states[1].clone()
    changed_round_one[:, 0, 0] += 0.5
    continuation_original = small.continue_update(round_states[1], anchor, start_round=1)
    continuation_changed = small.continue_update(changed_round_one, anchor, start_round=1)
    assert_true(checks, "real_recurrence_round_r_plus_1_consumes_round_r", not torch.equal(continuation_original, continuation_changed))

    shared_ids = small.shared_core_parameter_ids
    assert_true(checks, "shared_core_has_one_block", len(small.blocks) == 1)
    assert_true(checks, "shared_core_parameter_identity", all(tuple(id(parameter) for parameter in small.blocks[0].parameters()) == shared_ids for _ in range(small.rounds)))
    assert_true(checks, "depth_modulation_counted_apart", small.depth_embedding.num_embeddings == small.rounds and small.gate_logits.shape == (small.rounds, small.dimension))
    untied = OmegaCoreLM0(vocab_size=17, rounds=4, variant="untied")
    assert_true(checks, "untied_control_has_four_blocks", len(untied.blocks) == 4)
    assert_true(checks, "untied_control_parameter_addresses_distinct", len({id(parameter) for block in untied.blocks for parameter in block.parameters()}) == sum(1 for block in untied.blocks for _ in block.parameters()))

    reset_state = small.initial_state(2)
    _, reset_logits_a, _ = small.forward_token(tokens[:, 0], reset_state)
    contaminated = torch.randn_like(reset_state)
    _, reset_logits_b, _ = small.forward_token(tokens[:, 0], small.initial_state(2))
    _, contaminated_logits, _ = small.forward_token(tokens[:, 0], contaminated)
    assert_true(checks, "reset_reproduces_same_sequence", torch.equal(reset_logits_a, reset_logits_b))
    assert_true(checks, "previous_state_is_consumed", not torch.equal(reset_logits_a, contaminated_logits))

    vocab = 50257
    dimension = 128
    slots = 8
    p_core_block = 197888
    p_shell_expected = vocab * dimension + ((2 * dimension) * (slots * dimension) + slots * dimension) + (slots * dimension) + (slots * dimension) + ((slots * dimension) * vocab + vocab)
    p_mod_expected_k1 = dimension + dimension
    p_mod_expected_k4 = 4 * dimension + 4 * dimension
    k1 = OmegaCoreLM0(vocab_size=vocab, rounds=1)
    k4 = OmegaCoreLM0(vocab_size=vocab, rounds=4)
    untied_large = OmegaCoreLM0(vocab_size=vocab, rounds=4, variant="untied")
    b1 = parameter_breakdown(k1)
    b4 = parameter_breakdown(k4)
    bu = parameter_breakdown(untied_large)
    assert_true(checks, "parameter_count_k1", b1 == {"P_shell": p_shell_expected, "P_core": p_core_block, "P_modulation": p_mod_expected_k1, "P_total": p_shell_expected + p_core_block + p_mod_expected_k1})
    assert_true(checks, "parameter_count_k4", b4 == {"P_shell": p_shell_expected, "P_core": p_core_block, "P_modulation": p_mod_expected_k4, "P_total": p_shell_expected + p_core_block + p_mod_expected_k4})
    assert_true(checks, "parameter_count_untied_control", bu == {"P_shell": p_shell_expected, "P_core": 4 * p_core_block, "P_modulation": p_mod_expected_k4, "P_total": p_shell_expected + 4 * p_core_block + p_mod_expected_k4})
    bytes_k1 = {key: value * 4 for key, value in b1.items()}
    bytes_k4 = {key: value * 4 for key, value in b4.items()}
    assert_true(checks, "byte_count_fp32", bytes_k1["P_total"] == 233638724 and bytes_k4["P_total"] == 233641796)

    result = {
        "schema": "omega-core-lm-0-synthetic-contract-tests-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "training": False,
        "corpus_access": False,
        "checkpoint_access": False,
        "campaign_780x_access": False,
        "checks": checks,
        "parameter_breakdown": {"d": dimension, "m": slots, "vocab": vocab, "fp32_bytes": {"K1": bytes_k1, "K4": bytes_k4, "untied_K4": {key: value * 4 for key, value in bu.items()}}, "parameters": {"K1": b1, "K4": b4, "untied_K4": bu}},
        "contract": {"state_shape": "[B,m,d]", "anchor_shape": "[B,m,d]", "rounds": "state[r+1] consumes state[r] and same anchor A_t", "transmitted_state": "S_t^(K)"},
    }
    digest, file_sha = write_self_hashed(OUTPUT, result)
    print(json.dumps({"status": result["status"], "artifact": OUTPUT.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": file_sha, "checks_passed": sum(checks.values()), "checks_total": len(checks)}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

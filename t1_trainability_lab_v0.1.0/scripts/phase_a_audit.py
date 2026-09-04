"""Phase A autograd and computation-graph audit; performs no optimizer step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t1_trainability import InputAdapter, OutputReader, RecurrentCore, TokenVocabulary  # noqa: E402
from t1_trainability.data import encode_batch, load_jsonl  # noqa: E402


SEED = 101
TASK = "multi_hop"
BATCH_SIZE = 128
EPSILON = 0.1


def rms(value: Tensor) -> float:
    return float(value.square().mean().sqrt())


def per_round_rms(value: Tensor) -> list[float]:
    return [rms(item) for item in value]


def load_minibatch(vocabulary: TokenVocabulary) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    examples = load_jsonl(ROOT / "datasets" / TASK / "train.jsonl")[:BATCH_SIZE]
    return encode_batch(examples, vocabulary)


def forward_audit(
    adapter: InputAdapter,
    core: RecurrentCore,
    reader: OutputReader,
    input_ids: Tensor,
    mask: Tensor,
    query_ids: Tensor,
    *,
    alpha: float | None = None,
    retain_grad: bool = False,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    state = adapter(input_ids, mask)
    if retain_grad:
        state.retain_grad()
    states = [state]
    deltas: list[Tensor] = []
    for round_index in range(core.rounds):
        mixed = core.slot_mix(state)
        depth = core.depth_embedding.weight[round_index].view(1, 1, core.dimension)
        delta = core.cores[0](mixed + depth)
        if retain_grad:
            delta.retain_grad()
        gate = torch.sigmoid(core.gate_logits[round_index]) if alpha is None else torch.full_like(core.gate_logits[round_index], alpha)
        state = core.rms_norm(state + gate.view(1, 1, core.dimension) * delta)
        if retain_grad:
            state.retain_grad()
        deltas.append(delta)
        states.append(state)
    logits = reader(state, query_ids, TASK)
    return logits, states, deltas


def loss_for_alpha(
    adapter: InputAdapter,
    core: RecurrentCore,
    reader: OutputReader,
    batch: tuple[Tensor, Tensor, Tensor, Tensor],
    alpha: float,
) -> tuple[float, Tensor]:
    input_ids, mask, query_ids, targets = batch
    logits, _, _ = forward_audit(adapter, core, reader, input_ids, mask, query_ids, alpha=alpha)
    return float(F.cross_entropy(logits, targets).detach()), logits


def optimizer_membership(core: RecurrentCore, adapter: InputAdapter, reader: OutputReader) -> dict[str, object]:
    modules = nn.ModuleList((adapter, core, reader))
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in modules.parameters() if parameter is not core.gate_logits], "weight_decay": 1e-4},
            {"params": [core.gate_logits], "weight_decay": 0.0},
        ],
        lr=3e-4,
    )
    gate = core.gate_logits
    groups = [
        {
            "weight_decay": group["weight_decay"],
            "contains_gate": any(id(gate) == id(parameter) for parameter in group["params"]),
        }
        for group in optimizer.param_groups
    ]
    gate_groups = [group for group in groups if group["contains_gate"]]
    return {
        "gate_is_nn_parameter": isinstance(gate, nn.Parameter),
        "gate_in_optimizer": bool(gate_groups),
        "gate_optimizer_groups": gate_groups,
        "gate_excluded_from_weight_decay": bool(gate_groups) and all(group["weight_decay"] == 0.0 for group in gate_groups),
        "optimizer_group_count": len(optimizer.param_groups),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "campaign" / "phase_a_audit.json")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    vocabulary = TokenVocabulary()
    batch = load_minibatch(vocabulary)
    input_ids, mask, query_ids, targets = batch
    adapter = InputAdapter(len(vocabulary), 64, 1, max_length=64)
    core = RecurrentCore(64, 1, 4, "shared")
    reader = OutputReader(len(vocabulary), 64)
    criterion = nn.CrossEntropyLoss()

    # Baseline graph: retain every h_r and delta_r, then backprop once.
    adapter.zero_grad(set_to_none=True)
    core.zero_grad(set_to_none=True)
    reader.zero_grad(set_to_none=True)
    logits, states, deltas = forward_audit(adapter, core, reader, input_ids, mask, query_ids, retain_grad=True)
    loss = criterion(logits, targets)
    loss.backward()
    learned_gate = torch.sigmoid(core.gate_logits.detach())
    round_metrics = []
    for round_index, delta in enumerate(deltas):
        state = states[round_index]
        effective_delta = learned_gate[round_index].view(1, 1, core.dimension) * delta.detach()
        cosine = F.cosine_similarity(state.detach().flatten(1), delta.detach().flatten(1), dim=-1).mean()
        round_metrics.append(
            {
                "round": round_index + 1,
                "rms_h": rms(state.detach()),
                "rms_delta": rms(delta.detach()),
                "rms_alpha_delta": rms(effective_delta),
                "rms_alpha_delta_over_rms_h": rms(effective_delta) / max(rms(state.detach()), 1e-12),
                "cosine_h_delta": float(cosine),
                "grad_norm_h": rms(state.grad) if state.grad is not None else 0.0,
                "grad_norm_delta": rms(delta.grad) if delta.grad is not None else 0.0,
                "gate_probability_mean": float(learned_gate[round_index].mean()),
                "gate_logit_grad_rms": rms(core.gate_logits.grad[round_index]),
                "gate_logit_grad_max_abs": float(core.gate_logits.grad[round_index].abs().max()),
                "depth_embedding_grad_rms": rms(core.depth_embedding.weight.grad[round_index]),
                "depth_embedding_grad_max_abs": float(core.depth_embedding.weight.grad[round_index].abs().max()),
            }
        )

    # Force scalar alpha values on this exact model state and minibatch.
    forced = {}
    base_logits = None
    for alpha in (0.0, 0.5, 1.0, 2.0):
        forced_loss, forced_logits = loss_for_alpha(adapter, core, reader, batch, alpha)
        if base_logits is None:
            base_logits = forced_logits
        forced[str(alpha)] = {
            "loss": forced_loss,
            "max_abs_logit_delta_vs_alpha0": float((forced_logits - base_logits).abs().max().detach()),
            "mean_abs_logit_delta_vs_alpha0": float((forced_logits - base_logits).abs().mean().detach()),
        }

    # Central finite difference for one gate-logit coordinate, compared to the real autograd gradient.
    coordinate = (0, 0)
    real_grad = float(core.gate_logits.grad[coordinate])
    with torch.no_grad():
        original = core.gate_logits[coordinate].item()
        core.gate_logits[coordinate] = original + EPSILON
    # Evaluate learned-gate loss directly for perturbations.
    def learned_loss() -> float:
        perturbed_logits, _, _ = forward_audit(adapter, core, reader, input_ids, mask, query_ids)
        return float(criterion(perturbed_logits, targets))

    plus = learned_loss()
    with torch.no_grad():
        core.gate_logits[coordinate] = original - EPSILON
    minus = learned_loss()
    with torch.no_grad():
        core.gate_logits[coordinate] = original
    finite_difference = (plus - minus) / (2.0 * EPSILON)

    result = {
        "phase": "A",
        "training_performed": False,
        "model_source": "freshly initialized model, seed 101; existing S4 checkpoint is incompatible with required S=1 adapter",
        "task": TASK,
        "batch_size": BATCH_SIZE,
        "dimension": 64,
        "slots": 1,
        "rounds": 4,
        "init_gate_probability": 0.1,
        "baseline_loss": float(loss),
        "round_metrics": round_metrics,
        "forced_alpha": forced,
        "finite_difference": {
            "coordinate": list(coordinate),
            "epsilon": EPSILON,
            "loss_plus": plus,
            "loss_minus": minus,
            "central_difference": finite_difference,
            "autograd_gate_logit_grad_before_clipping": real_grad,
            "absolute_error": abs(finite_difference - real_grad),
            "relative_error": abs(finite_difference - real_grad) / max(abs(real_grad), 1e-12),
        },
        "optimizer_membership": optimizer_membership(core, adapter, reader),
        "graph_checks": {
            "gate_is_leaf_nn_parameter": isinstance(core.gate_logits, nn.Parameter),
            "state_gradients_nonzero": [metric["grad_norm_h"] > 0.0 for metric in round_metrics],
            "delta_gradients_nonzero": [metric["grad_norm_delta"] > 0.0 for metric in round_metrics],
            "no_detach_or_item_or_tensor_reconstruction_or_inplace_in_recurrent_forward": "verified by source inspection of model.py forward path",
        },
        "readout_audit": {
            "encoder_bypass": False,
            "h0_bypass": False,
            "pooled_intermediate_states_bypass": False,
            "direct_query_residual_bypass": False,
            "query_embedding_used_for_attention_pooling": True,
            "explicit_final_norm_before_head": False,
            "head_input_path": "OutputReader(state=h_R, query_ids) -> query-conditioned slot pooling -> output -> task head",
            "source_note": "OutputReader.forward receives only h_R and query_ids; it does not receive encoder output or saved states outside the loop, but it also does not apply an explicit final Norm(h_R).",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""T2-NOBYPASS-REF-GEOM-ALG frozen executor/codebook geometry audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ctrl2_common import BASE_CHECKPOINT, load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "t2_nobypass_ref_geom_alg"


def stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double()
    return {"min": float(values.min()), "median": float(values.median()), "mean": float(values.mean()), "max": float(values.max())}


def margins(logits: torch.Tensor) -> dict[str, Any]:
    top = logits.topk(2, dim=-1).values
    values = top[:, 0] - top[:, 1]
    return {"per_value": [float(x) for x in values], "summary": {"min": float(values.min()), "median": float(values.median()), "mean": float(values.mean()), "max": float(values.max())}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main() -> None:
    model = load_executor()
    ids = torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT, dtype=torch.long)
    codebook = model.token_embedding(ids).detach()
    head = codebook[:, :32]
    tail = codebook[:, 32:]
    norms = codebook.norm(dim=-1)
    rho = head.norm(dim=-1) / norms
    rho_tail = tail.norm(dim=-1) / norms

    p = head / norms.unsqueeze(-1)
    singular_values = torch.linalg.svdvals(p)
    rank = int(torch.linalg.matrix_rank(p).item())
    sigma_min = float(singular_values.min())
    sigma_max = float(singular_values.max())

    q_self = torch.cat((head, torch.zeros_like(tail)), dim=-1)
    self_logits = model.register_decoder(q_self, codebook)
    self_pred = self_logits.argmax(dim=-1)
    self_margin = margins(self_logits)

    solve_method = "inverse" if rank == VALUE_COUNT else "pinv"
    u = torch.linalg.inv(p) if rank == VALUE_COUNT else torch.linalg.pinv(p)
    u_norms = u.norm(dim=0)
    u_hat = u / u_norms.unsqueeze(0)
    q_constructive = torch.cat((u_hat.T, torch.zeros_like(u_hat.T)), dim=-1)
    constructive_logits = model.register_decoder(q_constructive, codebook)
    constructive_pred = constructive_logits.argmax(dim=-1)
    constructive_margin = margins(constructive_logits)
    predicted_margin = 20.0 / u_norms
    observed_margin = constructive_logits.diagonal() - torch.topk(constructive_logits, 2, dim=-1).values[:, 1]

    result: dict[str, Any] = {
        "task": "T2-NOBYPASS-REF-GEOM-ALG",
        "status": "passed" if rank == VALUE_COUNT and bool(torch.equal(self_pred, torch.arange(VALUE_COUNT))) and bool(torch.equal(constructive_pred, torch.arange(VALUE_COUNT))) else "failed",
        "executor_checkpoint": str(BASE_CHECKPOINT),
        "executor_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "codebook": {"value_base": VALUE_BASE, "value_count": VALUE_COUNT, "shape": list(codebook.shape), "frozen": True},
        "energy_by_half": {"rho": {**stats(rho), "per_value": [float(x) for x in rho]}, "rho_tail": {**stats(rho_tail), "per_value": [float(x) for x in rho_tail]}},
        "effective_decoder_geometry": {"P_shape": list(p.shape), "rank_numeric": rank, "singular_values": [float(x) for x in singular_values], "sigma_min": sigma_min, "sigma_max": sigma_max, "condition_number": sigma_max / sigma_min},
        "self_query": {"query": "Q_i=[C_i[:32],0]", "exact_match_count": int((self_pred == torch.arange(VALUE_COUNT)).sum()), "predicted_local_indices": [int(x) for x in self_pred], "margins": self_margin},
        "constructive": {"solve": "P @ U = I", "solve_method": solve_method, "identity_residual_max_abs": float((p @ u - torch.eye(VALUE_COUNT)).abs().max()), "exact_match_count": int((constructive_pred == torch.arange(VALUE_COUNT)).sum()), "predicted_local_indices": [int(x) for x in constructive_pred], "margins_observed": constructive_margin, "margins_predicted_20_over_norm_u": [float(x) for x in predicted_margin], "margin_comparison_max_abs": float((observed_margin - predicted_margin).abs().max()), "margin_comparison_exact": bool(torch.allclose(observed_margin, predicted_margin, rtol=1e-6, atol=1e-6))},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT / "results.json"
    artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(artifact), "sha256": sha256(artifact), "status": result["status"], "self_exact": result["self_query"]["exact_match_count"], "constructive_exact": result["constructive"]["exact_match_count"], "rank": rank, "sigma_min": sigma_min, "sigma_max": sigma_max, "condition_number": result["effective_decoder_geometry"]["condition_number"]}, indent=2))


if __name__ == "__main__":
    main()

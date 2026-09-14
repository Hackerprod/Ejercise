"""Materialize the executable OMEGA CORE-LM-0 design audit artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "campaign" / "omega_core_lm_0_design_audit" / "omega_core_lm_0_design_audit.json"
TEST_RESULT = ROOT / "campaign" / "omega_core_lm_0_design_audit" / "synthetic_contract_tests.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_self_hashed(payload: dict[str, Any]) -> tuple[str, str]:
    unsigned = dict(payload)
    unsigned["artifact_self_hash"] = "__SELF_HASH__"
    digest = hashlib.sha256((json.dumps(unsigned, indent=2, sort_keys=True) + "\n").encode("utf-8")).hexdigest()
    written = dict(payload)
    written["artifact_self_hash"] = digest
    encoded = (json.dumps(written, indent=2, sort_keys=True) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(encoded)
    return digest, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    tests = json.loads(TEST_RESULT.read_text(encoding="utf-8"))
    if tests.get("status") != "passed" or tests.get("checks_passed") != tests.get("checks_total"):
        raise RuntimeError("synthetic contract tests are not all passing")

    d = 128
    m = 8
    vocab = 50257
    p_core_block = 197888
    p_shell = 58211537
    p_mod = {"K1": 256, "K4": 1024}
    p_total = {"K1": p_shell + p_core_block + p_mod["K1"], "K4": p_shell + p_core_block + p_mod["K4"], "untied_K4": p_shell + 4 * p_core_block + p_mod["K4"]}
    bytes_fp32 = {key: value * 4 for key, value in p_total.items()}
    b_state = m * d * 4
    b_intermediates = {"formula_bytes": "4 * (d + m*d + 2*m*d + K*(6*m*d + 2*m*m + 2*d) + m*d + V)", "K1": 244036, "K4": 322372}
    flops = {"formula": "2*(2d)*(m*d) + K*(4*2*m*d*d + 2*m*m*d + 2*m*m*d + 2*m*d*4*d + 2*m*4*d*d) + 2*(m*d)*V; excludes embedding lookup, RMS rsqrt, softmax, and memory traffic", "K1": 106629120, "K4": 116164608, "untied_K4": 116164608}

    reuse_map = [
        {"component": "T1 RecurrentCore / CoreMLP / SlotMix / RMSNorm", "path": "t1_trainability_lab_v0.1.0/t1_trainability/model.py", "demonstrated_function": "State [B,S,D] -> shared slot mixing -> CoreMLP -> gated residual -> RMSNorm; shared variant stores one CoreMLP and reuses it across rounds; tests cover shapes, gradients, and shared/untied parameterization.", "limitation": "State-only core: no token Prelude, persistent token loop, language head, or general text objective. Reuse as Update block, not as complete LM."},
        {"component": "T1 WorkspaceCore", "path": "t1_trainability_lab_v0.1.0/t1_trainability/workspace.py", "demonstrated_function": "Residual/replaced workspace accumulator with length masking and returned intermediate states.", "limitation": "Single vector workspace and round-indexed evidence; no multi-slot content mixing or vocabulary readout."},
        {"component": "CNRL/T0 kernels and transitions", "path": "Trash/cpu_native_recurrence_lab_v0.4.1/README.md; Trash/cpu_native_recurrence_lab_v0.4.1/docs/ARCHITECTURE.md", "demonstrated_function": "T0-R shared-depth reuse, T0-M slot matrixization, T0-RM real double-buffered recurrence; scalar/AVX2 kernel registry, fixed-point/group-RMS/global-RMS transitions, shared/clone physical controls. Existing report references shared/Bclone separation ~2260x in one D=1472 configuration.", "limitation": "CNRL is not an LM and has no tokenizer, language loss, attention proof, normalization proof for this student, or physical result for d=128/m=8. An int8 kernel does not validate a new attention/normalization/block implementation."},
        {"component": "Executor/ISA and controllers", "path": "t1_trainability_lab_v0.1.0/scripts/ctrl2_common.py; t1_trainability_lab_v0.1.0/t1_trainability/unified.py", "demonstrated_function": "Approved executor operations, state/register transitions, and frozen 32-value decoder/codebook paths.", "limitation": "Learned transformations and decoder rely on catalog/harness-specific 32-value semantics. The 32-value decoder is not a general LM head and controller success is not language evidence."},
        {"component": "T7 binder", "path": "t1_trainability_lab_v0.1.0/scripts/evaluate_t7_noop_none_lexical_paired_fresh.py", "demonstrated_function": "A frozen-core binder scores role queries over token candidates, selects a candidate or NULL, and reads values through shared W_v plus approved decode_states path.", "limitation": "Four semantic queries and controlled 32-value lexical grammar. It is a later workspace/binding interface candidate, not the OMEGA recurrent language workspace and must not be relabeled as one."},
    ]

    artifact = {
        "schema": "omega-core-lm-0-design-audit-v1",
        "status": "PASS_DESIGN",
        "decision": "Build one minimal causal recurrent workspace LM pilot; do not add external retrieval, learned halting, hidden-state teacher alignment, or quantization in this unit.",
        "question": "Minimum trainable CPU/RAM-compatible shared core that processes a matrix workspace and emits causal language without an external bank or teacher at inference.",
        "scope": {"design_only": True, "training_executed": False, "corpus_downloaded": False, "teacher_traces_generated": False, "new_manifests": False, "new_checkpoints": False, "campaign_780x_loaded": False, "T7_reopened": False, "T8_started": False},
        "reuse_map": reuse_map,
        "student_specification": {
            "dimensions": {"d": d, "m": m, "K_comparison": [1, 4], "dtype_first_pilot": "FP32; Q8/Q4 deferred until a separate accumulated-error study", "vocab": vocab},
            "causal_graph": [
                "x_t -> E[x_t]",
                "(E[x_t], S_{t-1}^K) -> concat(E[x_t], mean_slots(S_{t-1}^K)) -> Prelude -> A_t in R^(m x d)",
                "S_t^(0) = A_t",
                "for r=0..K-1: (S_t^(r), A_t, e_r) -> SlotMix(S_t^(r)+A_t) -> shared CoreMLP -> gated residual -> RMSNorm -> S_t^(r+1)",
                "S_t^(K) -> flatten -> RMSNorm -> Head -> p(x_{t+1})",
                "S_t^(K) is the only recurrent state transmitted to token t+1; reset is zeros for a new sequence",
            ],
            "prelude": "token embedding E[x_t] in R^d concatenated with mean over previous full workspace S_{t-1}^K in R^d, Linear(2d,m*d), then RMSNorm(m*d), reshape to [m,d]",
            "update": "SlotMix uses T1 single-head Q/K/V over all m slots; add A_t before mixing so Update consumes both current state and anchor; shared CoreMLP is Linear(d,4d)->GELU->Linear(4d,d) per slot; apply per-round e_r in R^d; gate g_r=sigmoid(gate_logits[r]) per dimension; S=RMSNorm(S+g_r*update)",
            "anchor": "A_t is the token-local anchor and is reused by every round; no learned global anchor in v0",
            "normalization": "RMSNorm without bias, epsilon 1e-6, after Prelude and after each residual update; no quantization scale in this unit",
            "write_gate": "Per-dimension sigmoid gate, initialized near 0.1; gate parameters are per-round modulation and counted outside P_core",
            "readout": "Flatten [m,d] to m*d, RMSNorm(m*d), dense Linear(m*d,V) full vocabulary head; no selective vocabulary optimization in pilot",
            "strict_causality": "Forward at t receives only x_t and S_{t-1}^K; target x_{t+1}, future tokens, teacher states, resolved semantic attributes, and external lookup are absent from the forward",
            "real_recurrence": "Round r+1 receives the tensor produced by round r. Re-running a frozen transformation on A_t or summing repeated costs without consuming S_t^(r) is explicitly rejected as non-recurrence.",
            "weight_sharing": "K=4 shared uses one literal UpdateBlock object containing SlotMix, CoreMLP, and RMSNorm; only depth embeddings and gate vectors vary by r and are separately counted. Untied control has four distinct UpdateBlock objects.",
            "slots": "Latent slots are generic workspace addresses; no FLOOR/AVOID/MATCH/ANCHOR assumptions and no T7 grammar dependency.",
            "depth": "Fixed K=1 versus K=4 only; learned halting is deferred until useful fixed-depth work is demonstrated.",
            "reuse_choice": "Wrap/recompose existing T1 SlotMix, CoreMLP, and RMSNorm into a token Prelude + persistent-state shell rather than cloning a Transformer or treating RecurrentCore as a complete LM.",
        },
        "pilot_recipe": {
            "teacher": {"model": "distilgpt2", "role": "frozen training-time Transformer teacher only", "tokenizer": "GPT-2 byte-level BPE, shared vocabulary V=50257", "license": "MIT model/tokenizer distribution; verify exact artifact metadata before any download", "inference_dependency": "none after distillation"},
            "data": {"corpus": "WikiText-2 raw-v1", "license": "CC BY-SA 4.0 dataset metadata; verify exact release before download", "purpose": "small real prose corpus, not a 32-value serialization", "split": "80/10/10 by source document/article before tokenization", "deduplication": "normalize whitespace, hash complete source documents, reject duplicate hashes across splits", "leakage": "evaluation documents and generated answers never enter training"},
            "loss": {"formula": "L = 0.5 * L_token_CE + 0.5 * T^2 * D_KL(softmax(z_T/T) || softmax(z_S/T))", "temperature": 2.0, "teacher_input": "same true prefix x_<=t; target x_{t+1} is used only by CE/KL loss, never as student input", "hidden_state_alignment": "not in initial pilot; later explicit variant only"},
            "budget": {"ram_cap": "16 GiB", "cpu_threads": 8, "steps": 20000, "batch_sequences": 8, "unroll_tokens": 256, "checkpointing": "short truncated-BPTT windows only; no full conversational BPTT", "student_memory": "FP32 parameter totals below; optimizer/activation overhead must remain within cap and be measured in future execution"},
            "not_claimed": "This corpus and pilot budget cannot establish general language capability or equivalence to a large Transformer."},
        "baselines": [
            {"name": "shared_K1", "configuration": "one shared UpdateBlock, K=1, m=8, d=128", "purpose": "no additional recurrent depth control"},
            {"name": "shared_K4", "configuration": "one shared UpdateBlock reused four times, m=8, d=128", "purpose": "candidate recurrent depth"},
            {"name": "untied_K4", "configuration": "four distinct UpdateBlocks, same workspace and K=4", "purpose": "depth control with non-shared weights"},
        ],
        "baseline_accounting": {"comparison": "same corpus, tokenizer, batch, steps, optimizer family, and evaluation; shared_K1 and shared_K4 are near-parameter-matched but not FLOP-matched; shared_K4 and untied_K4 have equal nominal block FLOPs but different unique parameter storage; no claim is simultaneously iso-parameter and iso-FLOP.", "parameters": tests["parameter_breakdown"]["parameters"], "flops_approximate": flops, "recurrence_ablation": {"fixed_before_results": True, "variants": ["normal shared_K4", "anchor_reset: before each r>0 set S^(r)=A_t", "state_shuffle: permute batch state before each r>0"], "primary_decision": "shared_K4 depth is useful only if normal minus anchor_reset test NLL >= 0.01 nats/token with paired bootstrap 95% CI excluding 0 and normal shared_K4 beats shared_K1 by >=0.05 nats/token; otherwise stop depth claim."}},
        "evaluation_protocol": {"primary": ["test next-token NLL and perplexity under true prefixes", "student maintains its own S_t^K between tokens", "report CE and KD divergence separately"], "state_stress": ["evaluate processed lengths 32,64,128,256,512 and report NLL degradation relative to length 32", "run reset-between-document control"], "autonomous_diagnostic": "free-running generation without teacher at inference, report repetition rate, distinct-2, length completion, and human/error sample review; diagnostic only", "quality_decision": "predeclare paired document bootstrap intervals, no T7-style exact 139/139 criterion", "recurrence_decision": "use predeclared ablations above before inspecting outcomes", "physical_decision": "measure wall-clock tokens/s, peak RSS, cache/DRAM counters when available, and logical bytes separately; no estimate may be labeled measurement"},
        "physical_contract": {"shape": {"state": "[B,8,128]", "state_per_sequence_fp32_bytes": b_state, "workspace_values": 1024}, "parameters": {"P_core": p_core_block, "P_shell": p_shell, "P_modulation": p_mod, "P_total": p_total, "fp32_bytes": bytes_fp32, "note": "P_core is one UpdateBlock including slot mixing, CoreMLP, and update RMSNorm; untied_K4 has 4 P_core blocks."}, "buffers": {"B_state": "m*d*4 = 4096 bytes per sequence", "B_intermediates": b_intermediates, "logical_components": ["token embedding d", "Prelude anchor m*d", "two state ping-pong buffers 2*m*d", "per-round Q/K/V, attention, mixed/update/candidate buffers, depth and gate", "readout m*d", "head logits V"], "training_note": "B_intermediates is an inference live-buffer upper bound; autograd/checkpoint memory is separate and must be measured during the authorized pilot."}, "kernels": {"reusable": ["CNRL scalar/AVX2 vector dot/GEMM contracts after FP32 correctness is established", "CNRL double-buffer and state-transition layout concepts", "T1 RMSNorm/SlotMix/CoreMLP interfaces"], "new_required": ["FP32 Prelude and token loop", "general-vocabulary Head", "training data pipeline and KD loss", "a fused CPU implementation for this exact block if PyTorch/BLAS is insufficient", "instrumentation for real cache/DRAM traffic"], "warning": "An existing int8 T0 kernel does not validate attention, RMSNorm, Prelude, or the complete OMEGA block."}, "latency_formula": "T_total = T_entrada + T_core + T_head + T_estado + T_overhead", "traffic_rule": "logical bytes calculated from tensors are distinct from physical DRAM traffic; absent counters do not turn estimates into measurements", "vocab_rule": "full V=50257 head is intentionally retained for first correctness pilot; selective vocabulary cost is recorded but not optimized until measurements justify it."},
        "synthetic_tests": {"artifact": {"path": "campaign/omega_core_lm_0_design_audit/synthetic_contract_tests.json", "self_hash": tests["artifact_self_hash"], "file_sha256": sha256_file(TEST_RESULT)}, "status": tests["status"], "checks": tests["checks"], "six_contract_tests": ["forms", "strict causality", "real round recurrence", "literal shared weights plus separate modulation", "state reset", "parameter and FP32 byte accounting"]},
        "source_inventory": {"design_basis": [record("../Conversacion.md"), record("t1_trainability/model.py"), record("t1_trainability/workspace.py"), record("tests/test_model.py"), record("../Trash/cpu_native_recurrence_lab_v0.4.1/README.md"), record("../Trash/cpu_native_recurrence_lab_v0.4.1/docs/ARCHITECTURE.md"), record("../Trash/cpu_native_recurrence_lab_v0.4.1/validation/LOCAL_VALIDATION.md"), record("scripts/ctrl2_common.py"), record("t1_trainability/unified.py")], "synthetic_runner": record("scripts/audit_omega_core_lm_0_design.py"), "builder": record("scripts/build_omega_core_lm_0_design_audit.py")},
        "closure_gate": {"causal_graph_complete": True, "pilot_recipe_concrete": True, "evaluation_protocol_predeclared": True, "physical_costs_accounted": True, "reuse_map_limited": True, "synthetic_tests_all_passed": True, "localized_blockers": [], "decision": "PASS_DESIGN"},
    }
    digest, file_sha = write_self_hashed(artifact)
    print(json.dumps({"status": artifact["status"], "artifact": OUTPUT.relative_to(ROOT).as_posix(), "artifact_self_hash": digest, "file_sha256": file_sha, "synthetic_tests": tests["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

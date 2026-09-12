import hashlib
import json
from pathlib import Path
import torch
from t2_i1_instruction import TOKEN_IDS, parse_instruction_i1
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLED = ROOT / "campaign/t2_nobypass1_staged_nbf0/assembled.pt"
OUT = ROOT / "campaign/t2_nobypass1_staged_nb4_pairgeom_alg"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@torch.no_grad()
def main():
    assembled = torch.load(ASSEMBLED, weights_only=False)
    core = {k: v for k, v in assembled["core"].items() if k not in ("w_c", "b_c")}
    enc = GateOnlyEncoder(core)
    enc.w_c.data.copy_(assembled["core"]["w_c"])
    enc.b_c.data.copy_(assembled["core"]["b_c"])
    enc.eval()
    model = load_executor()
    model.eval()
    codebook = model.token_embedding(torch.arange(VALUE_BASE, VALUE_BASE + VALUE_COUNT))
    manifest = json.loads(MANIFEST_PATH.read_text())
    pairs = manifest["calibration"] + manifest["heldout"]
    token_names = [*(f"VALUE_{i}" for i in range(32)), "AT_LEAST", "AVOID", "AND"]
    token_ids = torch.tensor([TOKEN_IDS[x] for x in token_names])
    teacher_logits = model.register_decoder(torch.cat((enc.w_v(enc.embedding(token_ids)), torch.zeros((35, 32))), -1), codebook)
    binary = (teacher_logits.topk(2).values[:, 0] - teacher_logits.topk(2).values[:, 1] > 3.0545).float()
    binary_by_id = {TOKEN_IDS[n]: binary[i] for i, n in enumerate(token_names)}
    rows = []
    failures = []
    delta_a = delta_f = soft_a = soft_f = flat_a = flat_f = bin_a = bin_f = 0

    def decode(state):
        return int(model.register_decoder(torch.cat((state.unsqueeze(0), torch.zeros((1, 32))), -1), codebook)[0].argmax())

    for pair in pairs:
        lower, forbidden = int(pair["lower"]), int(pair["forbidden"])
        text = f"AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}"
        parsed = parse_instruction_i1(text)
        ids, lengths = tensorize([text])
        details = enc(ids, lengths, return_details=True)
        k, sf, sa, vt, gate = details[5][0], details[6][0], details[7][0], details[8][0], details[9][0]
        li, fi = parsed.tokens.index(f"VALUE_{lower}"), parsed.tokens.index(f"VALUE_{forbidden}")
        wf, wa = torch.softmax(2.5 * sf, 0), torch.softmax(2.5 * sa, 0)
        raw_f, raw_a = decode((wf.unsqueeze(-1) * vt * gate.unsqueeze(-1)).sum(0)), decode((wa.unsqueeze(-1) * vt * gate.unsqueeze(-1)).sum(0))
        soft_f_dec = decode(wf[li] * gate[li] * vt[li] + wf[fi] * gate[fi] * vt[fi])
        soft_a_dec = decode(wa[li] * gate[li] * vt[li] + wa[fi] * gate[fi] * vt[fi])
        flat_f_dec, flat_a_dec = decode(wf[li] * vt[li] + wf[fi] * vt[fi]), decode(wa[li] * vt[li] + wa[fi] * vt[fi])
        bg = torch.stack([binary_by_id[int(x)] for x in parsed.token_ids])
        bin_f_dec = decode((wf.unsqueeze(-1) * vt * bg.unsqueeze(-1)).sum(0))
        bin_a_dec = decode((wa.unsqueeze(-1) * vt * bg.unsqueeze(-1)).sum(0))
        delta_a += float(sa[fi] > sa[li])
        delta_f += float(sf[li] > sf[fi])
        soft_a += soft_a_dec == forbidden; soft_f += soft_f_dec == lower
        flat_a += flat_a_dec == forbidden; flat_f += flat_f_dec == lower
        bin_a += bin_a_dec == forbidden; bin_f += bin_f_dec == lower
        row = {"digest": pair["digest"], "L": lower, "F": forbidden, "decoded_floor": raw_f, "decoded_avoid": raw_a, "pair_soft_floor": soft_f_dec, "pair_soft_avoid": soft_a_dec, "pair_flat_floor": flat_f_dec, "pair_flat_avoid": flat_a_dec, "binary_floor": bin_f_dec, "binary_avoid": bin_a_dec, "delta_A_raw": float(sa[fi] - sa[li]), "delta_F_raw": float(sf[li] - sf[fi])}
        rows.append(row)
        if raw_a != forbidden:
            failures.append(row)
    table = []
    for n in range(32):
        _, f1, a1, _, _ = (lambda d: (d[5][0], d[6][0], d[7][0], d[8][0], d[9][0]))(enc(*tensorize([f"AT_LEAST VALUE_{n}"]), return_details=True))
        _, f2, a2, _, _ = (lambda d: (d[5][0], d[6][0], d[7][0], d[8][0], d[9][0]))(enc(*tensorize([f"AVOID VALUE_{n}"]), return_details=True))
        c = float(torch.sigmoid(enc.embedding(torch.tensor([TOKEN_IDS[f"VALUE_{n}"]])) @ enc.w_c + enc.b_c))
        table.append({"n": n, "A_self": float(a2[1]), "A_cross": float(a1[1]), "F_self": float(f1[1]), "F_cross": float(f2[1]), "c": c})
    result = {"task": "T2-NOBYPASS-1-STAGED-NB4-PAIRGEOM-ALG", "status": "diagnostic_complete", "assembled_sha256": sha(ASSEMBLED), "pair_count": len(pairs), "A": {"delta_A_raw_positive": delta_a, "delta_F_raw_positive": delta_f, "failures": failures}, "B": {"failures": failures}, "C": table, "D": failures, "E": {"PAIR-SOFT": {"floor": soft_f, "avoid": soft_a, "joint": min(soft_f, soft_a)}, "PAIR-FLAT": {"floor": flat_f, "avoid": flat_a, "joint": min(flat_f, flat_a)}, "BINARY-NC": {"floor": bin_f, "avoid": bin_a, "joint": min(bin_f, bin_a)}}, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(path), "sha256": sha(path), "pair_count": len(pairs), "failures": len(failures), "E": result["E"]}, indent=2))

if __name__ == "__main__":
    main()

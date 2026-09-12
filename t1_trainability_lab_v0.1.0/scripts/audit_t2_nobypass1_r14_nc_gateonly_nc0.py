from pathlib import Path
import hashlib,json,torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'campaign'/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; ART=ROOT/'campaign'/'t2_nobypass1_r14_nc_gateonly_seed6706'/'gate.pt'; OUT=ROOT/'campaign'/'t2_nobypass1_r14_nc_gateonly_seed6706'
def h(x): return hashlib.sha256(x.detach().numpy().tobytes()).hexdigest()
@torch.no_grad()
def main():
 b=torch.load(BASE,weights_only=False)['encoder']; r=R13RelKeyKVEncoder(); r.load_state_dict(b); r.eval(); e=GateOnlyEncoder(b); a=torch.load(ART,weights_only=False); e.w_c.copy_(a['w_c']); e.b_c.copy_(a['b_c']); e.eval(); frozen={n:h(p) for n,p in r.named_parameters()}; after={n:h(p) for n,p in e.named_parameters() if n not in ('w_c','b_c')}; rows=[]; m=json.loads(MANIFEST_PATH.read_text())
 for o in ('normal','reverse'):
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(t); x,n=tensorize([t]); d0=r(x,n,return_details=True); d1=e(x,n,return_details=True); rows.append(torch.equal(d0[6],d1[6]) and torch.equal(d0[7],d1[7]))
 result={'status':'passed' if frozen==after and all(rows) else 'failed','gate':'NC0','base_checkpoint_sha256':sha256(BASE),'gate_artifact_sha256':sha256(ART),'frozen_parameters_identical':frozen==after,'raw_scores_identical':all(rows),'frozen_hashes_before':frozen,'frozen_hashes_after':after}; p=OUT/'nc0.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(p),'sha256':sha256(p),**{k:v for k,v in result.items() if 'hash' not in k}},indent=2))
if __name__=='__main__': main()

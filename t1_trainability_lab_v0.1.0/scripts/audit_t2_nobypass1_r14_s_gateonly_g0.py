from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; BASE=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; ART=CAMPAIGN/'t2_nobypass1_r14_s_gateonly_seed6706'/'gate.pt'; BASE_SHA='77ce2051aee6d776b2054b0ec991cb35e20f83287694400be9629e7ffd5b7737'
def digest(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
@torch.no_grad()
def main():
 b=torch.load(BASE,weights_only=False)['encoder']; raw=R13RelKeyKVEncoder(); raw.load_state_dict(b); raw.eval(); e=GateOnlyEncoder(b); g=torch.load(ART,weights_only=False); e.w_c.copy_(g['w_c']); e.b_c.copy_(g['b_c']); e.eval(); m=json.loads(MANIFEST_PATH.read_text()); frozen={n:digest(p) for n,p in raw.named_parameters()}; after={n:digest(p) for n,p in e.named_parameters() if n not in ('w_c','b_c')}; score_rows=[]
 for order in ('normal','reverse'):
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); d0=raw(ti,le,return_details=True); d1=e(ti,le,return_details=True); s0f,s0a=d0[6],d0[7]; s1f,s1a=d1[6],d1[7]; lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); score_rows.append({'order':order,'digest':p['digest'],'raw_equal':bool(torch.equal(s0f,s1f) and torch.equal(s0a,s1a)),'sF_L_before':float(s0f[0,lp]),'sF_L_after':float(s1f[0,lp]),'sF_F_before':float(s0f[0,fp]),'sF_F_after':float(s1f[0,fp]),'sA_F_before':float(s0a[0,fp]),'sA_F_after':float(s1a[0,fp]),'sA_L_before':float(s0a[0,lp]),'sA_L_after':float(s1a[0,lp])})
 result={'status':'passed' if frozen==after and all(x['raw_equal'] for x in score_rows) else 'failed','task':'T2-NOBYPASS-1-R1.4-S-GATEONLY','gate':'G0','base_checkpoint_sha256':sha256(BASE),'expected_base_sha256':BASE_SHA,'gate_artifact_sha256':sha256(ART),'frozen_parameter_hashes_before':frozen,'frozen_parameter_hashes_after':after,'frozen_parameters_identical':frozen==after,'raw_score_rows':score_rows,'raw_scores_identical':all(x['raw_equal'] for x in score_rows)}; out=CAMPAIGN/'t2_nobypass1_r14_s_gateonly_seed6706'/'g0.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'frozen_parameters_identical':result['frozen_parameters_identical'],'raw_scores_identical':result['raw_scores_identical']},indent=2))
if __name__=='__main__': main()

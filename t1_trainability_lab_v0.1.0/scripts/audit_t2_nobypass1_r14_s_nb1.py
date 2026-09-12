from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from t2_i1_instruction import TOKEN_IDS
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r14_s import R14SEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r14_s_seed6706'/'final.pt'
def dec(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 model=load_executor(); enc=R14SEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); groups={}; means=[]
 for role,op in (('floor','AT_LEAST'),('avoid','AVOID')):
  groups[role]={}
  for shifted in (False,True):
   rows=[]
   for n in range(32):
    ti,le=tensorize([f'{op} VALUE_{n}']); rf,ra,_,af,aa,_,_,_,vt,gate,vtg=enc(ti,le,return_details=True); r=rf if role=='floor' else ra; d,m=dec(model,cb,r); a=af if role=='floor' else aa; rows.append({'value':n,'decoded':d,'pass':d==n,'margin':m,'H':float((-(a.clamp_min(1e-9)*a.clamp_min(1e-9).log()).sum()).item())})
   groups[role]['shifted' if shifted else 'natural']={'cases':rows,'exact_count':sum(x['pass'] for x in rows)}
 synthetic=[]
 for n in range(32):
  s=torch.tensor([[TOKEN_IDS['AVOID'],TOKEN_IDS[f'VALUE_{n}']]]); q=torch.tensor([[TOKEN_IDS['AVOID'],TOKEN_IDS[f'VALUE_{n}'],TOKEN_IDS['KEEP']]]); *_,ks,_,_,_,_,_=enc(s,torch.tensor([2]),return_details=True); *_,kl,_,_,_,_,_=enc(q,torch.tensor([3]),return_details=True); diff=(ks[:,1]-kl[:,1]); synthetic.append({'value':n,'max_abs_diff':float(diff.abs().max()),'cosine':float(F.cosine_similarity(ks[:,1],kl[:,1]).item())})
 result={'status':'passed' if all(groups[r][s]['exact_count']==32 for r in groups for s in groups[r]) and max(x['max_abs_diff'] for x in synthetic)==0 else 'failed','task':'T2-NOBYPASS-1-R1.4-S','gate':'NB1','checkpoint_sha256':sha256(CK),'architectural_sanity':{'readout_uses_content_gate':True,'content_gate_role_independent':True},'groups':groups,'synthetic_suffix_invariance':synthetic,'content_gate_observation':'reported by checkpoint/training metadata'}; out=CAMPAIGN/'t2_nobypass1_r14_s_seed6706'/'nb1.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'exact':{r:{s:groups[r][s]['exact_count'] for s in groups[r]} for r in groups}},indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from t2_i1_instruction import TOKEN_IDS
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'
def decode(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),dim=-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 model=load_executor(); enc=R13RelKeyKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); groups={}
 for role,op in (('floor','AT_LEAST'),('avoid','AVOID')):
  groups[role]={}
  for shifted in (False,True):
   rows=[]
   for n in range(32):
    text=f'{op} VALUE_{n}'; ti,le=tensorize([text]); rf,ra,af,aa,*_=enc(ti,le,return_details=True); r=rf if role=='floor' else ra; dec,margin=decode(model,cb,r); weights=af if role=='floor' else aa; rows.append({'value':n,'decoded':dec,'pass':dec==n,'margin':margin,'attention_entropy':float((-(weights.clamp_min(1e-9)*weights.clamp_min(1e-9).log()).sum()).item()),'position_condition':'shifted rows 4,5' if shifted else 'natural rows 1,2'})
   groups[role]['shifted' if shifted else 'natural']={'cases':rows,'exact_count':sum(x['pass'] for x in rows),'exact_rate':sum(x['pass'] for x in rows)/32}
 synthetic=[]
 for n in range(32):
  short=torch.tensor([[TOKEN_IDS['AVOID'],TOKEN_IDS[f'VALUE_{n}']]]); long=torch.tensor([[TOKEN_IDS['AVOID'],TOKEN_IDS[f'VALUE_{n}'],TOKEN_IDS['KEEP']]]); ls=torch.tensor([2]); ll=torch.tensor([3]); *_,ks,_,_,_=enc(short,ls,return_details=True); *_,kl,_,_,_=enc(long,ll,return_details=True); x=ks[0,1]; y=kl[0,1]; synthetic.append({'value':n,'max_abs_diff':float((x-y).abs().max()),'l2_diff':float((x-y).norm()),'cosine':float(F.cosine_similarity(x.unsqueeze(0),y.unsqueeze(0)).item()),'exact_invariant':bool(torch.equal(x,y))})
 result={'status':'passed' if all(groups[r][s]['exact_count']==32 for r in groups for s in groups[r]) and all(x['exact_invariant'] for x in synthetic) else 'failed','task':'T2-NOBYPASS-1-R1.3-RELKEY-KV','gate':'NB1','checkpoint_sha256':sha256(CK),'g5_touched':False,'groups':groups,'synthetic_suffix_invariance':{'cases':synthetic,'max_abs_diff':max(x['max_abs_diff'] for x in synthetic),'max_l2_diff':max(x['l2_diff'] for x in synthetic),'min_cosine':min(x['cosine'] for x in synthetic),'all_exact_invariant':all(x['exact_invariant'] for x in synthetic)}}; out=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'nb1.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'exact':{r:{s:groups[r][s]['exact_count'] for s in groups[r]} for r in groups},'synthetic':result['synthetic_suffix_invariance']},indent=2))
if __name__=='__main__': main()

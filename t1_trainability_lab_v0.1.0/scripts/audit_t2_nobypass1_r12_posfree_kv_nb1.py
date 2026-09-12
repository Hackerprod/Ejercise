from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r12_posfree_kv import R12POSFreeKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r12_posfree_kv_seed6704'/'final.pt'
def decode(model,c,r):
 q=torch.cat((r,torch.zeros_like(r)),dim=-1); z=model.register_decoder(q,c)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 model=load_executor(); enc=R12POSFreeKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); groups={}
 for role,op in (('floor','AT_LEAST'),('avoid','AVOID')):
  groups[role]={}
  for shifted in (False,True):
   rows=[]
   for n in range(32):
    text=f'{op} VALUE_{n}'; ti,le=tensorize([text]); rf,ra,_,af,aa,*_=enc(ti,le,return_details=True); r=rf if role=='floor' else ra; dec,margin=decode(model,cb,r); ent=float((-( (af if role=='floor' else aa).clamp_min(1e-9)*((af if role=='floor' else aa).clamp_min(1e-9)).log()).sum()).item()); rows.append({'value':n,'decoded':dec,'pass':dec==n,'margin':margin,'attention_entropy':ent,'position_condition':'shifted rows 4,5' if shifted else 'natural rows 1,2'})
   groups[role]['shifted' if shifted else 'natural']={'cases':rows,'exact_count':sum(x['pass'] for x in rows),'exact_rate':sum(x['pass'] for x in rows)/32}
 result={'status':'passed' if all(groups[r][s]['exact_count']==32 for r in groups for s in groups[r]) else 'failed','task':'T2-NOBYPASS-1-R1.2-POSFREE-KV','gate':'NB1','g5_touched':False,'checkpoint_sha256':sha256(CK),'architectural_sanity':{'address_key_uses_absolute_position':False},'groups':groups}; out=CAMPAIGN/'t2_nobypass1_r12_posfree_kv_seed6704'/'nb1.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'exact':{r:{s:groups[r][s]['exact_count'] for s in groups[r]} for r in groups}},indent=2))
if __name__=='__main__': main()

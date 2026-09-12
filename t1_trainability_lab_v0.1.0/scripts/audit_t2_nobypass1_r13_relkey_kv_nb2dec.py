from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'
def dec(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),dim=-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 m=json.loads(MANIFEST_PATH.read_text()); model=load_executor(); enc=R13RelKeyKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); groups={}
 for order in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,*_=enc(ti,le,return_details=True); df,mf=dec(model,cb,rf); da,ma=dec(model,cb,ra); rows.append({'digest':p['digest'],'expected_floor':l,'expected_avoid':f,'decoded_floor':df,'decoded_avoid':da,'floor_pass':df==l,'avoid_pass':da==f,'floor_margin':mf,'avoid_margin':ma})
  groups[order]={'cases':rows,'floor_exact':sum(x['floor_pass'] for x in rows),'avoid_exact':sum(x['avoid_pass'] for x in rows),'joint_exact':sum(x['floor_pass'] and x['avoid_pass'] for x in rows)}
 result={'status':'passed' if all(groups[o]['floor_exact']==139 and groups[o]['avoid_exact']==139 for o in groups) else 'failed','task':'T2-NOBYPASS-1-R1.3-RELKEY-KV','gate':'NB2-DEC','checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'g5_touched':False,'normal':groups['normal'],'reverse':groups['reverse']}; out=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'nb2_dec.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'normal':{k:v for k,v in groups['normal'].items() if k!='cases'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

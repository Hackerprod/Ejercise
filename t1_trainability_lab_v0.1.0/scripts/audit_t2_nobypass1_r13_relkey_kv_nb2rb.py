from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'
@torch.no_grad()
def main():
 m=json.loads(MANIFEST_PATH.read_text()); model=load_executor(); enc=R13RelKeyKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cases={}
 for order in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); _,_,_,_,_,_,sf,sa,*_=enc(ti,le,return_details=True); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); rows.append({'digest':p['digest'],'floor_pass':float(sf[0,lp])>float(sf[0,fp]),'avoid_pass':float(sa[0,fp])>float(sa[0,lp]),'sF_L':float(sf[0,lp]),'sF_F':float(sf[0,fp]),'sA_F':float(sa[0,fp]),'sA_L':float(sa[0,lp])})
  cases[order]={'cases':rows,'floor_order_exact':sum(x['floor_pass'] for x in rows),'avoid_order_exact':sum(x['avoid_pass'] for x in rows),'joint_order_exact':sum(x['floor_pass'] and x['avoid_pass'] for x in rows)}
 result={'status':'passed' if all(cases[o]['floor_order_exact']==139 and cases[o]['avoid_order_exact']==139 for o in cases) else 'failed','task':'T2-NOBYPASS-1-R1.3-RELKEY-KV','gate':'NB2-RB','checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'g5_touched':False,'normal':cases['normal'],'reverse':cases['reverse']}; out=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'nb2_rb.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'normal':{k:v for k,v in cases['normal'].items() if k!='cases'},'reverse':{k:v for k,v in cases['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

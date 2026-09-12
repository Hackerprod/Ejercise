from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from t2_i1_instruction import parse_instruction_i1
from t2_nobypass1_r12_posfree_kv import R12POSFreeKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r12_posfree_kv_seed6704'/'final.pt'; MAN=MANIFEST_PATH
@torch.no_grad()
def main():
 m=json.loads(MAN.read_text()); model=load_executor(); enc=R12POSFreeKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cases={}
 for order in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,_,_,_,sf,sa,*_=enc(ti,le,return_details=True); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); sf_l=float(sf[0,lp]); sf_f=float(sf[0,fp]); sa_f=float(sa[0,fp]); sa_l=float(sa[0,lp]); rows.append({'digest':p['digest'],'order':order,'floor_position':lp,'avoid_position':fp,'sF_L':sf_l,'sF_F':sf_f,'sA_F':sa_f,'sA_L':sa_l,'floor_order_pass':sf_l>sf_f,'avoid_order_pass':sa_f>sa_l})
  cases[order]={'cases':rows,'floor_order_exact':sum(x['floor_order_pass'] for x in rows),'avoid_order_exact':sum(x['avoid_order_pass'] for x in rows),'joint_order_exact':sum(x['floor_order_pass'] and x['avoid_order_pass'] for x in rows)}
 result={'status':'passed' if all(cases[o]['floor_order_exact']==139 and cases[o]['avoid_order_exact']==139 for o in cases) else 'failed','task':'T2-NOBYPASS-1-R1.2-POSFREE-KV','gate':'NB2-RB','checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'g5_touched':False,'normal':cases['normal'],'reverse':cases['reverse']}; out=CAMPAIGN/'t2_nobypass1_r12_posfree_kv_seed6704'/'nb2_rb.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'normal':{k:v for k,v in cases['normal'].items() if k!='cases'},'reverse':{k:v for k,v in cases['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

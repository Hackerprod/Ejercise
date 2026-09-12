from __future__ import annotations
import json
from pathlib import Path
import torch
from audit_t2_i3_comp0_reg_alg import load_runtime,sha256
from t2_nobypass1_r0 import run_r0
from t2_nobypass1_r11_kv import R11KVEncoder
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r11_kv_seed6703'/'final.pt'
def main():
 _,_,eps,manifest,_,model,ctrl1,scorer=load_runtime(); enc=R11KVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); enc.eval(); cases=[]
 for ep in eps.values():
  r=run_r0(model,ctrl1,scorer,sup,manifest,ep,enc,'KEEP'); cases.append({'episode':ep['episode'],'success':r['success'],'actions':r['action_ids'],'expected':[0,1,2,5]})
 result={'status':'passed' if all(x['success'] for x in cases) else 'failed','task':'T2-NOBYPASS-1-R1.1-KV','gate':'NB1','atomic_noop_cases':len(cases),'exact_pass':sum(x['success'] for x in cases),'g5_touched':False,'checkpoint_sha256':sha256(CK),'cases':cases}; out=CAMPAIGN/'t2_nobypass1_r11_kv_seed6703'/'nb1.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),**{k:v for k,v in result.items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

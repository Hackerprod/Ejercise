from __future__ import annotations
import inspect,json
from pathlib import Path
import torch
from audit_t2_i3_comp0_reg_alg import load_runtime,sha256
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r0 import R0Encoder,run_r0
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_t2_i0_baseline_b import LatentConditionedSupervisor

ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r0_seed6701'/'final.pt'
def main():
 m=json.loads(MANIFEST_PATH.read_text()); rt=load_runtime(); _a,_b,eps,manifest,latent,model,ctrl1,scorer=rt; enc=R0Encoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); rows=[]; forbidden={'lower','forbidden','constraints'}; static=not forbidden.intersection(inspect.signature(run_r0).parameters)
 sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval()
 for pair in m['calibration']:
  text=f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}"; x=run_r0(model,ctrl1,scorer,sup,manifest,eps[10],enc,text); y=run_r0(model,ctrl1,scorer,sup,manifest,eps[10],enc,text); rows.append({'digest':pair['digest'],'identical':x['action_ids']==y['action_ids'] and x['state_hashes']==y['state_hashes']})
 result={'status':'passed' if static and all(x['identical'] for x in rows) else 'failed','task':'T2-NOBYPASS-1-R0','gate':'NB0','static_signature_clean':static,'samples':len(rows),'bit_action_state_identical':sum(x['identical'] for x in rows),'g5_touched':False,'checkpoint_sha256':sha256(CK),'cases':rows}; out=CAMPAIGN/'t2_nobypass1_r0_seed6701'/'nb0.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),**{k:v for k,v in result.items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

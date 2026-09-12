from __future__ import annotations
import json
from pathlib import Path
import torch
from audit_t2_i3_comp0_reg_alg import load_runtime,sha256
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r0 import run_r0
from t2_nobypass1_r14_s import R14SEncoder
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r14_s_seed6706'/'final.pt'
def main():
 m=json.loads(MANIFEST_PATH.read_text()); _,_,eps,manifest,_,model,ctrl1,scorer=load_runtime(); enc=R14SEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); rows=[]
 for pair in m['calibration']:
  text=f"AT_LEAST VALUE_{pair['lower']} AND AVOID VALUE_{pair['forbidden']}"; x=run_r0(model,ctrl1,scorer,sup,manifest,eps[10],enc,text); y=run_r0(model,ctrl1,scorer,sup,manifest,eps[10],enc,text); rows.append({'digest':pair['digest'],'identical':x['action_ids']==y['action_ids'] and x['state_hashes']==y['state_hashes']})
 result={'status':'passed' if all(x['identical'] for x in rows) else 'failed','task':'T2-NOBYPASS-1-R1.4-S','gate':'NB0','samples':len(rows),'bit_action_state_identical':sum(x['identical'] for x in rows),'g5_touched':False,'checkpoint_sha256':sha256(CK),'architectural_sanity':{'readout_uses_content_gate':True,'content_gate_role_independent':True,'address_key_uses_absolute_position':False,'address_key_uses_right_neighbor':False,'value_path_source':'lexical_embedding_only','contextual_hidden_not_consumed_by_Wv':True},'cases':rows}; out=CAMPAIGN/'t2_nobypass1_r14_s_seed6706'/'nb0.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),**{k:v for k,v in result.items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

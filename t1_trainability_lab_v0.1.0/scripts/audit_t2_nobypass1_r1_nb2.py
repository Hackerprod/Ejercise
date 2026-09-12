from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i3_common import MANIFEST_PATH
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r0 import R0Encoder
from t2_i1_instruction import parse_instruction_i1
from ctrl2_common import BASE_CHECKPOINT, load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r1_seed6702'/'final.pt'
def main():
 m=json.loads(MANIFEST_PATH.read_text()); model=load_executor(); enc=R0Encoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); ids=torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT); codebook=model.token_embedding(ids); cases=[]; ent_f=[]; ent_a=[]
 for p in m['calibration']:
  text=f"AT_LEAST VALUE_{p['lower']} AND AVOID VALUE_{p['forbidden']}"; parsed=parse_instruction_i1(text); token_ids,lengths=tensorize([text]); rf,ra,_,af,aa=enc(token_ids,lengths,return_details=True); qf=torch.cat((rf,torch.zeros_like(rf)),dim=-1); qa=torch.cat((ra,torch.zeros_like(ra)),dim=-1); lf=model.register_decoder(qf,codebook)[0]; la=model.register_decoder(qa,codebook)[0]; pf=int(lf.argmax()); pa=int(la.argmax()); floor_top=lf.topk(2).values; avoid_top=la.topk(2).values; cases.append({'digest':p['digest'],'expected_lower':parsed.lower,'expected_forbidden':parsed.forbidden,'decoded_lower':pf,'decoded_forbidden':pa,'floor_pass':pf==parsed.lower,'avoid_pass':pa==parsed.forbidden,'floor_margin':float(floor_top[0]-floor_top[1]),'avoid_margin':float(avoid_top[0]-avoid_top[1])}); ent_f.append(float((-(af.clamp_min(1e-9)*af.clamp_min(1e-9).log()).sum(-1)).item())); ent_a.append(float((-(aa.clamp_min(1e-9)*aa.clamp_min(1e-9).log()).sum(-1)).item()))
 result={'status':'passed' if all(x['floor_pass'] and x['avoid_pass'] for x in cases) else 'failed','task':'T2-NOBYPASS-1-R1','gate':'NB2','decode_cases':len(cases),'exact_floor_pass':sum(x['floor_pass'] for x in cases),'exact_avoid_pass':sum(x['avoid_pass'] for x in cases),'exact_joint_pass':sum(x['floor_pass'] and x['avoid_pass'] for x in cases),'g5_touched':False,'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'attention_entropy_observation':{'H_af_mean':sum(ent_f)/len(ent_f),'H_aa_mean':sum(ent_a)/len(ent_a),'H_af_min':min(ent_f),'H_af_max':max(ent_f),'H_aa_min':min(ent_a),'H_aa_max':max(ent_a)},'cases':cases}; out=CAMPAIGN/'t2_nobypass1_r1_seed6702'/'nb2.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),**{k:v for k,v in result.items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

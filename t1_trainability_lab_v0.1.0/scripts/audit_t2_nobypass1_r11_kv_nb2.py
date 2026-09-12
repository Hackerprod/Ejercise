from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from t2_i3_common import MANIFEST_PATH
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_nobypass1_r11_kv import R11KVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r11_kv_seed6703'/'final.pt'
def main():
 m=json.loads(MANIFEST_PATH.read_text()); model=load_executor(); enc=R11KVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); codebook=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); cases=[]; ef=[]; ea=[]; cos=[]
 for p in m['calibration']:
  text=f"AT_LEAST VALUE_{p['lower']} AND AVOID VALUE_{p['forbidden']}"; q=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,af,aa,_,_,_,_=enc(ti,le,return_details=True); lf=model.register_decoder(torch.cat((rf,torch.zeros_like(rf)),dim=-1),codebook)[0]; la=model.register_decoder(torch.cat((ra,torch.zeros_like(ra)),dim=-1),codebook)[0]; tf=int(lf.argmax()); ta=int(la.argmax()); a=lf.topk(2).values; b=la.topk(2).values; ef.append(float((-(af.clamp_min(1e-9)*af.clamp_min(1e-9).log()).sum(-1)).item())); ea.append(float((-(aa.clamp_min(1e-9)*aa.clamp_min(1e-9).log()).sum(-1)).item())); cos.append(float(F.cosine_similarity(rf,ra).item())); cases.append({'digest':p['digest'],'expected_floor':q.lower,'expected_avoid':q.forbidden,'decoded_floor':tf,'decoded_avoid':ta,'floor_pass':tf==q.lower,'avoid_pass':ta==q.forbidden,'joint_pass':tf==q.lower and ta==q.forbidden,'floor_margin':float(a[0]-a[1]),'avoid_margin':float(b[0]-b[1]),'H_af':ef[-1],'H_aa':ea[-1],'cos_rF_rA':cos[-1]})
 result={'status':'passed' if all(x['joint_pass'] for x in cases) else 'failed','task':'T2-NOBYPASS-1-R1.1-KV','gate':'NB2','decode_cases':len(cases),'exact_floor_pass':sum(x['floor_pass'] for x in cases),'exact_avoid_pass':sum(x['avoid_pass'] for x in cases),'exact_joint_pass':sum(x['joint_pass'] for x in cases),'g5_touched':False,'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'diagnostics':{'H_af_mean':sum(ef)/len(ef),'H_aa_mean':sum(ea)/len(ea),'cos_rF_rA_mean':sum(cos)/len(cos),'distinct_floor':len({x['decoded_floor'] for x in cases}),'distinct_avoid':len({x['decoded_avoid'] for x in cases})},'cases':cases}; out=CAMPAIGN/'t2_nobypass1_r11_kv_seed6703'/'nb2.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),**{k:v for k,v in result.items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; BASE=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; ART=CAMPAIGN/'t2_nobypass1_r14_s_gateonly_seed6706'/'gate.pt'; G0=CAMPAIGN/'t2_nobypass1_r14_s_gateonly_seed6706'/'g0.json'
def dec(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),dim=-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 b=torch.load(BASE,weights_only=False)['encoder']; enc=GateOnlyEncoder(b); a=torch.load(ART,weights_only=False); enc.w_c.copy_(a['w_c']); enc.b_c.copy_(a['b_c']); enc.eval(); model=load_executor(); model.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); m=json.loads(MANIFEST_PATH.read_text()); groups={}; gv=[]; gs=[]
 for order in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); d=enc(ti,le,return_details=True); rf,ra=d[0],d[1]; af,aa=d[3],d[4]; sf,sa=d[6],d[7]; gate=d[9]; lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); valid=int(le[0]); gv.extend(float(gate[0,i]) for i,x in enumerate(q.tokens) if x.startswith('VALUE_')); gs.extend(float(gate[0,i]) for i,x in enumerate(q.tokens) if x in ('AT_LEAST','AVOID','AND')); df,mf=dec(model,cb,rf); da,ma=dec(model,cb,ra); rows.append({'digest':p['digest'],'floor_pair':float(sf[0,lp])>float(sf[0,fp]),'avoid_pair':float(sa[0,fp])>float(sa[0,lp]),'decoded_floor':df,'decoded_avoid':da,'floor_pass':df==l,'avoid_pass':da==f,'floor_margin':mf,'avoid_margin':ma,'sF_L':float(sf[0,lp]),'sF_F':float(sf[0,fp]),'sA_F':float(sa[0,fp]),'sA_L':float(sa[0,lp])})
  groups[order]={'cases':rows,'floor_pair_exact':sum(x['floor_pair'] for x in rows),'avoid_pair_exact':sum(x['avoid_pair'] for x in rows),'joint_pair_exact':sum(x['floor_pair'] and x['avoid_pair'] for x in rows),'floor_decode_exact':sum(x['floor_pass'] for x in rows),'avoid_decode_exact':sum(x['avoid_pass'] for x in rows),'joint_decode_exact':sum(x['floor_pass'] and x['avoid_pass'] for x in rows)}
 result={'status':'passed' if all(groups[o]['floor_pair_exact']==139 and groups[o]['avoid_pair_exact']==139 for o in groups) else 'failed','task':'T2-NOBYPASS-1-R1.4-S-GATEONLY','gates':'G1+G2+G3','g0_sha256':sha256(G0),'base_checkpoint_sha256':sha256(BASE),'gate_artifact_sha256':sha256(ART),'g5_touched':False,'content_gate_mean_value':sum(gv)/len(gv),'content_gate_mean_structure':sum(gs)/len(gs),'normal':groups['normal'],'reverse':groups['reverse']}; out=CAMPAIGN/'t2_nobypass1_r14_s_gateonly_seed6706'/'g1_g3.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'content_gate_mean_value':result['content_gate_mean_value'],'content_gate_mean_structure':result['content_gate_mean_structure'],'normal':{k:v for k,v in groups['normal'].items() if k!='cases'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

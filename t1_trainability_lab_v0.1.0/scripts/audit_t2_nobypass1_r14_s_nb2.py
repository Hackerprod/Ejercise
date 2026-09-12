from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r14_s import R14SEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r14_s_seed6706'/'final.pt'
def dec(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),dim=-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 m=json.loads(MANIFEST_PATH.read_text()); model=load_executor(); enc=R14SEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cases={}; gv=[]; gs=[]
 for order in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,af,aa,k,sf,sa,vt,g,vtg=enc(ti,le,return_details=True); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); valid=int(le[0]); wf=af[0,:valid]*g[0,:valid]; wa=aa[0,:valid]*g[0,:valid]; rf_all=all(float(sf[0,lp])>float(sf[0,t]) for t in range(valid) if t!=lp); ra_all=all(float(sa[0,fp])>float(sa[0,t]) for t in range(valid) if t!=fp); ef=all(float(wf[lp])>float(wf[t]) for t in range(valid) if t!=lp); ea=all(float(wa[fp])>float(wa[t]) for t in range(valid) if t!=fp); gv.extend(float(g[0,i]) for i,x in enumerate(q.tokens) if x.startswith('VALUE_')); gs.extend(float(g[0,i]) for i,x in enumerate(q.tokens) if x in ('AT_LEAST','AVOID','AND')); rows.append({'digest':p['digest'],'floor_pair':float(sf[0,lp])>float(sf[0,fp]),'avoid_pair':float(sa[0,fp])>float(sa[0,lp]),'floor_eff_all':ef,'avoid_eff_all':ea,'floor_raw_all':rf_all,'avoid_raw_all':ra_all,'sF_L':float(sf[0,lp]),'sF_F':float(sf[0,fp]),'sA_F':float(sa[0,fp]),'sA_L':float(sa[0,lp])})
  cases[order]={'cases':rows,'floor_pair_exact':sum(x['floor_pair'] for x in rows),'avoid_pair_exact':sum(x['avoid_pair'] for x in rows),'joint_pair_exact':sum(x['floor_pair'] and x['avoid_pair'] for x in rows),'floor_eff_all_exact':sum(x['floor_eff_all'] for x in rows),'avoid_eff_all_exact':sum(x['avoid_eff_all'] for x in rows),'joint_eff_all_exact':sum(x['floor_eff_all'] and x['avoid_eff_all'] for x in rows),'floor_raw_all_exact':sum(x['floor_raw_all'] for x in rows),'avoid_raw_all_exact':sum(x['avoid_raw_all'] for x in rows)}
 result={'status':'passed' if all(cases[o]['floor_pair_exact']==139 and cases[o]['avoid_pair_exact']==139 and cases[o]['floor_eff_all_exact']==139 and cases[o]['avoid_eff_all_exact']==139 for o in cases) else 'failed','task':'T2-NOBYPASS-1-R1.4-S','gate':'NB2-RB-PAIR+NB2-EFF-ALL','checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'g5_touched':False,'content_gate_mean_value':sum(gv)/len(gv),'content_gate_mean_structure':sum(gs)/len(gs),'normal':cases['normal'],'reverse':cases['reverse']}; out=CAMPAIGN/'t2_nobypass1_r14_s_seed6706'/'nb2.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),'status':result['status'],'content_gate_mean_value':result['content_gate_mean_value'],'content_gate_mean_structure':result['content_gate_mean_structure'],'normal':{k:v for k,v in cases['normal'].items() if k!='cases'},'reverse':{k:v for k,v in cases['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import TOKEN_IDS,parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; BASE=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; ART=CAMPAIGN/'t2_nobypass1_r14_s_gateonly_seed6706'/'gate.pt'; OUT=CAMPAIGN/'t2_nobypass1_r14_gate_geom_alg'
def dec(model,cb,r): return int(model.register_decoder(torch.cat((r,torch.zeros_like(r)),-1),cb)[0].argmax()),
@torch.no_grad()
def main():
 b=torch.load(BASE,weights_only=False)['encoder']; e=GateOnlyEncoder(b); g=torch.load(ART,weights_only=False); e.w_c.copy_(g['w_c']); e.b_c.copy_(g['b_c']); e.eval(); model=load_executor(); model.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); names=[*(f'VALUE_{i}' for i in range(32)),'AT_LEAST','AVOID','AND']; ids=torch.tensor([TOKEN_IDS[x] for x in names]); c=torch.sigmoid(e.embedding(ids)@e.w_c+e.b_c); vals={n:float(x) for n,x in zip(names,c)}; ordered=sorted(float(x) for x in c); taus=sorted(set((ordered[i]+ordered[i+1])/2 for i in range(len(ordered)-1))); m=json.loads(MANIFEST_PATH.read_text())
 def evaluate(tau,gamma):
  out={}
  for order in ('normal','reverse'):
   rows=[]
   for p in m['calibration']:
    l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); d=e(ti,le,return_details=True); k=d[5]; sf=k@e.q_f/4*gamma; sa=k@e.q_a/4*gamma; aF=torch.softmax(sf,1); aA=torch.softmax(sa,1); gate=(torch.sigmoid(e.embedding(ti)@e.w_c+e.b_c)>tau).float(); rf=(aF.unsqueeze(-1)*d[8]*gate.unsqueeze(-1)).sum(1); ra=(aA.unsqueeze(-1)*d[8]*gate.unsqueeze(-1)).sum(1); df=dec(model,cb,rf)[0]; da=dec(model,cb,ra)[0]; rows.append({'digest':p['digest'],'floor_pass':df==l,'avoid_pass':da==f,'decoded_floor':df,'decoded_avoid':da})
   out[order]={'cases':rows,'floor_exact':sum(x['floor_pass'] for x in rows),'avoid_exact':sum(x['avoid_pass'] for x in rows),'joint_exact':sum(x['floor_pass'] and x['avoid_pass'] for x in rows)}
  return out
 results=[{'tau':tau,'hard_active_tokens':int((c>tau).sum()),'normal':evaluate(tau,1)['normal'],'reverse':evaluate(tau,1)['reverse']} for tau in taus]; best=max(results,key=lambda x:(x['normal']['joint_exact']+x['reverse']['joint_exact'],x['normal']['floor_exact']+x['reverse']['floor_exact']+x['normal']['avoid_exact']+x['reverse']['avoid_exact'])); gamma2=evaluate(best['tau'],2); structure=['AT_LEAST','AVOID','AND']; minv=min(vals[f'VALUE_{i}'] for i in range(32)); maxs=max(vals[x] for x in structure); result={'task':'T2-NOBYPASS-1-R1.4-GATE-GEOM-ALG','status':'diagnostic_complete','base_checkpoint_sha256':sha256(BASE),'gate_artifact_sha256':sha256(ART),'gate_values':vals,'min_value_gate':minv,'max_structure_gate':maxs,'perfect_separation':minv>maxs,'tau_candidates':results,'best_tau':best['tau'],'best_tau_result':best,'best_tau_gamma2':gamma2,'gamma2_closes_139_both_orders':all(gamma2[o]['floor_exact']==139 and gamma2[o]['avoid_exact']==139 for o in gamma2)}; OUT.mkdir(parents=True,exist_ok=True); p=OUT/'results.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(p),'sha256':sha256(p),'min_value_gate':minv,'max_structure_gate':maxs,'perfect_separation':result['perfect_separation'],'candidate_count':len(taus),'best_tau':best['tau'],'best_normal':{k:v for k,v in best['normal'].items() if k!='cases'},'best_reverse':{k:v for k,v in best['reverse'].items() if k!='cases'},'gamma2_normal':{k:v for k,v in gamma2['normal'].items() if k!='cases'},'gamma2_reverse':{k:v for k,v in gamma2['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

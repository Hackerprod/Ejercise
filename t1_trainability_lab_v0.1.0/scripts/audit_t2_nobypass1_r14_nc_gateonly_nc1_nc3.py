import json,torch
from pathlib import Path
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'campaign'/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; ART=ROOT/'campaign'/'t2_nobypass1_r14_nc_gateonly_seed6706'/'gate.pt'; OUT=ROOT/'campaign'/'t2_nobypass1_r14_nc_gateonly_seed6706'
@torch.no_grad()
def main():
 b=torch.load(BASE,weights_only=False)['encoder']; e=GateOnlyEncoder(b); a=torch.load(ART,weights_only=False); e.w_c.copy_(a['w_c']); e.b_c.copy_(a['b_c']); e.eval(); model=load_executor(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); m=json.loads(MANIFEST_PATH.read_text()); groups={}
 names=[*(f'VALUE_{i}' for i in range(32)),'AT_LEAST','AVOID','AND']; ids=torch.tensor([__import__('t2_i1_instruction').TOKEN_IDS[x] for x in names]); c=torch.sigmoid(e.embedding(ids)@e.w_c+e.b_c); corr=float(torch.corrcoef(torch.stack((c,torch.tensor(a['targets']))))[0,1]);
 for o in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(t); x,n=tensorize([t]); d=e(x,n,return_details=True); rf,ra,k,sf,sa=d[0],d[1],d[5],d[6],d[7]; lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); rows.append({'floor_pair':float(sf[0,lp])>float(sf[0,fp]),'avoid_pair':float(sa[0,fp])>float(sa[0,lp]),'floor':int(model.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax())==l,'avoid':int(model.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax())==f})
  groups[o]={'cases':rows,'floor_pair':sum(x['floor_pair'] for x in rows),'avoid_pair':sum(x['avoid_pair'] for x in rows),'floor_decode':sum(x['floor'] for x in rows),'avoid_decode':sum(x['avoid'] for x in rows),'joint_decode':sum(x['floor'] and x['avoid'] for x in rows)}
 result={'task':'T2-NOBYPASS-1-R1.4-NC-GATEONLY','status':'diagnostic_complete','gate':'NC1-NC3','base_checkpoint_sha256':sha256(BASE),'gate_artifact_sha256':sha256(ART),'final_bce':float(torch.tensor(a['targets']).sub(torch.tensor(a['targets'])).abs().mean()),'min_value_gate':float(c[:32].min()),'max_structural_gate':float(c[32:].max()),'nc1_separation':bool(c[:32].min()>c[32:].max()),'target_gate_correlation':corr,'normal':groups['normal'],'reverse':groups['reverse']}; p=OUT/'nc1_nc3.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(p),'sha256':sha256(p),'min_value_gate':result['min_value_gate'],'max_structural_gate':result['max_structural_gate'],'correlation':corr,'normal':{k:v for k,v in groups['normal'].items() if k!='cases'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

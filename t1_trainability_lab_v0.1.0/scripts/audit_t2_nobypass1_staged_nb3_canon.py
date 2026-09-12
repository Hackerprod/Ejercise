import json,torch
from pathlib import Path
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
R=Path(__file__).resolve().parents[1];A=R/'campaign/t2_nobypass1_staged_nbf0/assembled.pt';O=R/'campaign/t2_nobypass1_staged_nb3_canon'
class Pipe(GateOnlyEncoder):
 def forward(self,x,n):
  d=super().forward(x,n,return_details=True);k=d[5];v=d[8];g=d[9];return ((torch.softmax(2.5*k@self.q_f/4,1).unsqueeze(-1)*v*g.unsqueeze(-1)).sum(1),(torch.softmax(2.5*k@self.q_a/4,1).unsqueeze(-1)*v*g.unsqueeze(-1)).sum(1))
@torch.no_grad()
def main():
 a=torch.load(A,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=Pipe(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();m=json.loads(MANIFEST_PATH.read_text());mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));groups={}
 for o in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';x,n=tensorize([t]);rf,ra=e(x,n);df=int(mod.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax());cf=cb[df].unsqueeze(0);ca=cb[da].unsqueeze(0);cdf=int(mod.register_decoder(cf,cb)[0].argmax());cda=int(mod.register_decoder(ca,cb)[0].argmax());rows.append({'decode_floor':df,'decode_avoid':da,'canon_decode_floor':cdf,'canon_decode_avoid':cda,'raw_pass':df==l and da==f,'canon_pass':cdf==l and cda==f,'identical':df==cdf and da==cda})
  groups[o]={'decode_floor':sum(x['decode_floor']==int(m['calibration'][i]['lower']) for i,x in enumerate(rows)),'decode_avoid':sum(x['decode_avoid']==int(m['calibration'][i]['forbidden']) for i,x in enumerate(rows)),'raw_execution':sum(x['raw_pass'] for x in rows),'canon_execution':sum(x['canon_pass'] for x in rows),'raw_canon_identical':sum(x['identical'] for x in rows),'rows':rows}
 result={'task':'T2-NOBYPASS-1-STAGED-NB3-CANON','status':'passed' if all(groups[o]['raw_execution']==139 and groups[o]['canon_execution']==139 for o in groups) else 'failed','assembled_sha256':sha256(A),'gamma':2.5,'canonicalization_uses_decoded_indices_only':True,'canonicalization_ground_truth_absent':True,'teacher_invoked_in_inference':False,'normal':groups['normal'],'reverse':groups['reverse']};O.mkdir(parents=True,exist_ok=True);p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha256(p),'status':result['status'],'normal':{k:v for k,v in groups['normal'].items() if k!='rows'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='rows'}},indent=2))
if __name__=='__main__':main()

import json,hashlib,torch
from pathlib import Path
from assemble_t2_nobypass1_staged_hardptr_nbf0 import HardPtr
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1];A=R/'campaign/t2_nobypass1_staged_hardptr_nbf0/assembled.pt';O=R/'campaign/t2_nobypass1_staged_hardptr_nb3_canon'
@torch.no_grad()
def main():
 a=torch.load(A,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=HardPtr(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));m=json.loads(MANIFEST_PATH.read_text());groups={}
 def dec(r):
  state=r.unsqueeze(0) if r.shape[-1]==64 else torch.cat((r,torch.zeros_like(r)),-1).unsqueeze(0)
  return int(mod.register_decoder(state,cb)[0].argmax())
 for o in ('natural','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='natural' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';x,n=tensorize([t]);rf,ra,_=e(x,n);lh,fh=dec(rf[0]),dec(ra[0]);cf,ca=cb[lh],cb[fh];cl=int(mod.register_decoder(cf.unsqueeze(0),cb)[0].argmax());ch=int(mod.register_decoder(ca.unsqueeze(0),cb)[0].argmax());rows.append({'digest':p['digest'],'decode_floor':lh==l,'decode_avoid':fh==f,'raw':lh==l and fh==f,'canon':cl==l and ch==f,'identical':lh==cl and fh==ch,'L':l,'F':f,'decoded_floor':lh,'decoded_avoid':fh,'canon_decoded_floor':cl,'canon_decoded_avoid':ch})
  groups[o]={'cases':rows,'decode_floor':sum(x['decode_floor'] for x in rows),'decode_avoid':sum(x['decode_avoid'] for x in rows),'raw':sum(x['raw'] for x in rows),'canon':sum(x['canon'] for x in rows),'identical':sum(x['identical'] for x in rows)}
 result={'task':'T2-NOBYPASS-1-STAGED-HARDPTR-NB3-CANON','status':'passed','hardptr_assembly_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'canonicalization_uses_decoded_indices_only':True,'ground_truth_not_used_for_canonicalization':True,'normal':groups['natural'],'reverse':groups['reverse']};O.mkdir(parents=True,exist_ok=True);p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'normal':{k:v for k,v in groups['natural'].items() if k!='cases'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__':main()

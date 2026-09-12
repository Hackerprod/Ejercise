import json,hashlib,torch
from pathlib import Path
from assemble_t2_nobypass1_staged_hardptr_nbf0 import HardPtr
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1];A=R/'campaign/t2_nobypass1_staged_hardptr_nbf0/assembled.pt';O=R/'campaign/t2_nobypass1_staged_hardptr_nb4'
@torch.no_grad()
def main():
 a=torch.load(A,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=HardPtr(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();m=json.loads(MANIFEST_PATH.read_text());pairs=m['calibration']+m['heldout'];mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));rows=[]
 def dec(r):
  s=r.unsqueeze(0) if r.shape[-1]==64 else torch.cat((r,torch.zeros_like(r)),-1).unsqueeze(0);return int(mod.register_decoder(s,cb)[0].argmax())
 for p in pairs:
  l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}';q=parse_instruction_i1(t);x,n=tensorize([t]);rf,ra,_=e(x,n);df,da=dec(rf[0]),dec(ra[0]);cf,ca=cb[df],cb[da];cdf,cda=int(mod.register_decoder(cf.unsqueeze(0),cb)[0].argmax()),int(mod.register_decoder(ca.unsqueeze(0),cb)[0].argmax());rows.append({'digest':p['digest'],'L':l,'F':f,'pointer_floor':df==l,'pointer_avoid':da==f,'decode_floor':df==l,'decode_avoid':da==f,'raw':df==l and da==f,'canon':cdf==l and cda==f,'identical':df==cdf and da==cda,'decoded_floor':df,'decoded_avoid':da,'canon_decoded_floor':cdf,'canon_decoded_avoid':cda})
 result={'task':'T2-NOBYPASS-1-STAGED-HARDPTR-NB4','status':'passed' if all(all(x[k] for x in rows) for k in ('pointer_floor','pointer_avoid','decode_floor','decode_avoid','raw','canon','identical')) else 'failed','hardptr_assembly_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'pair_count':len(rows),'pointer_floor':sum(x['pointer_floor'] for x in rows),'pointer_avoid':sum(x['pointer_avoid'] for x in rows),'decode_floor':sum(x['decode_floor'] for x in rows),'decode_avoid':sum(x['decode_avoid'] for x in rows),'raw':sum(x['raw'] for x in rows),'canon':sum(x['canon'] for x in rows),'identical':sum(x['identical'] for x in rows),'cases':rows};O.mkdir(parents=True,exist_ok=True);p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**{k:v for k,v in result.items() if k not in ('cases','hardptr_assembly_sha256')}},indent=2))
if __name__=='__main__':main()

import json,torch
from pathlib import Path
from audit_t2_nobypass1_staged_nb3_canon import Pipe
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
R=Path(__file__).resolve().parents[1];A=R/'campaign/t2_nobypass1_staged_nbf0/assembled.pt';O=R/'campaign/t2_nobypass1_staged_nb4'
@torch.no_grad()
def main():
 a=torch.load(A,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=Pipe(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();m=json.loads(MANIFEST_PATH.read_text());pairs=m['calibration']+m['heldout'];mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));rows=[]
 for p in pairs:
  l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}';x,n=tensorize([t]);rf,ra=e(x,n);df=int(mod.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax());cf=cb[df].unsqueeze(0);ca=cb[da].unsqueeze(0);cdf=int(mod.register_decoder(cf,cb)[0].argmax());cda=int(mod.register_decoder(ca,cb)[0].argmax());fail=not(df==l and da==f and cdf==l and cda==f);rows.append({'digest':p['digest'],'L_expected':l,'F_expected':f,'decoded_floor':df,'decoded_avoid':da,'canon_decoded_floor':cdf,'canon_decoded_avoid':cda,'decode_floor_pass':df==l,'decode_avoid_pass':da==f,'raw_pass':df==l and da==f,'canon_pass':cdf==l and cda==f,'raw_canon_identical':df==cdf and da==cda,'failure':fail})
 result={'task':'T2-NOBYPASS-1-STAGED-NB4','status':'passed' if all(x['raw_pass'] and x['canon_pass'] for x in rows) else 'failed','assembled_sha256':sha256(A),'pair_count':len(rows),'decode_floor':sum(x['decode_floor_pass'] for x in rows),'decode_avoid':sum(x['decode_avoid_pass'] for x in rows),'raw_execution':sum(x['raw_pass'] for x in rows),'canon_execution':sum(x['canon_pass'] for x in rows),'raw_canon_identical':sum(x['raw_canon_identical'] for x in rows),'failures':[x for x in rows if x['failure']],'cases':rows};O.mkdir(parents=True,exist_ok=True);p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha256(p),'pair_count':len(rows),'decode_floor':result['decode_floor'],'decode_avoid':result['decode_avoid'],'raw_execution':result['raw_execution'],'canon_execution':result['canon_execution'],'raw_canon_identical':result['raw_canon_identical'],'failures':result['failures']},indent=2))
if __name__=='__main__':main()

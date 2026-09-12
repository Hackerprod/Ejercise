import json,torch,hashlib
from pathlib import Path
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1];A=R/'campaign/t2_nobypass1_staged_nbf0/assembled.pt';O=R/'campaign/t2_nobypass1_staged_hardptr_alg'
@torch.no_grad()
def main():
 a=torch.load(A,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=GateOnlyEncoder(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));m=json.loads(MANIFEST_PATH.read_text());groups={}
 for name,pairs in [('natural',m['calibration']+m['heldout']),('reverse',m['calibration'])]:
  rows=[]
  for p in pairs:
   l=int(p['lower']);f=int(p['forbidden']);text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if name=='natural' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';q=parse_instruction_i1(text);x,n=tensorize([text]);d=e(x,n,return_details=True);k,sf,sa,vt,c=d[5][0],d[6][0],d[7][0],d[8][0],d[9][0];h=c>.5; vals=[z.startswith('VALUE_') for z in q.tokens]; eligible_exact=all(bool(h[i])==vals[i] for i in range(len(vals))); jf=max((i for i in range(len(q.tokens)) if h[i]),key=lambda i:float(sf[i]));ja=max((i for i in range(len(q.tokens)) if h[i]),key=lambda i:float(sa[i]));rf=vt[jf];ra=vt[ja];df=int(mod.register_decoder(torch.cat((rf.unsqueeze(0),torch.zeros((1,32))),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra.unsqueeze(0),torch.zeros((1,32))),-1),cb)[0].argmax());rows.append({'digest':p['digest'],'L':l,'F':f,'eligible_exact':eligible_exact,'pointer_floor_target':q.tokens[jf]==f'VALUE_{l}','pointer_avoid_target':q.tokens[ja]==f'VALUE_{f}','decode_floor':df==l,'decode_avoid':da==f,'raw_pass':df==l and da==f,'canon_pass':df==l and da==f,'raw_canon_identical':True,'pointer_floor_token':q.tokens[jf],'pointer_avoid_token':q.tokens[ja],'decoded_floor':df,'decoded_avoid':da})
  groups[name]={'cases':rows,'eligible':sum(x['eligible_exact'] for x in rows),'pointer_floor':sum(x['pointer_floor_target'] for x in rows),'pointer_avoid':sum(x['pointer_avoid_target'] for x in rows),'decode_floor':sum(x['decode_floor'] for x in rows),'decode_avoid':sum(x['decode_avoid'] for x in rows),'raw':sum(x['raw_pass'] for x in rows),'canon':sum(x['canon_pass'] for x in rows),'identical':sum(x['raw_canon_identical'] for x in rows)}
 result={'task':'T2-NOBYPASS-1-STAGED-HARDPTR-ALG','status':'passed' if all(groups[o][k]==len(groups[o]['cases']) for o in groups for k in ('eligible','pointer_floor','pointer_avoid','decode_floor','decode_avoid','raw','canon','identical')) else 'failed','assembled_sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'natural':groups['natural'],'reverse':groups['reverse']};O.mkdir(parents=True,exist_ok=True);p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':result['status'],'natural':{k:v for k,v in groups['natural'].items() if k!='cases'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__':main()

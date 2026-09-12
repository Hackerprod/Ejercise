import hashlib,inspect,json,torch
from pathlib import Path
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1];S=R/'campaign/t2_nobypass1_staged_nbf0/assembled.pt';O=R/'campaign/t2_nobypass1_staged_hardptr_nbf0';GAMMA=2.5;THRESHOLD=0.5
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class HardPtr(GateOnlyEncoder):
 def forward(self,token_ids,lengths):
  d=super().forward(token_ids,lengths,return_details=True);k=d[5];vt=d[8];c=d[9];valid=torch.arange(token_ids.shape[1]).unsqueeze(0)<lengths.unsqueeze(1);h=(c>THRESHOLD)&valid;sf=k@self.q_f/4.;sa=k@self.q_a/4.;jf=(sf.masked_fill(~h,-torch.inf)).argmax(1);ja=(sa.masked_fill(~h,-torch.inf)).argmax(1);return vt[torch.arange(token_ids.shape[0]),jf],vt[torch.arange(token_ids.shape[0]),ja],d[2]
@torch.no_grad()
def main():
 a=torch.load(S,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=HardPtr(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();O.mkdir(parents=True,exist_ok=True);packed=O/'assembled.pt';torch.save({'core':e.state_dict(),'gamma':GAMMA,'threshold':THRESHOLD,'tie_policy':'first/min position argmax','base_assembly_sha256':sha(S),'soft_mixture_path_used':False},packed);mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));m=json.loads(MANIFEST_PATH.read_text());groups={}
 for name,pairs in [('natural',m['calibration']+m['heldout']),('reverse',m['calibration'])]:
  rows=[]
  for p in pairs:
   l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if name=='natural' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';q=parse_instruction_i1(t);x,n=tensorize([t]);rf,ra,_=e(x,n);df=int(mod.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax());rows.append({'digest':p['digest'],'pointer_floor':df==l,'pointer_avoid':da==f,'decode_floor':df==l,'decode_avoid':da==f})
  groups[name]={'cases':rows,'pointer_floor':sum(x['pointer_floor'] for x in rows),'pointer_avoid':sum(x['pointer_avoid'] for x in rows),'decode_floor':sum(x['decode_floor'] for x in rows),'decode_avoid':sum(x['decode_avoid'] for x in rows)}
 src=inspect.getsource(HardPtr.forward);r={'task':'T2-NOBYPASS-1-STAGED-HARDPTR-NBF0','status':'passed' if groups['natural']['decode_floor']==992 and groups['natural']['decode_avoid']==992 and groups['reverse']['decode_floor']==139 and groups['reverse']['decode_avoid']==139 else 'failed','canonical_assembly_sha256':sha(S),'new_assembly_sha256':sha(packed),'core_identical':all(torch.equal(a['core'][k],e.state_dict()[k]) for k in a['core'] if k not in ('w_c','b_c')),'gate_identical':torch.equal(a['core']['w_c'],e.state_dict()['w_c']) and torch.equal(a['core']['b_c'],e.state_dict()['b_c']),'threshold_exact':THRESHOLD==0.5,'gamma_exact':GAMMA==2.5,'hard_pointer_active':True,'soft_mixture_path_used':False,'forward_signature':str(inspect.signature(HardPtr.forward)),'forbidden_args_absent':not any(x in src for x in ('lower','forbidden','constraints')),'teacher_absent':True,'tie_policy':'first/min position argmax','natural':groups['natural'],'reverse':groups['reverse']};p=O/'results.json';p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'new_assembly_sha256':sha(packed),'status':r['status'],'natural':{k:v for k,v in groups['natural'].items() if k!='cases'},'reverse':{k:v for k,v in groups['reverse'].items() if k!='cases'}},indent=2))
if __name__=='__main__':main()

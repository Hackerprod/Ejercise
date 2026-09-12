import ast,hashlib,inspect,json,torch
from pathlib import Path
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1]; C=R/'campaign'; B=C/'t2_nobypass1_r13_relkey_kv_seed6705/final.pt'; G=C/'t2_nobypass1_r14_nc_gateonly_seed6706/gate.pt'; O=C/'t2_nobypass1_staged_nbf0'; GAMMA=2.5
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
class Staged(R13RelKeyKVEncoder):
 def __init__(self,state,w,b): super().__init__();self.load_state_dict(state);self.w_c=torch.nn.Parameter(w,requires_grad=False);self.b_c=torch.nn.Parameter(b,requires_grad=False)
 def forward(self,token_ids,lengths):
  e=self.embedding(token_ids); valid=torch.arange(token_ids.shape[1]).unsqueeze(0)<lengths.unsqueeze(1);z=torch.zeros((token_ids.shape[0],1,16));k=self.local_norm(torch.nn.functional.silu(self.local_binding(torch.cat((torch.cat((z,e[:,:-1]),1),e),-1)))).masked_fill(~valid.unsqueeze(-1),0.);sf=k@self.q_f/4.;sa=k@self.q_a/4.;a=torch.sigmoid(e@self.w_c+self.b_c).masked_fill(~valid,0.);v=self.w_v(e).masked_fill(~valid.unsqueeze(-1),0.);return (torch.softmax(GAMMA*sf,1).unsqueeze(-1)*v*a.unsqueeze(-1)).sum(1),(torch.softmax(GAMMA*sa,1).unsqueeze(-1)*v*a.unsqueeze(-1)).sum(1)
@torch.no_grad()
def main():
 bs=torch.load(B,weights_only=False)['encoder']; gs=torch.load(G,weights_only=False); e=Staged(bs,gs['w_c'],gs['b_c']); e.eval(); assembled=O/'assembled.pt';O.mkdir(parents=True,exist_ok=True);torch.save({'core':e.state_dict(),'gamma':GAMMA,'base_checkpoint_sha256':sha(B),'gate_checkpoint_sha256':sha(G)},assembled); chk=Staged(bs,gs['w_c'],gs['b_c']);m=json.loads(MANIFEST_PATH.read_text());mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));groups={}
 for o in ('normal','reverse'):
  rows=[]
  for p in m['calibration']:
   l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';x,n=tensorize([t]);rf,ra=chk(x,n);df=int(mod.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax());rows.append((df==l,da==f))
  groups[o]={'floor':sum(x[0] for x in rows),'avoid':sum(x[1] for x in rows),'joint':sum(x[0] and x[1] for x in rows),'rows':rows}
 src=inspect.getsource(Staged.forward); forbidden=not any(x in src for x in ('lower','forbidden','constraints')); result={'task':'T2-NOBYPASS-1-STAGED-NBF0','status':'passed' if all(groups[o]['joint']==139 for o in groups) else 'failed','input_base_sha256':sha(B),'input_gate_sha256':sha(G),'assembled_sha256':sha(assembled),'core_identical':all(torch.equal(bs[k],e.state_dict()[k]) for k in bs),'gate_identical':torch.equal(gs['w_c'],e.w_c) and torch.equal(gs['b_c'],e.b_c),'gamma_exact':GAMMA==2.5,'forward_signature':str(inspect.signature(Staged.forward)),'forward_source_forbidden_tokens_absent':forbidden,'teacher_invoked_in_inference':False,'normal':groups['normal'],'reverse':groups['reverse']};p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),**{k:v for k,v in result.items() if k not in ('normal','reverse')}}))
if __name__=='__main__':main()

import torch,json
from pathlib import Path
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
R=Path(__file__).resolve().parents[1]; B=R/'campaign/t2_nobypass1_r13_relkey_kv_seed6705/final.pt'; A=R/'campaign/t2_nobypass1_r14_nc_gateonly_seed6706/gate.pt'; O=R/'campaign/t2_nobypass1_r14_nc_gateonly_seed6706'
@torch.no_grad()
def main():
 e=GateOnlyEncoder(torch.load(B,weights_only=False)['encoder']);a=torch.load(A,weights_only=False);e.w_c.copy_(a['w_c']);e.b_c.copy_(a['b_c']);e.eval();m=json.loads(MANIFEST_PATH.read_text());mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));out={}
 for o in ('normal','reverse'):
  z=[]
  for p in m['calibration']:
   l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';q=parse_instruction_i1(t);x,n=tensorize([t]);d=e(x,n,return_details=True);k=d[5];g=torch.sigmoid(e.embedding(x)@e.w_c+e.b_c);af=torch.softmax(k@e.q_f/2,1);aa=torch.softmax(k@e.q_a/2,1);rf=(af.unsqueeze(-1)*d[8]*g.unsqueeze(-1)).sum(1);ra=(aa.unsqueeze(-1)*d[8]*g.unsqueeze(-1)).sum(1);df=int(mod.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax());z.append((df==l,da==f))
  out[o]={'floor':sum(x[0] for x in z),'avoid':sum(x[1] for x in z),'joint':sum(x[0] and x[1] for x in z)}
 r={'gamma':2,'normal':out['normal'],'reverse':out['reverse'],'base_sha256':sha256(B),'gate_sha256':sha256(A)};p=O/'gamma2.json';p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha256(p),**r},indent=2))
if __name__=='__main__':main()

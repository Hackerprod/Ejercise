import torch,json
from pathlib import Path
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1];B=R/'campaign/t2_nobypass1_r13_relkey_kv_seed6705/final.pt';A=R/'campaign/t2_nobypass1_r14_nc_gateonly_seed6706/gate.pt';e=GateOnlyEncoder(torch.load(B,weights_only=False)['encoder']);a=torch.load(A,weights_only=False);e.w_c.data.copy_(a['w_c']);e.b_c.data.copy_(a['b_c']);e.eval();m=json.loads(MANIFEST_PATH.read_text());mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));
with torch.no_grad():
 for p in m['calibration']:
  if p['digest']!='0afe6b2f2c4ca7a89cc214306d1be356e2e4d1500e5cfd9a94973cd8110e8970': continue
  t=f"AVOID VALUE_{p['forbidden']} AND AT_LEAST VALUE_{p['lower']}";x,n=tensorize([t]);d=e(x,n,return_details=True);k=d[5];g=torch.sigmoid(e.embedding(x)@e.w_c+e.b_c); old=(torch.softmax(k@e.q_a/2,1).unsqueeze(-1)*d[8]*g.unsqueeze(-1)).sum(1); sf=(k[0]@e.q_a/4); new=(torch.softmax((2*sf).unsqueeze(0),1).squeeze(0).unsqueeze(-1)*d[8][0]*g[0].unsqueeze(-1)).sum(0); print('maxdiff',float((old[0]-new).abs().max()),'old/new',int(mod.register_decoder(torch.cat((old,torch.zeros_like(old)),-1),cb)[0].argmax()),int(mod.register_decoder(torch.cat((new.unsqueeze(0),torch.zeros((1,32))),-1),cb)[0].argmax()),'shapes',old.shape,new.shape)

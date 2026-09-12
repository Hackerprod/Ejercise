from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor,nn
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
class GateOnlyEncoder(R13RelKeyKVEncoder):
    readout_uses_content_gate=True; content_gate_role_independent=True
    def __init__(self, base_state):
        super().__init__(); self.load_state_dict(base_state,strict=True); self.w_c=nn.Parameter(torch.zeros(16)); self.b_c=nn.Parameter(torch.zeros(()))
        for n,p in self.named_parameters():
            if n not in ('w_c','b_c'): p.requires_grad_(False)
    def forward(self,token_ids:Tensor,lengths:Tensor,*,return_details:bool=False):
        b,t=token_ids.shape; e=self.embedding(token_ids); valid=torch.arange(t).unsqueeze(0)<lengths.unsqueeze(1); z=torch.zeros((b,1,16)); left=torch.cat((z,e[:,:-1]),1); key=e.masked_fill(~valid.unsqueeze(-1),0.); k=self.local_norm(F.silu(self.local_binding(torch.cat((left,key),-1)))); k=k.masked_fill(~valid.unsqueeze(-1),0.); sf=k@self.q_f/4.; sa=k@self.q_a/4.; af=torch.softmax(sf,1).masked_fill(~valid,0.); aa=torch.softmax(sa,1).masked_fill(~valid,0.); vt=self.w_v(e).masked_fill(~valid.unsqueeze(-1),0.); gate=torch.sigmoid(e@self.w_c+self.b_c).masked_fill(~valid,0.); vtg=vt*gate.unsqueeze(-1); rf=(af.unsqueeze(-1)*vtg).sum(1); ra=(aa.unsqueeze(-1)*vtg).sum(1); pos=torch.arange(1,t+1).unsqueeze(0); me=(e+self.position(pos)).masked_fill(~valid.unsqueeze(-1),0.); ml=torch.cat((z,me[:,:-1]),1); mh=self.local_norm(F.silu(self.local_binding(torch.cat((ml,me),-1)))).masked_fill(~valid.unsqueeze(-1),0.); mode=self.w_g(mh.sum(1)/lengths.clamp_min(1).unsqueeze(-1)); out=(rf,ra,mode); return (*out,af,aa,k,sf,sa,vt,gate,vtg) if return_details else out
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

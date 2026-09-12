import torch
import torch.nn.functional as F
from torch import Tensor, nn

class NB5Vocab:
    def __init__(self, manifest):
        self.ids=manifest["token_ids"]
        self.token_names=list(self.ids)
        self.external_to_internal={str(self.ids[n]):i for i,n in enumerate(self.token_names)}
        self.noop_index=len(self.token_names)
    def encode(self, external): return self.external_to_internal[str(external)]
    def encode_name(self, name): return self.noop_index if name == "NOOP" else self.encode(self.ids[name])

class NB5CoreEncoder(nn.Module):
    def __init__(self, manifest, seed):
        super().__init__(); torch.manual_seed(seed); self.vocab=NB5Vocab(manifest); self.embedding=nn.Embedding(len(self.vocab.token_names)+1,16); self.position=nn.Embedding(6,16); self.local_binding=nn.Linear(32,16); self.local_norm=nn.LayerNorm(16); self.q_f=nn.Parameter(torch.randn(16)*.02); self.q_a=nn.Parameter(torch.randn(16)*.02); self.w_v=nn.Linear(16,32); self.w_g=nn.Linear(16,32)
    def forward(self, token_ids: Tensor, lengths: Tensor, *, return_details=False):
        b,t=token_ids.shape; e=self.embedding(token_ids); valid=torch.arange(t).unsqueeze(0)<lengths.unsqueeze(1); z=torch.zeros((b,1,16)); left=torch.cat((z,e[:,:-1]),1); key=e.masked_fill(~valid.unsqueeze(-1),0.); k=self.local_norm(F.silu(self.local_binding(torch.cat((left,key),-1)))).masked_fill(~valid.unsqueeze(-1),0.); sf=k@self.q_f/4.; sa=k@self.q_a/4.; af=torch.softmax(sf,1).masked_fill(~valid,0.); aa=torch.softmax(sa,1).masked_fill(~valid,0.); vt=self.w_v(e).masked_fill(~valid.unsqueeze(-1),0.); rf=(af.unsqueeze(-1)*vt).sum(1); ra=(aa.unsqueeze(-1)*vt).sum(1); pos=torch.arange(1,t+1).unsqueeze(0); me=(e+self.position(pos)).masked_fill(~valid.unsqueeze(-1),0.); ml=torch.cat((z,me[:,:-1]),1); mh=self.local_norm(F.silu(self.local_binding(torch.cat((ml,me),-1)))).masked_fill(~valid.unsqueeze(-1),0.); mode=self.w_g(mh.sum(1)/lengths.clamp_min(1).unsqueeze(-1)); out=(rf,ra,mode); return (*out,af,aa,k,sf,sa,vt) if return_details else out
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

class NB5GateOnlyEncoder(NB5CoreEncoder):
    def __init__(self, state):
        super().__init__({'token_ids': {f'ARG_{i:02d}': i for i in range(32)} | {'LINK':32,'OP_X':33,'OP_Y':34}}, 0); self.load_state_dict(state, strict=True); self.w_c=nn.Parameter(torch.zeros(16)); self.b_c=nn.Parameter(torch.zeros(()))
        for n,p in self.named_parameters():
            if n not in ('w_c','b_c'): p.requires_grad_(False)
    def forward(self, token_ids, lengths, *, return_details=False):
        out=super().forward(token_ids,lengths,return_details=True); rf,ra,mode,af,aa,k,sf,sa,vt=out; e=self.embedding(token_ids); valid=torch.arange(token_ids.shape[1]).unsqueeze(0)<lengths.unsqueeze(1); gate=torch.sigmoid(e@self.w_c+self.b_c).masked_fill(~valid,0.); rf=(af.unsqueeze(-1)*vt*gate.unsqueeze(-1)).sum(1); ra=(aa.unsqueeze(-1)*vt*gate.unsqueeze(-1)).sum(1); result=(rf,ra,mode,af,aa,k,sf,sa,vt,gate,vt*gate.unsqueeze(-1)); return result if return_details else result[:3]

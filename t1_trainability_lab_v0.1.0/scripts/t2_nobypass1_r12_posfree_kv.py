"""T2-NOBYPASS-1-R1.2-POSFREE-KV encoder."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from t2_nobypass1_r0 import R0Encoder
class R12POSFreeKVEncoder(R0Encoder):
    address_key_uses_absolute_position=False
    value_path_source='lexical_embedding_only'
    contextual_hidden_not_consumed_by_Wv=True
    def forward(self, token_ids: Tensor, lengths: Tensor, *, return_details: bool=False):
         b,t=token_ids.shape; lexical=self.embedding(token_ids); valid=torch.arange(t).unsqueeze(0)<lengths.unsqueeze(1); z=torch.zeros((b,1,16)); left=torch.cat((z,lexical[:,:-1]),1); right=torch.cat((lexical[:,1:],z),1); key_tokens=lexical.masked_fill(~valid.unsqueeze(-1),0.); k=self.local_norm(F.silu(self.local_binding(torch.cat((left,key_tokens,right),-1)))); k=k.masked_fill(~valid.unsqueeze(-1),0.); sf=(k@self.q_f)/4.; sa=(k@self.q_a)/4.; af=torch.softmax(sf,1).masked_fill(~valid,0.); aa=torch.softmax(sa,1).masked_fill(~valid,0.); vt=self.w_v(lexical).masked_fill(~valid.unsqueeze(-1),0.); rf=(af.unsqueeze(-1)*vt).sum(1); ra=(aa.unsqueeze(-1)*vt).sum(1); pos=torch.arange(1,t+1).unsqueeze(0); mode_tokens=(lexical+self.position(pos)).masked_fill(~valid.unsqueeze(-1),0.); ml=torch.cat((z,mode_tokens[:,:-1]),1); mr=torch.cat((mode_tokens[:,1:],z),1); mode_h=self.local_norm(F.silu(self.local_binding(torch.cat((ml,mode_tokens,mr),-1)))); mode_h=mode_h.masked_fill(~valid.unsqueeze(-1),0.); pool=mode_h.sum(1)/lengths.clamp_min(1).unsqueeze(-1); output=(rf,ra,self.w_g(pool)); return (*output,af,aa,k,sf,sa,vt) if return_details else output

"""T2-NOBYPASS-1-R1.1-KV lexical-only value-path encoder."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from t2_nobypass1_r0 import R0Encoder

class R11KVEncoder(R0Encoder):
    value_path_source = "lexical_embedding_only"
    contextual_hidden_not_consumed_by_Wv = True
    def forward(self, token_ids: Tensor, lengths: Tensor, *, return_details: bool = False):
        b,t=token_ids.shape; lexical=self.embedding(token_ids); pos=torch.arange(1,t+1).unsqueeze(0); tokens=lexical+self.position(pos); valid=torch.arange(t).unsqueeze(0)<lengths.unsqueeze(1); tokens=tokens.masked_fill(~valid.unsqueeze(-1),0.); z=torch.zeros((b,1,16)); left=torch.cat((z,tokens[:,:-1]),1); right=torch.cat((tokens[:,1:],z),1); h=self.local_norm(F.silu(self.local_binding(torch.cat((left,tokens,right),-1)))); h=h.masked_fill(~valid.unsqueeze(-1),0.); score_f=(h@self.q_f)/4.; score_a=(h@self.q_a)/4.; af=torch.softmax(score_f,dim=1).masked_fill(~valid,0.); aa=torch.softmax(score_a,dim=1).masked_fill(~valid,0.); value_tokens=self.w_v(lexical); value_tokens=value_tokens.masked_fill(~valid.unsqueeze(-1),0.); uf=(af.unsqueeze(-1)*value_tokens).sum(1); ua=(aa.unsqueeze(-1)*value_tokens).sum(1); pool=h.sum(1)/lengths.clamp_min(1).unsqueeze(-1); output=(uf,ua,self.w_g(pool)); return (*output,af,aa,h,score_f,score_a,value_tokens) if return_details else output

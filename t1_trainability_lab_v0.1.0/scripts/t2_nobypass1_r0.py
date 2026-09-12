"""T2-NOBYPASS-1-R0 new local text encoder and clean runtime."""
from __future__ import annotations
import hashlib
from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from evaluate_u0c_c1_e_r_alu import DIMENSION, KEY_BASE, VALUE_BASE
from evaluate_u0c_ctrl4_preflight import COPY_E_R, DECREASE, EMIT, INCREASE, MAX_DECISIONS, READ_E, READ_P, dispatch_unified_action, pair_value, state_hash
from evaluate_u0c_ctrl7_preflight import GOALS, oracle_action, primitive_check, target_value
from evaluate_u0c_ctrl2_o_canon import canonical_value_view
from train_u0c_ctrl1 import SLOT_P, SLOT_R, SLOT_COUNT, materialize_graph_batch
from t1_trainability.unified import ROW_PAIR, ROW_REL
from t2_i1_instruction import parse_instruction_i1

class R0Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.embedding=nn.Embedding(39,16); self.position=nn.Embedding(6,16); self.local_binding=nn.Linear(48,16); self.local_norm=nn.LayerNorm(16); self.q_f=nn.Parameter(torch.randn(16)*.02); self.q_a=nn.Parameter(torch.randn(16)*.02); self.w_v=nn.Linear(16,32); self.w_g=nn.Linear(16,32)
    def forward(self, token_ids: Tensor, lengths: Tensor) -> tuple[Tensor,Tensor,Tensor]:
        b,t=token_ids.shape; pos=torch.arange(1,t+1).unsqueeze(0); tokens=self.embedding(token_ids)+self.position(pos); valid=torch.arange(t).unsqueeze(0)<lengths.unsqueeze(1); tokens=tokens.masked_fill(~valid.unsqueeze(-1),0.); z=torch.zeros((b,1,16)); left=torch.cat((z,tokens[:,:-1]),1); right=torch.cat((tokens[:,1:],z),1); h=self.local_norm(F.silu(self.local_binding(torch.cat((left,tokens,right),-1)))); h=h.masked_fill(~valid.unsqueeze(-1),0.); af=torch.softmax((h@self.q_f)/4.,dim=1).masked_fill(~valid,0.); aa=torch.softmax((h@self.q_a)/4.,dim=1).masked_fill(~valid,0.); uf=(af.unsqueeze(-1)*h).sum(1); ua=(aa.unsqueeze(-1)*h).sum(1); pool=h.sum(1)/lengths.clamp_min(1).unsqueeze(-1); return self.w_v(uf),self.w_v(ua),self.w_g(pool)
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

def condition_features(model,ctrl1,scorer,state,goal,r_f,r_a,v_e,v_r):
    nav=F.softmax(ctrl1(state[:,SLOT_P],goal),-1)
    if v_r:
        q,_=canonical_value_view(model,state[:,SLOT_R]); z=torch.zeros_like(r_f); cf=F.softmax(scorer.logits(q,torch.cat((r_f,z),-1)),-1); ca=F.softmax(scorer.logits(q,torch.cat((r_a,z),-1)),-1)
    else: cf=torch.zeros((state.shape[0],3)); ca=torch.zeros((state.shape[0],3))
    return torch.cat((nav,cf,ca,torch.tensor([[float(v_e),float(v_r)]])), -1)

@torch.no_grad()
def run_r0(model,ctrl1,scorer,supervisor,manifest,episode,encoder,instruction,canonicalized=False):
    parsed=parse_instruction_i1(instruction); ids=torch.tensor([parsed.token_ids]); lengths=torch.tensor([len(parsed.token_ids)]); rf,ra,c=encoder(ids,lengths); rf,ra,c=rf[0],ra[0],c[0]
    if canonicalized:
        df=int(canonical_value_view(model,torch.cat((rf,torch.zeros_like(rf))).unsqueeze(0))[1].item()); da=int(canonical_value_view(model,torch.cat((ra,torch.zeros_like(ra))).unsqueeze(0))[1].item()); rf=model.token_embedding(torch.tensor([VALUE_BASE+df]))[0,:32]; ra=model.token_embedding(torch.tensor([VALUE_BASE+da]))[0,:32]
    graph=manifest["graphs"][episode["graph"]]; mk,mv,mt,rm=materialize_graph_batch(model,[graph]); state=torch.zeros((1,SLOT_COUNT,DIMENSION)); state[:,SLOT_P]=model.token_embedding(torch.tensor([episode["start_key"]+KEY_BASE])); presence=torch.ones((1,SLOT_COUNT),dtype=torch.bool); goal=model.token_embedding(torch.tensor([episode["goal_key"]+KEY_BASE])); x0=pair_value(manifest,episode); pointer,value=episode["start_key"],x0; re=cp=ve=vr=False; target=target_value(x0,parsed.lower,parsed.forbidden,parsed.constraints); events=[]; emitted=False; bad=None
    for decision in range(MAX_DECISIONS):
        expected=oracle_action(pointer,episode["goal_key"],read_e_done=re,copied=cp,value=value,lower=parsed.lower,forbidden=parsed.forbidden,constraints=parsed.constraints); feat=condition_features(model,ctrl1,scorer,state,goal,rf.unsqueeze(0),ra.unsqueeze(0),ve,vr); action=int(supervisor(feat,c).argmax(-1).item()); before=state.clone(); row=next((i for i,x in enumerate(graph["rows"]) if (expected==READ_P and x["kind"]==ROW_REL and x["key"]==pointer) or (expected==READ_E and x["kind"]==ROW_PAIR and x["key"]==episode["goal_key"])), -1); state,op=dispatch_unified_action(model,mk,mv,mt,rm,state,presence,action,v_e=ve,v_r=vr); ok=primitive_check(model,before,state,action,{**op,"available_e":ve,"available_r":vr},row,graph,target); events.append({"action":action,"action_name":("READ_P","READ_E","COPY_E_R","INCREASE","DECREASE","EMIT")[action],"expected_action":expected,"state_hash":state_hash(state,SLOT_R),"valid":ok}); bad=decision if not ok and bad is None else bad; rej=op.get("rejected",False); ve=True if action==READ_E and not rej else False if action==READ_P and not rej else ve; re=True if action==READ_E and not rej else re; cp=True if action==COPY_E_R and not rej else cp; vr=True if action in (COPY_E_R,INCREASE,DECREASE) and not rej else vr; value=value+1 if action==INCREASE and not rej else value-1 if action==DECREASE and not rej else value; pointer=graph["rows"][op["selected_row"]]["value"] if action==READ_P and not rej and op.get("selected_row",-1)>=0 else pointer; emitted=action==EMIT and not rej
        if emitted: break
    final=int(canonical_value_view(model,state[:,SLOT_R])[1].item()) if vr else None; return {"actions":[e["action_name"] for e in events],"action_ids":[e["action"] for e in events],"state_hashes":[e["state_hash"] for e in events],"events":events,"parsed":{"tokens":list(parsed.tokens),"token_ids":list(parsed.token_ids),"constraints":list(parsed.constraints),"lower":parsed.lower,"forbidden":parsed.forbidden},"target":target,"final_value":final,"success":emitted and bad is None and final==target}

"""Train T2-NOBYPASS-1-R1.2-POSFREE-KV."""
from __future__ import annotations
import argparse,hashlib,json,random
from pathlib import Path
import torch
import torch.nn.functional as F
from train_t2_i2_r2 import load_source
from train_t2_i0_baseline_b import LatentConditionedSupervisor,CTRL7_CHECKPOINT
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r12_posfree_kv import R12POSFreeKVEncoder
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'campaign'/'t2_nobypass1_r12_posfree_kv_seed6704'
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def text_for(c,l,f):
 c=tuple(int(x) for x in c); return 'KEEP' if c==(0,0) else f'AT_LEAST VALUE_{l}' if c==(1,0) else f'AVOID VALUE_{f}'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=6704); p.add_argument('--output-root',type=Path,default=OUT); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed); obs,labels=load_source(); model=load_executor(); model.eval(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); enc=R12POSFreeKVEncoder(); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=0.); gen=torch.Generator().manual_seed(a.seed+1); codebook=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); actions=(0,1,2,3,5); goals=((0,0),(1,0),(0,1)); counts={(0,0):{0:8,1:8,2:8,5:8},(1,0):{0:8,1:8,2:8,3:8,5:16},(0,1):{0:8,1:8,2:8,3:8,5:16}}; buckets={(c,x):torch.where((labels['constraints'][:,0]==c[0])&(labels['constraints'][:,1]==c[1])&(labels['action']==x))[0] for c in goals for x in actions}; last_b=last_r=torch.tensor(0.); entf=[]; enta=[]
 for step in range(1,5001):
  selected=[]
  for c in goals:
   for action,count in counts[c].items(): selected.append(buckets[(c,action)][torch.randint(len(buckets[(c,action)]),(count,),generator=gen)])
  ix=torch.cat(selected); ti,le=tensorize([text_for(labels['constraints'][i].tolist(),int(labels['lower'][i]),int(labels['forbidden'][i])) for i in ix.tolist()]); rf,ra,mode,af,aa,_,_,_,_=enc(ti,le,return_details=True); last_b=F.cross_entropy(sup(obs['features'][ix],mode),labels['action'][ix]); floor=labels['constraints'][ix,0]==1; avoid=labels['constraints'][ix,1]==1; terms=[]; qf=torch.cat((rf,torch.zeros_like(rf)),dim=-1); qa=torch.cat((ra,torch.zeros_like(ra)),dim=-1); 
  if floor.any(): terms.append(F.cross_entropy(model.register_decoder(qf[floor],codebook),labels['lower'][ix][floor]))
  if avoid.any(): terms.append(F.cross_entropy(model.register_decoder(qa[avoid],codebook),labels['forbidden'][ix][avoid]))
  last_r=torch.stack(terms).mean(); (last_b+last_r).backward(); opt.step(); opt.zero_grad(set_to_none=True); progress=(step-1)/4999; opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*progress; entf.append(float((-(af.clamp_min(1e-9)*af.clamp_min(1e-9).log()).sum(-1)).mean().detach())); enta.append(float((-(aa.clamp_min(1e-9)*aa.clamp_min(1e-9).log()).sum(-1)).mean().detach()))
 ck=a.output_root/'final.pt'; torch.save({'encoder':enc.state_dict(),'seed':a.seed,'updates':5000,'trainable_parameters':enc.parameter_count(),'writer_inherited':False,'address_key_uses_absolute_position':False,'value_path_source':'lexical_embedding_only','contextual_hidden_not_consumed_by_Wv':True,'loss':'L_behavior + 1.0*L_ref','executor_checkpoint_sha256':sha256(BASE_CHECKPOINT)},ck); result={'status':'trained','task':'T2-NOBYPASS-1-R1.2-POSFREE-KV','seed':a.seed,'updates':5000,'batch_size':128,'trainable_parameters':enc.parameter_count(),'final_behavior_loss':float(last_b.detach()),'final_ref_loss':float(last_r.detach()),'lambda_ref':1.0,'writer_inherited':False,'address_key_uses_absolute_position':False,'value_path_source':'lexical_embedding_only','contextual_hidden_not_consumed_by_Wv':True,'attention_entropy_observation':{'H_af_mean_last_batch':entf[-1],'H_aa_mean_last_batch':enta[-1]},'checkpoint_sha256':sha256(ck)}; (a.output_root/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

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
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'campaign'/'t2_nobypass1_r13_relkey_kv_seed6705'
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def text_for(c,l,f):
 c=tuple(int(x) for x in c); return 'KEEP' if c==(0,0) else f'AT_LEAST VALUE_{l}' if c==(1,0) else f'AVOID VALUE_{f}'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=6705); p.add_argument('--output-root',type=Path,default=OUT); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed); obs,labels=load_source(); model=load_executor(); model.eval(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); enc=R13RelKeyKVEncoder(); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=0.); gen=torch.Generator().manual_seed(a.seed+1); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); acts=(0,1,2,3,5); goals=((0,0),(1,0),(0,1)); counts={(0,0):{0:8,1:8,2:8,5:8},(1,0):{0:8,1:8,2:8,3:8,5:16},(0,1):{0:8,1:8,2:8,3:8,5:16}}; buckets={(c,x):torch.where((labels['constraints'][:,0]==c[0])&(labels['constraints'][:,1]==c[1])&(labels['action']==x))[0] for c in goals for x in acts}; last_b=last_r=torch.tensor(0.)
 for step in range(1,5001):
  ix=torch.cat([buckets[(c,x)][torch.randint(len(buckets[(c,x)]),(n,),generator=gen)] for c in goals for x,n in counts[c].items()]); ti,le=tensorize([text_for(labels['constraints'][i].tolist(),int(labels['lower'][i]),int(labels['forbidden'][i])) for i in ix.tolist()]); rf,ra,mode,*_=enc(ti,le,return_details=True); last_b=F.cross_entropy(sup(obs['features'][ix],mode),labels['action'][ix]); floor=labels['constraints'][ix,0]==1; avoid=labels['constraints'][ix,1]==1; terms=[]; qf=torch.cat((rf,torch.zeros_like(rf)),dim=-1); qa=torch.cat((ra,torch.zeros_like(ra)),dim=-1); 
  if floor.any(): terms.append(F.cross_entropy(model.register_decoder(qf[floor],cb),labels['lower'][ix][floor]))
  if avoid.any(): terms.append(F.cross_entropy(model.register_decoder(qa[avoid],cb),labels['forbidden'][ix][avoid]))
  last_r=torch.stack(terms).mean(); (last_b+last_r).backward(); opt.step(); opt.zero_grad(set_to_none=True); prog=(step-1)/4999; opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*prog
 ck=a.output_root/'final.pt'; torch.save({'encoder':enc.state_dict(),'seed':a.seed,'updates':5000,'trainable_parameters':enc.parameter_count(),'writer_inherited':False,'address_key_uses_absolute_position':False,'address_key_uses_right_neighbor':False,'address_key_inputs':['previous_token','current_token'],'value_path_source':'lexical_embedding_only','contextual_hidden_not_consumed_by_Wv':True},ck); result={'status':'trained','task':'T2-NOBYPASS-1-R1.3-RELKEY-KV','seed':a.seed,'updates':5000,'batch_size':128,'trainable_parameters':enc.parameter_count(),'final_behavior_loss':float(last_b.detach()),'final_ref_loss':float(last_r.detach()),'lambda_ref':1.0,'writer_inherited':False,'address_key_uses_absolute_position':False,'address_key_uses_right_neighbor':False,'address_key_inputs':['previous_token','current_token'],'value_path_source':'lexical_embedding_only','contextual_hidden_not_consumed_by_Wv':True,'checkpoint_sha256':sha256(ck),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT)}; (a.output_root/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

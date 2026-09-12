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
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'campaign'/'t2_nobypass1_r14_s_gateonly_seed6706'; BASE_R13=ROOT/'campaign'/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; BASE_SHA='77ce2051aee6d776b2054b0ec991cb35e20f83287694400be9629e7ffd5b7737'
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def text_for(c,l,f): return 'KEEP' if tuple(int(x) for x in c)==(0,0) else f'AT_LEAST VALUE_{l}' if int(c[0]) else f'AVOID VALUE_{f}'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=6706); p.add_argument('--output-root',type=Path,default=OUT); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed); obs,labels=load_source(); model=load_executor(); model.eval(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); base=torch.load(BASE_R13,weights_only=False); base_state=base['encoder']; enc=GateOnlyEncoder(base_state); frozen_before={n:p.detach().clone() for n,p in enc.named_parameters() if n not in ('w_c','b_c')}; ids={id(p) for n,p in enc.named_parameters() if n in ('w_c','b_c')}; opt=torch.optim.AdamW([enc.w_c,enc.b_c],lr=1e-3,weight_decay=0.); assert {id(p) for g in opt.param_groups for p in g['params']}==ids; gen=torch.Generator().manual_seed(a.seed+1); acts=(0,1,2,3,5); goals=((0,0),(1,0),(0,1)); counts={(0,0):{0:8,1:8,2:8,5:8},(1,0):{0:8,1:8,2:8,3:8,5:16},(0,1):{0:8,1:8,2:8,3:8,5:16}}; buckets={(c,x):torch.where((labels['constraints'][:,0]==c[0])&(labels['constraints'][:,1]==c[1])&(labels['action']==x))[0] for c in goals for x in acts}; lb=lr=torch.tensor(0.)
 for step in range(5000):
  ix=torch.cat([buckets[(c,x)][torch.randint(len(buckets[(c,x)]),(n,),generator=gen)] for c in goals for x,n in counts[c].items()]); ti,le=tensorize([text_for(labels['constraints'][i].tolist(),int(labels['lower'][i]),int(labels['forbidden'][i])) for i in ix.tolist()]); rf,ra,mode,*_=enc(ti,le,return_details=True); lb=F.cross_entropy(sup(obs['features'][ix],mode),labels['action'][ix]); floor=labels['constraints'][ix,0]==1; avoid=labels['constraints'][ix,1]==1; qf=torch.cat((rf,torch.zeros_like(rf)),-1); qa=torch.cat((ra,torch.zeros_like(ra)),-1); terms=[]
  if floor.any(): terms.append(F.cross_entropy(model.register_decoder(qf[floor],model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT))),labels['lower'][ix][floor]))
  if avoid.any(): terms.append(F.cross_entropy(model.register_decoder(qa[avoid],model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT))),labels['forbidden'][ix][avoid]))
  lr=torch.stack(terms).mean(); (lb+lr).backward(); opt.step(); opt.zero_grad(set_to_none=True); opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*(step/4999)
 assert all(torch.equal(frozen_before[n],p.detach()) for n,p in enc.named_parameters() if n not in ('w_c','b_c')); ck=a.output_root/'gate.pt'; torch.save({'w_c':enc.w_c.detach(),'b_c':enc.b_c.detach(),'base_checkpoint':str(BASE_CHECKPOINT),'base_checkpoint_sha256':BASE_SHA,'seed':a.seed,'updates':5000},ck); result={'status':'trained','task':'T2-NOBYPASS-1-R1.4-S-GATEONLY','seed':a.seed,'updates':5000,'trainable_parameters':17,'frozen_parameter_count':sum(x.numel() for x in frozen_before.values()),'optimizer_parameter_ids_only_gate':True,'final_behavior_loss':float(lb),'final_ref_loss':float(lr),'checkpoint_base_sha256':BASE_SHA,'gate_artifact_sha256':sha256(ck)}; (a.output_root/'train.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

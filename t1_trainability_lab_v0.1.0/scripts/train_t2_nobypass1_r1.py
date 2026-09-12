"""Train T2-NOBYPASS-1-R1 with explicit frozen register-reference supervision."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from train_t2_i2_r2 import load_source
from train_t2_i0_baseline_b import LatentConditionedSupervisor, CTRL7_CHECKPOINT
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE, VALUE_COUNT
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r0 import R0Encoder

ROOT=Path(__file__).resolve().parents[1]; SOURCE_ROOT=ROOT/'campaign'/'u0c_ctrl7_pilot_seed4701'; DEFAULT_OUT=ROOT/'campaign'/'t2_nobypass1_r1_seed6702'
def sha256(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def text_for(c,lower,forbidden):
    c=tuple(int(x) for x in c)
    if c==(0,0): return 'KEEP'
    if c==(1,0): return f'AT_LEAST VALUE_{lower}'
    if c==(0,1): return f'AVOID VALUE_{forbidden}'
    raise ValueError(c)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=6702); p.add_argument('--output-root',type=Path,default=DEFAULT_OUT); a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed)
    obs,labels=load_source(); model=load_executor(); model.eval(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); enc=R0Encoder(); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=0.0); gen=torch.Generator().manual_seed(a.seed+1); ids=torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT); codebook=model.token_embedding(ids).detach(); actions=(0,1,2,3,5); goals=((0,0),(1,0),(0,1)); counts={(0,0):{0:8,1:8,2:8,5:8},(1,0):{0:8,1:8,2:8,3:8,5:16},(0,1):{0:8,1:8,2:8,3:8,5:16}}
    buckets={(c,action):torch.where((labels['constraints'][:,0]==c[0])&(labels['constraints'][:,1]==c[1])&(labels['action']==action))[0] for c in goals for action in actions}; last_b=last_r=torch.tensor(0.); ent_f=[]; ent_a=[]
    for step in range(1,5001):
        selected=[]
        for c in goals:
            for action,count in counts[c].items():
                bucket=buckets[(c,action)]; selected.append(bucket[torch.randint(len(bucket),(count,),generator=gen)])
        ix=torch.cat(selected)
        texts=[text_for(labels['constraints'][i].tolist(),int(labels['lower'][i]),int(labels['forbidden'][i])) for i in ix.tolist()]
        token_ids,lengths=tensorize(texts)
        rf,ra,mode,af,aa,_,_,_=enc(token_ids,lengths,return_details=True)
        last_b=F.cross_entropy(sup(obs['features'][ix],mode),labels['action'][ix])
        role_terms=[]; floor=(labels['constraints'][ix,0]==1); avoid=(labels['constraints'][ix,1]==1)
        qf=torch.cat((rf,torch.zeros_like(rf)),dim=-1); qa=torch.cat((ra,torch.zeros_like(ra)),dim=-1)
        if floor.any(): role_terms.append(F.cross_entropy(model.register_decoder(qf[floor],codebook),labels['lower'][ix][floor]))
        if avoid.any(): role_terms.append(F.cross_entropy(model.register_decoder(qa[avoid],codebook),labels['forbidden'][ix][avoid]))
        last_r=torch.stack(role_terms).mean(); loss=last_b+last_r; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); progress=(step-1)/4999; opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*progress; ent_f.append(float((-(af.clamp_min(1e-9)*af.clamp_min(1e-9).log()).sum(-1)).mean())); ent_a.append(float((-(aa.clamp_min(1e-9)*aa.clamp_min(1e-9).log()).sum(-1)).mean()))
    ck=a.output_root/'final.pt'; torch.save({'encoder':enc.state_dict(),'seed':a.seed,'updates':5000,'trainable_parameters':enc.parameter_count(),'writer_inherited':False,'loss':'L_behavior + 1.0*L_ref','lambda_ref':1.0,'executor_checkpoint':str(__import__('ctrl2_common').BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(__import__('ctrl2_common').BASE_CHECKPOINT)},ck); result={'status':'trained','task':'T2-NOBYPASS-1-R1','seed':a.seed,'updates':5000,'batch_size':128,'trainable_parameters':enc.parameter_count(),'final_behavior_loss':float(last_b),'final_ref_loss':float(last_r),'lambda_ref':1.0,'optimizer':'AdamW weight_decay=0','learning_rate':'linear 1e-3 -> 1e-5','writer_inherited':False,'attention_entropy_observation':{'H_af_mean_last_batch':ent_f[-1],'H_aa_mean_last_batch':ent_a[-1]},'checkpoint':str(ck),'checkpoint_sha256':sha256(ck)}; (a.output_root/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

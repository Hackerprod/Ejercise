"""Train new R0 text encoder against frozen CTRL-7 action labels."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from t2_i3_common import MANIFEST_PATH, encode_writer
from train_t2_i2_r2 import load_source
from train_t2_i0_baseline_b import LatentConditionedSupervisor
from train_t2_i0_b_r2 import CTRL7_CHECKPOINT
from train_u0c_ctrl2_o import sha256
from t2_nobypass1_r0 import R0Encoder
from t2_i2_r3_semantic_writer import tensorize

ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=6701); a=p.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); out=CAMPAIGN/f't2_nobypass1_r0_seed{a.seed}'; out.mkdir(parents=True,exist_ok=True)
 obs,labels=load_source(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); enc=R0Encoder(); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=0.); sched=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1.,end_factor=.01,total_iters=5000)
 gen=torch.Generator().manual_seed(a.seed+1); last=0.
 for _ in range(5000):
  idx=torch.randint(len(labels['action']),(128,),generator=gen); texts=[]
  for i in idx.tolist():
   c=tuple(int(x) for x in labels['constraints'][i]); l=int(labels['lower'][i]); f=int(labels['forbidden'][i]); texts.append('KEEP' if c==(0,0) else f'AT_LEAST VALUE_{l}' if c==(1,0) else f'AVOID VALUE_{f}')
  ids,lens=tensorize(texts); rf,ra,mode=enc(ids,lens); last=F.cross_entropy(sup(obs['features'][idx],mode),labels['action'][idx]); opt.zero_grad(); last.backward(); opt.step(); sched.step()
 ck=out/'final.pt'; torch.save({'encoder':enc.state_dict(),'seed':a.seed,'updates':5000,'trainable_parameters':enc.parameter_count(),'writer_checkpoint':'not_loaded','ctrl_checkpoint_sha256':sha256(CTRL7_CHECKPOINT)},ck); r={'status':'trained','task':'T2-NOBYPASS-1-R0','seed':a.seed,'updates':5000,'trainable_parameters':enc.parameter_count(),'final_action_loss':float(last.detach()),'checkpoint':sha256(ck),'writer_inherited':False}; (out/'results.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
if __name__=='__main__': main()

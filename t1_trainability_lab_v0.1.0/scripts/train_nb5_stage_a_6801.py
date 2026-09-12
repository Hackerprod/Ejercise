import hashlib,json,random
from pathlib import Path
import torch
import torch.nn.functional as F
from nb5_fresh import NB5CoreEncoder
from train_t2_i2_r2 import load_source
from train_t2_i0_baseline_b import LatentConditionedSupervisor,CTRL7_CHECKPOINT
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1]; M=R/'campaign/nb5_manifests/manifest_6801.json'; O=R/'campaign/nb5_stage_a_6801'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads(M.read_text()); assert sha(M)=='de2b47cbd183c871a539ebc019232a8e5130e2a8ce582088eadc1b7c1d1b0244'; rows=m['train']; assert len(rows)==96 and not any(x['kind']=='joint' for x in rows)
 torch.manual_seed(6801); random.seed(6801); enc=NB5CoreEncoder(m,6801); obs,lab=load_source(); sup=LatentConditionedSupervisor(CTRL7_CHECKPOINT); sup.eval(); exe=load_executor(); exe.eval(); cb=exe.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); perm=m['permutation']; inv={v:i for i,v in enumerate(perm)}; ids=[]; lens=[]; source=[]; floor=[]; avoid=[]
 for row in rows:
  c=row['constraints']; val=int(row['value']); op=row['operator'];
  if row['kind']=='noop': toks=[enc.vocab.noop_index]; candidates=torch.where((lab['constraints'][:,0]==0)&(lab['constraints'][:,1]==0))[0]
  else:
   opname=m['operator_floor'] if c=='floor' else m['operator_avoid']; toks=[enc.vocab.encode_name(opname),enc.vocab.encode_name(f'ARG_{inv[val]:02d}')]; candidates=torch.where((lab['constraints'][:,0]==(1 if c=='floor' else 0))&(lab['constraints'][:,1]==(1 if c=='avoid' else 0))&((lab['lower'] if c=='floor' else lab['forbidden'])==val))[0]
  assert len(candidates)>0; ids.append(toks);lens.append(len(toks));source.append(int(candidates[0])); floor.append(c=='floor');avoid.append(c=='avoid')
 t=torch.tensor([x+[0]*(2-len(x)) for x in ids]); le=torch.tensor(lens); ix=torch.tensor(source); floor=torch.tensor(floor); avoid=torch.tensor(avoid); opt=torch.optim.AdamW(enc.parameters(),lr=1e-3,weight_decay=0.); last_b=last_r=torch.tensor(0.)
 for step in range(1,5001):
  rf,ra,mode,*_=enc(t,le,return_details=True); last_b=F.cross_entropy(sup(obs['features'][ix],mode),lab['action'][ix]); terms=[]
  if floor.any(): terms.append(F.cross_entropy(exe.register_decoder(torch.cat((rf[floor],torch.zeros_like(rf[floor])),-1),cb),lab['lower'][ix][floor]))
  if avoid.any(): terms.append(F.cross_entropy(exe.register_decoder(torch.cat((ra[avoid],torch.zeros_like(ra[avoid])),-1),cb),lab['forbidden'][ix][avoid]))
  last_r=torch.stack(terms).mean(); (last_b+last_r).backward(); opt.step(); opt.zero_grad(set_to_none=True); opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*(step-1)/4999
 O.mkdir(parents=True,exist_ok=True); ck=O/'stage_a.pt'; torch.save({'encoder':enc.state_dict(),'seed':6801,'updates':5000,'manifest_sha256':sha(M),'joint_train_examples':0},ck)
 with torch.no_grad():
  rf,ra,mode=enc(t,le); lf=exe.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb).argmax(-1); la=exe.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb).argmax(-1); action=sup(obs['features'][ix],mode).argmax(-1)
 result={'status':'trained','seed':6801,'updates':5000,'manifest_sha256':sha(M),'checkpoint_sha256':sha(ck),'joint_train_examples':0,'forward_ground_truth_arguments':False,'teacher_in_inference':False,'train_rows':96,'atomic_floor_exact':int((lf[floor]==lab['lower'][ix][floor]).sum()),'atomic_avoid_exact':int((la[avoid]==lab['forbidden'][ix][avoid]).sum()),'noop_action_exact':int((action[~floor&~avoid]==lab['action'][ix][~floor&~avoid]).sum()),'noop_rows':int((~floor&~avoid).sum()),'final_behavior_loss':float(last_b),'final_ref_loss':float(last_r)}; (O/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

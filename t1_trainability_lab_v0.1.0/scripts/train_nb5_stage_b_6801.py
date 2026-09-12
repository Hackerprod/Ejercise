import hashlib,json,os
from pathlib import Path
import torch
import torch.nn.functional as F
from nb5_fresh import NB5GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
SEED=int(os.environ.get('NB5_SEED','6801')); R=Path(__file__).resolve().parents[1]; M=R/'campaign/nb5_manifests'/f'manifest_{SEED}_v2.json'; A=R/'campaign'/f'nb5_stage_a_{SEED}'/'stage_a.pt'; O=R/'campaign'/f'nb5_stage_b_{SEED}'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads(M.read_text()); payload=torch.load(A,map_location='cpu',weights_only=False); enc=NB5GateOnlyEncoder(payload['encoder']); frozen={n:p.detach().clone() for n,p in enc.named_parameters() if n not in ('w_c','b_c')}; exe=load_executor(); exe.eval(); cb=exe.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); ids=torch.tensor(list(enc.vocab.external_to_internal.values())+[enc.vocab.noop_index]); emb=enc.embedding(ids); logits=exe.register_decoder(torch.cat((enc.w_v(emb),torch.zeros_like(enc.w_v(emb))),-1),cb); top=logits.topk(2).values; margins=top[:,0]-top[:,1]; y=torch.zeros(len(ids)); y[:32]=1.; tau=3.0545; target=torch.sigmoid(2*(margins-tau)).detach(); opt=torch.optim.AdamW([enc.w_c,enc.b_c],lr=1e-3,weight_decay=0.)
 for step in range(1,5001):
  pred=emb@enc.w_c+enc.b_c; loss=F.binary_cross_entropy_with_logits(pred,target); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*(step-1)/4999
 assert all(torch.equal(frozen[n],p.detach()) for n,p in enc.named_parameters() if n not in ('w_c','b_c')); O.mkdir(parents=True,exist_ok=True); ck=O/'gate.pt'; torch.save({'w_c':enc.w_c.detach(),'b_c':enc.b_c.detach(),'stage_a_checkpoint_sha256':sha(A),'manifest_sha256':sha(M),'tau':tau,'updates':5000,'trainable_parameters':17},ck); result={'status':'trained','seed':6801,'updates':5000,'manifest_sha256':sha(M),'stage_a_checkpoint_sha256':sha(A),'gate_sha256':sha(ck),'tau':tau,'trainable_parameters':17,'joint_train_examples':0,'forward_ground_truth_arguments':False,'teacher_in_inference':False,'final_bce':float(loss.detach()),'numeric_confidence_targets':target.tolist(),'margins':margins.tolist()}; (O/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

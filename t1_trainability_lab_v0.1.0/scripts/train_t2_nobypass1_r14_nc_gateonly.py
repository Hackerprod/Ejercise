from pathlib import Path
import hashlib,json,torch
import torch.nn.functional as F
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import TOKEN_IDS
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'campaign'/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; OUT=ROOT/'campaign'/'t2_nobypass1_r14_nc_gateonly_seed6706'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True); torch.manual_seed(6706); b=torch.load(BASE,weights_only=False)['encoder']; model=load_executor(); model.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); names=[*(f'VALUE_{i}' for i in range(32)),'AT_LEAST','AVOID','AND']; ids=torch.tensor([TOKEN_IDS[x] for x in names]); base=GateOnlyEncoder(b); base.eval();
 with torch.no_grad():
  e=base.embedding(ids); v=base.w_v(e); logits=model.register_decoder(torch.cat((v,torch.zeros_like(v)),-1),cb); m=logits.topk(2).values; margins=m[:,0]-m[:,1]; tau=(margins[:32].min()+margins[32:].max())/2; y=torch.sigmoid(2*(margins-tau)).detach()
 enc=GateOnlyEncoder(b); opt=torch.optim.AdamW([enc.w_c,enc.b_c],lr=1e-3,weight_decay=0.); assert sum(p.numel() for p in opt.param_groups[0]['params'])==17
 for i in range(5000):
  c=torch.sigmoid(enc.embedding(ids)@enc.w_c+enc.b_c); loss=F.binary_cross_entropy(c,y); loss.backward(); opt.step(); opt.zero_grad(); opt.param_groups[0]['lr']=1e-3+(1e-5-1e-3)*i/4999
 with torch.no_grad(): c=torch.sigmoid(enc.embedding(ids)@enc.w_c+enc.b_c)
 ck=OUT/'gate.pt'; torch.save({'w_c':enc.w_c.detach(),'b_c':enc.b_c.detach(),'base_checkpoint':str(BASE),'base_checkpoint_sha256':'77ce2051aee6d776b2054b0ec991cb35e20f83287694400be9629e7ffd5b7737','tau':float(tau),'margins':margins,'targets':y},ck); result={'status':'trained','task':'T2-NOBYPASS-1-R1.4-NC-GATEONLY','trainable_parameters':17,'updates':5000,'final_bce':float(loss),'base_checkpoint_sha256':'77ce2051aee6d776b2054b0ec991cb35e20f83287694400be9629e7ffd5b7737','gate_artifact_sha256':sha(ck),'tau':float(tau),'margins':margins.tolist(),'targets':y.tolist(),'gates':c.tolist()}; (OUT/'train.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(OUT/'train.json'),'sha256':sha(OUT/'train.json'),'gate_sha':sha(ck),'tau':float(tau),'bce':float(loss)},indent=2))
if __name__=='__main__': main()

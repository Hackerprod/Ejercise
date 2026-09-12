"""T2-NOBYPASS-1-R1.2-RCTX-ALG frozen AND/PAD context controls."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from t2_i1_instruction import TOKEN_IDS,parse_instruction_i1
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r12_posfree_kv import R12POSFreeKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r12_posfree_kv_seed6704'/'final.pt'; OUT=CAMPAIGN/'t2_nobypass1_r12_posfree_kv_rctx_alg'
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def enc_keys(enc,ti,le,overrides=None):
 b,t=ti.shape; e=enc.embedding(ti); valid=torch.arange(t).unsqueeze(0)<le.unsqueeze(1); z=torch.zeros((b,1,16)); left=torch.cat((z,e[:,:-1]),1); right=torch.cat((e[:,1:],z),1); key_input=torch.cat((left,e,right),-1)
 if overrides:
  for pos,value in overrides.items(): key_input[:,pos,32:48]=value
 k=enc.local_norm(torch.nn.functional.silu(enc.local_binding(key_input))); k=k.masked_fill(~valid.unsqueeze(-1),0.); return k,(k@enc.q_f/4.),(k@enc.q_a/4.)
def compare(sf,sa,lp,fp): return {'sF_L':float(sf[lp]),'sF_F':float(sf[fp]),'sA_F':float(sa[fp]),'sA_L':float(sa[lp]),'floor_pass':float(sf[lp])>float(sf[fp]),'avoid_pass':float(sa[fp])>float(sa[lp])}
def summary(xs):
 t=torch.tensor(xs,dtype=torch.float64); return {'mean':float(t.mean()),'median':float(t.median()),'min':float(t.min()),'max':float(t.max())}
@torch.no_grad()
def main():
 model=load_executor(); enc=R12POSFreeKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); and_e=enc.embedding(torch.tensor(TOKEN_IDS['AND'])); pad=torch.zeros_like(and_e); manifest=json.loads(MANIFEST_PATH.read_text()); a_normal=[]; a_reverse=[]; b_normal=[]; b_reverse=[]
 for p in manifest['calibration']:
  l=int(p['lower']); f=int(p['forbidden']);
  for order,a_store,b_store in (('normal',a_normal,b_normal),('reverse',a_reverse,b_reverse)):
   text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); base_k,base_sf,base_sa=enc_keys(enc,ti,le); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); first=lp if order=='normal' else fp; sf,sa=base_sf[0],base_sa[0]; _,cf,ca=enc_keys(enc,ti,le,{first:pad}); remove_compare=compare(cf[0],ca[0],lp,fp); remove_compare['digest']=p['digest']; remove_compare['order']=order; remove_compare['first_clause_value']=l if order=='normal' else f; remove_compare['baseline']=compare(sf,sa,lp,fp); a_store.append(remove_compare); last=fp if order=='normal' else lp; _,bf,ba=enc_keys(enc,ti,le,{last:and_e}); add_compare=compare(bf[0],ba[0],lp,fp); add_compare['digest']=p['digest']; add_compare['order']=order; add_compare['last_clause_value']=f if order=='normal' else l; add_compare['baseline']=remove_compare['baseline']; b_store.append(add_compare)
  def agg(rows,value_key): return {'floor_exact':sum(x['floor_pass'] for x in rows),'avoid_exact':sum(x['avoid_pass'] for x in rows),'floor_rate':sum(x['floor_pass'] for x in rows)/139,'avoid_rate':sum(x['avoid_pass'] for x in rows)/139,'value_failures':{str(v):sum((not x['floor_pass'] if x['order']=='normal' else not x['avoid_pass']) and x[value_key]==v for x in rows) for v in range(32) if any((not x['floor_pass'] if x['order']=='normal' else not x['avoid_pass']) and x[value_key]==v for x in rows)}}
 table=[]
 for n in range(32):
  text=f'AT_LEAST VALUE_{n}'; ti,le=tensorize([text]); _,sf_and,sa_and=enc_keys(enc,ti,le,{1:and_e}); _,sf_pad,sa_pad=enc_keys(enc,ti,le,{1:pad}); table.append({'value':n,'sF_AND':float(sf_and[0,1]),'sF_PAD':float(sf_pad[0,1]),'delta_F':float(sf_and[0,1]-sf_pad[0,1]),'sA_AND':float(sa_and[0,1]),'sA_PAD':float(sa_pad[0,1]),'delta_A':float(sa_and[0,1]-sa_pad[0,1])})
  result={'task':'T2-NOBYPASS-1-R1.2-RCTX-ALG','status':'diagnostic_complete','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'definition':{'e_PAD':'zeros_like(lexical_embedding)','e_AND':'encoder.embedding(TOKEN_IDS[AND])','score':'q·k/4','value_path_unchanged':True},'RCTX_A_remove_AND':{'normal':{'cases':a_normal,'aggregate':agg(a_normal,'first_clause_value')},'reverse':{'cases':a_reverse,'aggregate':agg(a_reverse,'first_clause_value')}},'RCTX_B_add_AND':{'normal':{'cases':b_normal,'aggregate':agg(b_normal,'last_clause_value')},'reverse':{'cases':b_reverse,'aggregate':agg(b_reverse,'last_clause_value')}},'RCTX_C_table':table,'closure_assessment':{'remove_AND_closes':False,'add_AND_reproduces':False,'classification':'pending external interpretation'}}; OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'A_normal':agg(a_normal,'first_clause_value'),'A_reverse':agg(a_reverse,'first_clause_value'),'B_normal':agg(b_normal,'last_clause_value'),'B_reverse':agg(b_reverse,'last_clause_value'),'C':table},indent=2))
if __name__=='__main__': main()

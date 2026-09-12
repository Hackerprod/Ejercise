from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from t2_i1_instruction import TOKEN_IDS,parse_instruction_i1
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; OUT=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_readout_alg'
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dec(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),-1))[0] if False else model.register_decoder(torch.cat((r,torch.zeros_like(r)),-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1]),[float(x) for x in z]
def ent(a): return float((-(a.clamp_min(1e-9)*a.clamp_min(1e-9).log()).sum()).item())
def metric(model,cb,r,expected):
 d,m,l=dec(model,cb,r); return {'decoded':d,'expected':expected,'pass':d==expected,'margin':m,'logits':l}
def case_forward(enc,text):
 ti,le=tensorize([text]); return enc(ti,le,return_details=True)
@torch.no_grad()
def main():
 model=load_executor(); enc=R13RelKeyKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); manifest=json.loads(MANIFEST_PATH.read_text())
 lexical=[]
 for n in range(32):
  x=enc.w_v(enc.embedding(torch.tensor([TOKEN_IDS[f'VALUE_{n}']]))) ; d,m,l=dec(model,cb,x); lexical.append({'value':n,'decoded':d,'pass':d==n,'margin':m,'logits':l})
 forced={'normal':[],'reverse':[]}
 for p in manifest['calibration']:
  l=int(p['lower']); f=int(p['forbidden'])
  for order in forced:
   text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); rf,ra,_,_,_,_,_,_,vt=case_forward(enc,text); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); forced[order].append({'digest':p['digest'],'floor':metric(model,cb,vt[:,lp],l),'avoid':metric(model,cb,vt[:,fp],f)})
 a_pass=sum(x['pass'] for x in lexical)==32 and all(sum(x[r]['pass'] for x in forced[o])==139 for o in forced for r in ('floor','avoid'))
 result={'task':'T2-NOBYPASS-1-R1.3-READOUT-ALG','status':'diagnostic_complete' if a_pass else 'alert_halt','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'A_lexical_value_forced_target':{'lexical':lexical,'forced':forced,'lexical_exact':sum(x['pass'] for x in lexical),'forced_exact':{o:{r:sum(x[r]['pass'] for x in forced[o]) for r in ('floor','avoid')} for o in forced}}}
 if a_pass:
  all_cases={}
  for order in ('normal','reverse'):
   rows=[]
   for p in manifest['calibration']:
    l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); rf,ra,_,af,aa,h,sf,sa,vt=case_forward(enc,text); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); valid=list(range(5)); roles={'floor':(rf,sf[0],af[0],lp,l),'avoid':(ra,sa[0],aa[0],fp,f)}; row={'digest':p['digest'],'order':order}
    for role,(r,s,a,pos,expected) in roles.items():
     other=fp if role=='floor' else lp; rank=1+sum(float(s[i])>float(s[pos]) for i in valid); gap=float(s[pos]-max(s[i] for i in valid if i!=pos)); winner=max((i for i in valid if i!=pos),key=lambda i:float(s[i])); types={0:'AT_LEAST',1:'VALUE',2:'AND',3:'AVOID',4:'VALUE'}; learned=metric(model,cb,r,expected); mass={'m_T':float(a[pos]),'m_C':float(a[other]),'m_S':float(sum(a[i] for i in valid if i not in (pos,other)))}; zstruct=vt[:,[0,2,3]].mul(a[[0,2,3]].view(1,-1,1)).sum(1); zdec=metric(model,cb,zstruct,expected); zero=r-(vt[:,0]*a[0]+vt[:,2]*a[2]+vt[:,3]*a[3]); zero_d=metric(model,cb,zero,expected); norm_d=metric(model,cb,zero/(mass['m_T']+mass['m_C']),expected); gamma=[]
     for g in (1,2,4,8,16):
      w=torch.softmax(s*g,0); gamma.append({'gamma':g,**metric(model,cb,(w.unsqueeze(0)@vt).squeeze(0),expected)})
     row[role]={'full_rank':{'rank':rank,'gap_all':gap,'winner_position':winner,'winner_type':types[winner]},'mass_partition':mass,'learned':learned,'zero_struct':zero_d,'zero_struct_rescaled':norm_d,'zero_struct_rescale_decode_invariant':zero_d['decoded']==norm_d['decoded'],'structural_only':zdec,'sharpen':gamma}
    rows.append(row)
   all_cases[order]=rows
  def agg(order,role,path):
   vals=[x[role][path] for x in all_cases[order]]; return {'count':sum(v['pass'] for v in vals),'rate':sum(v['pass'] for v in vals)/139}
  result.update({'B_full_rank':{'normal':all_cases['normal'],'reverse':all_cases['reverse']},'C_mass_partition':all_cases,'D_zero_struct':{'normal':all_cases['normal'],'reverse':all_cases['reverse'],'summary':{o:{r:{'exact':agg(o,r,'zero_struct'),'rescaled_exact':agg(o,r,'zero_struct_rescaled'),'all_rescale_invariant':all(x[r]['zero_struct_rescale_decode_invariant'] for x in all_cases[o])} for r in ('floor','avoid')} for o in all_cases}},'E_structural_contribution':all_cases,'F_sharpen_sweep':all_cases})
 OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'status':result['status'],'A':result['A_lexical_value_forced_target']['lexical_exact'],'forced':result['A_lexical_value_forced_target']['forced_exact'],'sections':[x for x in ('B_full_rank','C_mass_partition','D_zero_struct','E_structural_contribution','F_sharpen_sweep') if x in result]},indent=2))
if __name__=='__main__': main()

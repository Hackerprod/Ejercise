from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; OUT=CAMPAIGN/'t2_nobypass1_r13_zs_sharp_alg'
GAMMAS=(1,2,4,8,16,32,64)
def sha256(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dec(model,cb,r):
 z=model.register_decoder(torch.cat((r,torch.zeros_like(r)),-1),cb)[0]; top=z.topk(2).values; return int(z.argmax()),float(top[0]-top[1])
@torch.no_grad()
def main():
 model=load_executor(); enc=R13RelKeyKVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); manifest=json.loads(MANIFEST_PATH.read_text()); all_cases={'normal':[],'reverse':[]}
 for order in all_cases:
  for p in manifest['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,af,aa,_,sf,sa,vt=enc(ti,le,return_details=True); lp=q.tokens.index(f'VALUE_{l}'); fp=q.tokens.index(f'VALUE_{f}'); row={'digest':p['digest'],'lower':l,'forbidden':f,'roles':{}}
   for role,r,s,a,pos,other,expected in (('floor',rf,sf[0],af[0],lp,fp,l),('avoid',ra,sa[0],aa[0],fp,lp,f)):
    struct=vt[:,[i for i in range(int(le[0])) if i not in (pos,other)]]; struct_positions=[i for i in range(int(le[0])) if i not in (pos,other)]; delta=float(s[pos]-s[other]); items=[]
    for gamma in GAMMAS:
     w=torch.softmax(s*gamma,0); z=w[pos]*vt[:,pos]+w[other]*vt[:,other]; d,m=dec(model,cb,z); rz=z/(w[pos]+w[other]); rd,rm=dec(model,cb,rz); items.append({'gamma':gamma,'decoded':d,'pass':d==expected,'margin':m,'target_mass':float(w[pos]),'competitor_mass':float(w[other]),'structural_mass':float(w[struct_positions].sum()),'rescaled_decoded':rd,'rescaled_invariant':d==rd})
    row['roles'][role]={'target_position':pos,'competitor_position':other,'expected':expected,'delta_s':delta,'gamma':items}
   all_cases[order].append(row)
 summaries={}
 for order,cases in all_cases.items():
  by_gamma=[]
  for gi,gamma in enumerate(GAMMAS):
   vals={'gamma':gamma}
   for role in ('floor','avoid'):
    selected=[c['roles'][role]['gamma'][gi] for c in cases]; vals[role]={'exact_count':sum(x['pass'] for x in selected),'exact_rate':sum(x['pass'] for x in selected)/139,'failed_cases':[{'digest':cases[i]['digest'],'lower':cases[i]['lower'],'forbidden':cases[i]['forbidden'],'decoded':x['decoded'],'margin':x['margin'],'delta_s':cases[i]['roles'][role]['delta_s']} for i,x in enumerate(selected) if not x['pass']],'all_rescaled_invariant':all(x['rescaled_invariant'] for x in selected)}
   vals['joint_exact']=sum(c['roles']['floor']['gamma'][gi]['pass'] and c['roles']['avoid']['gamma'][gi]['pass'] for c in cases)
   by_gamma.append(vals)
  summaries[order]={'per_gamma':by_gamma,'cases':cases}
 residuals=[]
 for order,cases in all_cases.items():
  for c in cases:
   for role in ('floor','avoid'):
    passing=[g for g in GAMMAS if c['roles'][role]['gamma'][GAMMAS.index(g)]['pass']]; residuals.append({'order':order,'digest':c['digest'],'lower':c['lower'],'forbidden':c['forbidden'],'role':role,'delta_s':c['roles'][role]['delta_s'],'first_pass_gamma':min(passing) if passing else None,'last_pass_gamma':max(passing) if passing else None,'gamma_star_if_needed':None})
 closure=all(x['first_pass_gamma'] is not None for x in residuals); result={'task':'T2-NOBYPASS-1-R1.3-ZS-SHARP-ALG','status':'diagnostic_complete','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'definition':{'gamma_grid':list(GAMMAS),'zero_struct':'r=a_T(gamma)*v_T+a_C(gamma)*v_C','rescaled':'r/(a_T+a_C)','raw_delta_s':'s_T-s_C','rescaled_decoder_invariance':True},'normal':summaries['normal'],'reverse':summaries['reverse'],'residuals':residuals,'two_factor_mechanism_confirmed':closure,'sanity_all_rescaled_invariant':all(x['rescaled_invariant'] for order in all_cases for c in all_cases[order] for role in ('floor','avoid') for x in c['roles'][role]['gamma'])}; OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'two_factor_mechanism_confirmed':closure,'normal':[{'gamma':g['gamma'],'floor':g['floor']['exact_count'],'avoid':g['avoid']['exact_count'],'joint':g['joint_exact']} for g in summaries['normal']['per_gamma']],'reverse':[{'gamma':g['gamma'],'floor':g['floor']['exact_count'],'avoid':g['avoid']['exact_count'],'joint':g['joint_exact']} for g in summaries['reverse']['per_gamma']]},indent=2))
if __name__=='__main__': main()

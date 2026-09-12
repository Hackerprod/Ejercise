"""T2-NOBYPASS-1-R1-MIXPATH-ALG frozen attention/oracle interpolation audit."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_nobypass1_r0 import R0Encoder
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r1_seed6702'/'final.pt'; OLD=CAMPAIGN/'t2_nobypass1_r1_attractor_alg'/'results.json'; OUT=CAMPAIGN/'t2_nobypass1_r1_mixpath_alg'
BETAS=(0,.125,.25,.375,.5,.625,.75,.875,1.)
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(model,codebook,r):
 q=torch.cat((r,torch.zeros_like(r)),dim=-1); logits=model.register_decoder(q,codebook)[0]; top=logits.topk(2).values; return int(logits.argmax()),float(top[0]-top[1]),[float(x) for x in logits]
def aggregate(old,section,role):
 rows=old['probe_1_q_rank']['cases']; field='exact_uniform' if section=='probe_3_exact_uniform' else 'clause_only'; count=0
 for row in rows:
  exp=row['expected'][role]; count += row[field][role]['decoded']==exp
 return {'exact_count':count,'exact_rate':count/len(rows),'samples':len(rows),'source_artifact_sha256':sha256(OLD)}
def classify(counts):
 base=counts[0]; interior=max(counts[1:-1]); max_i=counts.index(max(counts));
 if all(counts[i]<=counts[i-1] for i in range(1,len(counts))): return 'mono-decreasing from beta=0'
 if max_i not in (0,len(counts)-1) and interior>base and counts[-1] < interior: return 'improvement-but-beta=1-destroys'
 if max_i not in (0,len(counts)-1) and interior>base: return 'intermediate maximum'
 if max(counts)<=base: return 'no beta improves beta=0'
 return 'unclassified/non-monotone'
@torch.no_grad()
def main():
 model=load_executor(); enc=R0Encoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); ids=torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT); codebook=model.token_embedding(ids); old=json.loads(OLD.read_text()); by_role={r:[] for r in ('floor','avoid')}; case_bank={r:[] for r in ('floor','avoid')}
 for item in old['probe_1_q_rank']['cases']:
  lower=item['expected']['floor']; forbidden=item['expected']['avoid']; text=f'AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}'; parsed=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,af,aa,h,_,_=enc(ti,le,return_details=True); targets={'floor':(rf,parsed.lower,parsed.tokens.index(f'VALUE_{lower}')),'avoid':(ra,parsed.forbidden,parsed.tokens.index(f'VALUE_{forbidden}'))}; valid=list(range(int(le[0]))); learned={r:decode(model,codebook,x[0]) for r,x in targets.items()}
  for role in ('floor','avoid'):
   rlearn,expected,target_pos=targets[role]; attention=af[0] if role=='floor' else aa[0]; e=torch.zeros_like(attention); e[target_pos]=1.; role_cases={'digest':item['digest'],'expected':expected,'target_position':target_pos,'learned':{'decoded':learned[role][0],'margin':learned[role][1]},'betas':[]}
   first=None; last=None
   for beta in BETAS:
    weights=(1-beta)*attention+beta*e; r=enc.w_v(weights.unsqueeze(0)@h[0]); dec,margin,logits=decode(model,codebook,r); passed=dec==expected; index=len(role_cases['betas']); first=index if passed and first is None else first; last=index if passed else last; role_cases['betas'].append({'beta':beta,'decoded':dec,'correct':passed,'margin':margin,'logits':logits,'attention':[float(x) for x in weights]})
   role_cases['first_pass_beta']=None if first is None else BETAS[first]; role_cases['last_pass_beta']=None if last is None else BETAS[last]; case_bank[role].append(role_cases)
 for role in ('floor','avoid'):
  cases=case_bank[role]; results=[]
  for i,beta in enumerate(BETAS):
    selected=[c['betas'][i] for c in cases]; hist={str(v):sum(x['decoded']==v for x in selected) for v in range(VALUE_COUNT) if any(x['decoded']==v for x in selected)}; results.append({'beta':beta,'exact_count':sum(x['correct'] for x in selected),'exact_rate':sum(x['correct'] for x in selected)/139,'margin_min':min(x['margin'] for x in selected),'margin_median':float(torch.tensor([x['margin'] for x in selected]).median()),'prediction_histogram':hist,'attractor_counts':{str(v):hist.get(str(v),0) for v in (4,18,8)}})
  by_role[role]={'beta_grid':[float(x) for x in BETAS],'per_beta':results,'exact_count_sequence':[x['exact_count'] for x in results],'branch_classification':classify([x['exact_count'] for x in results]),'cases':cases}
 old_uniform={r:aggregate(old,'probe_3_exact_uniform',r) for r in ('floor','avoid')}; old_clause={r:aggregate(old,'probe_4_clause_only',r) for r in ('floor','avoid')}
 result={'task':'T2-NOBYPASS-1-R1-MIXPATH-ALG','status':'diagnostic_complete','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'definition':{'a_beta':'(1-beta)*a_learn + beta*e_target','renormalized':False,'betas':[float(x) for x in BETAS],'decoder':'register_decoder(cat(w_v(sum(a_beta*h)),zeros32), frozen VALUE codebook)'},'floor':by_role['floor'],'avoid':by_role['avoid'],'prior_aggregates':{'exact_uniform':old_uniform,'clause_only':old_clause,'source_artifact_sha256':sha256(OLD)}}; OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'floor_sequence':result['floor']['exact_count_sequence'],'avoid_sequence':result['avoid']['exact_count_sequence'],'floor_branch':result['floor']['branch_classification'],'avoid_branch':result['avoid']['branch_classification'],'uniform':old_uniform,'clause_only':old_clause},indent=2))
if __name__=='__main__': main()

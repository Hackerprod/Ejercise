"""T2-NOBYPASS-1-R1-ATTRACTOR-ALG frozen six-probe audit."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_nobypass1_r0 import R0Encoder
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r1_seed6702'/'final.pt'; NB2=CAMPAIGN/'t2_nobypass1_r1_seed6702'/'nb2.json'; OUT=CAMPAIGN/'t2_nobypass1_r1_attractor_alg'
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_hash(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def decode(model,codebook,r):
 q=torch.cat((r,torch.zeros_like(r)),dim=-1); logits=model.register_decoder(q,codebook); top=logits.topk(2,dim=-1).values; return {'decoded':int(logits.argmax(-1).item()),'margin':float((top[:,0]-top[:,1]).item()),'logits':[float(x) for x in logits[0]]},logits
def rank_gap(scores,target,valid):
 vals=scores[valid]; order=torch.argsort(vals,descending=True,stable=True); target_local=valid.index(target); rank=int(torch.where(order==target_local)[0][0])+1; others=[i for i in valid if i!=target]; gap=float(scores[target]-scores[others].max()) if others else None; return rank,gap,int(target)
def summary(xs):
 t=torch.tensor(xs,dtype=torch.float64); return {'mean':float(t.mean()),'median':float(t.median()),'min':float(t.min()),'max':float(t.max())}
@torch.no_grad()
def main():
 model=load_executor(); payload=torch.load(CK,weights_only=False); enc=R0Encoder(); enc.load_state_dict(payload['encoder']); enc.eval(); ids=torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT); codebook=model.token_embedding(ids)
 # Canonical T2-I1 manifest is resolved from the existing NB2 artifact's per-case records when available.
 nb2=json.loads(NB2.read_text()); cases=[]; rank_f=[]; rank_a=[]; observed=[{'floor':{},'avoid':{}} for _ in range(1)]
 for item in nb2['cases']:
  lower=int(item['expected_lower']); forbidden=int(item['expected_forbidden']); text=f'AT_LEAST VALUE_{lower} AND AVOID VALUE_{forbidden}'; parsed=parse_instruction_i1(text); token_ids,lengths=tensorize([text]); rf,ra,_,af,aa,h,sf,sa=enc(token_ids,lengths,return_details=True); valid=list(range(int(lengths[0]))); floor_pos=parsed.tokens.index(f'VALUE_{lower}'); avoid_pos=parsed.tokens.index(f'VALUE_{forbidden}'); rf0=rf; ra0=ra; learned_f,_=decode(model,codebook,rf0); learned_a,_=decode(model,codebook,ra0); rf_forced=enc.w_v(h[:,floor_pos]); ra_forced=enc.w_v(h[:,avoid_pos]); forced_f,_=decode(model,codebook,rf_forced); forced_a,_=decode(model,codebook,ra_forced); rf_uniform=enc.w_v(h[:,valid].mean(1)); ra_uniform=enc.w_v(h[:,valid].mean(1)); uniform_f,_=decode(model,codebook,rf_uniform); uniform_a,_=decode(model,codebook,ra_uniform); clause_f=enc.w_v(h[:,[0,1]].mean(1)); clause_a=enc.w_v(h[:,[3,4]].mean(1)); clause_f_d,_=decode(model,codebook,clause_f); clause_a_d,_=decode(model,codebook,clause_a); rnkf,gapf,_=rank_gap(sf[0],floor_pos,valid); rnka,gapa,_=rank_gap(sa[0],avoid_pos,valid); rank_f.append(rnkf); rank_a.append(rnka); tokens=[]
  for pos,token in enumerate(parsed.tokens):
   vf,_=decode(model,codebook,enc.w_v(h[:,pos])); tokens.append({'position':pos,'token':token,'floor':vf})
  cases.append({'digest':item['digest'],'tokens':list(parsed.tokens),'expected':{'floor':parsed.lower,'avoid':parsed.forbidden},'target_positions':{'floor':floor_pos,'avoid':avoid_pos},'q_rank':{'floor':{'rank':rnkf,'gap':gapf,'scores':[float(x) for x in sf[0]]},'avoid':{'rank':rnka,'gap':gapa,'scores':[float(x) for x in sa[0]]}},'forced_value':{'floor':forced_f,'avoid':forced_a},'exact_uniform':{'floor':uniform_f,'avoid':uniform_a,'learned_floor':learned_f,'learned_avoid':learned_a,'floor_vector_changed':tensor_hash(rf0)!=tensor_hash(rf_uniform),'avoid_vector_changed':tensor_hash(ra0)!=tensor_hash(ra_uniform)},'clause_only':{'floor':clause_f_d,'avoid':clause_a_d},'token_contrib':tokens})
 # Re-run token contribution for AVOID separately to keep full per-role data without changing shared encoder.
 for c in cases:
  text=' '.join(c['tokens']); ti,le=tensorize([text]); _,_,_,_,_,h,_,_=enc(ti,le,return_details=True)
  for item in c['token_contrib']:
   va,_=decode(model,codebook,enc.w_v(h[:,item['position']])); item['avoid']=va
 obs_counts={'floor':{},'avoid':{}}
 for c in nb2['cases']:
  for role,key in (('floor','decoded_lower'),('avoid','decoded_forbidden')):
   value=str(c[key]); obs_counts[role][value]=obs_counts[role].get(value,0)+1
 head=codebook[:,:32]; norms=codebook.norm(dim=-1); rho=head.norm(dim=-1)/norms; p=head/norms[:,None]; head_unit=F.normalize(head,dim=-1); attractors={}
 for target in (4,18,8):
  sims=head_unit[target]@head_unit.T; sims[target]=-float('inf'); nearest=torch.topk(sims,3); attractors[str(target)]={'rho':float(rho[target]),'nearest_values':[int(x) for x in nearest.indices],'nearest_cosines':[float(x) for x in nearest.values],'self_decoder_margin':float((model.register_decoder(torch.cat((head[target],torch.zeros(32)),0).unsqueeze(0),codebook).topk(2).values[0,0]-model.register_decoder(torch.cat((head[target],torch.zeros(32)),0).unsqueeze(0),codebook).topk(2).values[0,1]).item()),'empirical_nb2_frequency':{'floor':obs_counts['floor'].get(str(target),0)/139,'avoid':obs_counts['avoid'].get(str(target),0)/139}}
 g=torch.Generator().manual_seed(7373); random_dirs=F.normalize(torch.randn((100000,32),generator=g),dim=-1); basin=model.register_decoder(torch.cat((random_dirs,torch.zeros_like(random_dirs)),dim=-1),codebook).argmax(-1); basin_counts=torch.bincount(basin,minlength=VALUE_COUNT); basin_rates={str(x):int(basin_counts[x])/100000 for x in (4,18,8)}
 forced_f=sum(c['forced_value']['floor']['decoded']==c['expected']['floor'] for c in cases); forced_a=sum(c['forced_value']['avoid']['decoded']==c['expected']['avoid'] for c in cases)
 result={'task':'T2-NOBYPASS-1-R1-ATTRACTOR-ALG','status':'diagnostic_complete','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'probe_1_q_rank':{'cases':cases,'floor_rank_summary':summary(rank_f),'avoid_rank_summary':summary(rank_a),'floor_gap_summary':summary([c['q_rank']['floor']['gap'] for c in cases]),'avoid_gap_summary':summary([c['q_rank']['avoid']['gap'] for c in cases]),'temperature_interpretation':'rank/gap reported; no temperature change authorized'},'probe_2_forced_value':{'cases':cases,'FORCED_F_exact_match':forced_f,'FORCED_A_exact_match':forced_a,'FORCED_F_rate':forced_f/len(cases),'FORCED_A_rate':forced_a/len(cases)},'probe_3_exact_uniform':{'cases':cases},'probe_4_clause_only':{'cases':cases},'probe_5_token_contrib':{'cases':cases},'probe_6_codebook_basin':{'effective_P_definition':'P[i]=C[i,:32]/||C[i]||_64','random_seed':7373,'random_samples':100000,'attractors':attractors,'basin_rates_4_18_8':basin_rates,'observed_nb2_counts':obs_counts}}
 OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'status':result['status'],'q_rank_floor':result['probe_1_q_rank']['floor_rank_summary'],'q_rank_avoid':result['probe_1_q_rank']['avoid_rank_summary'],'FORCED_F_rate':result['probe_2_forced_value']['FORCED_F_rate'],'FORCED_A_rate':result['probe_2_forced_value']['FORCED_A_rate'],'basin_rates':basin_rates},indent=2))
if __name__=='__main__': main()

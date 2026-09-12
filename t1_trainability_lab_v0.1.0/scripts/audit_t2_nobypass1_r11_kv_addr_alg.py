"""T2-NOBYPASS-1-R1.1-ADDR-ALG frozen addressing audit."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from t2_i1_instruction import TOKEN_IDS,parse_instruction_i1
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r11_kv import R11KVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from train_t2_i2_r2 import load_source
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r11_kv_seed6703'/'final.pt'; OUT=CAMPAIGN/'t2_nobypass1_r11_kv_addr_alg'
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(model,codebook,r):
 q=torch.cat((r,torch.zeros_like(r)),dim=-1); logits=model.register_decoder(q,codebook)[0]; top=logits.topk(2).values; return int(logits.argmax()),float(top[0]-top[1]),[float(x) for x in logits]
def summary(values):
 t=torch.tensor(values,dtype=torch.float64); return {'mean':float(t.mean()),'median':float(t.median()),'min':float(t.min()),'max':float(t.max())}
def rankgap(scores,pos,valid):
 others=[i for i in valid if i!=pos]; return 1+sum(float(scores[i])>float(scores[pos]) for i in valid),float(scores[pos]-scores[others].max())
def role_case(model,codebook,enc,text,role,target):
 parsed=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,af,aa,h,sf,sa,vt=enc(ti,le,return_details=True); pos=parsed.tokens.index(f'VALUE_{target}'); r=rf if role=='floor' else ra; dec,margin,logits=decode(model,codebook,r); scores=sf[0] if role=='floor' else sa[0]; valid=list(range(int(le[0]))); others=[i for i in valid if i!=pos]; rank=1+sum(float(scores[i])>float(scores[pos]) for i in valid); gap=float(scores[pos]-scores[others].max()); ent=float((-( (af if role=='floor' else aa).clamp_min(1e-9)*((af if role=='floor' else aa).clamp_min(1e-9)).log()).sum()).item()); return {'text':text,'tokens':list(parsed.tokens),'target':target,'target_position':pos,'decoded':dec,'pass':dec==target,'margin':margin,'attention_entropy':ent,'target_attention_mass':float((af if role=='floor' else aa)[0,pos]),'target_rank':rank,'target_gap':gap,'scores':[float(x) for x in scores], 'logits':logits,'value_token_contributions':[decode(model,codebook,vt[:,i])[0] for i in valid]}
@torch.no_grad()
def main():
 model=load_executor(); enc=R11KVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); codebook=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));
 lexical=[]
 for n in range(VALUE_COUNT):
  token=torch.tensor([TOKEN_IDS[f'VALUE_{n}']]); v=enc.w_v(enc.embedding(token)); dec,margin,logits=decode(model,codebook,v); lexical.append({'value':n,'token_id':int(token),'decoded':dec,'exact':dec==n,'margin':margin,'logits':logits})
 manifest=json.loads(MANIFEST_PATH.read_text()); joint=[]
 for p in manifest['calibration']:
  l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}'; parsed=parse_instruction_i1(text); ti,le=tensorize([text]); rf,ra,_,af,aa,h,sf,sa,vt=enc(ti,le,return_details=True); lp=parsed.tokens.index(f'VALUE_{l}'); fp=parsed.tokens.index(f'VALUE_{f}'); af_scores=sf[0]; aa_scores=sa[0]; valid=list(range(int(le[0])))
  ar,ag=rankgap(aa_scores,fp,valid); fr,fg=rankgap(af_scores,lp,valid); da,ma,_=decode(model,codebook,ra); df,mf,_=decode(model,codebook,rf); joint.append({'digest':p['digest'],'floor':{'target':l,'position':lp,'score':float(af_scores[lp]),'other_score':float(af_scores[fp]),'target_rank':fr,'target_gap':fg,'F_beats_L':float(af_scores[lp])<float(af_scores[fp]),'L_beats_F':float(af_scores[lp])>float(af_scores[fp])},'avoid':{'target':f,'position':fp,'score':float(aa_scores[fp]),'other_score':float(aa_scores[lp]),'target_rank':ar,'target_gap':ag,'F_beats_L':float(aa_scores[fp])>float(aa_scores[lp]),'L_beats_F':float(aa_scores[fp])<float(aa_scores[lp]),'decoded':da,'decoded_matches_F':da==f},'floor_decoded':df,'H_af':float((-(af.clamp_min(1e-9)*af.clamp_min(1e-9).log()).sum()).item()),'H_aa':float((-(aa.clamp_min(1e-9)*aa.clamp_min(1e-9).log()).sum()).item())})
 a_pass=len(lexical)==sum(x['exact'] for x in lexical); forced_f=[]; forced_a=[]
 for p in manifest['calibration']:
  l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}'; parsed=parse_instruction_i1(text); ti,le=tensorize([text]); *_,h,_,_,vt=enc(ti,le,return_details=True); lp=parsed.tokens.index(f'VALUE_{l}'); fp=parsed.tokens.index(f'VALUE_{f}'); rf=vt[:,lp]; ra=vt[:,fp]; df,mf,lf=decode(model,codebook,rf); da,ma,la=decode(model,codebook,ra); forced_f.append({'digest':p['digest'],'expected':l,'decoded':df,'pass':df==l,'margin':mf,'logits':lf}); forced_a.append({'digest':p['digest'],'expected':f,'decoded':da,'pass':da==f,'margin':ma,'logits':la})
 safety={'lexical_value_exact_count':sum(x['exact'] for x in lexical),'lexical_value_exact_rate':sum(x['exact'] for x in lexical)/32,'forced_floor_exact_count':sum(x['pass'] for x in forced_f),'forced_floor_exact_rate':sum(x['pass'] for x in forced_f)/139,'forced_avoid_exact_count':sum(x['pass'] for x in forced_a),'forced_avoid_exact_rate':sum(x['pass'] for x in forced_a)/139}; result={'task':'T2-NOBYPASS-1-R1.1-ADDR-ALG','status':'diagnostic_complete' if a_pass and safety['forced_floor_exact_count']==139 and safety['forced_avoid_exact_count']==139 else 'alert_halt','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'safety':safety,'A_lexical_value':{'cases':lexical,'exact_match_count':safety['lexical_value_exact_count']},'B_forced_value_r11':{'floor_cases':forced_f,'avoid_cases':forced_a,'exact_floor':safety['forced_floor_exact_count'],'exact_avoid':safety['forced_avoid_exact_count']}}
 if result['status']=='diagnostic_complete':
  avoid_fbeats=sum(x['avoid']['F_beats_L'] for x in joint); avoid_lbeats=sum(x['avoid']['L_beats_F'] for x in joint); floor_fbeats=sum(x['floor']['F_beats_L'] for x in joint); floor_lbeats=sum(x['floor']['L_beats_F'] for x in joint); copy_lbeats=sum(x['avoid']['decoded_matches_F'] and x['avoid']['L_beats_F'] for x in joint); result['C_q_rank_joint']={'cases':joint,'avoid':{'target_rank_summary':summary([x['avoid']['target_rank'] for x in joint]),'target_gap_summary':summary([x['avoid']['target_gap'] for x in joint]),'F_beats_L':avoid_fbeats,'L_beats_F':avoid_lbeats,'ties':139-avoid_fbeats-avoid_lbeats,'decoded_A_equals_F':sum(x['avoid']['decoded_matches_F'] for x in joint),'decoded_A_equals_F_and_L_beats_F':copy_lbeats},'floor':{'target_rank_summary':summary([x['floor']['target_rank'] for x in joint]),'target_gap_summary':summary([x['floor']['target_gap'] for x in joint]),'F_beats_L':floor_fbeats,'L_beats_F':floor_lbeats,'ties':139-floor_fbeats-floor_lbeats}}
  atomic=[]
  for n in range(32): atomic.append(role_case(model,codebook,enc,f'AVOID VALUE_{n}','avoid',n))
  result['D_atomic_vs_joint_addressing']={'atomic_avoid':{'sample_basis':'32 canonical AVOID VALUE_n probes','cases':atomic,'H_A_summary':summary([x['attention_entropy'] for x in atomic]),'target_rank_summary':summary([x['target_rank'] for x in atomic]),'target_gap_summary':summary([x['target_gap'] for x in atomic]),'target_attention_mass_summary':summary([x['target_attention_mass'] for x in atomic])},'joint_avoid':{'sample_basis':'139 mixed probes','H_A_summary':summary([x['H_aa'] for x in joint]),'target_rank_summary':summary([x['avoid']['target_rank'] for x in joint]),'target_gap_summary':summary([x['avoid']['target_gap'] for x in joint]),'target_attention_mass_summary':{'not_available_in_C_cases':True}},'branch':'rama 1: AVOID bueno en atomico, malo solo en joint'}
 OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'status':result['status'],'safety':safety,'C_present':'C_q_rank_joint' in result,'D_present':'D_atomic_vs_joint_addressing' in result},indent=2))
if __name__=='__main__': main()

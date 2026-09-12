"""T2-NOBYPASS-1-R1.1-POS-ALG frozen position-confound controls."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from t2_i1_instruction import parse_instruction_i1
from t2_i2_r3_semantic_writer import tensorize
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r11_kv import R11KVEncoder
from ctrl2_common import BASE_CHECKPOINT,load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r11_kv_seed6703'/'final.pt'; OUT=CAMPAIGN/'t2_nobypass1_r11_kv_pos_alg'
def sha256(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(xs):
 t=torch.tensor(xs,dtype=torch.float64); return {'mean':float(t.mean()),'median':float(t.median()),'min':float(t.min()),'max':float(t.max())}
def decode(model,codebook,r):
 logits=model.register_decoder(torch.cat((r,torch.zeros_like(r)),dim=-1),codebook)[0]; top=logits.topk(2).values; return int(logits.argmax()),float(top[0]-top[1]),[float(x) for x in logits]
def encode_at(enc,token_ids,lengths,pos_rows):
 b,t=token_ids.shape; lexical=enc.embedding(token_ids); tokens=lexical+enc.position(pos_rows); valid=torch.arange(t).unsqueeze(0)<lengths.unsqueeze(1); tokens=tokens.masked_fill(~valid.unsqueeze(-1),0.); z=torch.zeros((b,1,16)); left=torch.cat((z,tokens[:,:-1]),1); right=torch.cat((tokens[:,1:],z),1); h=enc.local_norm(torch.nn.functional.silu(enc.local_binding(torch.cat((left,tokens,right),-1)))); h=h.masked_fill(~valid.unsqueeze(-1),0.); sf=(h@enc.q_f)/4.; sa=(h@enc.q_a)/4.; af=torch.softmax(sf,1).masked_fill(~valid,0.); aa=torch.softmax(sa,1).masked_fill(~valid,0.); vt=enc.w_v(lexical).masked_fill(~valid.unsqueeze(-1),0.); rf=(af.unsqueeze(-1)*vt).sum(1); ra=(aa.unsqueeze(-1)*vt).sum(1); return rf,ra,af,aa,h,sf,sa
def metric(model,codebook,r,scores,weights,target_position,expected_value,valid):
 dec,margin,logits=decode(model,codebook,r); others=[i for i in valid if i!=target_position]; return {'target':expected_value,'target_position':target_position,'decoded':dec,'pass':dec==expected_value,'margin':margin,'logits':logits,'score':float(scores[target_position]),'H':float((-(weights.clamp_min(1e-9)*weights.clamp_min(1e-9).log()).sum()).item()),'rank':1+sum(float(scores[i])>float(scores[target_position]) for i in valid),'gap':float(scores[target_position]-scores[others].max()),'mass':float(weights[target_position])}
def aggregate(cases,role):
 return {'exact_count':sum(x[role]['pass'] for x in cases),'exact_rate':sum(x[role]['pass'] for x in cases)/len(cases),'H':summary([x[role]['H'] for x in cases]),'rank':summary([x[role]['rank'] for x in cases]),'gap':summary([x[role]['gap'] for x in cases]),'mass':summary([x[role]['mass'] for x in cases])}
def p1_group(model,codebook,enc,op,role,shifted):
 rows=[]
 for n in range(32):
  text=f'{op} VALUE_{n}'; parsed=parse_instruction_i1(text); ti,le=tensorize([text]); pos=torch.arange(1,int(le[0])+1).unsqueeze(0)
  if shifted: pos[0,0],pos[0,1]=4,5
  rf,ra,af,aa,h,sf,sa=encode_at(enc,ti,le,pos); target_position=parsed.tokens.index(f'VALUE_{n}'); r,s,w=(rf,sf[0],af[0]) if role=='floor' else (ra,sa[0],aa[0]); rows.append({'value':n,'position_rows':[int(x) for x in pos[0]],'result':metric(model,codebook,r,s,w,target_position,n,list(range(int(le[0]))))})
 return {'cases':rows,'aggregate':{'exact_count':sum(x['result']['pass'] for x in rows),'exact_rate':sum(x['result']['pass'] for x in rows)/32,'H':summary([x['result']['H'] for x in rows]),'rank':summary([x['result']['rank'] for x in rows]),'gap':summary([x['result']['gap'] for x in rows]),'mass':summary([x['result']['mass'] for x in rows])}}
@torch.no_grad()
def main():
 model=load_executor(); enc=R11KVEncoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); codebook=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); manifest=json.loads(MANIFEST_PATH.read_text()); p1={};
 for role,op in (('avoid','AVOID'),('floor','AT_LEAST')):
  p1[role]={'natural':p1_group(model,codebook,enc,op,role,False),'shifted':p1_group(model,codebook,enc,op,role,True)}
 p2=[]
 for p in manifest['calibration']:
  l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}'; parsed=parse_instruction_i1(text); ti,le=tensorize([text]); pos=torch.arange(1,int(le[0])+1).unsqueeze(0); pos[0,3],pos[0,4]=1,2; rf,ra,af,aa,h,sf,sa=encode_at(enc,ti,le,pos); lp=parsed.tokens.index(f'VALUE_{l}'); fp=parsed.tokens.index(f'VALUE_{f}'); fm=metric(model,codebook,rf,sf[0],af[0],lp,l,list(range(5))); am=metric(model,codebook,ra,sa[0],aa[0],fp,f,list(range(5))); fm['other_score']=float(sf[0,fp]); am['other_score']=float(sa[0,lp]); p2.append({'digest':p['digest'],'positions':[int(x) for x in pos[0]],'floor':fm,'avoid':am})
 p3=[]
 for p in manifest['calibration']:
  l=int(p['lower']); f=int(p['forbidden']); text=f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; parsed=parse_instruction_i1(text); ti,le=tensorize([text]); pos=torch.arange(1,int(le[0])+1).unsqueeze(0); rf,ra,af,aa,h,sf,sa=encode_at(enc,ti,le,pos); lp=parsed.tokens.index(f'VALUE_{l}'); fp=parsed.tokens.index(f'VALUE_{f}'); fm=metric(model,codebook,rf,sf[0],af[0],lp,l,list(range(5))); am=metric(model,codebook,ra,sa[0],aa[0],fp,f,list(range(5))); fm['other_score']=float(sf[0,fp]); am['other_score']=float(sa[0,lp]); p3.append({'digest':p['digest'],'tokens':list(parsed.tokens),'floor':fm,'avoid':am})
  p2agg={r:aggregate(p2,r) for r in ('floor','avoid')}; p3agg={r:aggregate(p3,r) for r in ('floor','avoid')}; p2agg['avoid'].update({'F_beats_L':sum(x['avoid']['score']>x['avoid']['other_score'] for x in p2),'L_beats_F':sum(x['avoid']['score']<x['avoid']['other_score'] for x in p2)}); p3agg['avoid'].update({'F_beats_L':sum(x['avoid']['score']>x['avoid']['other_score'] for x in p3),'L_beats_F':sum(x['avoid']['score']<x['avoid']['other_score'] for x in p3)}); p3agg['floor'].update({'F_beats_L':sum(x['floor']['score']>x['floor']['other_score'] for x in p3),'L_beats_F':sum(x['floor']['score']<x['floor']['other_score'] for x in p3)})
 result={'task':'T2-NOBYPASS-1-R1.1-POS-ALG','status':'diagnostic_complete','checkpoint':str(CK),'checkpoint_sha256':sha256(CK),'executor_checkpoint':str(BASE_CHECKPOINT),'executor_checkpoint_sha256':sha256(BASE_CHECKPOINT),'P1_atomic_position_shift':p1,'P2_joint_second_clause_position_remap':{'cases':p2,'aggregate':p2agg,'baseline_ADDR_AVOID_exact':37,'baseline_ADDR_AVOID_F_beats_L':30},'P3_reverse_order_joint':{'cases':p3,'aggregate':p3agg},'classification':'pending_external_verification'}; OUT.mkdir(parents=True,exist_ok=True); artifact=OUT/'results.json'; artifact.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(artifact),'sha256':sha256(artifact),'P1':{r:{k:v['aggregate']['exact_count'] for k,v in x.items()} for r,x in p1.items()},'P2':p2agg,'P3':p3agg,'classification':result['classification']},indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from scipy.optimize import minimize
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import TOKEN_IDS,parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r13_relkey_kv import R13RelKeyKVEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; BASE=CAMPAIGN/'t2_nobypass1_relkey_kv_seed6705'/'final.pt'; BASE=CAMPAIGN/'t2_nobypass1_r13_relkey_kv_seed6705'/'final.pt'; OUT=CAMPAIGN/'t2_nobypass1_r14_gate_signal_alg'
def svm(X,y):
 X=np.asarray(X,float); y=np.asarray(y,float); n,d=X.shape; z=np.zeros(d+1); z[-1]=0.; fun=lambda q:.5*np.dot(q[:-1],q[:-1]); jac=lambda q:np.r_[q[:-1],0.]; cons={'type':'ineq','fun':lambda q:y*(X@q[:-1]+q[-1])-1,'jac':lambda q:y[:,None]*np.c_[X,np.ones(n)]}; r=minimize(fun,z,jac=jac,constraints=cons,method='SLSQP',options={'ftol':1e-12,'maxiter':5000}); q=r.x; scores=X@q[:-1]+q[-1]; margins=y*scores; active=np.where(np.abs(margins-1)<1e-5)[0].tolist(); feasible=bool(np.min(margins)>=1-1e-5); return {'linearly_separable':bool(r.success and feasible),'success':bool(r.success),'message':str(r.message),'w':q[:-1].tolist(),'b':float(q[-1]),'max_margin':float(1/np.linalg.norm(q[:-1])) if np.linalg.norm(q[:-1]) else None,'support_vectors':active,'min_score_positive':float(np.min(scores[y>0])),'max_score_negative':float(np.max(scores[y<0])),'min_signed_margin':float(np.min(margins)),'constraint_violation':float(max(0,1-np.min(margins))),'scores':scores.tolist()}
def entropy(z):
 p=torch.softmax(z,0); return float((-(p*p.clamp_min(1e-12).log()).sum()))
@torch.no_grad()
def main():
 b=torch.load(BASE,weights_only=False)['encoder']; enc=R13RelKeyKVEncoder(); enc.load_state_dict(b); enc.eval(); model=load_executor(); model.eval(); cb=model.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); m=json.loads(MANIFEST_PATH.read_text()); names=[*(f'VALUE_{i}' for i in range(32)),'AT_LEAST','AVOID','AND']; ids=torch.tensor([TOKEN_IDS[x] for x in names]); emb=enc.embedding(ids); A=svm(emb.numpy(),[1]*32+[-1]*3); relx=[]; rely=[]
 for n in range(32):
  for text in (f'AT_LEAST VALUE_{n}',f'AVOID VALUE_{n}'):
   ti,le=tensorize([text]); d=enc(ti,le,return_details=True); relx.append(d[5][0,1].numpy()); rely.append(1)
 for order in ('normal','reverse'):
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); d=enc(ti,le,return_details=True); k=d[5][0]; relx.extend([k[q.tokens.index(f'VALUE_{l}')].numpy(),k[q.tokens.index(f'VALUE_{f}')].numpy()]); rely.extend([1,1])
 for n in range(32):
  for text in (f'AT_LEAST VALUE_{n}',f'AVOID VALUE_{n}'):
   ti,le=tensorize([text]); q=parse_instruction_i1(text); relx.append(enc(ti,le,return_details=True)[5][0,q.tokens.index(f'VALUE_{n}')].numpy()); rely.append(1)
 # Add structural keys from atomic and all joint instruction positions to make B labels explicit.
 for text in ('AT_LEAST VALUE_0','AVOID VALUE_0'):
  ti,le=tensorize([text]); q=parse_instruction_i1(text); relx.append(enc(ti,le,return_details=True)[5][0,0].numpy()); rely.append(-1)
 for order in ('normal','reverse'):
  for p in m['calibration']:
   l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; ti,le=tensorize([text]); q=parse_instruction_i1(text); d=enc(ti,le,return_details=True); k=d[5][0]
   for i,x in enumerate(q.tokens):
    if x in ('AT_LEAST','AVOID','AND'): relx.append(k[i].numpy()); rely.append(-1)
 B=svm(np.array(relx),np.array(rely));
 cmetrics={}
 for n,x in zip(names,enc.w_v(emb)): 
  logits=model.register_decoder(torch.cat((x.unsqueeze(0),torch.zeros_like(x.unsqueeze(0))),-1),cb)[0]; cmetrics[n]={'max_logit':float(logits.max()),'margin':float(logits.topk(2).values.diff().abs().item()),'entropy':entropy(logits)}
 def oracle(fit,kind,gamma):
  w=torch.tensor(fit['w']); bias=fit['b']; groups={}
  for order in ('normal','reverse'):
   rows=[]
   for p in m['calibration']:
    l=int(p['lower']); f=int(p['forbidden']); text=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if order=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}'; q=parse_instruction_i1(text); ti,le=tensorize([text]); d=enc(ti,le,return_details=True); k=d[5]; e=d[2] if False else enc.embedding(ti); feat=e if kind=='A' else k; hard=((feat@w+bias)>0).float(); sf=(k@enc.q_f/4)*gamma; sa=(k@enc.q_a/4)*gamma; af=torch.softmax(sf,1); aa=torch.softmax(sa,1); vt=d[8]; rf=(af.unsqueeze(-1)*vt*hard.unsqueeze(-1)).sum(1); ra=(aa.unsqueeze(-1)*vt*hard.unsqueeze(-1)).sum(1); df=int(model.register_decoder(torch.cat((rf,torch.zeros_like(rf)),-1),cb)[0].argmax()); da=int(model.register_decoder(torch.cat((ra,torch.zeros_like(ra)),-1),cb)[0].argmax()); rows.append({'digest':p['digest'],'floor_pass':df==l,'avoid_pass':da==f})
   groups[order]={'cases':rows,'floor_exact':sum(x['floor_pass'] for x in rows),'avoid_exact':sum(x['avoid_pass'] for x in rows),'joint_exact':sum(x['floor_pass'] and x['avoid_pass'] for x in rows)}
  return groups
 oracle_results={}
 if A['linearly_separable']: oracle_results['A_gamma1']=oracle(A,'A',1); oracle_results['A_gamma2']=oracle(A,'A',2)
 if B['linearly_separable']: oracle_results['B_gamma1']=oracle(B,'B',1); oracle_results['B_gamma2']=oracle(B,'B',2)
 minv=min(cmetrics[f'VALUE_{i}']['margin'] for i in range(32)); maxs=max(cmetrics[x]['margin'] for x in ('AT_LEAST','AVOID','AND')); minvl=min(cmetrics[f'VALUE_{i}']['max_logit'] for i in range(32)); maxsl=max(cmetrics[x]['max_logit'] for x in ('AT_LEAST','AVOID','AND')); branch='L' if A['linearly_separable'] else 'R' if B['linearly_separable'] else 'N' if (minv>maxs or minvl>maxsl) else 'H'; result={'task':'T2-NOBYPASS-1-R1.4-GATE-SIGNAL-ALG','status':'diagnostic_complete','base_checkpoint_sha256':sha256(BASE),'A_LEX_LINSEP':A,'B_RELKEY_LINSEP':B,'C_NUMERIC_CONFIDENCE':cmetrics,'numeric_margin_separates':minv>maxs,'numeric_max_logit_separates':minvl>maxsl,'oracle_gate_DEC':oracle_results,'predeclared_branch':branch}; OUT.mkdir(parents=True,exist_ok=True); p=OUT/'results.json'; p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(p),'sha256':sha256(p),'branch':branch,'A_separable':A['linearly_separable'],'B_separable':B['linearly_separable'],'numeric_margin_separates':minv>maxs,'numeric_max_logit_separates':minvl>maxsl,'oracle_keys':list(oracle_results)},indent=2))
if __name__=='__main__': main()

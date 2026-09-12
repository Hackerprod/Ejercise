import json,torch
from pathlib import Path
from t2_i2_r3_semantic_writer import tensorize
from t2_i1_instruction import parse_instruction_i1
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
from audit_t2_i3_comp0_reg_alg import sha256
R=Path(__file__).resolve().parents[1];B=R/'campaign/t2_nobypass1_r13_relkey_kv_seed6705/final.pt';A=R/'campaign/t2_nobypass1_r14_nc_gateonly_seed6706/gate.pt';O=R/'campaign/t2_nobypass1_r14_nc_residual_alg'
@torch.no_grad()
def main():
 e=GateOnlyEncoder(torch.load(B,weights_only=False)['encoder']);a=torch.load(A,weights_only=False);e.w_c.copy_(a['w_c']);e.b_c.copy_(a['b_c']);e.eval();mod=load_executor();cb=mod.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT));m=json.loads(MANIFEST_PATH.read_text());cache={}
 for o in ('normal','reverse'):
  for p in m['calibration']:
   l=int(p['lower']);f=int(p['forbidden']);t=f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}' if o=='normal' else f'AVOID VALUE_{f} AND AT_LEAST VALUE_{l}';q=parse_instruction_i1(t);x,n=tensorize([t]);d=e(x,n,return_details=True);valid=int(n[0]); cache[(o,p['digest'])]=(l,f,q,d,valid)
 def ev(gamma):
  out={}; ranks=[]; fails=[]
  for o in ('normal','reverse'):
   rr=[]
   for p in m['calibration']:
     l,f,q,d,v=cache[(o,p['digest'])]; k=d[5][0,:v]; c=torch.sigmoid(e.embedding(torch.tensor([q.token_ids]))@e.w_c+e.b_c)[0,:v]; sf=k@e.q_f/4;sa=k@e.q_a/4;uF=gamma*sf[:v]+torch.log(c);uA=gamma*sa[:v]+torch.log(c);lp=q.tokens.index(f'VALUE_{l}');fp=q.tokens.index(f'VALUE_{f}');rf=(torch.softmax((gamma*sf).unsqueeze(0),1).squeeze(0).unsqueeze(-1)*d[8][0,:v]*c.unsqueeze(-1)).sum(0);ra=(torch.softmax((gamma*sa).unsqueeze(0),1).squeeze(0).unsqueeze(-1)*d[8][0,:v]*c.unsqueeze(-1)).sum(0);df=int(mod.register_decoder(torch.cat((rf.unsqueeze(0),torch.zeros((1,32))),-1),cb)[0].argmax());da=int(mod.register_decoder(torch.cat((ra.unsqueeze(0),torch.zeros((1,32))),-1),cb)[0].argmax()); pf=df==l;pa=da==f;rr.append((pf,pa)); ranks.append({'order':o,'digest':p['digest'],'gamma':gamma,'floor_target_rank':int((uF>uF[lp]).sum())+1,'avoid_target_rank':int((uA>uA[fp]).sum())+1}); fails.extend([{'order':o,'digest':p['digest'],'L':l,'F':f,'decoded':da,'s_T':float(sa[fp]),'s_C':float(sa[lp]),'s_struct_max':float(max(sa[i] for i,z in enumerate(q.tokens[:v]) if z in ('AT_LEAST','AVOID','AND'))),'c_T':float(c[fp]),'c_C':float(c[lp]),'c_struct_max':float(max(c[i] for i,z in enumerate(q.tokens[:v]) if z in ('AT_LEAST','AVOID','AND'))),'avoid_target_rank':int((uA>uA[fp]).sum())+1}] if (not pa and gamma==2) else []);
   out[o]={'rows':rr,'floor':sum(x[0] for x in rr),'avoid':sum(x[1] for x in rr),'joint':sum(x[0] and x[1] for x in rr)}
  return out,ranks,fails
 sweeps=[];allr={};allf={}
 for g in (1,1.5,2,2.5,3,4):
  z,r,f=ev(g);sweeps.append({'gamma':g,'normal':z['normal'],'reverse':z['reverse']});allr[str(g)]=r;allf[str(g)]=f
 regressions=[]
 for i in range(1,len(sweeps)):
  prev,cur=sweeps[i-1],sweeps[i]
  for o in ('normal','reverse'):
   for metric in ('floor','avoid','joint'):
    if cur[o][metric]<prev[o][metric]:regressions.append({'from':prev['gamma'],'to':cur['gamma'],'order':o,'metric':metric,'before':prev[o][metric],'after':cur[o][metric]})
 branch='A' if any(x['normal']['joint']==139 and x['reverse']['joint']==139 for x in sweeps if x['gamma'] in (2.5,3)) and not regressions else 'B' if all(x['avoid_target_rank']==1 and x['L']==29 and x['F']==27 for x in allf['2']) else 'C' if any(x['avoid_target_rank']>1 for x in allf['2']) else 'B'
 result={'task':'T2-NOBYPASS-1-R1.4-NC-RESIDUAL-ALG','status':'diagnostic_complete','base_checkpoint_sha256':sha256(B),'gate_artifact_sha256':sha256(A),'avoid_failures_gamma2':allf['2'],'effective_ranks':allr,'sweep':sweeps,'regressions':regressions,'branch':branch};O.mkdir(parents=True,exist_ok=True);p=O/'results.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha256(p),'branch':branch,'failures_gamma2':len(allf['2']),'regressions':regressions,'sweep':sweeps},indent=2))
if __name__=='__main__':main()

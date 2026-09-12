import json,torch,hashlib
from pathlib import Path
from t2_i1_instruction import parse_instruction_i1
from t2_i2_r3_semantic_writer import tensorize
from t2_nobypass1_r14_s_gateonly import GateOnlyEncoder
R=Path(__file__).resolve().parents[1]; O=R/'campaign/t2_nobypass1_staged_nb4_pairgeom_alg/results.json'; A=R/'campaign/t2_nobypass1_staged_nbf0/assembled.pt'
@torch.no_grad()
def main():
 p=json.loads(O.read_text());a=torch.load(A,weights_only=False);core={k:v for k,v in a['core'].items() if k not in ('w_c','b_c')};e=GateOnlyEncoder(core);e.w_c.data.copy_(a['core']['w_c']);e.b_c.data.copy_(a['core']['b_c']);e.eval();out=[]
 for x in p['A']['failures']:
  l,f=x['L'],x['F'];q=parse_instruction_i1(f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}');ids,n=tensorize([q and f'AT_LEAST VALUE_{l} AND AVOID VALUE_{f}']);d=e(ids,n,return_details=True);s=d[7][0];c=d[9][0];fi=q.tokens.index(f'VALUE_{f}');li=q.tokens.index(f'VALUE_{l}');raw=int((s>s[fi]).sum())+1;u=2.5*s+torch.log(c);eff=int((u>u[fi]).sum())+1;out.append({**x,'s_A_F':float(s[fi]),'s_A_L':float(s[li]),'c_F':float(c[fi]),'c_L':float(c[li]),'delta_A_eff':float(2.5*(s[fi]-s[li])+torch.log(c[li]/c[fi])),'raw_rank_target':raw,'effective_rank_target':eff})
 p['B']={'failures':out};O.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'sha256':hashlib.sha256(O.read_bytes()).hexdigest(),'B':out},indent=2))
if __name__=='__main__':main()

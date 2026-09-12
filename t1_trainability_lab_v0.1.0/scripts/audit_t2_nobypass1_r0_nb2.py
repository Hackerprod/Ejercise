from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from t2_i3_common import MANIFEST_PATH
from t2_nobypass1_r0 import R0Encoder
from t2_i1_instruction import parse_instruction_i1
from audit_t2_i3_comp0_reg_alg import sha256
ROOT=Path(__file__).resolve().parents[1]; CAMPAIGN=ROOT/'campaign'; CK=CAMPAIGN/'t2_nobypass1_r0_seed6701'/'final.pt'
def main():
 m=json.loads(MANIFEST_PATH.read_text()); enc=R0Encoder(); enc.load_state_dict(torch.load(CK,weights_only=False)['encoder']); enc.eval(); cases=[]
 for p in m['calibration']:
  text=f"AT_LEAST VALUE_{p['lower']} AND AVOID VALUE_{p['forbidden']}"; q=parse_instruction_i1(text); ids=torch.tensor([q.token_ids]); lens=torch.tensor([len(q.token_ids)]); rf,ra,_=enc(ids,lens); emb=F.normalize(enc.embedding.weight[:,-0+0:].new_zeros((39,32)),dim=-1) if False else None
  # R0 role vectors are 32-wide; nearest-token decode uses first 16-wide lexical embeddings.
  lex=F.normalize(enc.embedding.weight,dim=-1); df=int((lex @ F.normalize(rf[0,:16],dim=-1)).argmax()); da=int((lex @ F.normalize(ra[0,:16],dim=-1)).argmax()); cases.append({'digest':p['digest'],'expected_lower':p['lower'],'expected_forbidden':p['forbidden'],'decoded_lower':df,'decoded_forbidden':da,'pass':df==p['lower'] and da==p['forbidden']})
 result={'status':'passed' if all(x['pass'] for x in cases) else 'failed','task':'T2-NOBYPASS-1-R0','gate':'NB2','decode_cases':len(cases),'exact_pass':sum(x['pass'] for x in cases),'g5_touched':False,'checkpoint_sha256':sha256(CK),'cases':cases}; out=CAMPAIGN/'t2_nobypass1_r0_seed6701'/'nb2.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'path':str(out),'sha256':sha256(out),**{k:v for k,v in result.items() if k!='cases'}},indent=2))
if __name__=='__main__': main()

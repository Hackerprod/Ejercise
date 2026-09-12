import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'campaign/nb5_manifests'; SEEDS=[6801,6802,6803,6804,6805]
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def build(seed):
 r=random.Random(seed); perm=list(range(32));r.shuffle(perm); swap=r.choice([False,True]); floor='OP_X' if not swap else 'OP_Y'; avoid='OP_Y' if not swap else 'OP_X'; base=1000+(seed-6800)*1000; ids=r.sample(range(base,base+500),35); names=[*(f'ARG_{i:02d}' for i in range(32)),'OP_X','OP_Y','LINK']; token_ids=dict(zip(names,ids)); train=[]
 for i in range(32):
  train += [{'kind':'atomic','operator':floor,'argument':f'ARG_{i:02d}','value':perm[i],'constraints':'floor'},{'kind':'atomic','operator':avoid,'argument':f'ARG_{i:02d}','value':perm[i],'constraints':'avoid'},{'kind':'noop','operator':'NOOP','argument':f'ARG_{i:02d}','value':perm[i],'constraints':'none'}]
 test=[]
 for reverse in (False,True):
  for l in range(32):
   for f in range(32):
    if l==f:continue
    op1,op2=(floor,avoid) if not reverse else (avoid,floor);a1,a2=f'ARG_{next(i for i,x in enumerate(perm) if x==l):02d}',f'ARG_{next(i for i,x in enumerate(perm) if x==f):02d}';test.append({'order':'reverse' if reverse else 'natural','tokens':[op1,a1,'LINK',op2,a2],'token_ids':[token_ids[op1],token_ids[a1],token_ids['LINK'],token_ids[op2],token_ids[a2]],'lower':l,'forbidden':f})
 return {'schema':'NB5-fresh-lexical-cipher-v1','domain_seed':seed,'value_count':32,'permutation':perm,'operator_floor':floor,'operator_avoid':avoid,'token_ids':token_ids,'train':train,'test':test,'test_count':len(test),'train_count':len(train),'weights_fresh_init_required':True,'forbidden_in_training':'joint'}
def write(seed):
 d=build(seed);p=OUT/f'manifest_{seed}.json';p.write_text(json.dumps(d,sort_keys=True,indent=2)+'\n');return p,sha_bytes(p.read_bytes())
if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True); rows=[]
 for s in SEEDS:
  p,h=write(s); p2,h2=write(s); rows.append({'seed':s,'path':str(p),'sha256':h,'regenerated_sha256':h2,'reproducible':h==h2})
 print(json.dumps({'manifests':rows,'all_reproducible':all(x['reproducible'] for x in rows)},indent=2))

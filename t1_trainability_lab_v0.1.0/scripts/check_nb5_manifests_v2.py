import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'campaign/nb5_manifests';SEEDS=(6801,6802,6803,6804,6805)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def value(name,perm): return perm[int(name[4:])] if name.startswith('ARG_') else None
def main():
 out=[]
 for seed in SEEDS:
  p=D/f'manifest_{seed}_v2.json';m=json.loads(p.read_text()); rows=m['test']; natural={(x['lower'],x['forbidden']):x for x in rows if x['order']=='natural'}; errors=[]
  if len(rows)!=1984 or len(natural)!=992: errors.append(f'counts rows={len(rows)} natural={len(natural)}')
  for x in rows:
   t=x['tokens']; floor=m['operator_floor'];avoid=m['operator_avoid'];
   if t[0]==floor: al,aa=t[1],t[4]
   elif t[0]==avoid: aa,al=t[1],t[4]
   else: errors.append('bad operators');continue
   if value(al,m['permutation'])!=x['lower'] or value(aa,m['permutation'])!=x['forbidden']: errors.append(f'semantic {x["order"]} {x["lower"]},{x["forbidden"]}')
   if x['order']=='reverse':
    n=natural.get((x['lower'],x['forbidden'])); exp=[avoid,n['tokens'][4],'LINK',floor,n['tokens'][1]] if n else None
    if exp is None or t!=exp: errors.append(f'reverse pair {x["lower"]},{x["forbidden"]}')
  out.append({'seed':seed,'path':str(p),'sha256':sha(p),'rows':len(rows),'semantic_errors':len(errors),'errors':errors[:20]})
 result={'status':'passed' if all(not x['semantic_errors'] for x in out) else 'failed','manifests':out,'checked_rows':sum(x['rows'] for x in out),'checked_expected':9920};print(json.dumps(result,indent=2));(D/'nb5_v2_semantic_check.json').write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['status']=='passed' else 1)
if __name__=='__main__':main()

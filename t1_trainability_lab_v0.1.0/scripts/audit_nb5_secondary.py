import hashlib,json
from pathlib import Path
import torch
from nb5_fresh import NB5GateOnlyEncoder
from ctrl2_common import load_executor
from evaluate_u0c_c1_e_r_alu import VALUE_BASE,VALUE_COUNT
R=Path(__file__).resolve().parents[1]; D=R/'campaign/nb5_manifests'; OUT=R/'campaign/nb5_secondary'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(seed):
    m=json.loads((D/f'manifest_{seed}_v2.json').read_text()); A=R/f'campaign/nb5_stage_a_{seed}/stage_a.pt'; G=R/f'campaign/nb5_stage_b_{seed}/gate.pt'; a=torch.load(A,map_location='cpu',weights_only=False); g=torch.load(G,map_location='cpu',weights_only=False); e=NB5GateOnlyEncoder(a['encoder']); e.w_c.data.copy_(g['w_c']); e.b_c.data.copy_(g['b_c']); e.eval(); x=load_executor(); x.eval(); cb=x.token_embedding(torch.arange(VALUE_BASE,VALUE_BASE+VALUE_COUNT)); rows=[]
    for p in m['test']:
        n=p['tokens']; d=e(torch.tensor([[e.vocab.encode_name(z) for z in n]]),torch.tensor([5]),return_details=True); sf,sa,vt,c=d[6],d[7],d[8],d[9]; ok=torch.where(c[0]>.5)[0]; jf=ok[torch.argmax(2.5*sf[0,ok])]; ja=ok[torch.argmax(2.5*sa[0,ok])]; lf=x.register_decoder(torch.cat((vt[0,jf].unsqueeze(0)*c[0,jf],torch.zeros((1,32))),-1),cb)[0]; la=x.register_decoder(torch.cat((vt[0,ja].unsqueeze(0)*c[0,ja],torch.zeros((1,32))),-1),cb)[0]; df=int(lf.argmax()); da=int(la.argmax()); cf=int(x.register_decoder(cb[df].unsqueeze(0),cb)[0].argmax()); ca=int(x.register_decoder(cb[da].unsqueeze(0),cb)[0].argmax()); l=int(p['lower']); f=int(p['forbidden']); il=next(i for i,v in enumerate(m['permutation']) if v==l); iff=next(i for i,v in enumerate(m['permutation']) if v==f); tf=f'ARG_{il:02d}'; ta=f'ARG_{iff:02d}'; uf=lf.topk(2).values; ua=la.topk(2).values; rows.append({'order':p['order'],'L':l,'F':f,'pointer_F':n[int(jf)]==tf,'pointer_A':n[int(ja)]==ta,'decode_F':df==l,'decode_A':da==f,'RAW':df==l and da==f,'CANON':cf==l and ca==f,'RAW_CANON':df==cf and da==ca,'delta_F':float(sf[0,n.index(tf)]-sf[0,n.index(ta)]),'delta_A':float(sa[0,n.index(ta)]-sa[0,n.index(tf)]),'margin_F':float(uf[0]-uf[1]),'margin_A':float(ua[0]-ua[1])})
    summary={}
    for order in ('natural','reverse'):
        z=[r for r in rows if r['order']==order]; summary[order]={'cases':len(z),**{k:sum(r[k] for r in z) for k in ('pointer_F','pointer_A','decode_F','decode_A','RAW','CANON','RAW_CANON')},'margin_F_positive':sum(r['margin_F']>0 for r in z),'margin_A_positive':sum(r['margin_A']>0 for r in z),'min_delta_F':min(r['delta_F'] for r in z),'min_delta_A':min(r['delta_A'] for r in z),'fail_only_F':sum(not r['decode_F'] and r['decode_A'] for r in z),'fail_only_A':sum(r['decode_F'] and not r['decode_A'] for r in z),'fail_both':sum(not r['decode_F'] and not r['decode_A'] for r in z)}
    sets={role:{(r['L'],r['F'],role) for r in rows if r['order']=='natural' and not r['decode_F' if role=='F' else 'decode_A']}=={(r['L'],r['F'],role) for r in rows if r['order']=='reverse' and not r['decode_F' if role=='F' else 'decode_A']} for role in ('F','A')}
    return {'seed':seed,'manifest_sha256':sha(D/f'manifest_{seed}_v2.json'),'stage_a_sha256':sha(A),'gate_sha256':sha(G),'summary':summary,'sanity_failure_sets_equal':sets,'sanity_pass':all(sets.values())}
def main():
    res={'status':'passed','seeds':[run(s) for s in (6802,6803,6804,6805)]}; res['status']='sanity_failed' if not all(x['sanity_pass'] for x in res['seeds']) else 'passed'; OUT.mkdir(exist_ok=True); (OUT/'results.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()

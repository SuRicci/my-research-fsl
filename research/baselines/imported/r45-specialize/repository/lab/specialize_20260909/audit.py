"""Independent rank/prediction recomputation and parent identity checks."""
import argparse,subprocess,sys
from pathlib import Path
import numpy as np
from campaign_common import ROOT,OUT,HERE,ORDER,read,dump,sha,validate_sources

def audit_r4():
    from run_r4 import dataset,measure
    datasets={};results={};checked={}
    for method in ORDER[:5]:
        files=sorted((OUT/method).glob('*/*/complete.json'));assert len(files)==180
        s=read(OUT/method/'summary.json');assert s['complete'] and len(s['rows'])==60
        counts=0;maxerror=0.
        for p in files:
            m=read(p);f=p.parent/'predictions.npz';assert sha(f)==m['sha256'];z=np.load(f,allow_pickle=False)
            for record in m['inputs']:
                if record['path'] not in checked:checked[record['path']]=sha(record['path'])
                assert checked[record['path']]==record['sha256']
            if m['dataset'] not in datasets:datasets[m['dataset']]=dataset(m['dataset'])
            data=datasets[m['dataset']];truth=data[5];rank=z['rankings']
            np.testing.assert_array_equal(z['query_ids'],data[4]);np.testing.assert_array_equal(np.sort(rank,axis=1),np.broadcast_to(np.arange(rank.shape[1]),rank.shape))
            ap=measure(rank,truth);np.testing.assert_allclose(ap,z['ap'],atol=0,rtol=0,equal_nan=True)
            np.testing.assert_allclose(np.nanmean(ap,0),m['means'],atol=1e-12)
            for v in m['baseline_replay_max_map_error'].values():assert v<3e-5;maxerror=max(maxerror,v)
            assert m['diagnostics']['new_gallery_rows']==0;counts+=len(rank)
        results[method]=dict(cells=180,query_rankings=counts,rank_ap_checks=True,baseline_max_map_error=maxerror)
    dump(OUT/'audit_r4.json',results)

def audit_r5():
    results={}
    for method in ORDER[5:]:
        files=sorted((OUT/method/'test').glob('*.npz'));assert len(files)==50
        grouped={};maxerror=0.;episodes=0
        for p in files:
            m=read(p.with_suffix('.json'));assert sha(p)==m['sha256'] and sha(m['source'])==m['source_sha256']
            z=np.load(p,allow_pickle=False);b=np.load(m['source'],allow_pickle=False)
            for n in ['support_indices','query_indices','class_ids','seed','yq']:np.testing.assert_array_equal(z[n],b[n])
            np.testing.assert_array_equal(z['predictions'],z['scores'].argmax(-1));np.testing.assert_allclose(z['accuracy'],(z['predictions']==z['yq']).mean(-1),atol=0,rtol=0)
            i=list(b['names']).index('r2_overall');np.testing.assert_array_equal(z['predictions'][1],b['predictions'][i]);error=float(np.abs(z['scores'][1]-b['scores'][i]).max());assert error<2e-4 and error==m['baseline_error']
            maxerror=max(maxerror,error);episodes+=len(z['accuracy'][0]);grouped.setdefault(tuple(m['cell']),[]).append(z['accuracy'])
            if method=='r5_consensus':assert (z['retained_count']>=0).all() and (z['retained_count']<=64).all()
            if method=='r5_support_cv':np.testing.assert_allclose(z['fusion_weights'].sum(1),1,atol=2e-7);assert not m['method_gallery_access']
        s=read(OUT/method/'summary.json');assert s['complete'] and len(s['rows'])==10 and episodes==30000
        for row in s['rows']:
            a=np.stack(grouped[tuple(row['cell'])]);np.testing.assert_allclose([a[:,0].mean(),a[:,1].mean(),a[:,2].mean()],[row['accuracy'],row['baseline_accuracy'],row['no_retrieval_accuracy']],atol=1e-12)
        results[method]=dict(files=50,episodes=episodes,baseline_max_score_error=maxerror,baseline_prediction_flips=0,identity_prediction_checks=True)
    dump(OUT/'audit_r5.json',results)

def main():
    validate_sources()
    for m in ORDER:assert (OUT/m/'exit_code').read_text().strip()=='0'
    for part in ['r4','r5']:subprocess.run([sys.executable,__file__,'--part',part],check=True,cwd=ROOT)
    result=dict(complete=True,candidates=10,r4=read(OUT/'audit_r4.json'),r5=read(OUT/'audit_r5.json'),protocol_sha256=sha(OUT/'protocol.json'),
        scope='All candidate full ranks or predictions audited; raw external caches remain on server; inference boundaries also checked by isolated contract tests')
    dump(OUT/'audit.json',result);print('TEN_CANDIDATE_AUDIT_PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['r4','r5']);args=p.parse_args()
    if args.part=='r4':audit_r4()
    elif args.part=='r5':audit_r5()
    else:main()

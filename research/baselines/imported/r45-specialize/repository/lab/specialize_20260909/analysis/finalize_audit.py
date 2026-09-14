"""Combine completed independent audits and verify the conditional extra comparator."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from campaign_common import OUT,ROOT,HERE,ORDER,read,dump,sha,validate_sources

def main():
    validate_sources();assert read(OUT/'state.json')['status']=='all_complete'
    assert (OUT/'controller_exit_code').read_text().strip()=='0'
    for m in ORDER:assert (OUT/m/'exit_code').read_text().strip()=='0' and read(OUT/m/'summary.json')['complete']
    r4=read(OUT/'audit_r4.json');r5=read(OUT/'audit_r5.json');assert len(r4)==len(r5)==5
    for name,digest in read(OUT/'analysis_protocol.json')['sources'].items():assert sha(name)==digest
    extra=None;d=OUT/'cv_uniform_control'
    if (d/'summary.json').exists():
        summary=read(d/'summary.json');assert summary['complete'];files=sorted(d.glob('*.npz'));assert len(files)==50;groups={}
        for p in files:
            m=read(p.with_suffix('.json'));assert sha(p)==m['sha256'] and sha(m['source'])==m['source_sha256']
            z=np.load(p,allow_pickle=False);b=np.load(m['source'],allow_pickle=False);bm=read(Path(m['source']).with_suffix('.json'))
            np.testing.assert_array_equal(z['yq'],b['yq']);np.testing.assert_array_equal(z['predictions'],z['scores'].argmax(-1))
            np.testing.assert_allclose(z['accuracy'],(z['predictions']==z['yq']).mean(-1),atol=0,rtol=0)
            groups.setdefault(tuple(bm['cell']),[]).append((b['accuracy'][0],z['accuracy'],b['accuracy'][1]))
        for row in summary['rows']:
            a=np.array(groups[tuple(row['cell'])]);np.testing.assert_allclose(a.mean((0,2)),[row['cv_accuracy'],row['uniform_accuracy'],row['r2_accuracy']],atol=1e-12)
        extra=dict(status='complete',files=50,episodes=30000,summary_sha256=sha(d/'summary.json'))
    record=dict(complete=True,candidates=10,r4=r4,r5=r5,cv_uniform_control=extra,protocol_sha256=sha(OUT/'protocol.json'),
        summary_hashes={m:sha(OUT/m/'summary.json') for m in ORDER},completed_at=time.time(),aggregation_source_sha256=sha(__file__),
        scope='All 900 rank files and 250 candidate prediction files independently checked; earlier R4 audit reused without repeating computation; raw caches are external dependencies')
    dump(OUT/'audit.json',record);print('FINAL_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()

import os,subprocess,time,sys
from campaign_common import ROOT,HERE,OUT,ORDER,sources,bind,read,dump,sha,validate_sources

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(order=ORDER,sources=sources(),candidates=read(HERE/'candidates.json'),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        baseline_commit='4a54094bf6f95ceff853165767c0b8c5eb7e28a3',r4_cells_per_candidate=180,r5_episodes_per_candidate=30000,
        rule='User accepts domain-specific improvements; preserve all losses and unadjusted interval caveats',sequential=True)
    bind(OUT/'protocol.json',protocol);state=[]
    for method in ORDER:
        validate_sources();d=OUT/method;d.mkdir(parents=True,exist_ok=True)
        if (d/'exit_code').exists() and (d/'exit_code').read_text().strip()=='0' and (d/'summary.json').exists():
            assert read(d/'summary.json')['complete'];state.append(dict(method=method,status='complete_reused'));continue
        started=time.time();dump(OUT/'state.json',dict(method=method,status='running',completed=state,started=started))
        print('START',method,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
        script=HERE/('run_r4.py' if method.startswith('r4_') else 'run_r5.py')
        with open(d/'run.log','a') as log:r=subprocess.run([sys.executable,str(script),method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        (d/'exit_code').write_text(str(r.returncode)+'\n')
        row=dict(method=method,exit_code=r.returncode,seconds=time.time()-started,status='complete' if r.returncode==0 else 'failed');state.append(row)
        dump(OUT/'state.json',dict(status=row['status'],completed=state))
        if r.returncode:raise RuntimeError('Candidate failed: '+method)
        assert read(d/'summary.json')['complete'];print('FINISH',method,round(row['seconds'],1),flush=True)
    dump(OUT/'state.json',dict(status='all_complete',completed=state,protocol_sha256=sha(OUT/'protocol.json')))
    print('ALL_TEN_COMPLETE',flush=True)

if __name__=='__main__':main()

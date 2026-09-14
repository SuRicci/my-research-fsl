"""Official-source 6+6 support / 2 independent query continuation."""
import sys, json, time, hashlib, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.stats import permutation_test

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(next(ROOT.glob('实验11*'))))
from e11 import r3bridge,readout
ASSET=HERE/'assets'/'bow_official'
OUT=HERE/'e11';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
SIDX=list(range(6))+list(range(7,13));QIDX=[6,13]
Y=np.array([1]*6+[-1]*6,dtype=float)
GROUPS={'0':'Anything else','1':'HOI','2':'Taste / Nutrition / Food','3':'Color / Material / Shape',
        '4':'Functionality / Status / Affordance','5':'And / Or / Not','6':'Factual Knowledge',
        '7':'Meta Class','8':'Relationship','9':'Unusual Observations'}

def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def boot(x):
    rng=np.random.default_rng(909);x=np.asarray(x,dtype=float)
    means=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(50)])
    return np.quantile(means,[.025,.975]).tolist()

def metadata(split):
    p=HERE/'sources'/f'assets__data__bongard-ow__bongard_ow_{split}.json'
    return json.loads(p.read_text(encoding='utf-8'))

def imagekeys(p):
    # Preserve each side's original order; validate against the official loader's 0..6,7..13 ordering.
    files=p['imageFiles'];assert len(files)==14
    assert all(x.split('/')[-1].startswith('pos__') for x in files[:7])
    assert all(x.split('/')[-1].startswith('neg__') for x in files[7:])
    return [str(p['uid'])+'/'+'_'.join(x.split('/')[-1].split('__')[:2]) for x in files]

def audit():
    mapping=json.loads((ASSET/'map.json').read_text(encoding='utf-8'))
    allp={s:metadata(s) for s in ['train','val','test']}
    keys=sorted(set(k for ps in allp.values() for p in ps for k in imagekeys(p)))
    def one(k):
        path=ASSET/mapping[k]
        with Image.open(path) as im:
            a=np.asarray(im.convert('RGB'))
            digest=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
            small=np.asarray(im.convert('L').resize((9,8)))
            dh=np.packbits((small[:,1:]>small[:,:-1]).ravel()).tobytes().hex()
        return k,{'rgb_sha256':digest,'dhash':dh,'path':str(path.relative_to(ASSET)),'shape':list(a.shape)}
    with ThreadPoolExecutor(max_workers=8) as pool:ident=dict(pool.map(one,keys))
    dump(OUT/'image_identities.json',ident)
    previous=set();manifest={};stats={}
    for split,ps in allp.items():
        entries=[]
        for p in ps:
            kk=imagekeys(p);hh=[ident[k]['rgb_sha256'] for k in kk]
            reason=[]
            if len(set(hh))!=14:reason.append('within_problem_rgb_duplicate')
            if set(hh)&previous:reason.append('rgb_overlap_with_prior_split')
            entries.append({'uid':str(p['uid']),'keys':kk,'commonSense':str(p['commonSense']),
                            'clean':not reason,'excluded_reasons':reason})
        manifest[split]=entries
        stats[split]={'all_problems':len(entries),'clean_problems':sum(e['clean'] for e in entries),
                      'within_duplicate_problems':sum('within_problem_rgb_duplicate' in e['excluded_reasons'] for e in entries),
                      'cross_split_duplicate_problems':sum('rgb_overlap_with_prior_split' in e['excluded_reasons'] for e in entries)}
        previous.update(ident[k]['rgb_sha256'] for p in ps for k in imagekeys(p))
    dump(OUT/'manifest.json',manifest);dump(OUT/'data_audit.json',stats)
    print('E11 data audit',stats,flush=True)

@torch.no_grad()
def encode():
    ident=json.loads((OUT/'image_identities.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    # Training metadata is used for duplicate embargo only; no cross-problem feature adaptation.
    keys=sorted(set(k for split in ['val','test'] for e in manifest[split] for k in e['keys']))
    stat={}
    for b in ['clip_vitb16','dinov2_vits14']:
        target=ASSET/(b+'_features.pt')
        if target.exists():print('cache exists',b,flush=True);continue
        ext=r3bridge.get_extractor(b,'cuda');tf=r3bridge.center_transform(b)
        allcls=[];allpatch=[];t=time.perf_counter()
        for st in range(0,len(keys),64):
            xs=[]
            for k in keys[st:st+64]:
                with Image.open(ASSET/ident[k]['path']) as im:xs.append(tf(im.convert('RGB')))
            X=torch.stack(xs).cuda()
            if b=='clip_vitb16':cls=ext.encode_image(X);patch=None
            else:
                cls,patch,grid=ext.extract(X)
                patch=F.normalize(F.adaptive_avg_pool2d(patch.transpose(1,2).reshape(-1,384,*grid),(4,4)).flatten(2).transpose(1,2),dim=-1)
            allcls.append(cls.cpu().half())
            if patch is not None:allpatch.append(patch.cpu().half())
            if st%640==0:print('E11 encode',b,st,'/',len(keys),flush=True)
        data={'keys':keys,'cls':torch.cat(allcls)}
        if allpatch:data['patch']=torch.cat(allpatch)
        torch.save(data,target);stat[b]={'images':len(keys),'seconds':time.perf_counter()-t,'gradient_updates':0,'transform':'Resize short edge 224 + center crop, standard backbone normalization; not a reproduction of the original learned model transforms.'}
    dump(OUT/'encoding_cost.json',stat)

def ridge(Ks,Kq,y,lam):
    # Support-centering (intercept) and a support-only PSD diagonal correction for Chamfer kernels.
    mu=Ks.mean(0);grand=Ks.mean();A=Ks-mu[None,:]-mu[:,None]+grand
    B=Kq-mu[None,:]-Kq.mean(1)[:,None]+grand
    shift=max(0.,-float(np.linalg.eigvalsh(A).min()))
    return B@np.linalg.solve(A+(lam+shift)*np.eye(len(y)),y-y.mean())+y.mean()

def loo_kernel(Ks,y):
    s=[]
    for i in range(len(y)):
        keep=np.arange(len(y))!=i
        row=Ks[i,keep];yy=y[keep]
        s.append(row[yy>0].mean()-row[yy<0].mean())
    s=np.array(s);z=np.sort(s);ts=np.r_[z[0]-.001,(z[:-1]+z[1:])/2,z[-1]+.001]
    score=np.array([np.mean((s>t)==(y>0)) for t in ts]);best=ts[score==score.max()]
    # Symmetric tie-break: median of all maximizing thresholds.
    return float(np.median(best))

def predict_global(Z):
    Z=norm(Z);s=Z[SIDX];x=Z[QIDX];ks=s@s.T;kq=x@s.T
    st=torch.from_numpy(s).float();xt=torch.from_numpy(x).float();yt=torch.tensor(Y>0).long()
    pred={'contrast':kq[:,Y>0].mean(1)-kq[:,Y<0].mean(1),
          'contrast_loo':readout.run_method('contrast_loo',st,yt,xt).numpy(),
          'cosine_proto':x@(norm(s[Y>0].mean(0)[None])[0]-norm(s[Y<0].mean(0)[None])[0])}
    for k in [1,3,5]:pred['knn'+str(k)]=Y[np.argsort(-kq,axis=1)[:,:k]].mean(1)
    for lam in [.001,.01,.1,1.,10.]:pred['ridge'+str(lam)]=ridge(ks,kq,Y,lam)
    return pred

def score_problem(features,i):
    zs={b:norm(d['cls'][i[b]].float().numpy()) for b,d in features.items()}
    zs['clip_dino']=norm(np.concatenate(list(zs.values()),axis=-1))
    patch=norm(features['dinov2_vits14']['patch'][i['dinov2_vits14']].float().numpy())
    # Fixed 2x2 spatial pyramid, as a simpler local-evidence baseline.
    pooled=norm(patch.reshape(14,4,4,384).reshape(14,2,2,2,2,384).mean((2,4)))
    zs['spatial']=norm(np.concatenate([zs['dinov2_vits14'],pooled.reshape(14,-1)],axis=-1))
    out={}
    for b,z in zs.items():
        for m,v in predict_global(z).items():out[b+'__'+m]=v
    sim=np.einsum('iad,jbd->ijab',patch,patch,optimize=True)
    K=.5*(sim.max(-1).mean(-1)+sim.max(-2).mean(-1))
    ks=K[np.ix_(SIDX,SIDX)];kq=K[np.ix_(QIDX,SIDX)]
    out['local__contrast']=kq[:,Y>0].mean(1)-kq[:,Y<0].mean(1)
    out['local__contrast_loo']=out['local__contrast']-loo_kernel(ks,Y)
    for lam in [.01,.1,1.,10.]:out['local__ridge'+str(lam)]=ridge(ks,kq,Y,lam)
    return out

def evaluate(phase):
    split='val' if phase=='dev' else 'test'
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))[split]
    feats={b:torch.load(ASSET/(b+'_features.pt'),weights_only=True) for b in ['clip_vitb16','dinov2_vits14']}
    luts={b:{k:i for i,k in enumerate(d['keys'])} for b,d in feats.items()}
    predictions=[];times=[];names=None
    for e in manifest:
        ix={b:[lut[k] for k in e['keys']] for b,lut in luts.items()}
        t=time.perf_counter();pred=score_problem(feats,ix);times.append(time.perf_counter()-t)
        if names is None:names=list(pred)
        predictions.append(np.stack([pred[m] for m in names]))
    scores=np.stack(predictions);yq=np.array([True,False]);correct=(scores>0)==yq;acc=correct.mean(-1)
    clean=np.array([e['clean'] for e in manifest]);uids=np.array([e['uid'] for e in manifest]);groups=np.array([e['commonSense'] for e in manifest])
    np.savez_compressed(OUT/(split+'_predictions.npz'),scores=scores,yq=yq,clean=clean,uids=uids,groups=groups,names=np.array(names),seconds=np.array(times))
    rows=[{'method':m,'full_accuracy':float(acc[:,j].mean()),'clean_accuracy':float(acc[clean,j].mean()),'clean_ci95':boot(acc[clean,j])} for j,m in enumerate(names)]
    dump(OUT/(split+'_summary.json'),{'rows':rows,'n_full':len(clean),'n_clean':int(clean.sum()),'all_methods_seconds_median':float(np.median(times))})
    if phase=='dev':
        global_ix=[j for j,m in enumerate(names) if not m.startswith(('spatial__','local__'))]
        local_ix=[j for j,m in enumerate(names) if m.startswith(('spatial__','local__'))]
        bestg=max(global_ix,key=lambda j:acc[clean,j].mean());bestl=max(local_ix,key=lambda j:acc[clean,j].mean())
        dump(OUT/'selection.json',{'global':names[bestg],'local':names[bestl],'global_dev_accuracy':float(acc[clean,bestg].mean()),'local_dev_accuracy':float(acc[clean,bestl].mean()),'protocol_sha256':sha(HERE/'protocol_v1.json'),'runner_sha256':sha(Path(__file__)),'created_at':time.time()})
        print('E11 frozen selection',names[bestg],names[bestl],acc[clean,bestg].mean(),acc[clean,bestl].mean(),flush=True)
    else:
        sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));g=names.index(sel['global']);l=names.index(sel['local'])
        d=acc[clean,l]-acc[clean,g]
        report={'global':sel['global'],'local':sel['local'],'global_acc':float(acc[clean,g].mean()),'local_acc':float(acc[clean,l].mean()),'difference':float(d.mean()),'difference_ci95':boot(d),'full_global_acc':float(acc[:,g].mean()),'full_local_acc':float(acc[:,l].mean()),'n_clean':int(clean.sum()),'gate_passed':bool(d.mean()>=.03 and boot(d)[0]>0),'groups':{}}
        for group,label in GROUPS.items():
            mask=clean&(groups==group)
            if mask.any():report['groups'][group]={'label':label,'n':int(mask.sum()),'global_acc':float(acc[mask,g].mean()),'local_acc':float(acc[mask,l].mean()),'delta_ci95':boot(acc[mask,l]-acc[mask,g])}
        dump(OUT/'decision.json',report);print('E11 decision',report,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['audit','encode','dev','retest'],required=True);a=ap.parse_args()
    if a.phase=='audit':audit()
    elif a.phase=='encode':encode()
    else:evaluate(a.phase)

if __name__=='__main__':main()

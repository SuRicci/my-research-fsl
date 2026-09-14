"""Bounded, label-free local-token capability and resource check; no classification."""
from pathlib import Path
import sys,json,time,hashlib,shutil
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/main/representation-scatter-20260913'))
import extract_views as ex
OUT=Path(__file__).resolve().parent

def main():
    ex.guard();torch.set_num_threads(4);rows=ex.source_rows();result={'tier':'capability_only','accuracy_evaluated':False,'checks':{},'free_gib_before':shutil.disk_usage(OUT).free/2**30,'total_images':sum(len(r['ids']) for r in rows.values())}
    assert torch.backends.mps.is_available();sample={};encoded={};times={};counts={}
    for bi,b in enumerate(ex.enc.BACKBONES):
        m=ex.enc.model(b);elapsed=0.;num=0
        tf=ex.enc.T.Compose([ex.enc.T.ToTensor(),ex.enc.T.Normalize(*ex.enc.NORM[bi])])
        for key,r in rows.items():
            idx=np.linspace(0,len(r['paths'])-1,8,dtype=int);sample[key]={'indices':idx.tolist(),'rgb':[r['rgb'][i] for i in idx]}
            x=torch.stack([tf(ex.views(r['paths'][i])[0]) for i in idx]).to('mps')
            with torch.inference_mode():
                native=m.encode_image(x) if bi==0 else m(x)
                torch.mps.synchronize();start=time.perf_counter()
                if bi==0:
                    captured=[]
                    hook=m.visual.transformer.register_forward_hook(lambda module,inputs,output:captured.append(output))
                    repeat=m.encode_image(x);hook.remove()
                    z=m.visual.ln_post(captured[0].permute(1,0,2))
                    projected=z@m.visual.proj
                    cls=projected[:,0];patch=projected[:,1:];native_dim=z.shape[-1]
                else:
                    z=m.forward_features(x);cls=z['x_norm_clstoken'];patch=z['x_norm_patchtokens'];native_dim=patch.shape[-1]
                torch.mps.synchronize();elapsed+=time.perf_counter()-start;num+=len(x)
                side=int(patch.shape[1]**.5);assert side*side==patch.shape[1]
                pooled=F.adaptive_avg_pool2d(patch.float().cpu().reshape(len(x),side,side,-1).permute(0,3,1,2),(4,4)).flatten(2).transpose(1,2)
                pooled=F.normalize(pooled,dim=-1).float().cpu();cls=F.normalize(cls.float().cpu(),dim=-1);native=F.normalize(native.float().cpu(),dim=-1)
                error=float((cls-native).abs().max());assert error<1e-5,(key,b,error)
            ds,pool=key.split('_');old=torch.load(ex.ASSET/(ds+'_'+b+'_'+pool+'.pt'),weights_only=True)
            assert old['ids'].tolist()==r['ids'];saved=F.normalize(old['features'][idx].float(),dim=-1)
            saved_error=float((cls-saved).abs().max());saved_cos=float((cls*saved).sum(-1).min())
            assert saved_error<.001 and saved_cos>.99999,(key,b,saved_error,saved_cos)
            half=pooled.half().float();half_error=float((half-pooled).abs().max())
            assert torch.isfinite(pooled).all() and half_error<.001
            encoded[key+'_'+b]=pooled.numpy().astype(np.float16)
            result['checks'][key+'_'+b]={'sample_count':len(idx),'native_tokens_shape':list(patch.shape),'preprojection_dimension':int(native_dim),'pooled_shape':list(pooled.shape),'native_cls_max_abs':error,'saved_cls_max_abs':saved_error,'saved_cls_min_cosine':saved_cos,'half_max_abs':half_error,'pooled_pairwise_cosine_mean':float((pooled@pooled.transpose(1,2)).mean())}
            counts[b]={'tokens':int(patch.shape[1]),'dim':int(patch.shape[2])}
            print('CHECK',key,b,error,saved_error,flush=True)
        times[b]={'patch_pass_seconds':elapsed,'sample_count':num,'seconds_per_image':elapsed/num,'projected_full_pass_seconds':elapsed/num*result['total_images']}
        del m;torch.mps.empty_cache()
    n=result['total_images'];result['storage_bytes']={'all_native_projected_tokens_float16':n*2*sum(v['tokens']*v['dim'] for v in counts.values()),'both_encoders_4x4_float16':n*2*16*sum(v['dim'] for v in counts.values()),'dino_only_4x4_float16':n*2*16*counts['dinov2_vits14']['dim']}
    result['timings']=times;result['storage_caveat']='Full source estimate excludes metadata and classifier outputs; no duplicate chunk plus final copy allowed. Timing excludes image loading, model loading, and thermal slowdown.'
    result['clip_caveat']='CLIP patches use the CLS-trained final layer normalization and projection. They are derived tokens, not a separately validated dense semantic encoder.'
    result['samples']=sample;result['software']={'torch':torch.__version__,'numpy':np.__version__,'device':'mps','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    np.savez(OUT/'probe_features.npz',**encoded)
    result['free_gib_after']=shutil.disk_usage(OUT).free/2**30;result['status']='passed'
    result['file_sha256']=hashlib.sha256((OUT/'probe_features.npz').read_bytes()).hexdigest()
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

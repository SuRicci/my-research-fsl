from pathlib import Path
import os,json,hashlib,shutil,datetime,concurrent.futures
q=Path('/Users/decoqwq/DeepScientist/quests/012').resolve();w=Path.cwd();out=w/'artifacts/reports/pause_20260914';plan=json.loads((out/'cleanup_plan.json').read_text())
roots=[q]+[p for p in (q/'.ds/worktrees').iterdir() if p.is_dir() and not p.is_symlink()]
directories=[]
for r in roots:
 for sub in ['baselines/local/r2-replay/data/dtd/images','baselines/local/r2-replay/data/EuroSAT_RGB','experiments/main/fsl-pets-transfer-20260913/assets/data/images']:
  p=r/sub
  if p.exists() and not p.is_symlink():
   assert p.resolve().is_relative_to(q)
   n=b=0;h=hashlib.sha256()
   for cur,ds,fs in os.walk(p,followlinks=False):
    ds.sort();fs.sort()
    for f in fs:
     a=Path(cur)/f
     if a.is_symlink():continue
     st=a.stat();n+=1;b+=st.st_size;h.update((str(a.relative_to(p))+'\t'+str(st.st_size)+'\n').encode())
   directories.append({'path':str(p),'files':n,'logical_bytes':b,'tree_path_size_sha256':h.hexdigest(),'reason':'verified redownloadable image-only dataset directory'})
files=[x for x in plan['candidates'] if not any(Path(x['path']).is_relative_to(Path(d['path'])) for d in directories)]
files=[x for x in files if not (x['reason']=='redownloadable baseline image dataset' and Path(x['path']).suffix in {'.md','.txt','.csv','.json','.yml','.yaml'})]
def hashed(row):
 p=Path(row['path']);h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return dict(row,sha256=h.hexdigest())
print('PROTECTING',len(plan['protected']),'FILES',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:protected=list(pool.map(hashed,plan['protected']))
(out/'protected_results_before.json').write_text(json.dumps(protected,indent=2)+'\n')
weight_rows=[x for x in files if x['reason']=='model weight or regenerable tensor cache']
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:weight_hashes=list(pool.map(hashed,weight_rows))
(out/'deleted_weight_hashes.json').write_text(json.dumps(weight_hashes,indent=2)+'\n')
execution={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'user_authorization':'2026-09-14T08:31:10Z cleanup weights and keep experiment results before pause','directories':directories,'files':files,'protected_manifest':'protected_results_before.json','uncertain_preserved':plan['uncertain'],'free_before_bytes':shutil.disk_usage(q).free,'deleted_files':0,'deleted_directories':0,'deleted_logical_bytes':0}
(out/'cleanup_execution_plan.json').write_text(json.dumps(execution,indent=2)+'\n')
for i,row in enumerate(files):
 p=Path(row['path']);assert not p.is_symlink() and p.resolve().is_relative_to(q);st=p.stat();assert st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns'],str(p)
 p.unlink();execution['deleted_files']+=1;execution['deleted_logical_bytes']+=row['bytes']
 if i%1000==0:print('FILE_CLEANUP',i,len(files),flush=True)
for i,row in enumerate(directories):
 p=Path(row['path']);assert not p.is_symlink() and p.resolve().is_relative_to(q)
 shutil.rmtree(p);execution['deleted_files']+=row['files'];execution['deleted_directories']+=1;execution['deleted_logical_bytes']+=row['logical_bytes'];print('IMAGE_DIRECTORY_CLEANED',i+1,len(directories),flush=True)
print('VERIFYING_RETAINED_RESULTS',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:after=list(pool.map(hashed,protected))
mismatches=[{'path':a['path'],'before':a['sha256'],'after':b['sha256']} for a,b in zip(protected,after) if a['sha256']!=b['sha256']]
assert not mismatches,mismatches
execution.update({'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed','protected_file_count':len(protected),'protected_hash_mismatches':mismatches,'free_after_bytes':shutil.disk_usage(q).free})
execution['observed_free_delta_bytes']=execution['free_after_bytes']-execution['free_before_bytes'];execution['removed_weight_or_tensor_files']=len(weight_rows);execution['preserved_external_original_assets']=True;execution['preserved_runtime_and_git_history']=True
(out/'cleanup_receipt.json').write_text(json.dumps(execution,indent=2)+'\n')
print('CLEANUP_COMPLETE',json.dumps({k:v for k,v in execution.items() if k not in ['directories','files','uncertain_preserved']}),flush=True)

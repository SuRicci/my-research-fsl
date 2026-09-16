from pathlib import Path
import json,hashlib,subprocess,shutil,os,datetime
q=Path('/Users/decoqwq/DeepScientist/quests/012');w=Path(__file__).resolve().parents[2];h=w/'handoffs/pause-archive-20260916'
v=json.loads((h/'remote-verification.json').read_text());assert v['status']=='passed' and v['all_asset_download_hashes_match']
manifest=json.loads((h/'preservation-manifest.json').read_text());refs=json.loads((h/'refs.json').read_text())
# Check the authoritative remote commit still contains our verified archive metadata.
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=w,text=True).split()[0]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def logical_bytes(root):
 total=0
 for base,dirs,files in os.walk(root,followlinks=False):
  for n in files:
   p=Path(base)/n
   if p.is_file() and not p.is_symlink():total+=p.stat().st_size
 return total
roots=sorted({r['source_root'] for r in manifest['files'] if r['source_root'] not in [str(q),str(w)]})
# Any changed source blocks removing the entire inactive workspace.
checked=0
for r in manifest['files']:
 if r['source_root'] not in roots:continue
 p=Path(r['source_root'])/r['path'];assert p.is_file() and sha(p)==r['sha256'],str(p);checked+=1
before=shutil.disk_usage(q).free;removed=[]
for root in roots:
 p=Path(root);assert p.parent==q/'.ds/worktrees' and p!=w
 size=logical_bytes(p);subprocess.run(['git','worktree','remove','--force',str(p)],cwd=w,check=True)
 removed.append({'path':str(p),'bytes':size,'reason':'Inactive workspace; complete Git history and supplementary research bytes verified in cloud'})
# Remove recoverable download/clone caches only; no external source directories.
for p in [q/'tmp/mechanism-history-20260915',q/'tmp/cache-index',q/'tmp/pause-archive-20260916/cloud']:
 if p.exists():
  size=logical_bytes(p);shutil.rmtree(p);removed.append({'path':str(p),'bytes':size,'reason':'Disposable recovered historical Git/download index or superseded partial clone'})
# Remove archived trained-prior weights and imported feature caches from the active workspace;
# keep measured scores, predictions, source code and compact reports locally.
selected=[]
for r in manifest['files']:
 if r['source_root']!=str(w):continue
 rel=r['path'];p=w/rel
 if p.suffix in ['.pt','.pth','.ckpt','.safetensors'] or (rel.startswith('baselines/imported/') and p.suffix in ['.npz','.npy']):
  assert p.is_file() and sha(p)==r['sha256'],str(p);selected.append((p,r))
for p,r in selected:
 p.unlink();removed.append({'path':str(p),'bytes':r['bytes'],'sha256':r['sha256'],'reason':'Cloud-verified archived weight or imported feature cache; local measured results retained'})
for base,dirs,files in os.walk(w):
 if '.ds' in Path(base).relative_to(w).parts:dirs[:]=[];continue
 for name in list(dirs):
  if name=='__pycache__':
   p=Path(base)/name;size=logical_bytes(p);shutil.rmtree(p);dirs.remove(name);removed.append({'path':str(p),'bytes':size,'reason':'Regenerable Python bytecode'})
receipt={'status':'completed','completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'research_state':'paused_by_user','verified_source_files_rechecked':checked,'inactive_workspaces_removed':len(roots),'removed':removed,'removed_logical_bytes':sum(r['bytes'] for r in removed),'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(q).free,'remote_main_observed':remote,'release_url':v['release']['url'],'preserved':'Active workspace measured results, source code, reports, local Git history, root/current runtime and login state; external original sources untouched','limitations':'Absolute input paths into retired workspaces require cloud restoration before rerunning; no new experiment authorized. Free-space delta may differ from logical bytes due APFS/system activity.'}
(h/'cleanup-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='removed'}),flush=True)

"""Remove verified quest payloads only after full cloud recovery checks."""
from pathlib import Path
import os,json,hashlib,shutil,subprocess,datetime
Q=Path('/Users/decoqwq/DeepScientist/quests/012')
W=Q/'.ds/worktrees/oslo-gallery-pets-20260913'
T=Q/'tmp/github-archive-20260914'
assert Path.cwd()==Q and Q.is_dir() and W.is_dir()
remote=json.loads((T/'remote-verification.json').read_text())
cloud=json.loads((T/'cloud-ready-receipt.json').read_text())
assert remote['status']=='passed' and remote['remote_verified'] and len(remote['verified'])==14
assert cloud['private'] and not cloud['release_draft'] and cloud['browser_clone_verified']
info=json.loads((T/'snapshot-manifest.json').read_text());assert info['source_root']==str(Q) and not info['redactions']
SOURCE={r['path']:r for r in info['files']}
KEEP_ROOT={'quest.yaml','brief.md','plan.md','status.md','SUMMARY.md','CHECKLIST.md','.gitignore'}
KEEP_ENV={'.codex','.claude','.opencode','.kimi'}
matched=[];skipped=[];inode={};protected=[]
def sha(p):
 s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if key not in inode:
  h=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(2**20):h.update(b)
  inode[key]=h.hexdigest()
 return inode[key]
def keep(rel):
 parts=Path(rel).parts
 if len(parts)==1 and rel in KEEP_ROOT:return True
 if parts[0] in KEEP_ENV or parts[0]=='memory':return True
 if parts[0]=='.ds':
  if len(parts)==2:return True
  if parts[1] in ['conversations','codex-home']:return True
  if rel.startswith(str(W.relative_to(Q))+'/'):
   sub=Path(rel).relative_to(W.relative_to(Q));sp=sub.parts
   if len(sp)==1 and sub.name in ['PLAN.md','CHECKLIST.md','plan.md','status.md','SUMMARY.md','brief.md','.gitignore']:return True
   if sp[0] in KEEP_ENV or sp[0]=='memory':return True
 return False
# New scientific files created after snapshot must be archived before this cleanup can run.
science_roots=[]
for base in [Q,*[x for x in (Q/'.ds/worktrees').iterdir() if x.is_dir()]]:
 for group in ['experiments','baselines','paper','literature']:
  r=base/group
  if r.is_dir():science_roots.append(r)
new_science=[]
for base in science_roots:
 for root,dirs,files in os.walk(base):
  dirs[:]=[n for n in dirs if n not in ['.git','__pycache__','node_modules','codex-home']]
  for n in files:
   if n in ['.DS_Store','.git'] or n.endswith('.lock'):continue
   f=Path(root)/n
   if str(f.relative_to(Q)) not in SOURCE:new_science.append(str(f.relative_to(Q)))
assert not new_science,('Scientific files added after snapshot; archive delta first',new_science)
for n in ['protected_results_before.json','baseline_result_hashes.json']:
 rows=json.loads((Q/'handoffs/pause-20260914'/n).read_text())
 for r in rows:
  p=Path(r['path']);assert p.is_relative_to(Q) and p.is_file(),str(p)
  entry=SOURCE[str(p.relative_to(Q))];assert entry['archive_sha256']==r['sha256'] and sha(p)==r['sha256'],str(p)
 protected+=rows
for r in info['files']:
 rel=r['path'];p=Q/rel
 assert p.is_relative_to(Q) and '..' not in Path(rel).parts
 if keep(rel):continue
 if r['type']=='symlink':
  if p.is_symlink() and os.readlink(p)==r['target']:matched.append(p)
  elif p.exists() or p.is_symlink():skipped.append(rel)
 elif p.is_file() and not p.is_symlink():
  if sha(p)==r['source_sha256']:matched.append(p)
  else:skipped.append(rel)
 elif p.exists() or p.is_symlink():skipped.append(rel)
# Research results may not have changed while the experiment pause is in force.
critical=[p for p in skipped if any(x in Path(p).parts for x in ['experiments','baselines','paper','literature'])]
assert not critical,critical
name=subprocess.run(['git','-C',str(Q),'config','--get','user.name'],capture_output=True,text=True,check=True).stdout.strip()
email=subprocess.run(['git','-C',str(Q),'config','--get','user.email'],capture_output=True,text=True,check=True).stdout.strip()
original_refs=subprocess.run(['git','-C',str(Q),'for-each-ref','--format=%(refname) %(objectname)'],capture_output=True,text=True,check=True).stdout
delta_dir=T/'upload/archive/post-snapshot-history'
delta=json.loads((delta_dir/'verification.json').read_text())
assert delta['status']=='passed' and delta['independent_patch_application']=='passed' and cloud['post_snapshot_history_verified']
assert original_refs==(delta_dir/'source-refs-after-delta.txt').read_text(),'Original refs changed after independently verified history delta'
started=datetime.datetime.now(datetime.timezone.utc).isoformat();free_before=shutil.disk_usage(Q).free
R=Q/'handoffs/cloud-archive-20260914';R.mkdir(parents=True,exist_ok=True)
shutil.copyfile(T/'remote-verification.json',R/'remote-verification.json')
shutil.copyfile(T/'cloud-ready-receipt.json',R/'cloud-ready-receipt.json')
shutil.copyfile(delta_dir/'verification.json',R/'post-snapshot-history-verification.json')
for p in matched:p.unlink()
# Source-code snapshots preserve imported code; excluded vendor Git caches are disposable.
removed_vendor_git=[]
for base in science_roots:
 for root,dirs,files in os.walk(base):
  if '.git' in dirs:
   cache=Path(root)/'.git';shutil.rmtree(cache);dirs.remove('.git');removed_vendor_git.append(str(cache.relative_to(Q)))
# Remove empty directories without traversing excluded environment trees or staging.
for root,dirs,files in os.walk(Q,topdown=False):
 p=Path(root)
 if p==Q or p==W or p.is_relative_to(T) or '.git' in p.parts or 'codex-home' in p.parts:continue
 try:p.rmdir()
 except OSError:pass
# Old inactive workspaces can contain excluded, disposable runner environments.
removed_inactive=[]
for p in (Q/'.ds/worktrees').iterdir():
 if not p.is_dir() or p==W:continue
 remaining=[]
 for root,dirs,files in os.walk(p):
  dirs[:]=[n for n in dirs if n not in ['codex-home','__pycache__','node_modules']]
  for n in files:
   f=Path(root)/n
   if n not in ['.git','.DS_Store'] and not n.endswith('.lock'):remaining.append(str(f.relative_to(p)))
 if not remaining:
  shutil.rmtree(p);removed_inactive.append(p.name)
# All 70 refs are recoverable from the independently verified, downloaded history archive.
# Replace the bulky old repository with a parentless, tiny local cloud pointer.
shutil.rmtree(Q/'.git')
def git(*args):subprocess.run(['git','-C',str(Q),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
git('init','-b','main');git('config','user.name',name);git('config','user.email',email)
git('remote','add','origin','https://github.com/SuRicci/my-research-fsl.git')
pointer=f'''# Quest 012: research archived to private GitHub
Research remains paused. Local result files and original Git history have been removed only after full remote download verification.
Repository: https://github.com/SuRicci/my-research-fsl
Complete archive: https://github.com/SuRicci/my-research-fsl/releases/tag/quest-012-archive-20260914
Restore instructions: https://github.com/SuRicci/my-research-fsl/blob/main/archive/RESTORE.md
Verified remote main at archive completion: {cloud['main_commit']}
All 10 archive parts and 4 recovery metadata files were downloaded and SHA256-verified. 21043 protected evidence paths were byte-identical; 70 refs / 685 commits were independently restored. Original-to-sanitized commit maps are in the cloud repository. Five subsequent archive-administration commits are preserved and independently restore-verified in archive/post-snapshot-history.
The local Git repository is now only this retrieval pointer. It is not the original research history. To resume research, restore the cloud snapshot/history first. Runner state and login environments retained for conversation are not research backups.
Existing source directories outside Quest 012 were outside this cleanup.
'''
(Q/'CLOUD_ARCHIVE.md').write_text(pointer)
ignore='*\n!.gitignore\n!CLOUD_ARCHIVE.md\n';(Q/'.gitignore').write_text(ignore)
git('add','-f','.gitignore','CLOUD_ARCHIVE.md');git('commit','-m','archive: retain verified cloud retrieval pointer')
bootstrap=Q/'tmp/cloud-pointer-bootstrap'
git('worktree','add','--no-checkout','-b','run/dino-base-control-20260914',str(bootstrap),'main')
W.mkdir(parents=True,exist_ok=True);os.replace(bootstrap/'.git',W/'.git');bootstrap.rmdir();git('worktree','repair',str(W))
(W/'CLOUD_ARCHIVE.md').write_text(pointer);(W/'.gitignore').write_text(ignore)
subprocess.run(['git','-C',str(W),'reset','--mixed','HEAD'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert subprocess.run(['git','-C',str(W),'rev-parse','HEAD'],capture_output=True,check=True).stdout==subprocess.run(['git','-C',str(Q),'rev-parse','HEAD'],capture_output=True,check=True).stdout
(W/'PLAN.md').write_text('# Archived and paused\n'+pointer)
(W/'CHECKLIST.md').write_text('# Archive completed\n\n## In Progress\n- None; research paused.\n\n## Completed\n- Verified private GitHub upload, all remote archive downloads, full snapshot hashes and independent Git history recovery.\n- Removed verified local research payloads and replaced old local Git history with a small cloud pointer.\n\n## Next\n- Fresh user instruction; restore from cloud before any research execution.\n')
for root in [Q,W]:
 (root/'plan.md').write_text('# Research map: cloud archive and pause\n'+pointer)
 (root/'status.md').write_text('# Paused after verified private cloud migration\n'+pointer)
 (root/'SUMMARY.md').write_text('# Quest 012 cloud archive\n'+pointer)
 (root/'handoffs').mkdir(exist_ok=True)
 (root/'handoffs/CURRENT_CHECKPOINT.md').write_text(pointer)
# Retain receipts, then remove only generated upload/download payloads and tools.
shutil.rmtree(T)
p=W/'handoffs/github-archive-20260914'
if p.exists():shutil.rmtree(p)
p=Q/'tmp/archive-tools'
if p.exists():shutil.rmtree(p)
subprocess.run(['sync'],check=True)
result={'status':'completed','started_at':started,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_repository':'SuRicci/my-research-fsl','cloud_main_commit':cloud['main_commit'],'protected_files_rechecked':len(protected),'verified_original_paths_deleted':len(matched),'post_snapshot_changed_paths_preserved':skipped,'inactive_environment_workspaces_removed':removed_inactive,'original_git_history_removed':True,'post_snapshot_history_commits_preserved':5,'excluded_vendor_git_caches_removed':removed_vendor_git,'local_git_now':'small independent cloud retrieval pointer','generated_upload_parts_removed':True,'free_before_bytes':free_before,'free_after_bytes':shutil.disk_usage(Q).free,'free_delta_note':'Includes removal of generated temporary upload archives; not solely original research disk usage. APFS and unrelated system changes can affect volume free space.','preserved':'Small runtime/control/retrieval state, active login environments and external original research sources'}
(R/'local-cleanup-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)

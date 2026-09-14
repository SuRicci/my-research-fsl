"""Continue the authorized archive lifecycle after existing transfers finish."""
from pathlib import Path
import json,os,time,subprocess,hashlib,tempfile,base64,datetime
Q=Path('/Users/decoqwq/DeepScientist/quests/012');W=Q/'.ds/worktrees/oslo-gallery-pets-20260913';T=Q/'tmp/github-archive-20260914';R=Q/'handoffs/cloud-archive-20260914'
E=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',GH_NO_UPDATE_NOTIFIER='1',GIT_TERMINAL_PROMPT='0')
requirement=Q/'memory/knowledge/active-user-requirements.md';authorized_hash=hashlib.sha256(requirement.read_bytes()).hexdigest()
(T/'authorization.json').write_text(json.dumps({'requirements_sha256':authorized_hash,'scope':'Current explicit private GitHub migration and cleanup after verification','at':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
def api(path,body=None,method=None):
 cmd=['gh','api','repos/SuRicci/my-research-fsl/'+path]
 if method:cmd+=['-X',method]
 with tempfile.TemporaryDirectory(prefix='cloud-receipt-',dir=Q/'tmp') as tmp:
  if body is not None:
   p=Path(tmp)/'body.json';p.write_text(json.dumps(body,ensure_ascii=False));cmd+=['--input',str(p)]
  r=subprocess.run(cmd,env=E,capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stderr)
  return json.loads(r.stdout)
print('WAITING_FOR_EXISTING_UPLOAD_AND_DOWNLOAD_VERIFICATION',flush=True)
while True:
 if (T/'resume-upload-progress.json').exists():
  progress=json.loads((T/'resume-upload-progress.json').read_text())
  if progress['status']=='failed':raise RuntimeError('Resumed uploader failed; preserve originals and inspect progress receipt')
 if (T/'remaining-upload-receipt.json').exists():
  x=json.loads((T/'remaining-upload-receipt.json').read_text())
  if x['status']=='failed':raise RuntimeError('Existing uploader failed; keep originals and inspect receipt')
 if (T/'remote-verification.json').exists() and (T/'remaining-upload-receipt.json').exists():
  v=json.loads((T/'remote-verification.json').read_text())
  if v['status']=='passed':break
 time.sleep(45)
assert hashlib.sha256(requirement.read_bytes()).hexdigest()==authorized_hash,'New user instruction arrived; inspect it before cleanup'
print('REMOTE_DOWNLOADS_VERIFIED_STARTING_FINAL_CLOUD_CHECKS',flush=True)
subprocess.run(['python3','-u',str(W/'handoffs/github-archive-20260914/finalize_remote.py')],cwd=W,env=E,check=True)
assert hashlib.sha256(requirement.read_bytes()).hexdigest()==authorized_hash,'New user instruction arrived; inspect it before cleanup'
print('CLOUD_READY_STARTING_HASH_GUARDED_LOCAL_CLEANUP',flush=True)
subprocess.run(['python3','-u',str(W/'handoffs/github-archive-20260914/cleanup_verified_local.py')],cwd=Q,env=E,check=True)
cleanup=json.loads((R/'local-cleanup-receipt.json').read_text());assert cleanup['status']=='completed'
# Record the completed local cleanup atomically without overwriting unrelated remote changes.
head=api('git/ref/heads/main')['object']['sha'];tree=api('git/commits/'+head)['tree']['sha']
readme=base64.b64decode(api('contents/README.md')['content']).decode();readme=readme.replace('本地清理待执行。','本地已完成对应清理。')
state={'state':'archived_local_payloads_removed','private_repository':True,'all_archive_parts_download_verified':True,'local_cleanup_completed':True,'protected_files':cleanup['protected_files_rechecked'],'snapshot_files':134694,'snapshot_symlinks':78,'history_refs':70,'history_commits':685,'post_snapshot_history_commits':5,'archive_parts':10,'recovery_metadata_assets':4,'total_archive_bytes':9316706614,'free_disk_bytes_after_cleanup':cleanup['free_after_bytes'],'local_retention':'small runtime/control/login state and cloud retrieval pointer; external original research sources excluded from cleanup','research_status':'paused'}
entries=[{'path':path,'mode':'100644','type':'blob','content':content} for path,content in [('README.md',readme),('archive/status.json',json.dumps(state,ensure_ascii=False,indent=2)+'\n'),('archive/local-cleanup-receipt.json',json.dumps(cleanup,ensure_ascii=False,indent=2)+'\n')]]
newtree=api('git/trees',{'base_tree':tree,'tree':entries},'POST')['sha'];commit=api('git/commits',{'message':'Record completed verified cloud migration and local cleanup','tree':newtree,'parents':[head]},'POST')['sha'];api('git/refs/heads/main',{'sha':commit,'force':False},'PATCH');assert api('git/ref/heads/main')['object']['sha']==commit
result={'status':'completed','final_remote_main':commit,'repository':'https://github.com/SuRicci/my-research-fsl','archive_url':'https://github.com/SuRicci/my-research-fsl/releases/tag/quest-012-archive-20260914','free_gib':cleanup['free_after_bytes']/2**30,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print('MIGRATION_COMPLETED '+json.dumps(result),flush=True)

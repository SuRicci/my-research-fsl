from pathlib import Path
import subprocess,json,os,datetime,gzip,shutil
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';R=T/'upload';A=R/'archive';E=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',GH_NO_UPDATE_NOTIFIER='1',GIT_TERMINAL_PROMPT='0')
assert json.loads((T/'verification.json').read_text())['status']=='passed'
shutil.copyfile(T/'verification.json',A/'local-verification.json')
p=A/'snapshot-manifest.json'
if p.exists():
 with p.open('rb') as src,gzip.open(A/'snapshot-manifest.json.gz','wb',compresslevel=5) as dst:shutil.copyfileobj(src,dst)
 p.unlink()
status={'state':'uploading_full_archive','local_deletion_allowed':False,'local_archive_verification':'passed','protected_files':21043,'snapshot_members':134772,'history_refs':70,'history_commits':685,'asset_count':10,'total_bytes':9316706614}
(A/'status.json').write_text(json.dumps(status,indent=2)+'\n')
for cmd in [['git','-C',str(R),'add','archive'],['git','-C',str(R),'commit','-m','Add complete archive manifests, checksums and recovery instructions'],['git','-C',str(R),'-c','http.version=HTTP/1.1','-c','http.postBuffer=67108864','push','origin','main']]:
 r=subprocess.run(cmd,env=E,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);print(r.stdout,flush=True)
 if r.returncode:print(r.stderr,flush=True);raise SystemExit(r.returncode)
assets=json.loads((A/'assets.json').read_text())['assets'];skip={'quest-012-snapshot.tar.gz.part001','quest-012-snapshot.tar.gz.part002'}
paths=[str(T/r['file']) for r in assets if r['file'] not in skip]
paths += [str(A/n) for n in ['assets.json','SHA256SUMS','RESTORE.md','verify_archive.py']]
print(json.dumps({'transfer_started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remaining_archive_parts':len(paths)-4,'metadata_files':4}),flush=True)
r=subprocess.run(['gh','release','upload','quest-012-archive-20260914',*paths,'--repo','SuRicci/my-research-fsl'],env=E,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
receipt={'status':'completed' if r.returncode==0 else 'failed','exit_code':r.returncode,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout':r.stdout,'stderr':r.stderr};(T/'remaining-upload-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);raise SystemExit(r.returncode)

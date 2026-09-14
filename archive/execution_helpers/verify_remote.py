from pathlib import Path
import subprocess,json,os,datetime,time,hashlib,shutil
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';A=T/'upload/archive';D=T/'download-check';D.mkdir(exist_ok=True)
E=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',GH_NO_UPDATE_NOTIFIER='1',GIT_TERMINAL_PROMPT='0')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(2**20):h.update(b)
 return h.hexdigest()
def run(cmd):
 r=subprocess.run(cmd,env=E,capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout
expected=json.loads((A/'assets.json').read_text())['assets']
expected += [{'file':n,'bytes':(A/n).stat().st_size,'sha256':sha(A/n),'role':'metadata'} for n in ['assets.json','SHA256SUMS','RESTORE.md','verify_archive.py']]
receipt=T/'remote-verification-progress.json'
verified=json.loads(receipt.read_text()).get('verified',{}) if receipt.exists() else {}
last=None
while len(verified)<len(expected):
 rows=json.loads(run(['gh','api','repos/SuRicci/my-research-fsl/releases/388290702/assets','--paginate']));remote={x['name']:x for x in rows};(T/'remote-assets-latest.json').write_text(json.dumps(rows,indent=2)+'\n')
 available=sum(r['file'] in remote and remote[r['file']]['state']=='uploaded' for r in expected)
 state=(available,len(verified))
 if state!=last:print(json.dumps({'at':now(),'remote_ready':available,'expected':len(expected),'download_verified':len(verified)}),flush=True);last=state
 for row in expected:
  name=row['file'];item=remote.get(name)
  if name in verified:
   assert item and item['id']==verified[name]['remote_asset_id'] and item['size']==row['bytes'],name
   continue
  if not item or item['state']!='uploaded':continue
  assert item['size']==row['bytes'],name
  if item.get('digest'):assert item['digest']=='sha256:'+row['sha256'],name
  target=D/name
  if target.exists():target.unlink()
  print(json.dumps({'download_started':name,'at':now()}),flush=True)
  run(['gh','release','download','quest-012-archive-20260914','--repo','SuRicci/my-research-fsl','--pattern',name,'--dir',str(D)])
  actual=sha(target);assert target.stat().st_size==row['bytes'] and actual==row['sha256'],name
  verified[name]={'remote_asset_id':item['id'],'bytes':row['bytes'],'sha256':actual,'download_verified_at':now(),'server_digest':item.get('digest'),'role':row['role']}
  receipt.write_text(json.dumps({'status':'in_progress','verified':verified,'expected_count':len(expected)},indent=2)+'\n');target.unlink()
  print(json.dumps({'download_verified':name,'done':len(verified),'expected':len(expected),'at':now()}),flush=True)
 if len(verified)<len(expected):time.sleep(120)
local=json.loads((T/'verification.json').read_text());history=json.loads((T/'history-audit.json').read_text());assert local['status']=='passed' and history['bundle_fsck']=='passed'
result={'status':'passed','remote_verified':True,'all_parts_downloaded_and_sha256_identical':True,'part_count':10,'metadata_count':4,'verified':verified,'verified_at':now(),'local_snapshot_structural_verification':local,'history_independent_restore':history,'verification_method':'Every remote part downloaded, hashed and compared to structurally verified local parts; local full snapshot stream and independent history bundle restore passed. Temporary downloads deleted after checking to maintain disk headroom.'}
(T/'remote-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['verified','history_independent_restore']}),flush=True)

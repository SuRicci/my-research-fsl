from pathlib import Path
import subprocess,json,os,datetime,time,hashlib,shutil,threading
from concurrent.futures import ThreadPoolExecutor
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';A=T/'upload/archive';D=T/'download-check';D.mkdir(exist_ok=True)
E=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',GH_NO_UPDATE_NOTIFIER='1',GIT_TERMINAL_PROMPT='0')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(2**20):h.update(b)
 return h.hexdigest()
def run(cmd):
 attempts=3 if cmd[:2]==['gh','api'] else 1
 for attempt in range(attempts):
  r=subprocess.run(cmd,env=E,capture_output=True,text=True)
  if not r.returncode:return r.stdout
  if attempt+1<attempts:
   print('API_CONNECTION_RETRY '+str(attempt+1),flush=True);time.sleep(5)
 raise RuntimeError(r.stderr)
expected=json.loads((A/'assets.json').read_text())['assets']
expected += [{'file':n,'bytes':(A/n).stat().st_size,'sha256':sha(A/n),'role':'metadata'} for n in ['assets.json','SHA256SUMS','RESTORE.md','verify_archive.py']]
receipt=T/'remote-verification-progress.json'
verified=json.loads(receipt.read_text()).get('verified',{}) if receipt.exists() else {}
lock=threading.Lock()
def verify_one(row,item):
 name=row['file'];target=D/name
 print(json.dumps({'download_started':name,'at':now()}),flush=True)
 if target.exists() and target.stat().st_size==row['bytes'] and sha(target)==row['sha256']:
  print('REUSED_COMPLETE_LOCAL_DOWNLOAD '+name,flush=True)
 else:
  if target.exists():target.unlink()
  run(['gh','release','download','quest-012-archive-20260914','--repo','SuRicci/my-research-fsl','--pattern',name,'--dir',str(D)])
 actual=sha(target);assert target.stat().st_size==row['bytes'] and actual==row['sha256'],name
 with lock:
  verified[name]={'remote_asset_id':item['id'],'bytes':row['bytes'],'sha256':actual,'download_verified_at':now(),'server_digest':item.get('digest'),'role':row['role']}
  temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps({'status':'in_progress','verified':verified,'expected_count':len(expected)},indent=2)+'\n');temp.replace(receipt)
  count=len(verified)
 target.unlink()
 print(json.dumps({'download_verified':name,'done':count,'expected':len(expected),'at':now()}),flush=True)
while len(verified)<len(expected):
 rows=json.loads(run(['gh','api','repos/SuRicci/my-research-fsl/releases/388290702/assets','--paginate']));remote={x['name']:x for x in rows};(T/'remote-assets-latest.json').write_text(json.dumps(rows,indent=2)+'\n')
 pending=[]
 for row in expected:
  name=row['file'];item=remote.get(name)
  if name in verified:
   assert item and item['id']==verified[name]['remote_asset_id'] and item['size']==row['bytes'] and item.get('digest')=='sha256:'+row['sha256'],name
  elif item and item['state']=='uploaded':
   assert item['size']==row['bytes'] and item.get('digest')=='sha256:'+row['sha256'],name
   pending.append((row,item))
 print(json.dumps({'at':now(),'available_to_verify':len(pending),'verified':len(verified),'expected':len(expected),'concurrency':3}),flush=True)
 pending.sort(key=lambda pair:pair[0]['bytes'])
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda pair:verify_one(*pair),pending))
 if len(verified)<len(expected):time.sleep(45)
local=json.loads((T/'verification.json').read_text());history=json.loads((T/'history-audit.json').read_text());assert local['status']=='passed' and history['bundle_fsck']=='passed'
result={'status':'passed','remote_verified':True,'all_parts_downloaded_and_sha256_identical':True,'part_count':10,'metadata_count':4,'verified':verified,'verified_at':now(),'local_snapshot_structural_verification':local,'history_independent_restore':history,'verification_method':'Every remote part downloaded, hashed and compared to structurally verified local parts; local full snapshot stream and independent history bundle restore passed. Temporary downloads deleted after checking to maintain disk headroom.'}
(T/'remote-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['verified','history_independent_restore']}),flush=True)

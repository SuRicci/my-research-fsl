"""Resume only missing assets after the user-requested interruption."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,os,hashlib,datetime,threading
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';A=T/'upload/archive'
E=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',GH_NO_UPDATE_NOTIFIER='1',GIT_TERMINAL_PROMPT='0')
BASE='repos/SuRicci/my-research-fsl/releases/388290702';lock=threading.Lock()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(args):
 p=subprocess.run(args,env=E,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
def remote():return {r['name']:r for r in json.loads(run(['gh','api',BASE+'/assets','--paginate']))}
def check(row,item):
 return item and item['state']=='uploaded' and item['size']==row['bytes'] and item.get('digest')=='sha256:'+row['sha256']
rows=json.loads((A/'assets.json').read_text())['assets']
meta=[{'file':n,'bytes':(A/n).stat().st_size,'sha256':hashlib.sha256((A/n).read_bytes()).hexdigest(),'role':'metadata'} for n in ['assets.json','SHA256SUMS','RESTORE.md','verify_archive.py']]
assert json.loads((T/'verification.json').read_text())['status']=='passed'
state={'status':'running','started_at':now(),'already_complete':[],'uploaded':[],'failures':[],'concurrency':2}
def save():
 p=T/'resume-upload-progress.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(T/'resume-upload-progress.json')
def upload(row):
 name=row['file'];item=remote().get(name)
 if item:
  assert check(row,item),('Existing remote asset differs; do not overwrite',name)
  with lock:state['already_complete'].append(name);save()
  print('REUSED '+name,flush=True);return
 path=(A if row['role']=='metadata' else T)/name
 assert path.stat().st_size==row['bytes']
 print('UPLOAD_STARTED '+json.dumps({'file':name,'bytes':row['bytes'],'at':now()}),flush=True)
 try:
  run(['gh','release','upload','quest-012-archive-20260914',str(path),'--repo','SuRicci/my-research-fsl'])
  item=remote().get(name);assert check(row,item),name
  with lock:state['uploaded'].append({'file':name,'id':item['id'],'at':now()});save()
  print('UPLOAD_VERIFIED_SERVER_DIGEST '+name,flush=True)
 except Exception as ex:
  with lock:state['failures'].append({'file':name,'error':str(ex),'at':now()});save()
  raise
try:
 for row in meta:upload(row)
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(upload,rows))
 rem=remote();assert all(check(row,rem.get(row['file'])) for row in rows+meta)
 state.update(status='completed',completed_at=now());save()
 receipt={'status':'completed','exit_code':0,'at':now(),'method':'Reused matching assets; uploaded missing assets; all 14 server digests verified'}
 (T/'remaining-upload-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('ALL_UPLOADS_COMPLETE '+json.dumps(receipt),flush=True)
except Exception:
 state.update(status='failed',finished_at=now());save();raise

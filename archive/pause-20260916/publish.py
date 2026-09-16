from pathlib import Path
import subprocess,json,os,hashlib,shutil,sys
q=Path('/Users/decoqwq/DeepScientist/quests/012');w=Path(__file__).resolve().parents[2];h=w/'handoffs/pause-archive-20260916';stage=q/'tmp/pause-archive-20260916';cloud=stage/'cloud-index';repo='SuRicci/my-research-fsl';tag='quest-012-pause-20260916'
def run(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True).strip()
def save(name,v):(h/name).write_text(json.dumps(v,indent=2)+'\n')
def git(*args):return run(['git',*args],cloud)
def publish(extra=False):
 parent=git('rev-parse','refs/heads/main');git('read-tree',parent)
 files=[p for p in h.iterdir() if p.is_file() and p.name not in ['inventory-before.json']]
 for p in files:
  dest='archive/pause-20260916/'+p.name;oid=git('hash-object','-w',str(p));git('update-index','--add','--cacheinfo','100644,'+oid+','+dest)
 # Give the owner a report link on the repository landing page without replacing old content.
 try:readme=git('show',parent+':README.md')
 except subprocess.CalledProcessError:readme='# Research archive\n'
 marker='## September 16 research pause and results'
 if marker not in readme:
  readme+='\n\n'+marker+'\n\n[中文结果增补报告](archive/pause-20260916/RESULTS_REPORT.md) · [Recovery instructions](archive/pause-20260916/RESTORE.md) · [Full incremental archive](https://github.com/'+repo+'/releases/tag/'+tag+')\n\nExperiments are paused at the owner’s request. The previous September14 archive remains intact.\n'
 p=stage/'README-publish.md';p.write_text(readme+'\n');oid=git('hash-object','-w',str(p));git('update-index','--add','--cacheinfo','100644,'+oid+',README.md')
 for key in ['user.name','user.email']:
  v=run(['git','config','--get',key],w);git('config',key,v)
 tree=git('write-tree');commit=git('commit-tree',tree,'-p',parent,'-m','archive: '+('verify cloud recovery and record cleanup' if extra else 'preserve September16 results and pause research'))
 git('update-ref','refs/heads/main',commit,parent);git('push','origin',commit+':refs/heads/main')
 remote=git('ls-remote','origin','refs/heads/main').split()[0];assert remote==commit
 save('git-publication.json',{'status':'pushed_and_remote_verified','commit':commit,'parent':parent,'repository':repo,'url':'https://github.com/'+repo+'/tree/'+commit+'/archive/pause-20260916'})
 return commit
if '--metadata-only' in sys.argv:
 print('METADATA_COMMIT',publish(True),flush=True);sys.exit(0)
a=json.loads((h/'package-verification.json').read_text());assert a['status']=='passed';assert not json.loads((h/'history-safety-audit.json').read_text())['credential_hits']
commit=publish();print('GIT_PUSH_VERIFIED',commit,flush=True)
notes=stage/'release-notes.md';notes.write_text('September16 paused research evidence. Includes complete currently reachable research history and supplemental uncommitted research evidence. Results and recovery details are on the main branch. Earlier September14 archive remains unchanged.\n')
subprocess.run(['gh','release','create',tag,'--repo',repo,'--target',commit,'--draft','--title','Quest012 — research pause and evidence, 2026-09-16','--notes-file',str(notes)],check=True)
paths=[str(stage/a['name']) for a in a['assets']]+[str(h/'SHA256SUMS'),str(h/'RESTORE.md')]
subprocess.run(['gh','release','upload',tag,'--repo',repo,*paths],check=True)
print('UPLOAD_COMPLETE',flush=True)
verification=stage/'remote-downloads';verification.mkdir(exist_ok=True)
for a in a['assets']:
 subprocess.run(['gh','release','download',tag,'--repo',repo,'--pattern',a['name'],'--dir',str(verification)],check=True)
 with (verification/a['name']).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert digest==a['sha256'],a['name'];assert (verification/a['name']).stat().st_size==a['bytes']
 print('REMOTE_HASH_VERIFIED',a['name'],flush=True)
# Independent reconstruction from the downloaded bundle, not the local source Git repository.
restored=stage/'restored-history.git';subprocess.run(['git','clone','--mirror',str(verification/'research-history.bundle'),str(restored)],check=True,stdout=subprocess.DEVNULL)
expected=json.loads((h/'refs.json').read_text());actual=dict(s.split(' ',1) for s in run(['git','--git-dir='+str(restored),'for-each-ref','--format=%(refname) %(objectname)','refs/heads']).splitlines());assert expected==actual
subprocess.run(['git','--git-dir='+str(restored),'fsck','--full'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['gh','release','edit',tag,'--repo',repo,'--draft=false'],check=True)
meta=json.loads(run(['gh','release','view',tag,'--repo',repo,'--json','url,isDraft,assets']))
assert not meta['isDraft'];assert len(meta['assets'])==4
save('remote-verification.json',{'status':'passed','all_asset_download_hashes_match':True,'independently_restored_refs':len(actual),'git_fsck':'passed','git_commit':commit,'release':meta,'assets':json.loads((h/'package-verification.json').read_text())['assets']})
print('CLOUD_RECOVERY_VERIFIED',meta['url'],flush=True)
publish(True)

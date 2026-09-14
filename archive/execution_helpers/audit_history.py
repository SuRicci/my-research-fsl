from pathlib import Path
import subprocess,json,re,shutil,hashlib
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';H=T/'history.git'
def git(*args):return subprocess.run(['git','--git-dir='+str(H),*args],capture_output=True,text=True,check=True).stdout
rows=git('rev-list','--objects','--all').splitlines();bad=[x for x in rows if '.ds/' in x or any(x.endswith('/'+y) for y in ['auth.json','hosts.yml','.env'])];assert not bad,bad
refs=git('for-each-ref','--format=%(refname) %(objectname)');orig=(T/'original-refs.txt').read_text();assert sorted(x.split()[0] for x in refs.splitlines())==sorted(x.split()[0] for x in orig.splitlines())
(T/'sanitized-refs.txt').write_text(refs)
for n in ['commit-map','ref-map','changed-refs']:
 p=H/'filter-repo'/n
 if p.exists():shutil.copyfile(p,T/('history-'+n+'.txt'))
ids=[x.split()[0] for x in rows];ip=T/'history-object-ids.txt';ip.write_text('\n'.join(ids)+'\n')
pat=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}')
blobs=0;size=0;hits=[]
with ip.open('rb') as src:
 p=subprocess.Popen(['git','--git-dir='+str(H),'cat-file','--batch'],stdin=src,stdout=subprocess.PIPE)
 for i in range(len(ids)):
  header=p.stdout.readline().decode().split();oid,kind,n=header;data=p.stdout.read(int(n));assert len(data)==int(n);assert p.stdout.read(1)==b'\n'
  if kind=='blob':
   blobs+=1;size+=int(n)
   if pat.search(data):hits.append(oid)
  if i%10000==0:print(json.dumps({'objects_scanned':i,'total':len(ids),'blob_bytes':size}),flush=True)
 assert p.wait()==0
receipt={'status':'passed' if not hits else 'failed','refs':len(refs.splitlines()),'commits':int(git('rev-list','--all','--count')),'objects':len(rows),'blobs_scanned':blobs,'logical_blob_bytes':size,'credential_path_count':len(bad),'credential_pattern_hit_oids':hits,'history_export':'same 70 refs and 685 commits; credential/runtime trees and previously deleted r2-domains raw data/downloads excluded; content redactions mapped by git-filter-repo'}
(T/'history-audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);assert not hits

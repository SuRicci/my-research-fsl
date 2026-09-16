from pathlib import Path
import os,json,subprocess,hashlib,re,tarfile,shutil,time
q=Path('/Users/decoqwq/DeepScientist/quests/012');w=Path(__file__).resolve().parents[2];h=w/'handoffs/pause-archive-20260916';stage=q/'tmp/pause-archive-20260916';stage.mkdir(parents=True,exist_ok=True)
def git(*args,cwd=w):return subprocess.check_output(['git',*args],cwd=cwd)
def sha(p):
 hsh=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):hsh.update(b)
 return hsh.hexdigest()
def save(name,v):(h/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
secret=re.compile(rb'(?<![A-Za-z0-9_])(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
refs=dict(line.split(' ',1) for line in git('for-each-ref','--format=%(refname) %(objectname)','refs/heads').decode().splitlines());save('refs.json',refs)
objects=git('rev-list','--objects',*refs)
p=subprocess.Popen(['git','cat-file','--batch'],cwd=w,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
text_count=0;hits=[];seen=set()
for line in objects.splitlines():
 oid,_,path=line.partition(b' ')
 if oid in seen:continue
 seen.add(oid)
 if any(x in path.lower().split(b'/') for x in [b'auth.json',b'.env',b'credentials.json']):hits.append({'path':path.decode(),'reason':'credential path'})
 p.stdin.write(oid+b'\n');p.stdin.flush();header=p.stdout.readline().split();size=int(header[-1]);body=p.stdout.read(size);p.stdout.read(1)
 if header[1]==b'blob' and b'\0' not in body[:8192]:
  text_count+=1
  if secret.search(body):hits.append({'path':path.decode(),'oid':oid.decode(),'reason':'secret pattern'})
p.stdin.close();p.wait();save('history-safety-audit.json',{'objects':len(seen),'text_blobs_scanned':text_count,'credential_hits':hits})
assert not hits,'Credential-pattern hits: inspect audit before upload'
print('HISTORY_SAFETY_PASSED',len(seen),text_count,flush=True)
subprocess.run(['git','bundle','create',str(stage/'research-history.bundle'),*refs],cwd=w,check=True)
subprocess.run(['git','bundle','verify',str(stage/'research-history.bundle')],cwd=w,check=True,stdout=subprocess.DEVNULL)
print('BUNDLE_CREATED',flush=True)
roots=[q]+[Path(line[9:]) for line in git('worktree','list','--porcelain').decode().splitlines() if line.startswith('worktree ') and line[9:]!=str(q)]
allowed={'experiments','artifacts','baselines','paper','literature','handoffs','memory'}
skip={'__pycache__','.git','.ds','node_modules','progress','answers','milestones','interactions','approvals','userfiles'}
manifest=[];supp=[];links=[];exclusions=[];uniq={}
with tarfile.open(stage/'research-supplement.tar.gz','w:gz',compresslevel=3) as tf:
 for root in roots:
  label='quest-root' if root==q else root.name
  tree={}
  for row in git('ls-tree','-rz','HEAD',cwd=root).split(b'\0'):
   if not row:continue
   meta,path=row.split(b'\t',1);tree[path.decode()]=meta.split()[2].decode()
  rootfiles=[root/n for n in ['brief.md','plan.md','analysis_plan.md','CHECKLIST.md','status.md','SUMMARY.md','CLOUD_ARCHIVE.md'] if (root/n).is_file()]
  files=list(rootfiles)
  for top in sorted(allowed):
   if not (root/top).is_dir():continue
   for base,dirs,names in os.walk(root/top,followlinks=False):
    dirs[:]=[d for d in dirs if d not in skip]
    for n in names:
     f=Path(base)/n;rel=str(f.relative_to(root))
     if n in ['_index.jsonl','active-user-requirements.md','inventory-before.json'] or rel.startswith('handoffs/pause-archive-20260916/') or n=='auth.json':continue
     files.append(f)
  for f in sorted(set(files)):
   rel=str(f.relative_to(root))
   if f.is_symlink():links.append({'workspace':label,'path':rel,'target':os.readlink(f)});continue
   if not f.is_file():continue
   b=f.read_bytes();size=len(b);digest=hashlib.sha256(b).hexdigest();oid=hashlib.sha1(b'blob '+str(size).encode()+b'\0'+b).hexdigest()
   if b'\0' not in b[:8192] and secret.search(b):raise RuntimeError('Secret pattern in '+str(f))
   rec={'workspace':label,'path':rel,'bytes':size,'sha256':digest,'git_blob':oid,'source_root':str(root),'storage':'git' if tree.get(rel)==oid else 'supplement'}
   manifest.append(rec)
   if rec['storage']=='git':continue
   member=label+'/'+rel;rec['member']=member
   if digest in uniq:
    ti=tarfile.TarInfo(member);ti.type=tarfile.LNKTYPE;ti.linkname=uniq[digest];tf.addfile(ti)
   else:
    tf.add(f,arcname=member,recursive=False);uniq[digest]=member
   supp.append(rec)
  print('WORKSPACE_PACKED',label,len(manifest),len(supp),flush=True)
save('preservation-manifest.json',{'files':manifest,'symlinks':links,'excluded_categories':['runtime/login/session trees','conversation/progress/milestone artifacts','temporary downloads','Python bytecode','active user requirements'],'refs':refs})
save('supplement-manifest.json',{'count':len(supp),'files':supp,'symlinks':links})
with tarfile.open(stage/'research-supplement.tar.gz','r:gz') as tf:
 for r in supp:
  stream=tf.extractfile(r['member']);digest=hashlib.file_digest(stream,'sha256').hexdigest();assert digest==r['sha256'],r['member']
print('SUPPLEMENT_CONTENT_VERIFIED',len(supp),flush=True)
assets=[]
for name in ['research-history.bundle','research-supplement.tar.gz']:
 p=stage/name;assets.append({'name':name,'bytes':p.stat().st_size,'sha256':sha(p)});print('ASSET',name,p.stat().st_size,flush=True)
save('package-verification.json',{'status':'passed','files':len(manifest),'supplement_files':len(supp),'refs':len(refs),'assets':assets,'credential_hits':0})
(h/'SHA256SUMS').write_text(''.join(a['sha256']+'  '+a['name']+'\n' for a in assets))
print('PACKAGE_COMPLETE',json.dumps(assets),flush=True)

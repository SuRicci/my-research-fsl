from pathlib import Path
import os, json, hashlib, re, tarfile, io, gzip, datetime, shutil
Q=Path('/Users/decoqwq/DeepScientist/quests/012')
OUT=Q/'tmp/github-archive-20260914';OUT.mkdir(exist_ok=True)
PAT=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}')
EXDIR={'.git','codex-home','node_modules','__pycache__'}
EXFILE={'auth.json','hosts.yml','.env','credentials.json','id_rsa','id_ed25519','.DS_Store'}
paths=[];excluded=[]
for root,dirs,files in os.walk(Q,followlinks=False):
    root=Path(root)
    for n in list(dirs):
        p=root/n;r=str(p.relative_to(Q))
        if n in EXDIR or r=='tmp':
            excluded.append({'path':r,'reason':'credentials/environment/git-managed separately' if r!='tmp' else 'disposable scratch and archive staging'});dirs.remove(n)
        elif p.is_symlink():paths.append(p);dirs.remove(n)
    for n in files:
        p=root/n
        if n in EXFILE or n.endswith('.pem') or n.endswith('.lock'):
            excluded.append({'path':str(p.relative_to(Q)),'reason':'credential/system/active lock file'})
        else:paths.append(p)
paths.sort()
manifest=[];content={};inode={};redacted=[];counts={'files':0,'symlinks':0,'unique_content_bytes':0,'original_bytes':0,'hardlinks':0}
class Parts:
    def __init__(self):self.n=0;self.used=0;self.f=None;self.parts=[];self.sha=None
    def write(self,b):
        total=len(b)
        while b:
            if self.f is None:
                self.n+=1;self.path=OUT/f'quest-012-snapshot.tar.gz.part{self.n:03d}';self.f=self.path.open('wb');self.sha=hashlib.sha256();self.used=0
            chunk=b[:1024**3-self.used];self.f.write(chunk);self.sha.update(chunk);self.used+=len(chunk);b=b[len(chunk):]
            if self.used==1024**3:self.endpart()
        return total
    def flush(self):
        if self.f:self.f.flush()
    def endpart(self):
        if self.f:
            self.f.close();self.parts.append({'file':self.path.name,'bytes':self.used,'sha256':self.sha.hexdigest()});self.f=None
    def close(self):self.endpart()
parts=Parts()
with gzip.GzipFile(fileobj=parts,mode='wb',compresslevel=3,mtime=0) as gz:
    with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        for i,p in enumerate(paths):
            rel=str(p.relative_to(Q));name='quest-012/'+rel
            if p.is_symlink():
                target=os.readlink(p);ti=tarfile.TarInfo(name);ti.type=tarfile.SYMTYPE;ti.linkname=target;tar.addfile(ti)
                manifest.append({'path':rel,'type':'symlink','target':target});counts['symlinks']+=1;continue
            try:s=p.stat()
            except FileNotFoundError:excluded.append({'path':rel,'reason':'ephemeral file disappeared before snapshot'});continue
            if not p.is_file():excluded.append({'path':rel,'reason':'socket or non-regular runtime entry'});continue
            key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            if key in inode:
                old=inode[key];row=dict(old,path=rel);ti=tarfile.TarInfo(name);ti.type=tarfile.LNKTYPE;ti.linkname=content[row['archive_sha256']];counts['hardlinks']+=1
            else:
                data=p.read_bytes();original=hashlib.sha256(data).hexdigest();hits=len(PAT.findall(data))
                if hits:
                    data=PAT.sub(b'[REDACTED_CREDENTIAL]',data);redacted.append({'path':rel,'matches':hits})
                digest=hashlib.sha256(data).hexdigest();row={'path':rel,'type':'file','bytes':len(data),'source_bytes':s.st_size,'source_sha256':original,'archive_sha256':digest,'redactions':hits};inode[key]=row
                ti=tarfile.TarInfo(name);ti.mode=s.st_mode&0o777;ti.mtime=int(s.st_mtime)
                if digest in content:ti.type=tarfile.LNKTYPE;ti.linkname=content[digest];counts['hardlinks']+=1
                else:
                    ti.size=len(data);content[digest]=name;counts['unique_content_bytes']+=len(data)
                    tar.addfile(ti,io.BytesIO(data));manifest.append(row);counts['files']+=1;counts['original_bytes']+=s.st_size
                    if i%1000==0:print(json.dumps({'processed':i,'total':len(paths),**counts}),flush=True)
                    continue
            tar.addfile(ti);manifest.append(row);counts['files']+=1;counts['original_bytes']+=s.st_size
        info={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(Q),'counts':counts,'files':manifest,'exclusions':excluded,'redactions':redacted,'coverage_note':'All retained quest files except listed credential/runtime environment/scratch exclusions. Git history exported separately. Symlinks retained without following external targets.'}
        encoded=(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode();ti=tarfile.TarInfo('ARCHIVE_MANIFEST.json');ti.size=len(encoded);tar.addfile(ti,io.BytesIO(encoded))
parts.close()
(OUT/'snapshot-manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
by={str(Q/x['path']):x for x in manifest};coverage=[];failures=[]
for n in ['protected_results_before.json','baseline_result_hashes.json']:
    rows=json.loads((Q/'handoffs/pause-20260914'/n).read_text())
    for row in rows:
        item=by.get(row['path']);ok=item is not None and item.get('source_sha256')==row['sha256'] and item.get('archive_sha256')==row['sha256']
        if not ok:failures.append({'path':row['path'],'reason':'not byte-identical to protected hash','present':item is not None})
    coverage.append({'manifest':n,'expected':len(rows),'verified':sum(by.get(row['path'],{}).get('archive_sha256')==row['sha256'] for row in rows)})
receipt={'status':'ready' if not failures else 'coverage_failure','parts':parts.parts,'counts':counts,'coverage':coverage,'failures':failures,'redacted_file_count':len(redacted),'exclusion_count':len(excluded),'free_gib':shutil.disk_usage(Q).free/2**30}
(OUT/'snapshot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
assert not failures, 'Protected evidence coverage failed; originals must remain'

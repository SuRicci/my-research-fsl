"""Verify downloaded archive assets and every snapshot file without extracting."""
from pathlib import Path, PurePosixPath
import sys,json,hashlib,tarfile,io,gzip
ASSETS=Path(sys.argv[1]) if len(sys.argv)>1 else Path('.')
META=Path(__file__).resolve().parent
manifest=json.loads((META/'assets.json').read_text())
def digest_stream(f):
    h=hashlib.sha256()
    while b:=f.read(2**20):h.update(b)
    return h.hexdigest()
for row in manifest['assets']:
    p=ASSETS/row['file'];assert p.stat().st_size==row['bytes'],p
    with p.open('rb') as f:assert digest_stream(f)==row['sha256'],p
    print('Verified asset:',p.name,flush=True)
class Joined(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.f=None
    def readable(self):return True
    def readinto(self,b):
        while True:
            if self.f is None:
                try:self.f=next(self.paths).open('rb')
                except StopIteration:return 0
            n=self.f.readinto(b)
            if n:return n
            self.f.close();self.f=None
    def close(self):
        if self.f:self.f.close()
        super().close()
parts=[ASSETS/r['file'] for r in manifest['assets'] if r['role']=='snapshot_part']
seen={};embedded=None
with io.BufferedReader(Joined(parts)) as f:
    with tarfile.open(fileobj=f,mode='r|gz') as t:
        for m in t:
            pp=PurePosixPath(m.name);assert not pp.is_absolute() and '..' not in pp.parts,m.name
            if m.name=='ARCHIVE_MANIFEST.json':embedded=json.load(t.extractfile(m));continue
            assert m.name.startswith('quest-012/'),m.name
            if m.isfile():seen[m.name]={'type':'file','sha256':digest_stream(t.extractfile(m))}
            elif m.islnk():
                assert m.linkname in seen and seen[m.linkname]['type']=='file',m.linkname
                seen[m.name]=dict(seen[m.linkname])
            elif m.issym():seen[m.name]={'type':'symlink','target':m.linkname}
            else:raise AssertionError('Unexpected archive member: '+m.name)
assert embedded is not None
assert len(seen)==len(embedded['files']),(len(seen),len(embedded['files']))
for r in embedded['files']:
    item=seen['quest-012/'+r['path']];assert item['type']==r['type'],r['path']
    if r['type']=='file':assert item['sha256']==r['archive_sha256'],r['path']
    else:assert item['target']==r['target'],r['path']
report={'status':'passed','asset_count':len(manifest['assets']),'snapshot_members':len(seen),'snapshot_files':sum(x['type']=='file' for x in seen.values()),'snapshot_symlinks':sum(x['type']=='symlink' for x in seen.values()),'all_asset_sha256_passed':True,'all_embedded_file_hashes_passed':True}
(ASSETS/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

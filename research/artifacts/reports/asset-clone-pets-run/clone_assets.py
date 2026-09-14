from pathlib import Path
import ctypes,os,json,hashlib,shutil,time
root=Path('/Users/decoqwq/DeepScientist/quests/012')
folder=Path(__file__).resolve().parent
pairs=json.loads((folder/'candidate_manifest.json').read_text())
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
clone=lib.clonefile;clone.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];clone.restype=ctypes.c_int
before=shutil.disk_usage(root).free;started=time.time();done=[];source_hash={}
for i,item in enumerate(pairs):
 src=Path(item['source']);dst=Path(item['copy']);temp=dst.with_name(dst.name+'.ds-cow-tmp')
 assert not temp.exists()
 if str(src) not in source_hash:source_hash[str(src)]=hashlib.sha256(src.read_bytes()).hexdigest()
 assert source_hash[str(src)]==item['sha256']==hashlib.sha256(dst.read_bytes()).hexdigest()
 st=dst.stat()
 if clone(os.fsencode(src),os.fsencode(temp),0)!=0:raise OSError(ctypes.get_errno(),str(temp))
 try:
  assert hashlib.sha256(temp.read_bytes()).hexdigest()==item['sha256']
  shutil.copystat(dst,temp)
  assert temp.stat().st_ino!=src.stat().st_ino
  os.replace(temp,dst)
  assert dst.stat().st_size==st.st_size
  done.append(item)
 finally:
  if temp.exists():temp.unlink()
 if (i+1)%1000==0:print('CLONED',i+1,'/',len(pairs),'FREE_GiB',shutil.disk_usage(root).free/2**30,flush=True)
after=shutil.disk_usage(root).free
summary={'status':'completed','file_count':len(done),'logical_bytes':sum(x['size'] for x in done),'free_before_bytes':before,'free_after_bytes':after,'observed_free_change_bytes':after-before,'elapsed_seconds':time.time()-started,'mechanism':'macOS clonefile, independent inodes and copy-on-write; SHA256 identity checked before replacement','scope':'Only quest-created exact duplicate baseline files; source files untouched. Available-space delta also reflects concurrent OS activity.'}
(folder/'completion.json').write_text(json.dumps(summary,indent=2))
print('COMPLETE',json.dumps(summary),flush=True)

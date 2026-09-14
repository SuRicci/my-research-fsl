from pathlib import Path
import os,json,hashlib,shutil,datetime,collections
q=Path('/Users/decoqwq/DeepScientist/quests/012');w=Path.cwd();o=w/'artifacts/reports/pause_20260914';a=json.loads((o/'protected_results_before.json').read_text());groups=collections.defaultdict(list)
for r in a:
 if Path(r['path']).suffix in {'.npz','.npy'}:groups[(r['bytes'],r['sha256'])].append(r)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(2**20),b''):h.update(c)
 return h.hexdigest()
before=shutil.disk_usage(q).free;changed=[];already=0
for key,rows in groups.items():
 if len(rows)<2:continue
 rows.sort(key=lambda r:(not Path(r['path']).is_relative_to(w),len(r['path']),r['path']));canon=Path(rows[0]['path']);assert digest(canon)==key[1]
 for row in rows[1:]:
  p=Path(row['path']);assert p.resolve().is_relative_to(q) and not p.is_symlink();assert p.stat().st_dev==canon.stat().st_dev
  if os.path.samefile(canon,p):already+=1;continue
  assert digest(p)==key[1];temp=p.with_name(p.name+'.pause-dedup-link');assert not temp.exists()
  os.link(canon,temp);os.replace(temp,p);assert os.path.samefile(canon,p)
  changed.append({'path':str(p),'canonical':str(canon),'bytes':key[0],'sha256':key[1]})
  if len(changed)%500==0:print('DEDUP_LINKS',len(changed),flush=True)
cache={};verified=0
for row in a:
 p=Path(row['path']);st=p.stat();k=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 if k not in cache:cache[k]=digest(p)
 assert cache[k]==row['sha256'],str(p);verified+=1
r={'status':'completed','completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'only byte-identical saved experiment npz/npy result arrays; no code/config/runtime files','links_created':len(changed),'already_shared_paths':already,'logical_duplicate_bytes_consolidated':sum(x['bytes'] for x in changed),'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(q).free,'protected_files_reverified':verified,'hash_mismatches':0,'changes':changed,'resume_note':'Saved result arrays are immutable shared hard links. To modify an old result path deliberately, first copy it to a new file; normal new-run outputs use new paths.'};r['observed_free_delta_bytes']=r['free_after_bytes']-before
(o/'result_dedup_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('DEDUP_COMPLETE',json.dumps({k:v for k,v in r.items() if k!='changes'}),flush=True)

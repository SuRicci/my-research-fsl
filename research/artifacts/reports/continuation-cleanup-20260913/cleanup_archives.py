from pathlib import Path
import hashlib, json, tarfile, zipfile, shutil, datetime
Q=Path('/Users/decoqwq/DeepScientist/quests/012')
OUT=Path(__file__).resolve().parent
P=Q/'.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/assets'
JOBS=[(Q/'tmp/dtd-r1.0.1.tar.gz',Q/'baselines/local/r2-replay/data'),(Q/'tmp/EuroSAT_RGB.zip',Q/'baselines/local/r2-replay/data'),(Q/'tmp/dtd-r1.0.1-labels.tar.gz',Q/'baselines/local/r2-replay/data/dtd'),(P/'downloads/images.tar.gz',P/'data'),(P/'downloads/annotations.tar.gz',P/'data')]
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
report={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'free_before_bytes':shutil.disk_usage(Q).free,'scope':'Five explicitly enumerated redundant archives only; extracted images, labels, features, outputs, environments and Git history retained.','archives':[]}
for archive,base in JOBS:
 assert archive.is_file() and not archive.is_symlink() and Q in archive.parents
 row={'archive':str(archive),'extracted_root':str(base),'bytes':archive.stat().st_size,'regular_files':0,'failures':[],'verified_members':[]}
 with archive.open('rb') as f:row['archive_sha256']=digest(f)
 iszip=archive.suffix=='.zip'
 with (zipfile.ZipFile(archive) if iszip else tarfile.open(archive,'r|gz')) as z:
  entries=z.infolist() if iszip else z
  for member in entries:
   if (member.is_dir() if iszip else member.isdir()):continue
   name=member.filename if iszip else member.name
   if not iszip and not member.isfile():row['failures'].append({'name':name,'reason':'nonregular entry'});continue
   row['regular_files']+=1
   target=base/name
   if '..' in Path(name).parts or Path(name).is_absolute() or not target.is_file() or target.is_symlink():row['failures'].append({'name':name,'reason':'missing or unsafe extracted member'});continue
   with (z.open(member) if iszip else z.extractfile(member)) as f:a=digest(f)
   with target.open('rb') as f:b=digest(f)
   if a!=b:row['failures'].append({'name':name,'reason':'content mismatch'})
   else:row['verified_members'].append({'name':name,'sha256':b})
 row['status']='retained' if row['failures'] else 'verified_redundant'
 report['archives'].append(row)
 (OUT/'archive_verification.json').write_text(json.dumps(report,indent=2))
 if not row['failures']:
  archive.unlink();row['status']='removed'
 print(json.dumps({k:v for k,v in row.items() if k!='verified_members'}),flush=True)
 report['free_after_bytes']=shutil.disk_usage(Q).free
 report['removed_logical_bytes']=sum(x['bytes'] for x in report['archives'] if x['status']=='removed')
 (OUT/'archive_verification.json').write_text(json.dumps(report,indent=2))
report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUT/'archive_verification.json').write_text(json.dumps(report,indent=2))
print('COMPLETE',report['removed_logical_bytes'],report['free_after_bytes']/2**30,flush=True)

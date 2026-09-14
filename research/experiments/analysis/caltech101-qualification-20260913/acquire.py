from pathlib import Path
import datetime,hashlib,json,shutil,time,urllib.request,zipfile
HERE=Path(__file__).resolve().parent; C=json.loads((HERE/'CONTRACT.json').read_text()); p=Path(C['archive_path']); p.parent.mkdir(parents=True,exist_ok=True)
assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,9,15,9,tzinfo=datetime.timezone.utc)
assert shutil.disk_usage(p.parent).free/2**30>=C['free_floor_gib']+C['acquisition_headroom_gib']
start=time.time(); downloaded=0
if not p.exists():
    part=p.with_suffix('.partial');assert not part.exists(),'Inspect existing partial before retry'
    req=urllib.request.Request(C['url'],headers={'User-Agent':'DeepScientist research data audit'})
    with urllib.request.urlopen(req,timeout=C['download_timeout_seconds']) as response,part.open('wb') as f:
        assert int(response.headers.get('Content-Length',C['expected_bytes']))==C['expected_bytes']
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            f.write(chunk);downloaded+=len(chunk)
            assert downloaded<=C['archive_limit_mib']*2**20
            assert shutil.disk_usage(p.parent).free/2**30>=C['free_floor_gib']
    assert part.stat().st_size==C['expected_bytes']
    assert hashlib.md5(part.read_bytes()).hexdigest()==C['expected_md5']
    part.rename(p)
assert p.stat().st_size==C['expected_bytes']
assert hashlib.md5(p.read_bytes()).hexdigest()==C['expected_md5']
with zipfile.ZipFile(p) as z:
    members=[]
    for info in z.infolist():
        name=Path(info.filename);assert not name.is_absolute() and '..' not in name.parts
        members.append({'name':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size})
    assert z.testzip() is None
result={'status':'passed','archive_path':str(p),'bytes':p.stat().st_size,'md5':C['expected_md5'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'archive_crc_passed':True,'members':members,'member_count':len(members),'uncompressed_member_bytes':sum(x['bytes'] for x in members),'downloaded_bytes':downloaded,'elapsed_seconds':time.time()-start,'free_gib':shutil.disk_usage(p.parent).free/2**30,'no_target_accuracy_computed':True}
(HERE/'acquisition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

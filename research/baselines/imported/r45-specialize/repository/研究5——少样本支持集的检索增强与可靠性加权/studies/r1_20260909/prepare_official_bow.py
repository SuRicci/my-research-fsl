"""Fetch the publisher's archival dataset; never overwrite pilot assets."""
import json, time, zipfile, hashlib, re
from pathlib import Path
import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).resolve().parent
ASSET = OUT/'assets'/'bow_official'
SOURCE = OUT/'sources'

def main():
    ASSET.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(exist_ok=True)
    s = requests.Session()
    api = 'https://api.github.com/repos/rujiewu/Bongard-OpenWorld/git/trees/main?recursive=1'
    r=s.get(api,timeout=40);r.raise_for_status()
    tree=r.json(); (SOURCE/'github_tree.json').write_text(json.dumps(tree),encoding='utf-8')
    paths=[x['path'] for x in tree['tree'] if x['path'].endswith('.py') and ('dataset' in x['path'] or 'data_loader' in x['path'])]
    paths += ['README.md'] + [x['path'] for x in tree['tree'] if re.search('bongard_ow_(train|val|test)\\.json$',x['path'])]
    for path in paths:
        r=s.get('https://raw.githubusercontent.com/rujiewu/Bongard-OpenWorld/main/'+path, timeout=40);r.raise_for_status()
        (SOURCE/path.replace('/','__')).write_bytes(r.content)
    print('source files', paths, flush=True)
    target=ASSET/'images.zip'
    if not target.exists():
        r=s.get('https://drive.google.com/uc?export=download&id=1aXr3ihVq0mtzbl6ZNJMogYEyEY-WALNr',timeout=40)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,'html.parser');form=soup.find('form',id='download-form')
        if form is None: raise RuntimeError('Drive returned no download form')
        params={i['name']:i.get('value','') for i in form.find_all('input') if i.get('name')}
        t=time.time(); total=0;last=0
        with s.get(form['action'],params=params,stream=True,timeout=(30,60)) as data:
            data.raise_for_status()
            if 'text/html' in data.headers.get('Content-Type',''): raise RuntimeError('Drive download is HTML')
            with target.with_suffix('.part').open('wb') as f:
                for chunk in data.iter_content(8*1024*1024):
                    f.write(chunk); total+=len(chunk)
                    if time.time()-last>30:
                        print('download MB',round(total/1e6),'seconds',round(time.time()-t),flush=True);last=time.time()
        target.with_suffix('.part').replace(target)
    mapping={}; n=0
    with zipfile.ZipFile(target) as z:
        for info in z.infolist():
            parts=info.filename.split('/')
            if len(parts)<2 or not re.match(r'(pos|neg)__',parts[-1]):continue
            uid=parts[-2]; side_idx='_'.join(parts[-1].split('__')[:2])
            dest=ASSET/'images'/uid/(side_idx+'.img')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(z.read(info))
            mapping[uid+'/'+side_idx]=str(dest.relative_to(ASSET));n+=1
    (ASSET/'map.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print('official extracted',n,flush=True)

if __name__=='__main__':main()

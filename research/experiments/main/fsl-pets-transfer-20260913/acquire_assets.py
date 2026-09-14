"""Official Pets acquisition, identity audit and canonical frozen encoding."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib, json, os, shutil, sys, tarfile, time
import requests
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'protocol.json').read_text())
ROOT = Path('/Users/decoqwq/DeepScientist/quests/012')
sys.path.insert(0, str(ROOT / 'baselines/local/r2-replay'))
import recover_gallery as encoder
ASSETS = HERE / 'assets'
RESOURCES = [('images.tar.gz', '5c4f3ee8e5d25df40f4fd59a7f44e54c'),
             ('annotations.tar.gz', '95a8c909bbe2e81eed6a22bccdf3f68f')]

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)

def guard():
    assert shutil.disk_usage(HERE).free >= 10 * 2**30, 'Disk reserve below 10 GiB'
    assert datetime.now(timezone.utc) < datetime.fromisoformat(CFG['resources']['deadline_utc']), 'Round deadline reached'

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()

def acquire():
    sources = []
    for name, md5 in RESOURCES:
        guard()
        target = ASSETS / 'downloads' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/' + name
        if not target.exists():
            temp = target.with_suffix('.part')
            with requests.get(url, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                total = int(response.headers.get('Content-Length', 0))
                assert total < 1024**3, 'Unexpected source size'
                n = 0
                with temp.open('wb') as out:
                    for chunk in response.iter_content(2**20):
                        guard()
                        out.write(chunk)
                        n += len(chunk)
                        assert n <= 1024**3
                        if n // (64 * 2**20) != (n - len(chunk)) // (64 * 2**20):
                            print('DOWNLOAD', name, n, total, flush=True)
                if total:
                    assert n == total
            assert digest(temp, 'md5') == md5, 'Official checksum mismatch'
            os.replace(temp, target)
        assert digest(target, 'md5') == md5
        sources.append({'url': url, 'path': str(target), 'bytes': target.stat().st_size,
                        'md5': md5, 'sha256': digest(target),
                        'checksum_source': 'installed torchvision OxfordIIITPet._RESOURCES'})
        with tarfile.open(target) as archive:
            for member in archive:
                keep = member.name.lower().endswith('.jpg') if name == 'images.tar.gz' else member.name in ['annotations/trainval.txt', 'annotations/test.txt', 'annotations/README']
                if not member.isfile() or not keep:
                    continue
                rel = Path(member.name)
                assert not rel.is_absolute() and '..' not in rel.parts
                path = ASSETS / 'data' / rel
                if path.exists() and path.stat().st_size == member.size:
                    continue
                guard()
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as inp, path.open('wb') as out:
                    shutil.copyfileobj(inp, out)
        print('SOURCE_VERIFIED', name, flush=True)
    dump(ASSETS / 'source_manifest.json', {'sources': sources, 'license': 'CC BY-SA 4.0 per Oxford official page',
         'page': 'https://www.robots.ox.ac.uk/~vgg/data/pets/', 'no_redistribution': True})

def identities():
    splits = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for split in ['test', 'trainval']:
            rows = [line.split() for line in (ASSETS / 'data/annotations' / (split + '.txt')).read_text().splitlines() if line and not line.startswith('#')]
            paths = [ASSETS / 'data/images' / (row[0] + '.jpg') for row in rows]
            hashes = list(pool.map(encoder.rgb, paths))
            splits[split] = [{'name': row[0], 'label': int(row[1]) - 1, 'rgb': h, 'path': str(p)} for row, h, p in zip(rows, hashes, paths)]
    query = []; seen = set()
    for row in splits['test']:
        if row['rgb'] not in seen:
            seen.add(row['rgb']); query.append(row)
    test_hashes = set(seen); gallery = []; seen = set(test_hashes)
    for row in splits['trainval']:
        if row['rgb'] not in seen:
            seen.add(row['rgb']); gallery.append(row)
    rng = np.random.RandomState(CFG['gallery_selection_seed'])
    gallery = [gallery[i] for i in rng.permutation(len(gallery))[:CFG['gallery_size']]]
    assert len(gallery) == CFG['gallery_size']
    labels, counts = np.unique([row['label'] for row in query], return_counts=True)
    assert len(labels) == 37 and counts.min() >= 20
    assert not set(r['rgb'] for r in query) & set(r['rgb'] for r in gallery)
    previous = set()
    for folder in [ROOT / 'baselines/local/r2-canonical/assets', ROOT / 'baselines/local/r2-domains/assets']:
        for path in folder.glob('*identities.json'):
            d = json.loads(path.read_text())
            for key, values in d.items():
                if 'rgb' in key and isinstance(values, list):
                    previous.update(values)
    overlap = (set(r['rgb'] for r in query) | set(r['rgb'] for r in gallery)) & previous
    assert not overlap, 'Pets identities overlap an existing canonical pool'
    value = {'query': query, 'gallery': gallery, 'query_total': len(splits['test']),
             'gallery_total': len(splits['trainval']), 'query_unique': len(query),
             'gallery_selected': len(gallery), 'class_counts': dict(zip(map(str, labels), map(int, counts))),
             'query_gallery_rgb_overlap': 0, 'known_prior_rgb_count': len(previous), 'known_prior_rgb_overlap': len(overlap),
             'selection_uses_gallery_labels': False, 'scope': 'not a claim about pretrained encoder corpora'}
    dump(ASSETS / 'identities.json', value)
    print('IDENTITIES_VERIFIED', len(query), len(gallery), 'prior RGB hashes', len(previous), flush=True)
    return value

def encode(ident):
    assert torch.backends.mps.is_available(), 'MPS required for this selected local encoding path'
    manifest = {'variant': 'canonical-openai-quickgelu-native-dinov2', 'precision': 'float32 MPS, normalized float16 storage',
                'encoder_source_sha256': digest(Path(encoder.__file__)), 'protocol_sha256': digest(HERE / 'protocol.json'),
                'identity_sha256': digest(ASSETS / 'identities.json'), 'torch_version': torch.__version__, 'files': {}}
    assert manifest['encoder_source_sha256'] == CFG['sources_sha256'][str(Path(encoder.__file__))]
    for index, backbone in enumerate(encoder.BACKBONES):
        model = encoder.model(backbone)
        tf = encoder.T.Compose([encoder.T.Resize(224, interpolation=encoder.T.InterpolationMode.BICUBIC), encoder.T.CenterCrop(224), encoder.T.ToTensor(), encoder.T.Normalize(*encoder.NORM[index])])
        with ThreadPoolExecutor(max_workers=6) as pool:
            for split in ['query', 'gallery']:
                rows = ident[split]; chunks = []
                for start in range(0, len(rows), 32):
                    guard()
                    target = ASSETS / 'chunks' / backbone / split / ('%06d.pt' % start)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        features = torch.load(target, weights_only=True)
                    else:
                        features = encoder.encode(model, backbone, [row['path'] for row in rows[start:start+32]], tf, pool).half()
                        tmp = target.with_suffix('.tmp'); torch.save(features, tmp); os.replace(tmp, target)
                    assert features.shape[0] == min(32, len(rows) - start) and torch.isfinite(features).all()
                    chunks.append(features)
                    if start % 320 == 0:
                        print('ENCODE', backbone, split, start, len(rows), flush=True)
                full = torch.cat(chunks)
                assert float((full.float().norm(dim=1) - 1).abs().max()) < .002
                path = ASSETS / 'features' / (backbone + '_' + split + '.pt')
                path.parent.mkdir(exist_ok=True)
                torch.save({'features': full, 'ids': torch.arange(len(rows))}, path)
                manifest['files'][backbone + '_' + split] = {'path': str(path), 'sha256': digest(path), 'shape': list(full.shape)}
                dump(ASSETS / 'feature_manifest.json', manifest)
        del model
        torch.mps.empty_cache()
    return manifest

if __name__ == '__main__':
    start = time.time()
    try:
        acquire()
        ident = identities()
        encode(ident)
        added = sum(p.stat().st_size for p in ASSETS.rglob('*') if p.is_file())
        assert added <= CFG['resources']['added_gib_max'] * 2**30
        dump(HERE / 'outputs/assets_complete.json', {'status': 'success', 'elapsed_seconds': time.time()-start, 'asset_bytes': added, 'free_gib': shutil.disk_usage(HERE).free/2**30})
        print('ASSETS_COMPLETE', flush=True)
    except Exception as error:
        dump(HERE / 'outputs/assets_blocker.json', {'error': repr(error), 'time': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.time()-start})
        raise

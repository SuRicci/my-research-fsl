"""Restore only the existing v0/r224 query tensors; never re-encode or relabel them."""
import io, json, hashlib, tarfile, time
from pathlib import Path
from collections import defaultdict
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'baselines/local/r2-replay/assets'
ARCHIVE = Path('/Users/decoqwq/Desktop/learning/科研/individual research/.scratch/takeover_20260907_1443/backups/r3_cache.tar.gz')
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
features, hashes = defaultdict(dict), defaultdict(dict)
started = time.time()
with tarfile.open(ARCHIVE, 'r|gz') as archive:
    for member in archive:
        if '/cache/cls/' not in member.name or not member.name.endswith('__v0__r224.pt'):
            continue
        scope, backbone, filename = member.name.split('/')[-3:]
        key = (scope, backbone)
        index = int(filename.split('#')[1].split('__')[0])
        if index in features[key]:
            raise ValueError('duplicate feature index')
        raw = archive.extractfile(member).read()
        tensor = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        assert tensor.ndim == 1 and torch.isfinite(tensor).all()
        assert len(tensor) == (512 if backbone == 'clip_vitb16' else 384)
        features[key][index] = tensor
        hashes[key][index] = hashlib.sha256(raw).hexdigest()
        if sum(len(v) for v in features.values()) % 10000 == 0:
            print('RESTORED_ROWS', sum(len(v) for v in features.values()), flush=True)
manifest = []
for (scope, backbone), values in features.items():
    ids = sorted(values)
    assert ids == list(range(len(ids)))
    target = OUT / (scope + '_' + backbone + '_query.pt')
    torch.save({'ids': torch.tensor(ids), 'features': torch.stack([values[i] for i in ids])}, target)
    source_index = ROOT / 'tmp/cache-index' / scope / backbone / 'index.json'
    index = json.loads(source_index.read_text())
    for i in ids:
        assert index[backbone + '|' + scope + '#' + str(i) + '|v0|r224']['shape'] == list(values[i].shape)
    manifest.append({'scope': scope, 'backbone': backbone, 'rows': len(ids), 'path': str(target),
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                     'index_sha256': hashlib.sha256(source_index.read_bytes()).hexdigest(),
                     'member_sha256': hashes[(scope, backbone)]})
(OUT / 'query_cache_manifest.json').write_text(json.dumps({'archive': str(ARCHIVE), 'view': 'v0', 'resolution': 224,
    'elapsed_seconds': time.time()-started, 'items': manifest}, indent=2))
print('COMPLETE', [(x['scope'], x['backbone'], x['rows']) for x in manifest], flush=True)

"""Package completed evidence and verify every archive member without extraction."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path('/data/liuhaoyu/individual-research')
OUT = Path('/data/liuhaoyu/r45-specialize-20260909')
BACKUP = Path('/data/liuhaoyu/backups/r45-specialize-20260909')
ACTIVE = Path('lab/specialize_20260909')


def sha_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return sha_stream(stream)


def verify(path):
    path = Path(path)
    with tarfile.open(path, 'r:*') as archive:
        manifest = json.load(archive.extractfile('MANIFEST.json'))
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members), 'duplicate members'
        assert {m.name for m in members} == set(manifest['files']) | {'MANIFEST.json'}
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute()
            assert '..' not in Path(member.name).parts
            if member.name == 'MANIFEST.json':
                continue
            expected = manifest['files'][member.name]
            assert member.size == expected['bytes'], member.name
            assert sha_stream(archive.extractfile(member)) == expected['sha256'], member.name
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                business_files=len(manifest['files']), tar_members=len(members),
                member_hashes_verified=True, verified_at=time.time())


def build():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    audit = json.loads((OUT / 'audit.json').read_text())
    assert audit['complete'] and audit['candidates'] == 10
    assert json.loads((OUT / 'state.json').read_text())['status'] == 'all_complete'
    assert (OUT / 'controller_exit_code').read_text().strip() == '0'
    for name, expected in protocol['sources'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in audit['summary_hashes'].items():
        assert sha(OUT / name / 'summary.json') == expected, name
    analysis = json.loads((OUT / 'analysis_protocol.json').read_text())
    for name, expected in analysis['sources'].items():
        assert sha(name) == expected, name

    BACKUP.mkdir(parents=True, exist_ok=False)
    dependencies = {
        'scope': 'Complete new campaign ranks, predictions, scores, logs, summaries, reports and source snapshots. Historical data, models, feature caches and parent-run NPZ files are external dependencies, not bundled.',
        'baseline_commit': protocol['baseline_commit'],
        'checkout_at_archival': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'frozen_source_validation': True,
        'python': subprocess.check_output(['/data/liuhaoyu/.conda/envs/torch/bin/python', '--version'], text=True).strip(),
        'external_run_roots': ['/data/liuhaoyu/r45-scale-20260909', '/data/liuhaoyu/r45-refine-20260909', '/data/liuhaoyu/r45-followup-20260909'],
        'restoration': 'repository/ contains source snapshots and new outputs under lab/specialize_20260909/results/. Restore the source tree at the recorded ROOT and the results directory at OUT, and supply external dependencies at their original paths before running audit scripts. The archive alone is not a data-complete fresh training environment.',
        'manifest_rule': 'MANIFEST.json lists business files only and does not hash itself. Whole archive SHA256 is recorded externally.',
        'created_at': time.time(),
    }
    dep = BACKUP / 'DEPENDENCIES.json'
    dep.write_text(json.dumps(dependencies, ensure_ascii=False, indent=2))
    files = {'DEPENDENCIES.json': dep}
    for p in OUT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            files[str(Path('repository') / ACTIVE / 'results' / p.relative_to(OUT))] = p

    sources = {ROOT / p for p in protocol['sources']}
    sources.update(p for p in (ROOT / ACTIVE).rglob('*') if p.is_file()
                   and 'results' not in p.relative_to(ROOT / ACTIVE).parts
                   and '__pycache__' not in p.parts and p.name != 'CHECKLIST.md')
    for relative in ['research_common', '研究3——少样本学习测试时计算扩展/src',
                     '研究5——少样本支持集的检索增强与可靠性加权']:
        sources.update(p for p in (ROOT / relative).rglob('*.py') if '__pycache__' not in p.parts)
    for p in sources:
        files[str(Path('repository') / p.relative_to(ROOT))] = p

    details = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in sorted(files.items())}
    receipts = {}
    for label in ['full', 'core']:
        selected = files if label == 'full' else {n: p for n, p in files.items()
                    if p.suffix not in {'.npz', '.npy', '.pt', '.pth'} and p.stat().st_size < 20 * 1024 * 1024}
        manifest = dict(kind=label, files={n: details[n] for n in sorted(selected)},
                        scope=dependencies['scope'], created_at=time.time())
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        path = BACKUP / ('r45-specialize-' + label + ('.tar' if label == 'full' else '.tar.gz'))
        with tarfile.open(path, 'w' if label == 'full' else 'w:gz') as archive:
            for name, p in sorted(selected.items()):
                archive.add(p, arcname=name, recursive=False)
            member = tarfile.TarInfo('MANIFEST.json')
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        receipts[label] = verify(path)
        print(json.dumps({label: receipts[label]}), flush=True)
    receipt = dict(server='222.20.99.52', archives=receipts, dependencies=dependencies)
    (BACKUP / 'server_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), ensure_ascii=False, indent=2))
    else:
        build()

# -*- coding: utf-8 -*-
"""Accept only recorded path-only migrations; retain original selection snapshots."""
import json,hashlib
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'research_directions.json').is_file())
def verify_artifact(path,expected):
    path=Path(path);actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual==expected:return
    record=json.loads((ROOT/'maintenance/20260909_reorganize/path_edits.json').read_text(encoding='utf-8'))
    rel=path.resolve().relative_to(ROOT).as_posix()
    row=next((r for r in record['edits'] if r['after_path']==rel),None)
    assert row and row['before_sha256']==expected and row['after_sha256']==actual,rel
    original=ROOT/row['original_copy']
    assert hashlib.sha256(original.read_bytes()).hexdigest()==expected,rel

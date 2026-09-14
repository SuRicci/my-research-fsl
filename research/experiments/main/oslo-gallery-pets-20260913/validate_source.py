"""Numerical parity to pinned author class and inductive prediction invariants."""
from pathlib import Path
from collections import defaultdict
from typing import Tuple
import ast, hashlib, json, platform, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sklearn.metrics import precision_recall_curve, auc
import sklearn
import oslo

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / 'literature/oslo_source'

def prototypes(x, y):
    return torch.stack([x[y == label].mean(0) for label in y.unique(sorted=True)])


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
    tree = ast.parse((SOURCE / 'src/all_in_one/osem.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    env = dict(torch=torch, F=F, Tensor=Tensor, Tuple=Tuple, AllInOne=object,
               compute_prototypes=prototypes, precision_recall_curve=precision_recall_curve, auc_fn=auc)
    exec(compile(tree, 'pinned_author_osem.py', 'exec'), env)
    torch.set_num_threads(2); torch.manual_seed(26091330)
    rows = []
    for shot in [1, 5]:
        S = F.normalize(torch.randn(2, 5, shot, 24), dim=-1)
        G = F.normalize(torch.randn(50, 24), dim=-1)
        Q = F.normalize(torch.randn(2, 17, 24), dim=-1)
        for steps, enabled in [(2, True), (2, False), (0, True)]:
            state, stats = oslo.fit(S, G, steps=steps, use_inlier=enabled)
            for b in range(2):
                reference = env['OSEM'](steps, .05, .1, 1., enabled)
                origin = torch.cat([S[b].flatten(0, 1), G]).mean(0, keepdim=True)
                rs = F.normalize(S[b].flatten(0, 1) - origin, dim=-1)
                rg = F.normalize(G - origin, dim=-1)
                labels = torch.arange(5).repeat_interleave(shot)
                saved = []
                def logits(p, x):
                    saved.append(p.clone())
                    return reference.cosine(x, p)
                reference.get_logits = logits
                outliers = torch.arange(len(G)) % 2 == 0
                _, probs, _ = reference(rs, rg, labels, outliers=outliers,
                    query_labels=torch.arange(len(G)) % 5,
                    intra_task_metrics={'main_metrics': defaultdict(list), 'secondary_metrics': defaultdict(list)})
                pred = oslo.predict((state[0][b:b+1], state[1][b:b+1]), G[None])[0].softmax(-1)
                error = float((probs - pred).abs().max())
                proto_error = float((F.normalize(saved[-1], dim=-1) - F.normalize(state[1][b], dim=-1)).abs().max())
                assert error < 2e-6 and proto_error < 2e-6, (shot, steps, enabled, error, proto_error)
                rows.append({'shot': shot, 'steps': steps, 'inlier': enabled, 'batch_index': b, 'probability_error': error, 'prototype_error': proto_error})
            expected = oslo.predict(state, Q)
            permutation = torch.randperm(Q.shape[1])
            assert torch.allclose(oslo.predict(state, Q[:, permutation]), expected[:, permutation], atol=2e-6, rtol=0)
            assert torch.allclose(torch.cat([oslo.predict(state, Q[:, :7]), oslo.predict(state, Q[:, 7:])], 1), expected, atol=2e-6, rtol=0)
            extra = torch.randn(2, 9, 24)
            assert torch.allclose(oslo.predict(state, torch.cat([Q, extra], 1))[:, :17], expected, atol=2e-6, rtol=0)
            assert torch.isfinite(expected).all() and torch.isfinite(stats['effective_mass']).all()
    result = {'status': 'passed', 'source_commit': manifest['commit'], 'source_cases': rows,
        'query_permutation_split_insertion': 'passed', 'fit_arguments': ['support', 'gallery', 'fixed_hyperparameters'],
        'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'platform': platform.platform(), 'mps_available': torch.backends.mps.is_available()},
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'oslo.py', Path(__file__), HERE/'protocol.json']}}
    (HERE/'outputs').mkdir(exist_ok=True)
    (HERE/'outputs/source_validation.json').write_text(json.dumps(result, indent=2))
    print('SOURCE_AND_INDUCTIVE_VALIDATION_PASSED',len(rows),flush=True)

if __name__ == '__main__':
    main()

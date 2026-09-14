import numpy as np
import torch
from r4_refine import candidate_scores, cover_select
from r5_methods import menu, candidates, baseline, gallery_reference
from r4_geometry import corrected


def test_r4_anchor_order_query_independence_and_identity():
    rng = np.random.default_rng(13001)
    go, qo, qn, ao, an = [rng.normal(size=s) for s in [(30, 12), (3, 12), (3, 9), (32, 12), (32, 9)]]
    v, _ = candidate_scores(go, qo, qn, ao, an)
    order = rng.permutation(32)
    other, _ = candidate_scores(go, qo, qn, ao[order], an[order])
    single, _ = candidate_scores(go, qo[:1], qn[:1], ao, an)
    for name in v:
        np.testing.assert_allclose(v[name], other[name], atol=1e-8)
        np.testing.assert_allclose(v[name][:1], single[name], atol=1e-8)
    identity, _ = candidate_scores(go, qo, qo, ao, ao)
    old = (qo / np.linalg.norm(qo, axis=-1, keepdims=True)) @ (go / np.linalg.norm(go, axis=-1, keepdims=True)).T
    for name in ['linear_raw', 'rbf_raw']:
        np.testing.assert_allclose(identity[name], old, atol=1e-8)
    ids = cover_select(ao, go, 16)
    assert len(np.unique(ids)) == 16


def test_r4_geometry_identity_and_query_independence():
    rng = np.random.default_rng(13002)
    go, qo, ao = [rng.normal(size=s) for s in [(40, 12), (3, 12), (32, 12)]]
    identity = corrected(go, qo, qo, ao, ao)
    for name in ['centered_residual_0.5', 'centered_residual_1.0', 'centered_residual_2.0']:
        np.testing.assert_allclose(identity[name], identity['old_centered'], atol=1e-8)
    qn, an = rng.normal(size=(3, 9)), rng.normal(size=(32, 9))
    full = corrected(go, qo, qn, ao, an)
    single = corrected(go, qo[:1], qn[:1], ao, an)
    for name in full:
        np.testing.assert_allclose(full[name][:1], single[name], atol=1e-8)


def test_r5_queries_episodes_and_classes_are_independent():
    torch.manual_seed(13001)
    gc, gd = torch.randn(100, 18), torch.randn(100, 12)
    for shot in [1, 5]:
        sc, sd = torch.randn(2, 5, shot, 18), torch.randn(2, 5, shot, 12)
        xc, xd = torch.randn(2, 75, 18), torch.randn(2, 75, 12)
        cfg = menu(shot); thresholds = {.05: .1, .25: .2}
        full, _ = candidates(sc, sd, xc, xd, gc, gd, cfg, thresholds)
        one, _ = candidates(sc[:1], sd[:1], xc[:1, :1], xd[:1, :1], gc, gd, cfg, thresholds)
        perm = torch.tensor([2, 4, 0, 3, 1])
        reordered, _ = candidates(sc[:, perm], sd[:, perm], xc, xd, gc, gd, cfg, thresholds)
        for name in full:
            torch.testing.assert_close(full[name][:1, :1], one[name], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(full[name][:, :, perm], reordered[name], atol=2e-5, rtol=2e-5)
        b = baseline(sc, sd, xc, xd, gc, gd, shot)
        if shot == 5:
            c = next(c for c in cfg if c['family'] == 'center' and c['w'] == .5 and c['center'] == 0 and c['lam'] == 1.)
        else:
            c = next(c for c in cfg if c['family'] == 'constant' and c['mix'] == .5)
        torch.testing.assert_close(full[c['name']], b['r2_overall'], atol=2e-5, rtol=2e-5)

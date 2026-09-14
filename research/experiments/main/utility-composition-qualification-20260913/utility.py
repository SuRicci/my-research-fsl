"""Fixed-gallery composition; inference consumes features only."""
import torch
import torch.nn.functional as F


def prepare(S, G, retrieve):
    P = F.normalize(S.mean(2), dim=-1)
    idx = retrieve(P, G, 64)
    N = G[idx]
    cos = []
    winners = []
    for lo, hi in [(0, 512), (512, S.shape[-1])]:
        sim = F.normalize(S[..., lo:hi].mean(2), dim=-1) @ F.normalize(G[:, lo:hi], dim=-1).T
        cos.append(sim.gather(2, idx))
        winners.append(sim.argmax(1).gather(1, idx.flatten(1)).reshape_as(idx))
    sim = P @ G.T
    eye = torch.eye(5, dtype=torch.bool)[None, :, :, None]
    other = sim[:, None].expand(-1, 5, -1, -1).masked_fill(eye, -torch.inf).max(2).values
    margin = (sim - other).gather(2, idx)
    cls = torch.arange(5)[None, :, None]
    agree = ((winners[0] == cls) & (winners[1] == cls)).to(S.dtype)
    feat = torch.stack([*cos, margin, agree], -1)
    return P, N, feat, idx


def scores(cache, Q, coef, alpha_logit, ridge_scores, uniform=False):
    P, N, feat, _ = cache
    weights = torch.softmax(feat @ (5 * coef.tanh()), dim=-1)
    if uniform:
        weights = torch.ones_like(weights) / weights.shape[-1]
    M = F.normalize((N * weights[..., None]).sum(2), dim=-1)
    alpha = alpha_logit.sigmoid()
    A = F.normalize((1 - alpha) * P + alpha * M, dim=-1)
    Y = torch.eye(5, dtype=A.dtype)[None].expand(len(A), -1, -1)
    return ridge_scores(A, Y, Q, .1)


def validate(ref):
    torch.manual_seed(26091363)
    S = F.normalize(torch.randn(2, 5, 1, 896, dtype=torch.double), dim=-1)
    G = F.normalize(torch.randn(80, 896, dtype=torch.double), dim=-1)
    Q = F.normalize(torch.randn(2, 15, 896, dtype=torch.double), dim=-1)
    cache = prepare(S, G, ref.retrieve)
    c = torch.zeros(4, dtype=torch.double, requires_grad=True)
    a = torch.tensor(0., dtype=torch.double, requires_grad=True)
    v = scores(cache, Q, c, a, ref.ridge_scores)
    # R2 source is float32; compare within its numerical precision.
    r2 = ref.r2_scores(S.float(), Q.float(), G.float())
    parity = float((v - r2).abs().max())
    assert parity < 1e-6
    y = torch.arange(5).repeat_interleave(3)[None].expand(2, -1)
    loss = F.cross_entropy((10*v).flatten(0, 1), y.flatten())
    grad = torch.autograd.grad(loss, (c, a))
    finite = []
    eps = 1e-5
    with torch.no_grad():
        for k in range(5):
            values = []
            for sign in [1, -1]:
                cc = c.clone(); aa = a.clone()
                if k < 4: cc[k] += sign*eps
                else: aa += sign*eps
                z = scores(cache, Q, cc, aa, ref.ridge_scores)
                values.append(F.cross_entropy((10*z).flatten(0, 1), y.flatten()))
            finite.append((values[0]-values[1])/(2*eps))
        analytical = torch.cat([grad[0], grad[1][None]])
        err = float((torch.stack(finite)-analytical).abs().max())
        assert err < 1e-6
        split = torch.cat([scores(cache, q, c, a, ref.ridge_scores) for q in Q.split(4, dim=1)], 1)
        assert torch.allclose(split, v, atol=1e-10)
        perm = torch.tensor([3, 1, 4, 0, 2])
        vperm = scores(prepare(S[:, perm], G, ref.retrieve), Q, c, a, ref.ridge_scores)
        assert torch.allclose(vperm, v[..., perm], atol=1e-10)
        gp = torch.randperm(len(G))
        assert torch.allclose(scores(prepare(S, G[gp], ref.retrieve), Q, c, a, ref.ridge_scores), v, atol=1e-10)
        # Independent primal solution on a small synthetic problem checks the dual solver.
        X = torch.randn(2, 5, 7, dtype=torch.double); T = torch.eye(5, dtype=torch.double)[None].expand(2, -1, -1)
        Z = torch.randn(2, 9, 7, dtype=torch.double); xc=X-X.mean(1,keepdim=True);tc=T-T.mean(1,keepdim=True)
        w=torch.linalg.solve(xc.transpose(1,2)@xc+.1*torch.eye(7),xc.transpose(1,2)@tc)
        primal=(Z-X.mean(1,keepdim=True))@w+T.mean(1,keepdim=True)
        pe=float((primal-ref.ridge_scores(X,T,Z,.1)).abs().max());assert pe<1e-10
    return {'status':'passed','uniform_r2_error':parity,'gradient_error':err,'primal_dual_error':pe,
            'query_partition_invariant':True,'class_permutation_equivariant':True,'gallery_permutation_invariant':True}

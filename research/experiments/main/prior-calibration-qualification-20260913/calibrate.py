"""Source-calibrated relevance. Inference accepts features only."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ["clip_max_cosine", "clip_top2_gap", "dino_max_cosine", "dino_top2_gap",
            "encoder_argmax_agreement", "fused_max_cosine", "fused_top2_gap"]


def features(S, G):
    blocks, winners = [], []
    for lo, hi in [(0, 512), (512, S.shape[-1]), (0, S.shape[-1])]:
        P = F.normalize(S[..., lo:hi].mean(2), dim=-1)
        H = F.normalize(G[:, lo:hi], dim=-1)
        sim = (P @ H.T).transpose(1, 2)
        top = sim.topk(2, dim=-1)
        blocks.extend([top.values[..., 0], top.values[..., 0] - top.values[..., 1]])
        winners.append(top.indices[..., 0])
    blocks.insert(4, (winners[0] == winners[1]).float())
    return torch.stack(blocks, dim=-1).numpy()


def fit_calibrator(x, y, cfg):
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"],
        solver=cfg["solver"], max_iter=cfg["max_iter"], tol=cfg["tol"], random_state=0)
    model.fit(scaler.transform(x), y)
    assert int(model.n_iter_.max()) < cfg["max_iter"], "calibration failed to converge"
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                reference_prior=0.5, features=FEATURES,
                train_rows=int(len(y)), train_prevalence=float(np.mean(y)),
                iterations=int(model.n_iter_.max()))


def predict_calibrator(model, x):
    z = ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"])
    z += model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def adapt_prior(p, cfg):
    p = np.clip(p, cfg["probability_clip"], 1 - cfg["probability_clip"])
    pi = np.full((len(p), 1), cfg["em_init"], dtype=np.float64)
    for iteration in range(cfg["em_max_iter"]):
        post = pi * p / (pi * p + (1 - pi) * (1 - p))
        new = post.mean(1, keepdims=True)
        delta = np.abs(new - pi)
        pi = new
        if delta.max() < cfg["em_tolerance"]:
            break
    post = pi * p / (pi * p + (1 - pi) * (1 - p))
    return post, pi[:, 0], iteration + 1


def retrieval_scores(S, Q, G, posterior, ridge_scores, retrieve, mode="adaptive", mix=0.5, lam=0.1):
    P = F.normalize(S.mean(2), dim=-1)
    idx = retrieve(P, G, 64)
    neighbors = G[idx]
    w = torch.from_numpy(np.asarray(posterior, dtype=np.float32))
    w = w[torch.arange(len(S))[:, None, None], idx]
    mass = w.mean(2, keepdim=True)
    if mode == "same_mass":
        mu = F.normalize(neighbors.mean(2), dim=-1)
    else:
        weighted = (neighbors * w[..., None]).sum(2)
        fallback = neighbors.mean(2)
        weighted = torch.where(w.sum(2, keepdim=True) > 1e-12, weighted, fallback)
        mu = F.normalize(weighted, dim=-1)
    alpha = mix * mass if mode in ["adaptive", "same_mass"] else mix
    augmented = F.normalize((1 - alpha) * P + alpha * mu, dim=-1)
    return ridge_scores(augmented, torch.eye(5)[None].expand(len(S), -1, -1), Q, lam)


def self_check(cfg, ref):
    from scipy.optimize import minimize_scalar
    rng = np.random.RandomState(26091349)
    membership = rng.rand(20000) < 0.2
    x = rng.normal(np.where(membership, 1, -1), 1)
    prob = 1 / (1 + np.exp(-2 * x))
    post, pi, it = adapt_prior(prob[None], cfg)
    result = minimize_scalar(lambda a: -np.log(a * prob + (1-a) * (1-prob)).mean(),
                             bounds=(1e-6, 1-1e-6), method="bounded")
    assert abs(pi[0] - result.x) < 1e-5
    assert abs(pi[0] - 0.2) < 0.03
    torch.manual_seed(26091349)
    S = F.normalize(torch.randn(2, 5, 1, 896), dim=-1)
    Q = F.normalize(torch.randn(2, 75, 896), dim=-1)
    G = F.normalize(torch.randn(100, 896), dim=-1)
    ones = np.ones((2, 100))
    score = retrieval_scores(S, Q, G, ones, ref.ridge_scores, ref.retrieve)
    error = float((score - ref.r2_scores(S, Q, G)).abs().max())
    assert error < 1e-6, error
    split = torch.cat([retrieval_scores(S, part, G, ones, ref.ridge_scores, ref.retrieve)
                       for part in Q.split(17, dim=1)], dim=1)
    assert torch.allclose(score, split, atol=1e-6)
    feat = features(S, G)
    order = torch.tensor([3, 0, 4, 2, 1])
    assert np.allclose(feat, features(S[:, order], G), atol=1e-6)
    return dict(status="passed", em_estimate=float(pi[0]), mle_estimate=float(result.x),
                planted_prior=0.2, em_iterations=it, r2_max_error=error,
                query_partition_invariant=True, class_permutation_invariant_features=True)

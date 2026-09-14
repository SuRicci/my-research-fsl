from pathlib import Path
import json
import numpy as np
import torch
from scipy.optimize import minimize_scalar
import prevalidated as method
import assets_helper as assets
import reference_eval as ref

HERE = Path(__file__).resolve().parent
data, _ = assets.load_pool("original")
si, qi, _, _ = ref.tasks(data["dtd"]["ident"], "dtd", 5, "dev")
S = data["dtd"]["fusion"][si[:4]].double()
rng = np.random.RandomState(731)
synthetic = torch.tensor(rng.randn(2, 5, 5, 3) @ rng.randn(3, 12), dtype=torch.double)
errors = {"loo_refit": 0., "scalar_loss": 0., "query_batch": 0.}
for batch in [S, synthetic]:
    model = method.fit(batch)
    A, Y = batch.flatten(1, 2), model["Y"]
    for li, penalty in enumerate(method.LAMBDAS):
        explicit = []
        for i in range(25):
            keep = [j for j in range(25) if j != i]
            explicit.append(ref.ridge_scores(A[:, keep], Y[:, keep], A[:, i:i+1], penalty)[:, 0])
        error = float((torch.stack(explicit, 1) - model["loo"][li]).abs().max())
        errors["loo_refit"] = max(errors["loo_refit"], error)
        for e in range(len(batch)):
            Z = model["loo"][li, e].numpy()
            def loss(k):
                score = k * Z
                mx = score.max(1)
                return float((mx + np.log(np.exp(score - mx[:, None]).sum(1)) - score[np.arange(25), np.repeat(np.arange(5), 5)]).mean())
            independent = minimize_scalar(loss, bounds=(0., 100.), method="bounded", options={"xatol": 1e-9})
            reference = min(loss(0.), loss(100.), independent.fun)
            errors["scalar_loss"] = max(errors["scalar_loss"], abs(reference - float(model["nll"][li, e])))
    Q = batch.flatten(1, 2)
    whole = method.predict(model, Q)["preval"]
    one = method.predict(model, Q[:, :1])["preval"]
    errors["query_batch"] = max(errors["query_batch"], float((whole[:, :1] - one).abs().max()))
assert errors["loo_refit"] < 1e-8 and errors["scalar_loss"] < 1e-8 and errors["query_batch"] < 1e-8
(HERE / "validation.json").write_text(json.dumps({"status": "passed", "errors": errors, "actual_tasks": 4, "rank_deficient_tasks": 2, "deleted_rows_per_task": 25, "lambda_count": 4}, indent=2))
print("VALIDATION_PASSED", json.dumps(errors))

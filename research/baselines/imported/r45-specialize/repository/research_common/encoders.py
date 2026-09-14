from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .records import WORKSPACE, download_public, read_json, sha256, write_json

DINO_REVISION = "2302b6bf46953431b969155307b9bed152754069"
DINO_REPO = WORKSPACE / "research_common/extern/dinov2"


def prepare_dino_source():
    import requests
    receipt_path = DINO_REPO / "source_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["revision"] != DINO_REVISION:
            raise ValueError("DINO source revision mismatch")
        for row in receipt["files"]:
            if sha256(DINO_REPO / row["relative_path"]) != row["sha256"]:
                raise ValueError("DINO source modified after download")
        return receipt
    response = requests.get(f"https://api.github.com/repos/facebookresearch/dinov2/git/trees/{DINO_REVISION}?recursive=1", timeout=30)
    response.raise_for_status()
    files = [row["path"] for row in response.json()["tree"] if row["type"] == "blob" and (
        row["path"] in ("hubconf.py", "dinov2/__init__.py", "LICENSE") or
        row["path"].endswith(".py") and row["path"].startswith(("dinov2/layers/", "dinov2/models/", "dinov2/hub/")))]
    def one(relative):
        url = f"https://raw.githubusercontent.com/facebookresearch/dinov2/{DINO_REVISION}/{relative}"
        result = download_public(url, DINO_REPO / relative)
        result["relative_path"] = relative
        return result
    with ThreadPoolExecutor(6) as pool:
        records = list(pool.map(one, files))
    receipt = {"revision": DINO_REVISION, "files": records}
    write_json(receipt_path, receipt)
    return receipt


class FrozenEncoder:
    def __init__(self, name, device="cuda"):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no mock fallback")
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.name, self.device = name, device
        checkpoints = Path(os.environ.get('R5_CLIP_WEIGHTS_DIR',str(Path.home() / '.cache/torch/hub/checkpoints')))
        if name in ("clip_b32", "clip_b16"):
            import clip
            model_name = "ViT-B/32" if name == "clip_b32" else "ViT-B/16"
            weights = checkpoints / (model_name.replace("/", "-") + ".pt")
            if not weights.is_file():
                raise FileNotFoundError(weights)
            self.model, self.preprocess = clip.load(str(weights), device=device, jit=False)
            if float(self.model.logit_scale.exp()) < 50:
                raise ValueError("CLIP pretrained scale check failed")
            self.identity = {"kind": name, "weights_sha256": sha256(weights),
                "package_source": str(Path(clip.__file__).parent), "clip_model_source_sha256": sha256(Path(clip.__file__).parent / "model.py"),
                "preprocessing": "OpenAI CLIP official resize/center-crop 224, RGB, mean/std", "checkpoint_path": str(weights)}
        elif name == "dino_s14":
            from torchvision import transforms
            prepare_dino_source()
            weights = Path(os.environ.get('R5_DINO_WEIGHTS',str(WORKSPACE / '研究1——CLIP结合DinoV2/pretrained/dinov2_vits14_pretrain.pth')))
            if not weights.is_file():
                raise FileNotFoundError(weights)
            self.model = torch.hub.load(str(DINO_REPO), "dinov2_vits14", source="local", pretrained=False)
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
            self.model.to(device)
            self.preprocess = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
            self.identity = {"kind": name, "weights_sha256": sha256(weights), "source_revision": DINO_REVISION,
                "source_receipt_sha256": sha256(DINO_REPO / "source_receipt.json"),
                "preprocessing": "RGB resize short edge 256 bicubic, center-crop 224, ImageNet normalization", "checkpoint_path": str(weights)}
        else:
            raise ValueError(name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.identity.update(frozen=True, mock=False, torch=torch.__version__, device=device)

    @torch.inference_mode()
    def encode_pil(self, images):
        tensor = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        if self.name.startswith("clip"):
            out = self.model.encode_image(tensor)
        else:
            out = self.model.forward_features(tensor)["x_norm_clstoken"]
        out = F.normalize(out.float(), dim=-1)
        if not torch.isfinite(out).all():
            raise ValueError("Non-finite visual embedding")
        return out.cpu().numpy()

    def close(self):
        del self.model
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

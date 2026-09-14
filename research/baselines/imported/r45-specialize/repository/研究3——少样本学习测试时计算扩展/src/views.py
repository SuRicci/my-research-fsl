# -*- coding: utf-8 -*-
"""多视图采样器（V 轴）+ 分辨率档位（r 轴）。

视图 v=0 恒为确定性中心视图（resize + center crop）；v≥1 为增广视图
（RandomResizedCrop / 水平翻转 / 轻颜色抖动），由 (img_id, v) 派生种子，
保证同键同视图 → 特征缓存键 (img_id, v, r, backbone) 可复现。
"""
import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from PIL import Image
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD

# CLIP / DINOv2 的归一化常量（构造 transform 时选择）
NORM_STATS = {
    "clip": (IMAGENET_MEAN, IMAGENET_STD),
    "dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def cache_key(img_id: str, view: int, res: int, backbone: str) -> str:
    """特征缓存键：全局唯一且稳定（跨进程/跨机器一致）。"""
    return f"{backbone}|{img_id}|v{view}|r{res}"


def _view_seed(img_id: str, view: int) -> int:
    """由 (img_id, view) 派生确定性种子（与进程无关）。"""
    h = hashlib.md5(f"{img_id}#v{view}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


@dataclass
class ViewSpec:
    """一个视图的描述（供缓存索引与审计）。"""
    view: int
    res: int
    kind: str  # "center" | "rrc" | "rrc+flip" | "rrc+flip+jitter"


class ViewSampler:
    """对单张图生成 V 个确定性视图（v=0 为中心视图）。"""

    def __init__(self, backbone_type: str = "clip"):
        mean, std = NORM_STATS.get(backbone_type, NORM_STATS["clip"])
        self.mean, self.std = mean, std

    def specs(self, V: int) -> List[ViewSpec]:
        kinds = ["center", "rrc", "rrc+flip", "rrc+flip+jitter"]
        return [ViewSpec(view=v, res=0, kind=kinds[0] if v == 0 else kinds[1 + (v - 1) % 3]) for v in range(V)]

    def make_transform(self, img_id: str, view: int, res: int) -> transforms.Compose:
        """构造 (img_id, view, res) 对应的确定性预处理流水线。"""
        norm = transforms.Normalize(self.mean, self.std)
        if view == 0:
            # 中心视图：与标准 eval 预处理一致
            return transforms.Compose([
                transforms.Resize(res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(res),
                transforms.ToTensor(),
                norm,
            ])
        # 增广视图：种子固定 → 同键同图
        seed = _view_seed(img_id, view)
        ops: List = [
            DeterministicRandomResizedCrop(res, scale=(0.6, 1.0), seed=seed),
            DeterministicHorizontalFlip(p=0.5 if view % 2 == 0 else 0.0, seed=seed + 1),
        ]
        if (view - 1) % 3 == 2:
            ops.append(DeterministicColorJitter(0.2, 0.2, seed=seed + 2))
        ops += [transforms.ToTensor(), norm]
        return transforms.Compose(ops)

    def render(self, image: Image.Image, img_id: str, V: int, res: int) -> List:
        """生成前 V 个视图的 tensor 列表。"""
        return [self.make_transform(img_id, v, res)(image) for v in range(V)]


class _SeededOp:
    """以固定种子驱动单个 torchvision 变换，保证跨进程可复现。"""

    def __init__(self, op, seed: int):
        self.op = op
        self.seed = seed

    def __call__(self, img):
        import torch

        st = torch.get_rng_state()
        torch.manual_seed(self.seed)
        try:
            return self.op(img)
        finally:
            torch.set_rng_state(st)


def DeterministicRandomResizedCrop(res: int, scale: Tuple[float, float], seed: int):
    return _SeededOp(
        transforms.RandomResizedCrop(res, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC),
        seed,
    )


def DeterministicHorizontalFlip(p: float, seed: int):
    return _SeededOp(transforms.RandomHorizontalFlip(p=p), seed)


def DeterministicColorJitter(brightness: float, contrast: float, seed: int):
    return _SeededOp(transforms.ColorJitter(brightness=brightness, contrast=contrast), seed)


def resolution_ladder(backbone: str) -> Sequence[int]:
    """各 backbone 的分辨率档位（r 轴）。"""
    if backbone.startswith("dinov2"):
        return (224, 448, 518)
    return (224, 336, 448)


if __name__ == "__main__":
    # 单元验证：缓存键与视图确定性
    k1 = cache_key("img_001", 0, 224, "clip_vitb16")
    k2 = cache_key("img_001", 0, 224, "clip_vitb16")
    k3 = cache_key("img_001", 1, 224, "clip_vitb16")
    assert k1 == k2 and k1 != k3, "缓存键失败"

    import numpy as np

    img = Image.fromarray((np.random.rand(64, 80, 3) * 255).astype("uint8"))
    vs = ViewSampler("clip")
    a = vs.render(img, "img_001", V=3, res=224)
    b = vs.render(img, "img_001", V=3, res=224)
    import torch

    for x, y in zip(a, b):
        assert x.shape == (3, 224, 224)
        assert torch.equal(x, y), "同键视图不可复现"
    assert not torch.equal(a[0], a[1]), "中心视图与增广视图不应相同"
    # 不同 img_id 的同序增广视图应不同
    c = vs.render(img, "img_002", V=3, res=224)
    assert not torch.equal(a[1], c[1]), "不同键的增广视图不应相同"
    print("[views] 缓存键/视图确定性单元验证通过; specs:", [(s.view, s.kind) for s in vs.specs(4)])

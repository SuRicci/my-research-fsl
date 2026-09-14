# -*- coding: utf-8 -*-
"""MuSC 式测试集互评分（P1-AD，可后补；T 轴：互评分轮次）。

设计接口（实现后补，参考官方 xrli-U/MuSC）：
- 输入：测试集 patch 特征集合与记忆库距离初值；
- 每轮：各图像用其余图像（正常打分高置信的）作为互参照库重打分；
- rounds = T（预算轴），T=0 退化为纯记忆库距离。
当前显式 raise，避免静默伪结果；run_scaling_ad.py 中 T 轴档位默认只含 0。
"""
from typing import Optional

import torch


def mutual_score(patch_feats: torch.Tensor, base_scores: torch.Tensor,
                 rounds: int, **kw) -> torch.Tensor:
    """互评分细化（可后补）。rounds=0 时应返回 base_scores（该路径已实现）。"""
    if rounds <= 0:
        return base_scores
    raise NotImplementedError(
        "MuSC 互评分为 P1-AD 可选项（T 轴），实现后补（实验计划 §4）；T>0 暂不可用"
    )

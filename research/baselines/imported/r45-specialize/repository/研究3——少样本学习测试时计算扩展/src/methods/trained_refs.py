# -*- coding: utf-8 -*-
"""需轻量梯度的高预算参照（P2，可后补）：Tip-Adapter-F / TPT / LIMO。

本文件只固定接口与注册表，实现后补（实验计划 §4：P2 级仅在 2–3 个代表性
数据集小子集上跑，作为 iso-FLOPs 图的高预算端锚点，非主贡献）。
所有实现必须：显式声明训练步数/学习率（计入预算记账），限定子集规模。
当前任何调用显式 raise，避免静默伪结果。
"""
from typing import List

# 已实现的参照方法（后补时加入）
AVAILABLE: List[str] = []

# 规划中的参照方法（未实现）
PLANNED: List[str] = ["tip_adapter_f", "tpt", "limo"]


def run_trained_ref(name: str, *args, **kwargs):
    """统一入口（P2 可后补）。name ∈ PLANNED；实现后移入 AVAILABLE。"""
    if name not in PLANNED:
        raise KeyError(f"未知参照方法: {name}（规划: {PLANNED}）")
    raise NotImplementedError(
        f"{name} 为 P2 级高预算参照（含轻量梯度，限小子集），实现后补（实验计划 §4/§5 E5）"
    )

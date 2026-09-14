# -*- coding: utf-8 -*-
"""特征磁盘缓存：读写 / 索引 / 完整性校验。

工程红线（实验计划 §2）：特征只算一次，所有 scaling/分配实验读缓存做向量运算。
布局：cache/{scope}/{backbone}/{key}.pt（float16 CPU tensor，省空间），
索引 index.json 记录 key→(shape, dtype, mtime) 供完整性校验与断点续跑。
"""
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..config import CACHE_ROOT


class FeatureCache:
    """单 (scope, backbone) 的特征缓存命名空间。

    flush_every: put() 时每多少条落盘一次索引（大批量写缓存任务可调大，
    避免 index.json 随条目数增长导致的 O(n²) 重写）。
    read_only: v3.5 P0-8——大矩阵确认 run 全程只读：put/flush 立即
    RuntimeError，命名空间目录不存在即失败，cache miss 由 get() 抛
    KeyError（fail closed）；并发进程禁止写同一 namespace。
    """

    def __init__(self, scope: str, backbone: str, root: Path = CACHE_ROOT,
                 flush_every: int = 64, read_only: bool = False):
        self.read_only = bool(read_only)
        self.dir = Path(root) / scope / backbone
        if self.read_only:
            if not self.dir.is_dir():
                raise RuntimeError(
                    f"read_only 缓存命名空间不存在: {self.dir}（fail closed）")
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.flush_every = max(1, int(flush_every))
        self._index: Optional[Dict] = None

    # ---------------- 索引 ----------------
    def _load_index(self) -> Dict:
        if self._index is None:
            if self.index_path.exists():
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            else:
                self._index = {}
        return self._index

    def _save_index(self) -> None:
        """原子写索引；Windows 下偶发索引/杀软锁占用，重试后回退直接写。"""
        tmp = self.index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f)
        for attempt in range(5):
            try:
                tmp.replace(self.index_path)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        # 回退：直接覆盖写（非原子，但避免中断长任务）
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f)
        tmp.unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("|", "__").replace("/", "_")
        return self.dir / f"{safe}.pt"

    # ---------------- 读写 ----------------
    def has(self, key: str) -> bool:
        return key in self._load_index() and self._path(key).exists()

    def _require_writable(self, op: str) -> None:
        if self.read_only:
            raise RuntimeError(
                f"read_only 缓存禁止 {op}: {self.dir}（v3.5 P0-8 fail closed）")

    def index_hash(self) -> str:
        """index.json 内容 sha256（run 指纹用；索引缺失返回 'no-index'）。"""
        import hashlib
        if not self.index_path.exists():
            return "no-index"
        return hashlib.sha256(self.index_path.read_bytes()).hexdigest()

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """写入特征（转 float16 CPU）；索引每 flush_every 条落盘一次（防 O(n²)）。"""
        self._require_writable("put")
        t = tensor.detach().to("cpu", dtype=torch.float16)
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        torch.save(t, tmp)
        for attempt in range(5):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        else:
            torch.save(t, path)  # 回退直接写
            tmp.unlink(missing_ok=True)
        idx = self._load_index()
        idx[key] = {"shape": list(t.shape), "mtime": time.time()}
        if len(idx) % self.flush_every == 0:
            self._save_index()

    def flush(self) -> None:
        """显式落盘索引（批量写入结束后调用）。"""
        self._require_writable("flush")
        if self._index is not None:
            self._save_index()

    def get(self, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """读取特征并校验形状与索引一致。"""
        idx = self._load_index()
        if key not in idx:
            raise KeyError(f"缓存未命中: {key}")
        t = torch.load(self._path(key), map_location="cpu", weights_only=True)
        if list(t.shape) != idx[key]["shape"]:
            raise RuntimeError(f"缓存损坏（形状不符）: {key}: {list(t.shape)} != {idx[key]['shape']}")
        return t.to(dtype)

    def get_many(self, keys: Iterable[str], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.stack([self.get(k, dtype) for k in keys])

    def keys(self) -> List[str]:
        return sorted(self._load_index().keys())

    def missing(self, keys: Iterable[str]) -> List[str]:
        return [k for k in keys if not self.has(k)]

    def verify(self) -> Tuple[int, int]:
        """完整性校验：返回 (ok, bad) 条目数。"""
        idx = self._load_index()
        ok = bad = 0
        for key, meta in idx.items():
            p = self._path(key)
            try:
                t = torch.load(p, map_location="cpu", weights_only=True)
                if list(t.shape) == meta["shape"]:
                    ok += 1
                else:
                    bad += 1
            except Exception:
                bad += 1
        return ok, bad

    def __len__(self) -> int:
        return len(self._load_index())


if __name__ == "__main__":
    # 单元验证：读写/索引/校验/断点语义
    import shutil

    root = CACHE_ROOT / "_unittest"
    shutil.rmtree(root, ignore_errors=True)
    fc = FeatureCache("_unittest", "clip_vitb16", root=CACHE_ROOT)
    x = torch.randn(512)
    fc.put("img1|v0|r224", x)
    assert fc.has("img1|v0|r224") and not fc.has("img9|v0|r224")
    y = fc.get("img1|v0|r224")
    assert y.dtype == torch.float32 and torch.allclose(x, y, atol=1e-3), "float16 往返误差超阈"
    z = torch.randn(4, 512)
    fc.put("img1|v1|r224", z)
    got = fc.get_many(["img1|v0|r224", "img1|v1|r224"]) if False else None  # 形状不同不应 stack
    ok, bad = fc.verify()
    assert ok == 2 and bad == 0
    assert fc.missing(["img1|v0|r224", "zzz"]) == ["zzz"]
    fc.flush()  # 索引落盘后新实例可见（断点续跑）
    fc2 = FeatureCache("_unittest", "clip_vitb16", root=CACHE_ROOT)
    assert len(fc2) == 2 and fc2.has("img1|v0|r224")
    shutil.rmtree(root, ignore_errors=True)
    print("[cache_io] 单元验证通过: ok=2, bad=0")

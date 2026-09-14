# 上传期间新增的历史记录

主历史分卷保留原先冻结的 70 个引用、685 次提交。本目录另保留上传期间自动产生的 5 次归档管理提交，未修改实验、基线、论文或文献文件。

在恢复主历史后，可在工作副本应用增量：

```bash
git clone restored-history.git restored-working
git -C restored-working checkout run/dino-base-control-20260914
git -C restored-working am /absolute/path/to/my-research-fsl/archive/post-snapshot-history/*.patch
```

增量已在原脱敏基点上独立应用，所有变更文件均与源提交逐字节核对。原提交 ID、脱敏基点、补丁哈希和最终树哈希见 verification.json。提交者配置及应用时间会改变重新生成的提交 ID，不改变恢复文件内容。主历史分卷及原映射保持原样。

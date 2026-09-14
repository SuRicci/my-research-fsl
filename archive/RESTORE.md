# 取回完整归档

此仓库为私有。先在本机完成 GitHub CLI 登录，再运行：

```bash
git clone https://github.com/SuRicci/my-research-fsl.git
cd my-research-fsl
mkdir downloaded
gh release download quest-012-archive-20260914 --repo SuRicci/my-research-fsl --dir downloaded
python3 archive/verify_archive.py downloaded
```

验证器检查所有分卷及归档内每一个文件的 SHA256，无需先解压。分卷和哈希的完整列表在 assets.json。

在新的空目录中恢复项目快照：

```bash
mkdir restored
cat downloaded/quest-012-snapshot.tar.gz.part* | tar -xzf - -C restored
```

恢复 Git 历史：

```bash
cat downloaded/quest-012-history.bundle.part* > downloaded/quest-012-history.bundle
git clone --mirror downloaded/quest-012-history.bundle restored-history.git
git --git-dir=restored-history.git fsck --full
```

快照保留原目录结构和符号链接；部分链接可能指向原机器路径，不能据此认为外部数据也已归档。快照中的工作树 Git 指针为历史记录，移动机器后应从 Git bundle 重新建立工作树，而不要直接使用旧的绝对指针。原始报告的绝对路径可将原 Quest 根路径替换为 restored/quest-012；所有逐任务结果按原相对路径保留。

历史已经移除登录凭据、运行器目录及此前已清理的 r2-domains 原始数据与下载包；分支和提交数保留，提交对应关系在 history-commit-map.txt。完整重跑仍需按报告说明重新获取已清理输入。此次仅迁移 Quest 012，既有外部科研目录未包含在本地清理范围内。

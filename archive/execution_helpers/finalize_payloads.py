from pathlib import Path
import json,hashlib,shutil,subprocess,datetime
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';A=T/'upload/archive'
s=json.loads((T/'snapshot-receipt.json').read_text());h=json.loads((T/'history-audit.json').read_text());assert s['status']=='ready' and not s['failures'];assert h['status']=='passed' and h['bundle_fsck']=='passed' and h['all_restored_refs_identical']
assets=[dict(r,role='snapshot_part') for r in s['parts']]
bundle=T/'quest-012-history.bundle';full=hashlib.sha256();pieces=hashlib.sha256();partrows=[]
with bundle.open('rb') as f:
 for i in range(1,1000):
  data=f.read(1024**3)
  if not data:break
  full.update(data);p=T/f'quest-012-history.bundle.part{i:03d}';p.write_bytes(data);digest=hashlib.sha256(data).hexdigest();pieces.update(p.read_bytes());partrows.append({'file':p.name,'bytes':len(data),'sha256':digest,'role':'history_part'})
assert full.hexdigest()==pieces.hexdigest();assets+=partrows
info={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':'SuRicci/my-research-fsl','release_tag':'quest-012-archive-20260914','asset_count':len(assets),'assets':assets,'total_bytes':sum(r['bytes'] for r in assets),'history_bundle_sha256':full.hexdigest(),'history_bundle_bytes':bundle.stat().st_size,'snapshot_protected_coverage':s['coverage'],'history_restore_audit':h,'exclusion_note':'Runner credentials and environments, disposable scratch, raw Git metadata exported separately. Sanitized history excludes .ds and previously removed r2-domains data/downloads. Existing external source directories are outside quest-012 cleanup scope.'}
(A/'assets.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');(A/'SHA256SUMS').write_text('\n'.join(r['sha256']+'  '+r['file'] for r in assets)+'\n')
for n in ['snapshot-receipt.json','history-audit.json','history-commit-map.txt','history-ref-map.txt','history-changed-refs.txt','original-refs.txt','sanitized-refs.txt','browser-push-receipt.json']:
 p=T/n
 if p.exists():shutil.copyfile(p,A/n)
shutil.copyfile(T/'snapshot-manifest.json',A/'snapshot-manifest.json')
(A/'RESTORE.md').write_text('''# 取回完整归档

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
''')
# Full bundle is now reproducible byte-for-byte from verified parts; drop only this generated duplicate.
bundle.unlink()
print(json.dumps({'asset_count':len(assets),'total_gib':info['total_bytes']/2**30,'snapshot_coverage':s['coverage'],'history_refs':h['refs'],'history_commits':h['commits'],'free_gib':shutil.disk_usage(Q).free/2**30}),flush=True)

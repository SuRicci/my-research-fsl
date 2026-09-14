from pathlib import Path
import subprocess,json,re,hashlib,shutil,datetime,os
Q=Path('/Users/decoqwq/DeepScientist/quests/012');T=Q/'tmp/github-archive-20260914';D=T/'upload/archive/post-snapshot-history';D.mkdir(exist_ok=True)
OLD='97afd4ee654866dacf159bf3a75bc9a88b43282a';NEW='d361970f5dab76c8f4b696747d9ec8e291ee8b38';BASE='035cb6b9815f6aba04e6e483c410543be987ae2f';BRANCH='run/dino-base-control-20260914'
def git(root,*args,binary=False):
 p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=not binary)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
assert git(Q,'rev-parse',BRANCH).strip()==NEW
commits=git(Q,'rev-list','--reverse','--parents',OLD+'..'+NEW).splitlines();assert len(commits)==5 and all(len(x.split())==2 for x in commits)
changed=set()
for line in commits:
 c,p=line.split();changed.update(git(Q,'diff','--name-only',p,c).splitlines())
assert all(x in ['plan.md','CHECKLIST.md'] or x.startswith(('artifacts/','handoffs/github-archive-20260914/')) for x in changed),sorted(changed)
assert not any('/auth.json' in x or '/hosts.yml' in x or x.startswith('.ds/') for x in changed)
git(Q,'format-patch','--binary','--full-index','--no-signature','--output-directory',str(D),OLD+'..'+NEW)
patches=sorted(D.glob('*.patch'));assert len(patches)==5
pat=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}')
for p in patches:assert not pat.search(p.read_bytes()),'Credential pattern found in delta'
C=T/'history-delta-recovery';assert not C.exists()
git(Q,'clone','--shared','--no-checkout',str(T/'history.git'),str(C));git(C,'config','user.name',git(Q,'config','user.name').strip());git(C,'config','user.email',git(Q,'config','user.email').strip());git(C,'config','core.sparseCheckout','true');(C/'.git/info/sparse-checkout').write_text('/artifacts/\n/handoffs/\n/CHECKLIST.md\n/plan.md\n');git(C,'checkout','--detach',BASE)
git(C,'am',*[str(p) for p in patches]);rebuilt=git(C,'rev-parse','HEAD').strip();tree=git(C,'rev-parse','HEAD^{tree}').strip()
# Every path modified in any of the five commits must have the same final bytes.
checks=[]
for p in sorted(changed):
 original=git(Q,'show',NEW+':'+p,binary=True);restored=git(C,'show','HEAD:'+p,binary=True);assert original==restored,p
 checks.append({'path':p,'sha256':hashlib.sha256(restored).hexdigest()})
git(C,'fsck','--full','--no-dangling')
refs=git(Q,'for-each-ref','--format=%(refname) %(objectname)');assert refs==(T/'current-refs-preflight.txt').read_text();(D/'source-refs-after-delta.txt').write_text(refs)
receipt={'status':'passed','scope':'Five archive-administration commits created after the frozen scientific history; no experiment, baseline, paper or literature files changed','commit_count':5,'original_base':OLD,'original_head':NEW,'sanitized_base':BASE,'affected_branch':BRANCH,'source_commits':[x.split()[0] for x in commits],'independent_patch_application':'passed','reconstructed_head_example':rebuilt,'reconstructed_tree':tree,'changed_files_sha256':checks,'credential_pattern_hits':0,'patches':[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in patches],'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(D/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
(D/'README.md').write_text('''# 上传期间新增的历史记录

主历史分卷保留原先冻结的 70 个引用、685 次提交。本目录另保留上传期间自动产生的 5 次归档管理提交，未修改实验、基线、论文或文献文件。

在恢复主历史后，可在工作副本应用增量：

```bash
git clone restored-history.git restored-working
git -C restored-working checkout run/dino-base-control-20260914
git -C restored-working am /absolute/path/to/my-research-fsl/archive/post-snapshot-history/*.patch
```

增量已在原脱敏基点上独立应用，所有变更文件均与源提交逐字节核对。原提交 ID、脱敏基点、补丁哈希和最终树哈希见 verification.json。提交者配置及应用时间会改变重新生成的提交 ID，不改变恢复文件内容。主历史分卷及原映射保持原样。
''')
shutil.rmtree(C)
print(json.dumps({'status':'passed','commits':5,'changed_paths':len(checks),'patch_bytes':sum(p.stat().st_size for p in patches),'source_refs':len(refs.splitlines())}),flush=True)

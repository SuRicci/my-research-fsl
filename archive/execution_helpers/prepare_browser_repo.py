from pathlib import Path
import subprocess,shutil,re,json
Q=Path('/Users/decoqwq/DeepScientist/quests/012');W=Path.cwd();T=Q/'tmp/github-archive-20260914';D=T/'upload';D.mkdir(exist_ok=True)
pattern=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}')
files=[]
for src in (Q/'handoffs/pause-20260914').iterdir():
 if src.is_file():files.append((src,D/'report'/src.name))
for group in ['experiments','baselines','paper','artifacts/reports']:
 for src in (W/group).rglob('*'):
  if src.is_file() and not src.is_symlink() and not any(x in src.relative_to(W).parts for x in ['.git','.ds','__pycache__','data','downloads']):
   if src.suffix in ['.py','.md','.json','.yaml','.yml','.toml','.tex','.bib','.sty','.css','.pdf'] and src.stat().st_size<20*2**20:files.append((src,D/'research'/src.relative_to(W)))
for src,dst in files:
 data=src.read_bytes();assert not pattern.search(data),str(src)+' contains credential-like content';dst.parent.mkdir(exist_ok=True,parents=True);dst.write_bytes(data)
(D/'README.md').write_text('''# CV/FSL 研究归档（Quest 012）

本仓库保存实验 4、实验 5 及后续自主研究的私有归档。研究已于 2026-09-14 按要求暂停。

- [研究总报告（17 页 PDF）](report/研究总报告.pdf)
- [可编辑报告](report/研究总报告.md)：45 个条目，区分已证实优势、局部收益、失败和未解问题。
- [研究代码、协议和结果摘要](research/)
- [原始交付与证据索引](report/evidence_index.json)

## 完整数据与历史

网页目录是便于阅读的入口。完整逐任务结果、不同工作树、运行记录和文献，以及去除登录凭据的 Git 历史，保存在本仓库 `quest-012-archive-20260914` Release 的分卷附件中。必须下载所有分卷及校验清单；仅 clone 此仓库不能获得完整结果。

完整归档的范围、分卷哈希和验证结果见 `archive/`。Git 历史保留分支关系，并提供原提交到去凭据提交的映射；运行器登录目录、凭据和此前已删除的原始数据下载包不进入历史归档。当前保留研究结果必须逐文件核对，不以报告包代替。

此归档是暂停交付包；已有论文仍为草稿，不声称达到投稿标准。恢复研究时先读报告中的限制和恢复说明，按需要重新获取此前已清理的输入。
''')
(D/'.gitignore').write_text('*.part[0-9]*\n*.bundle\n__pycache__/\n')
if (D/'.git').exists():
 print('BROWSER_FILES_UPDATED',len(files)+2);raise SystemExit(0)
subprocess.run(['git','init','-b','main',str(D)],check=True)
for k in ['user.name','user.email']:
 value=subprocess.run(['git','config','--get',k],capture_output=True,text=True,check=True).stdout.strip();subprocess.run(['git','-C',str(D),'config',k,value],check=True)
subprocess.run(['git','-C',str(D),'config','http.proxy','http://127.0.0.1:7890'],check=True)
subprocess.run(['git','-C',str(D),'config','credential.helper',''],check=True)
subprocess.run(['git','-C',str(D),'config','--add','credential.helper','!gh auth git-credential'],check=True)
subprocess.run(['git','-C',str(D),'remote','add','origin','https://github.com/SuRicci/my-research-fsl.git'],check=True)
print(json.dumps({'browser_files':len(files)+2,'browser_bytes':sum(p.stat().st_size for p in D.rglob('*') if p.is_file() and '.git' not in p.parts)},indent=2))

from pathlib import Path
import subprocess,json,os,shutil,datetime,hashlib
Q=Path('/Users/decoqwq/DeepScientist/quests/012');W=Path.cwd();T=Q/'tmp/github-archive-20260914';R=T/'upload';A=R/'archive'
E=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',GH_NO_UPDATE_NOTIFIER='1',GIT_TERMINAL_PROMPT='0')
def run(cmd):
 p=subprocess.run(cmd,env=E,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
delta_dir=A/'post-snapshot-history';delta=json.loads((delta_dir/'verification.json').read_text());assert delta['status']=='passed' and delta['commit_count']==5
v=json.loads((T/'remote-verification.json').read_text());assert v['status']=='passed' and len(v['verified'])==14
repo=json.loads(run(['gh','api','repos/SuRicci/my-research-fsl']));assert repo['private']
shutil.copyfile(T/'remote-verification.json',A/'remote-verification.json')
helpers=A/'execution_helpers';helpers.mkdir(exist_ok=True)
for p in (W/'handoffs/github-archive-20260914').iterdir():
 if p.is_file() and p.suffix in ['.py','.json']:shutil.copyfile(p,helpers/p.name)
for name in ['interrupted-placeholders-removed.json','resume-upload-first-attempt.json','resume-upload-progress.json']:
 p=T/name
 if p.exists():shutil.copyfile(p,helpers/name)
state={'state':'remote_verified_cleanup_pending','local_deletion_allowed':True,'all_archive_parts_download_verified':True,'protected_files':21043,'snapshot_files':134694,'snapshot_symlinks':78,'history_refs':70,'history_commits':685,'archive_parts':10,'recovery_metadata_assets':4,'total_archive_bytes':9316706614,'private_repository':True}
(A/'status.json').write_text(json.dumps(state,indent=2)+'\n')
p=R/'README.md';s=p.read_text();s=s.replace('**当前状态：完整归档正在上传准备中，本地尚未删除。最终状态以 archive/status.json 为准。**','**当前状态：全部归档已上传并逐份回下载校验通过；本地清理待执行。最终状态以 archive/status.json 和 archive/local-cleanup-receipt.json 为准。**');s+='\n上传期间新增的 5 条归档管理提交另见 [历史增量](archive/post-snapshot-history/README.md)，已独立验证恢复；原 10 个分卷保持原样。\n';p.write_text(s)
for cmd in [['git','-C',str(R),'add','README.md','archive'],['git','-C',str(R),'commit','-m','Verify complete private research archive and recovery'],['git','-C',str(R),'-c','http.version=HTTP/1.1','-c','http.postBuffer=67108864','push','origin','main']]:print(run(cmd),flush=True)
notes='''完整研究归档已验证。10 个分卷共 9,316,706,614 字节，另附 4 个恢复文件；每个远端附件均已实际下载并核对 SHA256。快照包含 134,694 个文件、78 个符号链接；21,043 个受保护结果路径字节一致。去凭据 Git 历史保留 70 个分支引用、685 次提交，独立恢复和 Git 完整性检查通过。

恢复方法请阅读仓库 archive/RESTORE.md。登录凭据和运行器环境不属于此研究备份；旧历史中的 r2-domains 原始数据与下载包按此前清理要求排除。上传期间另有 5 条管理提交以可恢复补丁保存在仓库 archive/post-snapshot-history。此为私有暂停归档，不是投稿就绪声明。
'''
(T/'release-notes.md').write_text(notes)
print(run(['gh','release','edit','quest-012-archive-20260914','--repo','SuRicci/my-research-fsl','--draft=false','--latest=false','--notes-file',str(T/'release-notes.md')]),flush=True)
release=json.loads(run(['gh','api','repos/SuRicci/my-research-fsl/releases/388290702']));assert not release['draft'] and release['tag_name']=='quest-012-archive-20260914'
recovery=T/'browser-recovery'
run(['git','-c','http.proxy=http://127.0.0.1:7890','-c','credential.helper=','-c','credential.helper=!gh auth git-credential','clone','--depth','1','--no-checkout','https://github.com/SuRicci/my-research-fsl.git',str(recovery)])
head=run(['git','-C',str(recovery),'rev-parse','HEAD']).strip();expected=run(['git','-C',str(R),'rev-parse','HEAD']).strip();assert head==expected
run(['git','-C',str(recovery),'fsck','--full','--no-dangling'])
for p in delta_dir.iterdir():
 if p.is_file():
  fetched=subprocess.run(['git','-C',str(recovery),'show','HEAD:archive/post-snapshot-history/'+p.name],capture_output=True,env=E,check=True).stdout
  assert hashlib.sha256(fetched).digest()==hashlib.sha256(p.read_bytes()).digest(),p.name
shutil.rmtree(recovery)
result={'status':'ready_for_local_cleanup','private':True,'release_draft':False,'release_id':release['id'],'release_tag':release['tag_name'],'release_url':release['html_url'],'main_commit':head,'browser_clone_verified':True,'post_snapshot_history_verified':True,'post_snapshot_history_commits':5,'remote_archive_download_verified':True,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(T/'cloud-ready-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

from pathlib import Path
import subprocess,json,shutil,datetime
q=Path('/Users/decoqwq/DeepScientist/quests/012');w=Path.cwd();o=w/'artifacts/reports/pause_20260914'
def run(args):return subprocess.run(args,cwd=q,text=True,capture_output=True,check=True).stdout
refs=run(['git','for-each-ref','--format=%(refname) %(objectname)']);head=run(['git','rev-parse','HEAD']).strip();count=run(['git','count-objects','-vH']);before=shutil.disk_usage(q).free
(o/'git_before_compaction.json').write_text(json.dumps({'refs':refs,'head':head,'count_objects':count,'free_bytes':before},indent=2)+'\n')
cmd=['git','-c','pack.threads=4','-c','pack.windowMemory=128m','-c','gc.reflogExpire=never','-c','gc.reflogExpireUnreachable=never','-c','gc.pruneExpire=never','-c','gc.worktreePruneExpire=never','gc','--no-prune']
print('COMPRESS_HISTORY_WITHOUT_EXPIRY',flush=True);p=subprocess.run(cmd,cwd=q,text=True,capture_output=True);print(p.stdout,flush=True);print(p.stderr,flush=True)
afterrefs=run(['git','for-each-ref','--format=%(refname) %(objectname)']);afterhead=run(['git','rev-parse','HEAD']).strip();aftercount=run(['git','count-objects','-vH']);same=refs==afterrefs and head==afterhead
r={'status':'completed' if p.returncode==0 and same else 'needs_inspection','command':cmd,'exit_code':p.returncode,'refs_and_head_unchanged':same,'before_count_objects':count,'after_count_objects':aftercount,'stderr':p.stderr,'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(q).free,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'history_policy':'no reflog expiry, no object pruning, no worktree expiry, no ref rewrite'}
r['observed_free_delta_bytes']=r['free_after_bytes']-before;(o/'git_compaction_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('COMPACTION_RESULT',json.dumps(r),flush=True)
assert p.returncode==0 and same

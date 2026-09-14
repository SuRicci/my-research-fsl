"""Build tables, figures and a reproducible evidence index from saved predictions."""
import json,csv,hashlib,sys,time,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pp(x):return f'{100*x:.2f}'
def ci(v):return f'[{100*v[0]:.2f}, {100*v[1]:.2f}]'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def evidence():
    snap=HERE/'code_at_selection';snap.mkdir(exist_ok=True)
    old=(HERE/'run_e12.py').read_text(encoding='utf-8').replace("[('uniform_mix','raw'),('weighted_mix','uniform_mix')","[('weighted_mix','uniform_mix')")
    (snap/'run_e12.py').write_text(old,encoding='utf-8',newline='') if sys.version_info>=(3,10) else (snap/'run_e12.py').write_bytes(old.encode('utf-8'))
    expected=read(HERE/'e12'/'selection_receipt.json')['runner_sha256']
    assert digest(snap/'run_e12.py')==expected,'Frozen E12 source hash mismatch'
    shutil.copy2(HERE/'run_e11.py',snap/'run_e11.py')
    assert digest(snap/'run_e11.py')==read(HERE/'e11'/'selection.json')['runner_sha256']
    before=HERE/'input_snapshots';before.mkdir(exist_ok=True)
    for folder in [next(ROOT.glob('实验11*')),next(ROOT.glob('实验12*'))]:
        target=before/folder.name;target.mkdir(exist_ok=True)
        for name in ['README.md','protocol.json','run_pilot.py']:
            if not (target/name).exists():shutil.copy2(folder/name,target/name)
        for name in ['pilot_notes_20260908.md','pilot_main.json','pilot_main_test.json']:
            p=folder/'results'/name
            if p.exists() and not(target/name).exists():shutil.copy2(p,target/name)
    # Script snapshots, relevant local sources, result artifacts, model hashes and publisher inputs.
    items={}
    for p in HERE.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['__pycache__','assets','dev_before_query_dedup']):continue
        if p.name=='evidence_manifest.json':continue
        items[str(p.relative_to(HERE))]={'bytes':p.stat().st_size,'sha256':digest(p)}
    pack=HERE/'assets'/'bow_official'/'images.zip'
    out={'created_at':time.time(),'files':items,'publisher_zip':{'bytes':pack.stat().st_size,'sha256':digest(pack)},
         'snapshot_note':'E12 inference source exactly reconstructs the selection-time hash; current source only adds uniform-minus-raw post-analysis. Input snapshots predate README/protocol status updates.',
         'commands':['python audit_prepare.py','python prepare_official_bow.py','python run_e12.py --phase dev','python run_e12.py --phase retest','python run_e12.py --phase summarize','python analyze_e12.py','python run_e11.py --phase audit','python run_e11.py --phase encode','python run_e11.py --phase dev','python run_e11.py --phase retest','python check_experiments.py','python audit_feature_parity.py'],
         'python_executable':sys.executable}
    (HERE/'evidence_manifest.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    d11=read(HERE/'e11'/'decision.json');s11=read(HERE/'e11'/'test_summary.json');a11=read(HERE/'e11'/'data_audit.json')
    d12=read(HERE/'e12'/'decision.json');rows=read(HERE/'e12'/'summary.json')['rows'];e11cost=read(HERE/'e11'/'encoding_cost.json')
    mapping={r['method']:r for r in s11['rows']}
    with (HERE/'e11'/'all_methods.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(s11['rows'][0]));w.writeheader();w.writerows(s11['rows'])
    cols=['cell','raw','uniform_mix','weighted_mix','uniform_gate','weighted_gate','true25_diag','legacy_uniform','legacy_weighted','weighted_gate_minus_uniform_mix','weighted_gate_minus_uniform_mix_ci95','weighted_gate_fallback_rate']
    with (HERE/'e12'/'all_cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.3),gridspec_kw={'width_ratios':[.85,1.55]})
    vals=[d11['global_acc']*100,d11['local_acc']*100]
    for j,(key,color) in enumerate([(d11['global'],'#4472a6'),(d11['local'],'#dc8151')]):
        bounds=np.array(mapping[key]['clean_ci95'])*100
        axes[0].errorbar(j,vals[j],yerr=[[vals[j]-bounds[0]],[bounds[1]-vals[j]]],fmt='o',color=color,capsize=7,markersize=9,lw=2)
    axes[0].set_xticks([0,1],['全局融合 + 岭回归','局部空间描述 + 岭回归'])
    axes[0].set_ylim(45,70);axes[0].set_xlim(-.6,1.6);axes[0].set_ylabel('准确率 (%)，题级95% CI');axes[0].set_title('实验 11：去重测试集 124 题')
    for i,v in enumerate(vals):axes[0].text(i+.10,v+.4,f'{v:.2f}%',ha='left')
    axes[0].text(.5,.92,f"局部 − 全局：{100*d11['difference']:.2f} pp\n配对 95% CI {ci(d11['difference_ci95'])}",ha='center',va='top',transform=axes[0].transAxes,fontsize=9)
    one=[x for x in rows if x['shot']==1]
    labels=[]
    for x in one:
        tag='CLIP' if x['backbone'].startswith('clip') else 'DINO'
        cell=x['cell']
        label='CIFAR / 共享类' if cell.startswith('cifar') else ('DTD / 共享类' if cell.startswith('dtd_dtd') else ('DTD / 跨域' if cell.startswith('dtd_mini') else 'miniIN / 类不交'))
        labels.append(label+' / '+tag)
    y=np.arange(len(one));u=np.array([x['uniform_mix_minus_raw']*100 for x in one]);w=np.array([x['weighted_gate_minus_raw']*100 for x in one])
    axes[1].barh(y-.17,u,.32,label='均匀混入',color='#4472a6');axes[1].barh(y+.17,w,.32,label='加权 + 回退',color='#dc8151')
    axes[1].set_yticks(y,labels);axes[1].invert_yaxis();axes[1].axvline(0,color='#333',lw=.8)
    axes[1].set_xlim(-3,9);axes[1].set_xlabel('相对纯支持集的提升 (百分点)');axes[1].set_title('实验 12：1-shot，600 episode / 条件');axes[1].legend(loc='lower right')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax is axes[1] else 'y',alpha=.14);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=180);fig.savefig(HERE/'comparison.pdf');plt.close(fig)
    lines=['# 实验11、12续验与价值判断（2026-09-09）','',
      '**优先保留实验12的简单检索增强基线；两条当前新增机制均未达到继续扩大的门槛。** 实验11在完整数据与更强对照下仍没有显示局部读出的收益；实验12的真实收益主要来自适量混入近邻，可靠性权重与逐类回退没有稳定的额外贡献。','',
      '| 方向 | 本轮关键结果 | 投入判断 |','| --- | --- | --- |',
      f"| 11：抽象概念绑定 | 去重测试：选定全局对照 {pp(d11['global_acc'])}%，局部候选 {pp(d11['local_acc'])}%；差 {pp(d11['difference'])}pp，95% CI {ci(d11['difference_ci95'])} | 停止扩大当前全局/局部闭式机制；保留严格协议和基线 |",
      '| 12：检索增强与可靠性加权 | 共享类1-shot，均匀混入提升3.15–7.86pp；加权相对均匀的1-shot宏平均变化为−0.055pp | 保留简单检索作为工具/基线；停止当前可靠性加权与回退的论文主张 |','',
      '![两方向续验对比](comparison.png)','',
      '## 实际完成的实验','',
      '- 实验11：从作者备份恢复14,140张图像与完整官方元数据；做RGB身份审计；验证/测试共5,600张图分别提取冻结CLIP-B/16、DINOv2-S/14特征；比较50项全局、融合、空间和局部核读出。只在验证集选方法，随后冻结复测。',
      '- 实验12：两编码器 × 两shot × 四种查询/图库关系，共16条件；每条件200开发episode、600复测episode（三种子各200）。完成3,200开发和9,600复测episode；每个复测episode有75查询。所有配置和逐查询预测均保存。',
      '- 9组证据检查通过，包括支持/查询RGB身份隔离、开发/复测类别隔离、独立重新聚合与18项旧实现预测对拍；另有24项真实编码器特征复核通过。GPU为本机RTX 4080，编码器梯度更新为0。','',
      '## 11：按原来的闭式读出思路，加入局部证据继续验证','',
      '### 先修正旧证据','',
      '旧训练JSON被截断；旧测试采用变动shot，190题每题实际2–5个查询（56/81/36/17题分别为2/3/4/5查询），不是报告中的固定2查询。旧knn_vote与contrast在正负标签预测上代数等价。由此不能把旧64.2%解释为严格协议追平官方方法。旧commonSense代码未映射，组间/组内方差的简单比较也不能否定概念族结构。','',
      '本轮保持官方每侧6支持+第7张查询的任务定义，使用统一224中心裁剪。RGB审计如下：','',
      '| 划分 | 全部题 | 题内有重复图 | 与此前划分共享图 | 去重后题数 |','| --- | ---: | ---: | ---: | ---: |']
    for s,label in [('train','训练'),('val','验证'),('test','测试')]:
        v=a11[s];lines.append(f"| {label} | {v['all_problems']} | {v['within_duplicate_problems']} | {v['cross_split_duplicate_problems']} | {v['clean_problems']} |")
    lines+=['','去重规则以训练→验证→测试顺序排除整题；重复图条件可能重叠，按并集排除。完整官方划分作为敏感性结果保留。训练图只用于身份排除，没有训练读出器或学习跨题特征。','',
      '### 冻结选择后的结果','',
      '全局对照覆盖原对比方向、LOO定界、余弦原型、真正kNN（1/3/5邻居）、带截距的闭式岭回归，以及等比例CLIP+DINO融合。局部候选包括DINO固定空间描述、4×4局部描述子的对称平均最大匹配核及其LOO/岭回归读出。','',
      f"验证集选中 `{d11['global']}` 和 `{d11['local']}`；下表保留这次选择，未用测试集重选。",'',
      '| 方法 | 去重124题 | 完整200题 |','| --- | ---: | ---: |']
    listed=[('旧DINO对比+LOO','dinov2_vits14__contrast_loo'),('DINO余弦原型','dinov2_vits14__cosine_proto'),('DINO真正1-NN','dinov2_vits14__knn1'),('验证集选定全局对照',d11['global']),('验证集选定局部候选',d11['local'])]
    for label,key in listed:lines.append(f"| {label} | {pp(mapping[key]['clean_accuracy'])}% | {pp(mapping[key]['full_accuracy'])}% |")
    lines+=['',f"局部相对全局为 {pp(d11['difference'])}pp，题级配对bootstrap 95% CI {ci(d11['difference_ci95'])}。区间包含零，结果不支持局部候选有优势；预设的≥3pp且区间下界>0门槛未通过。",'',
      'commonSense已按官方十类全部报告。去重测试中“其他”有87题，其余每族仅2–9题，当前分族结果主要用于描述，不能据此建立“哪种概念原则上不可读”的结论。详见[分族结果](e11/decision.json)和[50方法全表](e11/all_methods.csv)。','',
      '原论文附录E.4已经包含冻结CLIP/DINO/DINOv2的平均相似度基线；因此此基本思路也面临直接的已有工作覆盖。[Bongard-OpenWorld原文](https://arxiv.org/html/2310.10207v3#A5.SS4)。','',
      '## 12：把检索收益与可靠性权重的贡献分开','',
      '### 公平对照与隔离','',
      '方法只看到支持特征、支持标签和无标签图库特征。图库真标签只参与结果完成后的污染诊断。以统一的混入比例控制伪支持总质量，均匀和加权获得相同选择预算；选择网格包含关闭增强。覆盖分数改为支持到近邻的相似度减去近邻自身图库密度，回退在每个类别单独触发。','',
      '开发与复测用确定性随机类别半分；所有四个条件按宏平均共同选参数，避免用每个测试条件的成绩单独选开关。CIFAR/DTD/miniImageNet图库分别去掉与查询池同RGB的2/1/1张图，另外清理图库自身重复图；DTD查询池清除3张重复图。清理后的图库为49,984/1,876/38,399图，并重新计算图库密度。','',
      '### 1-shot 主表','',
      '| 查询×图库 | 编码器 | 纯支持 | 均匀混入 | 加权+回退 | 均匀−纯支持 | 加权−均匀 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in one:
        cell=row['cell'];label='CIFAR×CIFAR（共享类）' if cell.startswith('cifar') else ('DTD×DTD（共享类）' if cell.startswith('dtd_dtd') else ('DTD×miniIN（跨域）' if cell.startswith('dtd_mini') else 'miniIN×miniIN（类不交）'))
        b='CLIP' if row['backbone'].startswith('clip') else 'DINO'
        lines.append(f"| {label} | {b} | {pp(row['raw'])}% | {pp(row['uniform_mix'])}% | {pp(row['weighted_gate'])}% | {pp(row['uniform_mix_minus_raw'])}pp | {pp(row['weighted_gate_minus_uniform_mix'])}pp |")
    p=d12['pooled']['1']['weighted_gate_minus_uniform_mix'];p5=d12['pooled']['5']['weighted_gate_minus_uniform_gate']
    lines+=['',f"1-shot加权对均匀的等条件宏平均差为 **{100*p['mean']:.3f}pp**，配对95% CI {ci(p['paired_ci95'])}；可靠性权重没有独立收益。5-shot下CLIP选择关闭增强；DINO的回退实际触发，但加权相对同样回退的均匀策略只增加{100*p5['mean']:.3f}pp，区间{ci(p5['paired_ci95'])}。完整5-shot与逐种子成绩见[16条件表](e12/all_cells.csv)和[含区间的JSON](e12/summary.json)。",'',
      '均匀混入在miniImageNet类不相交条件下也出现小幅正收益，说明旧“图库含目标类决定收益符号”的概括不成立；控制混入强度也影响结果。跨域DTD+DINO仍退化1.65pp，超过本轮允许的1pp损失；平均收益无法替代逐压力条件的门槛。','',
      '最终三个门槛：共享类1-shot较纯支持提升≥3pp通过；较均匀混入提升≥1pp未通过；全部压力条件退化≤1pp未通过。保留简单检索有实际价值，继续扩大当前加权/回退实现的证据不足。','',
      '## 证据范围、成本和后续投入','',
      '本轮比较的是两条思路各自相对强对照的价值，任务准确率不能跨实验直接排名。BOW测试曾在旧pilot查看；E12复用历史基准与缓存，类别隔离仅保证本轮选择与复测分开。区间是既定有限图像池下的配对不确定性，E12另保存类别宏平均bootstrap敏感性；这些结果没有新增外部数据集确认。RGB排除覆盖完全相同的解码图像，近重复与预训练重叠没有全面排除。','',
      'CIFAR条件使用Bertinetto的CIFAR-FS类别划分，但图像来自CIFAR-100官方test、每类100张；不能与采用每类600张的标准CIFAR-FS论文数字直接比较。BOW也只复用了官方题目协议，编码器/预处理与原学习式方法不同，64%旧论文点估计仅作历史背景。真25-shot诊断包含原支持集；去重后个别DTD类不足25张时用可用数量，逐episode索引保留−1填充标记。','',
      f"BOW新编码共11,200次图像前向（5,600图×两编码器），CLIP {e11cost['clip_vitb16']['seconds']:.1f}秒、DINO {e11cost['dinov2_vits14']['seconds']:.1f}秒，含本机图像读取与转换；读出在CPU进行。E12直接复用缓存、重新建去重图库密度；逐条件JSON保存批量检索用时。时间属于本机批量运行，尚未测独立单查询端到端P95。",'',
      '**资源安排：实验12作为可复用基线优先保留；实验11保留完整数据、严格评估器和全局岭回归基线。两条当前机制停止继续加网格。** 如果再次立项，11应先提出并验证实质不同的对象/关系证据表示；12应先给出能超越统一混入强度、在无图库标签条件下识别有害近邻的新信号，再用新的开发来源和外部确认集验证。原实验资产和失败结果均保留。','',
      '检索增强少样本学习、无标签干扰抑制已有直接相关工作；本轮未声称方法新颖。定向查新与权限差异见[查新说明](查新与设计差异.md)，包括[ICLR2018半监督少样本学习](https://mengyeren.com/research/2018/meta-learning-for-semi-supervised-few-shot-classification/)和[CVPR2025 SWAT](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Few-Shot_Recognition_via_Stage-Wise_Retrieval-Augmented_Finetuning_CVPR_2025_paper.html)。','',
      '## 文件与复现','',
      '- [运行前冻结协议](protocol_v1.json) · [数据初查](audit_initial.json) · [实现与结果验证](verification.json) · [真实特征复核](feature_parity.json)',
      '- [E11选择](e11/selection.json) · [E11主结论](e11/decision.json) · [E12选择](e12/selection.json) · [E12门槛](e12/decision.json)',
      '- `e11/*_predictions.npz`、`e12/*_dev.npz`、`e12/*_retest.npz` 保存逐查询分数/预测、身份、种子和类别；`evidence_manifest.json` 保存文件SHA256，`code_at_selection/` 保存匹配选择时哈希的源码。',
      '- `input_snapshots/` 保留更新前的方案/README/旧pilot；`e12/dev_before_query_dedup/` 保留修正查询重复前的开发结果；原实验目录的旧结果不覆盖。','',
      '以下命令使用已就绪的本地资产，先开发后复测；重新执行会更新本续验目录输出。','',
      '```powershell',
      "Set-Location 'D:\\科研\\续验11与12_20260909'",
      "$env:PYTHONUTF8='1'; $env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'",
      "& 'C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe' run_e11.py --phase dev",
      "& 'C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe' run_e11.py --phase retest",
      "& 'C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe' run_e12.py --phase dev",
      "& 'C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe' run_e12.py --phase retest",
      "& 'C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe' analyze_e12.py",
      "& 'C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe' check_experiments.py",
      '```','']
    (HERE/'续验与价值判断.md').write_text('\n'.join(lines),encoding='utf-8')
    evidence()
    print('report, figures, tables, evidence manifest complete',flush=True)

if __name__=='__main__':main()

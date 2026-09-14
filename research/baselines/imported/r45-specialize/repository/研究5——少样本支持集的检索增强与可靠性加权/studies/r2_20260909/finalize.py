# -*- coding: utf-8 -*-
"""Produce R2 report, figures, status integration and immutable input snapshots."""
import csv,json,hashlib,time,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from assets_io import HERE,ROOT,E12,dump

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pct(v):return f'{100*v:.2f}'
def interval(v):return f'[{100*v[0]:.2f}, {100*v[1]:.2f}]'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    s=read(HERE/'summary.json');rows=s['rows'];sel=read(HERE/'selection.json');ab=read(HERE/'ablation_summary.json')['rows'];api=read(HERE/'packaged_api_verification.json')
    assert s['practical_upgrade_passed'] and api['status']=='passed'
    order=[('cifar_fs','cifar100'),('dtd','dtd'),('miniimagenet','miniimagenet'),('dtd','miniimagenet'),('cub200','cub200'),('eurosat','eurosat'),('cub200','miniimagenet'),('eurosat','miniimagenet')]
    labels=['CIFAR / 同域共享类','DTD / 同域共享类','miniIN / 同域类不交','DTD / 跨域miniIN图库','CUB / 同域共享类（新增）','EuroSAT / 同域共享类（新增）','CUB / 跨域miniIN图库（新增）','EuroSAT / 跨域miniIN图库（新增）']
    lookup={(tuple(r['cell']),r['shot']):r for r in rows}
    table=[]
    for cell,label in zip(order,labels):
        for k in [1,5]:
            r=lookup[(cell,k)];table.append({'condition':label,'shot':k,'R1_DINO_accuracy':r['r1_dino'],'R2_accuracy':r['overall'],'delta_pp':100*r['overall_minus_r1_dino']['mean'],'delta_ci95_pp':interval(r['overall_minus_r1_dino']['ci95']),'control_accuracy':r['control'],'candidate_accuracy':r['candidate'],'source_query_images':r['n_distinct_query_images']})
    with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),gridspec_kw={'width_ratios':[1.45,1]})
    data=np.array([[100*lookup[(c,k)]['overall_minus_r1_dino']['mean'] for k in [1,5]] for c in order])
    im=axes[0].imshow(data,cmap='RdBu',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=5),aspect='auto')
    axes[0].set_xticks([0,1],['1-shot','5-shot']);axes[0].set_yticks(range(8),labels);axes[0].set_title('各条件相对R1的变化（百分点）')
    for i in range(8):
        for j in range(2):axes[0].text(j,i,f'{data[i,j]:+.2f}',ha='center',va='center',color='white' if abs(data[i,j])>3 else '#222')
    fig.colorbar(im,ax=axes[0],fraction=.04,pad=.025)
    x=0;xl=[]
    for phase,label in [('retest','旧条件'),('external','新增条件')]:
        for k in ['1','5']:
            v=s['pooled'][phase][k]['overall_minus_r1_dino'];m=100*v['mean'];lo,hi=np.array(v['ci95'])*100
            axes[1].errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=5,color='#286690',markersize=7)
            axes[1].text(x,m+.17,f'+{m:.2f}',ha='center');xl.append(label+'\n'+k+'-shot');x+=1
    axes[1].set_xticks(range(4),xl);axes[1].set_ylim(0,2.8);axes[1].set_xlim(-.5,3.5);axes[1].set_ylabel('宏平均提升（百分点），配对95% CI');axes[1].set_title('冻结后宏平均提升')
    axes[1].grid(axis='y',alpha=.2);axes[1].spines[['top','right']].set_visible(False)
    fig.suptitle('实验12 R2：双编码器 + 按shot选择闭式读出',fontsize=15)
    fig.text(.5,.01,'R1为冻结DINO检索基线；R2使用CLIP+DINO。区间对应既定图像池中的episode配对重采样。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.045,1,.95]);fig.savefig(HERE/'improvement.png',dpi=180);plt.close(fig)
    fast=[r for r in api['rows'] if r['shot']==1];slow=np.median([r['fit_ms'] for r in fast]);quick=np.median([r['prepared_gallery_fit_ms'] for r in fast])
    lines=['# 实验12第二轮优化：R2结果与使用（2026-09-09）','',
      '**R2通过本轮实用升级门槛，已封装为可直接使用的版本。** 相对上一轮冻结DINO均匀检索基线，旧四条件宏平均提升1-shot **0.78pp**、5-shot **0.85pp**；实验12新增的四条件分别提升 **2.02pp**、**2.09pp**。新增方法相对加强对照的独立贡献门槛仍未通过，当前定位是更强的可复用实验基线。','',
      '![提升总览](improvement.png)','',
      '## 最终采用的优化','',
      '| 设置 | 冻结配置 | 图库访问 |','| --- | --- | --- |',
      '| 5-way 1-shot | CLIP与DINO等比例融合；每类检索64邻居；真实支持与邻居原型各占0.5；带截距岭回归，λ=0.1 | 每题按支持原型检索一次 |',
      '| 5-way 5-shot | CLIP与DINO等比例融合；直接在25张真实支持图上拟合带截距岭回归，λ=1 | 读出不访问图库 |','',
      '两个编码器保持冻结。方法输入只有支持特征/标签和无标签图库，查询逐张独立处理。两种编码器必须对齐到同一批图像；融合增加了编码成本，报告同时保留单DINO对照。图库融合与归一化可以预处理一次，在多个episode间复用。','',
      '本轮比较53种配方×4种特征配置，共212个候选配置/shot：真实支持原型与岭回归、16/64/256邻居均匀混入、2/4轮带原支持锚点的再检索、保留邻居分布的核读出、检索后岭回归。按四个开发条件共同选择，候选与对照都可使用同样的编码器组合。5-shot最终选择真实支持岭回归，重复检索和核读出没有获选为默认。','',
      '## 冻结后的全条件结果','',
      '下表所有R1、R2方法使用完全相同的新采样episode。R1是上一轮固定的DINO均匀检索配置；R2始终使用开发集选定的overall配置，未按各测试条件重新挑选。','',
      '| 条件 | 1-shot：R1→R2 | 变化 | 5-shot：R1→R2 | 变化 |','| --- | ---: | ---: | ---: | ---: |']
    for cell,label in zip(order,labels):
        a=lookup[(cell,1)];b=lookup[(cell,5)]
        lines.append(f"| {label} | {pct(a['r1_dino'])}%→{pct(a['overall'])}% | {pct(a['overall_minus_r1_dino']['mean'])}pp | {pct(b['r1_dino'])}%→{pct(b['overall'])}% | {pct(b['overall_minus_r1_dino']['mean'])}pp |")
    lines+=['','| 汇总范围 | 1-shot提升与95% CI | 5-shot提升与95% CI |','| --- | --- | --- |']
    for phase,label in [('retest','旧四条件复测'),('external','新增四条件验证')]:
        a=s['pooled'][phase]['1']['overall_minus_r1_dino'];b=s['pooled'][phase]['5']['overall_minus_r1_dino']
        lines.append(f"| {label} | {pct(a['mean'])}pp，{interval(a['ci95'])} | {pct(b['mean'])}pp，{interval(b['ci95'])} |")
    lines+=['',
      '预设实用门槛是宏平均提升≥0.5pp且配对区间下界>0、共享类条件损失≤0.5pp、压力条件损失≤1pp；两个阶段、两种shot均通过。DTD同域1-shot下降0.14pp，CUB同域几乎持平，收益有明显任务差异。[完整区间、逐种子成绩与门槛](summary.json)。','',
      '## 提升来源与保留边界','',
      '1-shot在相同融合权重、相同检索深度、相同混入强度下，检索后岭回归相对均匀原型读出的宏平均提升约为旧条件0.73pp、新增条件0.60pp。5-shot相对同一融合特征的纯支持原型，岭回归分别提升0.40pp、0.90pp。因此提升同时来自特征融合和判别读出，不能全部归因于检索机制。固定参数消融为同episode诊断，见[消融结果](ablation_summary.json)。','',
      '1-shot候选相对开发选出的最强简单对照，在旧条件提升0.47pp，在新增条件提升1.38pp；旧条件未达预定1pp贡献门槛，且CUB同域候选略低于简单对照。5-shot获选版本本身就是加强后的简单对照。当前没有足够证据把R2包装成独立原创方法。','',
      '跨域问题仍然存在：1-shot R2相对同一融合特征的纯支持原型，在DTD、CUB、EuroSAT使用miniImageNet图库时分别下降0.18、0.89、1.54pp。相对R1的提升不能解释成“跨域图库一定有益”。在新领域使用时，应同时报告同特征的纯支持基线。','',
      '若预算严格限制为单DINO，开发选出的1-shot新机制在旧/新增条件相对R1仅变化+0.06/−0.04pp，区间均跨零；当前稳定提升主要需要双编码器或5-shot判别读出。单DINO的5-shot岭回归在旧/新增条件分别改善约0.31/1.42pp，保留在选择记录和实验输出中。','',
      '## 数据、验证与成本','',
      '- 开发：旧4条件×2shot×200=1,600个episode。冻结后：旧4条件和新增4条件各×2shot×600，共9,600个episode、720,000次查询判定/方法。每条件使用3个种子，各200个episode。',
      '- CUB使用官方train图库和test查询；图库RGB去重后5,993张、查询池5,794张。EuroSAT沿用已有确定性训练/测试划分，按预先固定种子从train每类取500张，共5,000张图库，查询池5,400张。图库/查询池的完全相同RGB重叠均为0，跨域组合也逐对排除查询池图像。',
      '- CUB、EuroSAT是实验12本轮新增的评价条件，曾被其他本地项目使用；旧基准的历史复测数据也已查看。开发/复测用固定类别半分，配置在任何本轮复测/新增条件成绩生成前已冻结。区间为固定图像池内配对episode估计，另提供类别宏平均敏感性，不把重复抽样当成新增源图。近重复和预训练重叠尚未全面审计。',
      '- 9组公式/来源/结果验证通过；53种公式全部检查查询批次独立、episode独立和类别顺序等变；独立原始空间岭回归与对偶实现最大误差4.17e−9。新增资产另有16项真实编码器特征复核。新API与冻结实验对拍48个episode、3,600条预测，全部一致；5-shot传入空图库仍得到同样预测。',
      '- R1的146份证据文件逐一SHA256检查保持不变。本轮特征和输出放在独立目录；原pilot、R1失败与停止条件保留。',
      f'- CPU四线程、缓存特征已在内存的48个API对拍episode中，1-shot含图库融合的适配中位数约{slow:.1f}ms；先预处理图库后降至{quick:.1f}ms，约{slow/quick:.1f}倍。此为不同图库条件下抽测的适配时间，图像编码另计；5-shot读出图库访问为0。新图库共新增21,988次编码器图像前向，原查询特征复用已核验缓存。','',
      '## 使用优化版','',
      '实现已加入原实验目录：[e12/optimized.py](../实验12——少样本支持集的检索增强与可靠性加权/e12/optimized.py)。输入为冻结CLIP-B/16与DINOv2-S/14的特征，须先按图像身份完成支持/查询/图库隔离。','',
      '```python',
      'from e12.optimized import prepare_gallery, fit_optimized',
      '',
      '# 1-shot：图库融合只准备一次，多题复用。',
      'gallery = prepare_gallery(gallery_clip, gallery_dino)',
      'model = fit_optimized(support_clip, support_dino, support_labels,',
      '                      prepared_gallery=gallery)',
      'predictions = model.predict(query_clip, query_dino)',
      '',
      '# 5-shot：自动使用真实支持岭回归，省去图库参数。',
      'model = fit_optimized(support_clip_5shot, support_dino_5shot, labels_5shot)',
      'predictions = model.predict(query_clip, query_dino)',
      '```','',
      '复现入口：本目录`run.py --phase retest`、`run.py --phase external`；随后运行`analyze.py`、`verify.py`、`verify_packaged.py`。使用`C:\\ProgramData\\anaconda3\\envs\\torch\\python.exe`，已有本地资产可直接运行。重新运行会更新本轮同名输出；冻结配置及其源码指纹见[selection.json](selection.json)、[selection_receipt.json](selection_receipt.json)。','',
      '## 查新与定位','',
      '本轮继续采用知识库的失败条件与已查看数据规则；检索未发现实验12已归档的独立新机制。方法上，缓存核读出可参照[Tip-Adapter（ECCV2022）](https://arxiv.org/abs/2207.09519)，该工作使用有标签支持缓存和CLIP先验，本轮核候选改用检索得到的伪支持，权限不同。少样本学习中的闭式岭回归已有[Meta-learning with differentiable closed-form solvers（ICLR2019）](https://robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/)；本轮使用冻结编码器，未运行其元训练。实际提升已通过上述本地对照验证，新颖性仍须独立建立。','',
      '[全条件CSV](comparison.csv) · [公式与数据检查](verification.json) · [API对拍与计时](packaged_api_verification.json) · [外部特征复核](external_feature_parity.json) · [冻结协议](protocol.json) · [上一轮报告](../续验11与12_20260909/续验与价值判断.md)','']
    (HERE/'优化报告.md').write_text('\n'.join(lines),encoding='utf-8')
    # Snapshot status files once before integrating the new, verified result.
    snaps=HERE/'input_snapshots';snaps.mkdir(exist_ok=True)
    for p in [E12/'README.md',E12/'protocol.json',ROOT/'research_directions.json',ROOT/'当前研究方向.md']:
        name=('e12_' if p.parent==E12 else 'root_')+p.name
        if not (snaps/name).exists():shutil.copy2(p,snaps/name)
    p=E12/'README.md';txt=p.read_text(encoding='utf-8');start=txt.index('编号12');end=txt.index('\n',start)
    intro='编号12，第一代；研究标识`E12-20260908`。**2026-09-09第二轮R2优化通过实用升级门槛：旧条件1/5-shot宏平均较R1提升0.78/0.85pp，新增CUB与EuroSAT相关条件提升2.02/2.09pp。** 当前版本为CLIP+DINO融合：1-shot检索后岭回归，5-shot真实支持岭回归；图库可一次预处理。见[优化报告](../实验12_优化提升_20260909/优化报告.md)与[可调用实现](e12/optimized.py)。独立新方法贡献门槛尚未通过；R1的可靠性权重失败及[历史续验](../续验11与12_20260909/续验与价值判断.md)保留。'
    txt=txt[:start]+intro+txt[end:]
    txt=txt.replace('[冻结续验协议](../续验11与12_20260909/protocol_v1.json)','[冻结优化协议](../实验12_优化提升_20260909/protocol.json)')
    p.write_text(txt,encoding='utf-8')
    p=E12/'protocol.json';obj=read(p);obj.update(status='r2_practical_upgrade_validated_method_increment_gate_not_passed',latest_report='../实验12_优化提升_20260909/优化报告.md',continuation_protocol='../实验12_优化提升_20260909/protocol.json')
    obj['current_implementation']={'module':'e12/optimized.py','entry':'fit_optimized','precompute':'prepare_gallery','encoder_configuration':'equal CLIP-B/16 plus DINOv2-S/14','selection':'../实验12_优化提升_20260909/selection.json'};dump(p,obj)
    p=ROOT/'research_directions.json';obj=read(p);obj['updated']='2026-09-09'
    for d in obj['directions']:
        if d['number']==12:d.update(status='r2_practical_upgrade_validated_method_increment_gate_not_passed',latest_report='实验12_优化提升_20260909/优化报告.md')
    dump(p,obj)
    p=ROOT/'当前研究方向.md';txt=p.read_text(encoding='utf-8');marker='## 2026-09-09：实验12第二轮优化R2'
    if marker not in txt:p.write_text(txt+'\n'+marker+'\n\nR2通过实用升级门槛，已加入双编码器融合与按shot选择的闭式读出；旧条件提升0.78/0.85pp，新增条件提升2.02/2.09pp。独立方法贡献门槛未通过，保留所有负结果。[报告与使用入口](实验12_优化提升_20260909/优化报告.md)。\n',encoding='utf-8')
    items={}
    for p in HERE.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['assets','__pycache__']):continue
        if p.name=='evidence_manifest.json':continue
        items[str(p.relative_to(HERE))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    dump(HERE/'evidence_manifest.json',{'created_at':time.time(),'files':items,'packaged_implementation':{'path':str(E12/'e12'/'optimized.py'),'sha256':sha(E12/'e12'/'optimized.py')},'preserved_prior_run':'../续验11与12_20260909/evidence_manifest.json'})
    print('R2 report, chart, integration and evidence manifest saved',flush=True)

if __name__=='__main__':main()

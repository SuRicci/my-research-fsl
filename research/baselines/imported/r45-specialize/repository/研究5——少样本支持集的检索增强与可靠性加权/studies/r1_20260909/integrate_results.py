# -*- coding: utf-8 -*-
"""Update only E11/E12 status and links; retain input snapshots and pilot outputs."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
report='续验11与12_20260909/续验与价值判断.md'
status={11:'bounded_continuation_completed_local_readout_no_go',12:'bounded_continuation_completed_retrieval_baseline_useful_weighting_no_go'}
for number in [11,12]:
    folder=next(ROOT.glob('实验'+str(number)+'*'))
    path=folder/'README.md';txt=path.read_text(encoding='utf-8')
    start=txt.index('编号'+str(number));end=txt.index('\n',start)
    if number==11:
        paragraph='编号11，第一代；研究标识`E11-20260908`。**2026-09-09有界续验完成：恢复官方完整图像与6-shot协议，去重测试124题；验证集选定全局融合岭回归60.08%，局部空间读出57.66%，当前局部机制未过门槛。** 查新确认原论文已包含冻结特征均值相似度基线。见[续验与价值判断](../'+report+')。首轮[历史结果](results/pilot_notes_20260908.md)保留，其“追平官方基线”“概念族无结构”的过强判断已由新报告修正。'
    else:
        paragraph='编号12，第一代；研究标识`E12-20260908`。**2026-09-09有界续验完成：16条件×600复测episode，均匀检索在共享类1-shot提升3.15–7.86pp；同总质量可靠性加权没有额外收益，跨域DINO仍退化1.65pp。** 保留简单检索基线，停止扩大当前加权/回退机制。见[续验与价值判断](../'+report+')及[查新说明](../续验11与12_20260909/查新与设计差异.md)。首轮[历史结果](results/pilot_notes_20260908.md)保留，其“类不相交一致负收益”概括已修正。'
    txt=txt[:start]+paragraph+txt[end:]
    marker='以下为首轮立项时的研究问题与门槛；本轮执行以[冻结续验协议](../续验11与12_20260909/protocol_v1.json)和新报告为准。'
    if marker not in txt:txt=txt.replace('\n\n**研究问题', '\n\n'+marker+'\n\n**研究问题',1)
    path.write_text(txt,encoding='utf-8')
    p=folder/'protocol.json';obj=json.loads(p.read_text(encoding='utf-8'))
    obj['status']=status[number];obj['last_updated']='2026-09-09';obj['latest_report']='../'+report
    obj['continuation_protocol']='../续验11与12_20260909/protocol_v1.json'
    obj['novelty_check']='targeted_primary_source_audit_completed_20260909_no_novelty_claim'
    if number==11:obj['data']['primary']='Bongard-OpenWorld publisher image backup and official metadata restored in continuation assets; 14140 images; official support/query protocol, audited subset and full-split sensitivity'
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
p=ROOT/'research_directions.json';obj=json.loads(p.read_text(encoding='utf-8'));obj['updated']='2026-09-09'
for entry in obj['directions']:
    if entry['number'] in [11,12]:entry.update(status=status[entry['number']],latest_report=report)
p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
p=ROOT/'当前研究方向.md';txt=p.read_text(encoding='utf-8');marker='## 2026-09-09：实验11、12有界续验'
if marker not in txt:
    txt+='\n'+marker+'\n\n实验11恢复完整数据和严格任务协议后，局部读出未胜全局岭回归对照；当前机制停止扩大。实验12的简单检索有1-shot实际收益，可靠性加权和回退未过新增贡献门槛；优先保留简单检索基线。\n\n[完整结果、代码与复现入口]('+report+')。旧数据与pilot结果保留。\n'
    p.write_text(txt,encoding='utf-8')
print('E11/E12 README, protocols, direction registry and current-direction page updated')

from pathlib import Path
import json,datetime,hashlib,html
w=Path.cwd();q=Path('/Users/decoqwq/DeepScientist/quests/012');o=w/'artifacts/reports/pause_20260914'
rows=json.loads((o/'directions_authored.json').read_text());ledger={r['item_id']:r for r in json.loads((o/'ledger_inventory.json').read_text())};cleanup=json.loads((o/'cleanup_receipt.json').read_text())
aliases={'prior-calibration-source-transfer':'prior-calibration-qualification-20260913','oslo-retained-influence':'oslo_influence','oslo-final-step-counterfactual':'oslo_final_step_intervention'}
evidence=[]
for i,r in enumerate(rows,1):
 ident=r[0];item=ledger.get(ident,{})
 paths=[(q/Path(s) if s.startswith('.ds/') else (Path(s) if Path(s).is_absolute() else w/Path(s))) for s in item.get('source_paths',[]) if isinstance(s,str)]
 for top in ['experiments/main','experiments/analysis']:
  root=w/top/aliases.get(ident,ident)
  for f in ['REPORT.md','RESULT.json','outputs/analysis.json','outputs/result_summary.json','outputs/metrics_summary.json','outputs/eval/metrics_summary.json','analysis.json','protocol.json']:
   if (root/f).is_file():paths.append(root/f)
 if ident=='dino-base-control-20260914':paths.insert(0,w/'artifacts/reports/dino-base-control-20260914.md')
 paths=list(dict.fromkeys(str(p) for p in paths))
 if not paths:paths=[str(w/'paper/evidence_ledger.json')]
 evidence.append({'id':f'E{i:02d}','direction_id':ident,'title':r[1],'sources':[{'path':p,'exists':Path(p).exists(),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() and Path(p).stat().st_size<2**20 else None} for p in paths],'ledger_item_id':item.get('item_id'),'original_scope':item.get('setup'),'original_result_summary':item.get('result_summary')})
(o/'evidence_index.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
main=[x.name for x in (w/'experiments/main').iterdir() if x.is_dir()]
covered={aliases.get(r[0],r[0]) for r in rows};missing=[x for x in main if x not in covered];assert not missing,missing
assert set(ledger)<=set(r[0] for r in rows)
now=datetime.datetime.now(datetime.timezone.utc)
L=['# 实验4与实验5：可信复用、自主改进与暂停总报告','',f'整理日期：{now:%Y-%m-%d}。状态：当前实验结束，结果归档与清理完成，按要求暂停。','',
'## 1. 结论先行','',
'本轮最值得保留的是几类有边界的正面结果与一套可信比较记录，尚未得到“在相同权限下稳定胜过强基线、且可跨独立数据集推广”的新方法。更复杂的权重、训练目标、局部描述子或更大编码器，多次带来局部改善，但未满足各自预先设定的稳健标准。','',
'1. **实验5的低成本中心化有可信小收益。** 在原始规范比较中，单样本宏平均由72.7733%提高到73.1333%，配对增益0.3600个百分点；在同数据集的新图像池上仍有0.3250个百分点改善。不过收益不均匀，且中心化不是新算法。','2. **真实多视图与散度是较有价值的开发域方向。** 在另一套六视图、已用于开发的固定任务中，累积方案达到77.374%；相对同视图均值R2提高2.1653个百分点，相对均值CS提高1.8340个百分点。增益主要来自EuroSAT，不能与上一条72.7733%的不同协议直接相减。','3. **独立域强对照没有被稳定超越。** Caltech上两个冻结累积配置为98.0320%和97.9787%，低于增强支持岭回归98.3307%和C10逻辑回归98.3920%。原数值审计失败及后续修复均保留；修复没有改变“缺乏强对照优势”的判断。','4. **最后一轮更强编码器没有通过升级标准。** DINOv2-B候选均值77.598%，旧方案77.374%；差值+0.2240个百分点，95%条件区间[-0.0700,0.5240]，既未达到0.5门槛，下界也未大于零。因此不把最高点估计当成稳健新方案。','5. **实验4仍有域内潜力，但本轮桥接几何未得到整体提升。** mAP43.0543对44.3259；DTD改善而EuroSAT下降。保留这一域差异，拒绝把事后参数诊断写成新成功。','6. **论文线最清楚的是一条诊断发现。** 图库样本“排序较好”不等于累计权重有益；任务外类别可能仍占据大部分影响。现有OSLO技术短文围绕这一窄结论，尚不是可投稿完成包。','',
'### 阅读范围与计数','',
f'本报告逐项覆盖当前工作区全部{len(main)}个主研究目录，整合{len(rows)}个方法与分析条目，包括原论文账本的42条及未纳入该账本的残差描述子、特征重构、DINOv2-B三项。另将精度修复、错误互补、参数选择不稳定与执行问题放入专节。441条去重归档记录只是本次读取的管理记录快照，包含重复汇报与状态记录，不等于441个实验；45个条目也不等于45次独立复现。','',
'本报告为面向项目负责人的证据与问题汇总，不是将全部历史实验强行拼接成同一篇论文。局部正收益、独立测试失败、已有方法与特权诊断在下文分别标明。','',
'## 2. 比较协议与可信基线','',
'### 2.1 历史材料如何复用','',
'最初实验5仅在DTD五样本、原三个种子的600任务上得到独立重放：90.1533%，原记录90.1489%，相差0.0044个百分点。这一近似重放只支持该切片，原16行历史结果保留为参考，不代表全部已复现。历史完整逐任务预测或排名缺失，不能仅凭汇总表补造统计区间。','',
'实验4历史CUB64桥接方向有mAP65.85对62.51的局部记录，但oracle保留率39.12%未达到50%目标。它不是本次DTD/EuroSAT规范检索的新证据。初始专项核心快照1406个文件哈希匹配，只证明该核心快照一致；完整预测、排名不在该快照中。','',
'### 2.2 三套规范比较各自保留','',
'| 比较范围 | 基线核心值 | 可支持什么 | 不能混同什么 |','|---|---:|---|---|','| 原始DTD/EuroSAT单视图分类 | R2宏平均72.7733% | 同冻结特征、同任务、四单样本条件及五样本守护比较 | 六视图新开发协议、Pets分数 |','| DTD/EuroSAT语义桥接检索 | 旧空间中心化mAP44.3259；V0变体43.3067 | 固定32/64桥接、同全图库排名比较 | 历史CUB/Oxford实例检索或SOTA |','| Pets冻结图库迁移 | 同域单样本96.2400%，异域93.9307% | 同权限图库与支持集控制，五样本亦保留 | 原SWAT/iLPC/OSLO所有论文设定 |','',
'Pets五样本同域和异域R2均为98.4427%，原因是其五样本头仅用支持集。两个相同图库条件不是独立证据；Pets单样本宏平均95.0853%也不能与DTD/EuroSAT宏平均直接排行。','',
'### 2.3 统计与权限边界','',
'置信区间主要采用按原种子分层、任务配对的bootstrap，条件于固定图像池。它们不是随机新域的置信区间，也没有自动包含反复研究选择造成的不确定性。图库条件共享任务时先在任务内平均；训练种子、任务种子、图库重复条件和独立图像池分别计数。','',
'早期固定图库方法禁止查询批次适应；后期逐查询一致性允许单个查询及其视图参与特定计算，仍不使用整个查询批次或答案。两者权限不同。用真实图库类别作的组成过滤、oracle选专家仅用于诊断，不能与可部署算法同列。预训练数据是否与目标语义或图片完全独立尚未获证；解码RGB去重不等于语义重复排除。','',
'## 3. 最有用的结果对照','',
'### 3.1 已开发六视图协议内的公平累积比较','',
'| 方法 | DTD (%) | EuroSAT (%) | 两域均值 (%) |','|---|---:|---:|---:|','| 原始单视图R2，仅作信息量参照 | 76.5853 | 70.3360 | 73.4607 |','| 六视图均值R2 | 77.9520 | 72.4653 | 75.2087 |','| 六视图均值CS | 77.8133 | 73.2667 | 75.5400 |','| 增强支持集岭回归 | 76.7440 | 72.1573 | 74.4507 |','| 固定散度预测融合 | 78.0773 | 76.5907 | 77.3340 |','| 加逐查询一致性 | 78.1573 | 76.5907 | 77.3740 |','',
'累积方案相对六视图均值R2为+2.1653个百分点，区间[1.9600,2.3640]；相对六视图均值CS为+1.8340，[1.6380,2.0214]。前一比较中DTD仅+0.2053，EuroSAT+4.1253。表中单视图到累积方案的+3.9133包含额外编码信息，不能宣称为等预算几何改进。','',
'### 3.2 最后完成的DINOv2-B对照','',
'| 比较对象 | B候选减对照 (百分点) | 95%条件区间 |','|---|---:|---|','| 原小编码器累积方案 | +0.2240 | [-0.0700, 0.5240] |','| 同B编码器均值R2 | +2.3380 | [2.1493, 2.5194] |','| 同B编码器均值CS | +2.0647 | [1.8840, 2.2494] |','| 同B编码器支持集岭回归 | +3.5807 | [3.3373, 3.8287] |','| 同B编码器逻辑回归C1 | +4.2593 | [4.0093, 4.5207] |','| 同B编码器逻辑回归C10 | +3.8607 | [3.5960, 4.1340] |','| 同B编码器增强支持岭回归 | +3.1847 | [2.9547, 3.4247] |','',
'该轮唯一未满足汇总优越性要求的比较对象是原小编码器累积方案；各域点估计无害及四图库条件下界大于-0.5的要求均满足。结论是“不足以升级这一固定配方”，不是“更强表征一定无用”。','',
'14,153图像的六视图特征完成验证；1,000独特任务、2,000图库任务条件、9种方法的1,350,000条预测算术及34个区间通过复核。原串联审核只重建每条件的第99任务；补充审核完成协议全部五个指定索引，共40个方法-任务-条件独立代数案例，最大绝对分数误差1.4433e-15。逻辑回归优化器本身未独立重写。','',
'## 4. 各方向的结果、优点与疑问','',
'“优点”包括已测出的性能优势和有证据支持的诊断价值；若只是合理动机，会明确写成检验目的，绝不当成已证实的收益。以下编号可在evidence_index.json找到原始报告、数据和协议路径。','']
for r,e in zip(rows,evidence):
 ident,title,verdict,result,benefit,problem,question=r
 L += [f'### {e["id"]} {title}', '',f'**当前判断：{verdict}。** {result}','',f'**值得保留的优点或价值：** {benefit}','',f'**问题与限制：** {problem}','',f'**尚未解决：** {question}','']
 existing=[x for x in e['sources'] if x['exists']]
 primary=next((x for x in existing if x['path'].endswith('.md')),existing[0] if existing else None)
 if primary:L += [f'证据：[{e["id"]} 原始记录]({Path(primary["path"]).resolve().as_uri()})。','']
L += ['## 5. 执行中遇到的问题与已完成的排查','',
'### 5.1 数值精度：相同预测不等于完整分数等价','',
'散度原审核87/88通过，DTD源单样本第99任务在gamma0.1下有一个top64邻居因float32近并列而交换，完整分数误差约0.003034超过固定2e-5。固定邻居后回归误差小、预测不变，只能定位问题，不能把原失败改成通过。','',
'随后完整检查1,200个单样本源配置、90,000个查询预测，保存float32与独立float64的预测计数和gamma选择均不变：DTD选择10，EuroSAT选择1。独立float64两种代数路径最大误差约2.03e-14，原float32失败仍为同一例。数值选择稳定已经核实，只有两计数的开发领先是否对新任务稳定仍未知。','',
'Caltech的确定性float64控制修复使固定15个真实任务和1个合成任务通过，112,500条保存预测未变；相对修复控制，两个配置差值分别-0.0471和-0.1004个百分点，仍无稳健优越性。原失败、诊断与修复作为三个证据层同时保留。','',
'### 5.2 任务采样与泄漏检查','',
'源身份池对半划分时，DTD每类最少37图像，一半可能只有18张，无法同时抽取5支持与15查询。实际曾在两项单样本条件之后失败；修正仅把源训练/选择的五样本查询数设为13，目标仍15，保留已完成单样本记录。该变更是可行性修复，不能将修复前后当作同一个未经修改的协议。','',
'Pets类别和DTD类别编号不能直接数值比较。新图像确认检查解码RGB身份重叠；Caltech目标中的500任务跨三图库复用，只是1,500条件，不是1,500独立任务。任务出现次数、独特图片、类别和独立数据集均不混算。','',
'Pets失败覆盖也有具体证据：同域单样本37类中21类低于R2，异域37类全受损；同域选入的图库样本约80%属于任务外类别，异域为100%。分类头线性残差很小使纯求解器错误解释较不可信，但不能据此认定唯一建模原因。','', '### 5.3 表征与实现细节','',
'CLIP激活与DINO实现差异促成新的规范特征变体；没有把旧缓存分数与新实现强行要求完全相等。MPS位置插值采用经过实测的CPU回退；硬件支持Metal不直接视为软件路径可用。','',
'EuroSAT原图是64×64，直接Resize224后做FiveCrop224会得到重复视图。实际采用不同尺度/裁剪，并显式记录视图与编码预算；视图个数增加并不是免费改进。局部16描述子的有效参与秩约1.4，共享图像成分占约84%；这只是几何观察，后续残差化的大幅下降说明删掉共享成分并未得到实用方法。','',
'最后一轮分类直接入口曾在产生结果之前崩溃。通过已验证的依赖导入顺序与精确模块路径恢复，没有修改科学代码或参数；正式计算和后续审核均退出成功。原始错误与修复命令保存在recovery.json和运行日志。','',
'### 5.4 选择不稳定、弱对照与元数据误导','',
'实验4开发集对alpha2相对alpha1的偏好很小，配对区间重叠；bootstrap最优频次不是泛化概率，不能事后改选alpha1宣称成功。逻辑回归C1或岭回归固定惩罚1可能是弱控制，必须与等调参简单控制比较。','',
'某些自动摘要把负总体结果标成“major”，或把最新候选称作incumbent；这些是管理标签，不是科学判定。主张以逐任务文件、固定门槛和审计报告为准。论文42条账本中仅4组直接支撑OSLO短文，38项是参考历史，不是42组独立支撑。','',
'发生过NumPy整数JSON序列化错误、导入路径与配置修复、Git分页等待、完成任务未及时登记、特征完成后约两小时未自动接上分类。修复与原失败均保留。后续应由一个可恢复的计算-检查链驱动，避免把模型继续生成消息当成作业调度器；本次已暂停，不再启动新链。','',
'## 6. 尚未回答的研究问题','',
'1. **何时图库真正有益？** 同数据集图库也可能大部分不属于当前五类；域相同不等于任务相关。缺少能稳定识别有益影响、同时不读图库标签的办法。','2. **为何多视图散度主要帮助EuroSAT？** 已有中心化、目标函数、局部描述子和更强表征对照缩小了可能解释，但没有唯一因果归因。','3. **能否从专家互补中得到可部署选择器？** oracle空间1.8833个百分点是真实诊断，训练选择器却修坏多于修好；可观测可靠性信号仍是缺口。','4. **为什么源域优化会伤害迁移？** 混合源、最差组、视图对齐等目标改进弱训练对照，却不能稳定超过不适配参考。未来需要明确的新信息或监督条件，不能继续无依据加正则。','5. **简单方法的优势能否推广？** 中心化与强正则在一些协议上有效，但独立域、少数类别、不同预训练覆盖和计算成本还不能统一概括。','6. **更大编码器是否值得？** 最后点估计提高0.224个百分点但区间包含零；当前不能证明性能收益抵得上编码成本，更不能推出容量因果规律。','7. **论文贡献是否足够？** 现有正收益大多基于已知组件或暴露开发集，跨域强基线资格未过。诊断短文有清楚现象，但不等于已经具备高水平投稿的新颖性与外部充分性。','',
'MIV式支持验证头作为结构不同的方向只完成了先前文献与输入需求核对，未在本任务产生新实验结果；接续评估已按本次暂停指令取消。需要训练编码器或引入类文本的直接方法不满足当前固定权限，属于适用范围排除，不属于测量失败。以上均是恢复后的候选问题，本次不安排继续运行。','',
'## 7. 论文与交付状态','',
'现有OSLO技术短文、LaTeX、PDF、证据账本和审阅记录保留。其中心是“排序质量与累计图库影响需分别检查”，不是所有45项历史都支持这一机制。新残差、重构和更大编码器结果在本总报告中完整呈现，没有临时改写原论文结论。','',
'已有检查显示草稿可编译、引用和表格已有核对，但包类型仍为draft_checkpoint，缺少投稿包与完成清单；本报告不宣称论文投稿就绪。原有文章质量提醒、失败实验和未解问题全部保留。暂停是按用户要求形成稳定交付点，不是对论文成功的认证。','',
'## 8. 工作区清理与保留内容','',
f'- 清理完成时间：{cleanup["completed_at"]}。删除{cleanup["removed_weight_or_tensor_files"]:,}个权重或可再生张量缓存文件；合计删除{cleanup["deleted_files"]:,}个文件及{cleanup["deleted_directories"]}个已核实原图目录。大量文件来自工作树中的重复数据副本，不代表相同数量的独立资产。',f'- 删除文件的逻辑体积合计{cleanup["deleted_logical_bytes"]/2**30:.2f} GiB；实测可用空间由{cleanup["free_before_bytes"]/2**30:.2f}增加至{cleanup["free_after_bytes"]/2**30:.2f} GiB，释放{cleanup["observed_free_delta_bytes"]/2**30:.2f} GiB。逻辑总量不等于实际释放量，二者不混报。',f'- {cleanup["protected_file_count"]:,}个结果、配置、源码和论文复核文件在清理前后逐文件SHA256一致，差异为0。逐任务预测、完整分数、聚合统计、协议、错误日志和失败结论均保留。','- 原始外部用户材料、共享环境、用户目录下的公共模型缓存、Git历史及DeepScientist运行状态未清理。少量用途混合的图库邻居特征档案和用于已完成输入一致性诊断的张量仍保留，具体列在cleanup_receipt.json的uncertain_preserved中；它们没有被冒称模型权重。','- 完整清理计划、删除权重哈希、目录路径/大小摘要、受保护文件哈希和执行回执均在本交付目录。没有做Git历史重写。','',
'**清理后的能力边界：** 可以继续核对保存的预测和统计结果，但不能承诺立即从原图重新编码或重训所有头。需要重新下载模型/数据或恢复缓存的命令与来源仍在原协议；重跑前必须按删除清单恢复所需输入。权重清理属于此次明确要求的存储取舍，不掩盖为“所有实验仍可零准备复现”。','',
'## 9. 暂停与恢复入口','',
'本轮在“最新结果已审计、所有方向有报告、清理有回执”的状态暂停。无新候选、训练或目标评估等待自动启动。恢复需新的用户消息或/resume，并从这份总报告、evidence_index.json、cleanup_receipt.json与handoffs/CURRENT_CHECKPOINT.md读取最新状态。','',
'恢复时先确定新问题和预算，再按所需路径恢复输入。默认不重跑已失败的权重/层/混合强度网格，不重复已完成的精度审核；只有新假设、实际实现错误或明确的新权限才重新打开相应方向。所有未证实的问题保留为问题，不预设下一轮一定成功。','',
'## 附录：证据与文件导航','',
'同目录文件：研究总报告.md（可编辑原稿）、研究总报告.html（浏览版）、研究总报告.pdf（便携版）、evidence_index.json（逐方向来源）、cleanup_receipt.json（清理回执）、deleted_weight_hashes.json（权重哈希）、protected_results_before.json（保留结果哈希）、review_report.md（本报告审阅）、coverage_validation.json（覆盖核验）。源实验结果留在各自原位置，未为了合并报告复制巨型预测银行。','',
'关键补充原始报告：累积同身份比较位于experiments/analysis/accumulated-stack-comparison-20260913；目标精度修复位于caltech-control-precision-20260913；完整源选择精度核验位于source-selection-precision-20260914。上述名称是检索线索，绝对路径与报告SHA256见evidence_index.json及原始清理/运行记录。','',
'本报告的研究结论来自本地文件、已保存日志和审计记录。原论文中的外部方法仅用于定位与权限比较，未将其公开表格分数当成本任务的可比实测值。']
text='\n'.join(L)+'\n'
text=text.replace('## 9. 暂停与恢复入口', (o/'storage_breakdown.md').read_text()+'\n## 9. 暂停与恢复入口')
(o/'研究总报告.md').write_text(text)
checks={'created_at':now.isoformat(),'direction_count':len(rows),'main_directory_count':len(main),'all_main_directories_covered':not missing,'missing_main_directories':missing,'ledger_items':len(ledger),'all_ledger_items_covered':set(ledger)<=set(r[0] for r in rows),'extra_measured_entries':['dino-residual-20260913','dino-reconstruction-20260913','dino-base-control-20260914'],'each_direction_has_result_advantage_limit_question':all(len(r)==7 and all(r) for r in rows),'all_directions_have_existing_evidence':all(any(s['exists'] for s in e['sources']) for e in evidence),'chars':len(text),'cleanup_protected_hash_mismatches':len(cleanup['protected_hash_mismatches'])}
(o/'coverage_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))

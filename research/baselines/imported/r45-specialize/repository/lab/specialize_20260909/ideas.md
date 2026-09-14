# 候选论证与相关工作

## 重要矛盾与选择

研究4中心化在Holidays有效，残差却多数退步；Oxford/Paris某些桥接下仍有收益。问题可能分别来自旧空间各向异性、新旧相似度尺度、桥接外插噪声和修正范围。研究5的题级门控几乎不能在域内判断检索收益，但域间关闭检索有用；这不等价于候选选择、分布建模和支持标签校准都无效。

本轮选择十个互不完全等价的切入点，属于“有明确失败问题支撑的增量改进”，不宣称机制原创。机制、目标、度量和部署范围均覆盖。更激进的编码器微调、增加图库原图权限、查询批次转导会改变任务契约，故不混入本轮。常数alpha、支持中心化、固定late fusion、覆盖sigmoid、一般margin/hub权重、核读出和重复检索已有尝试，不重复计数。

## 五项研究4预案

|ID|核心假设与实现|最强反对解释及失败检验|与已有路线的差异|
|---|---|---|---|
|r4_whiten|旧图库协方差含高方差干扰方向；用旧图库均值、收缩协方差和逆四次根变换后归一化检索。|低方差方向可能是噪声，白化使其放大；若Hard也退步，不再认为进一步几何校正自然有效。|只改旧空间，不冒充跨编码器贡献；此前只有中心化和桥接相对空间白化。|
|r4_rank|桥接相似度的幅值不可靠，但相对排序可传递；两侧相似度转为高斯秩，差值乘旧桥接标准差后用同RBF/PRESS插值。|少桥接下秩本身粗糙、OOD排序无意义；若CUB与地标都差，排序不变性不能救当前外插。|不是先前用rank误差选择lambda：这里改变拟合目标。|
|r4_diffusion|旧图库邻域结构可平滑桥接外残差；构建旧图库16NN行归一图，把桥接相似度软权重迭代扩散10步，再传递桥接残差。|图连接错误会跨实例传播；若Hard明显下降，局部图不可信。|加入旧图库图结构，不使用任何新图库图，不是生成式扩散。|
|r4_ensemble|少桥接结果受个别锚点驱动；8个75%子集集成，用逐查询-图库分歧收缩平均修正。|子集会丢失仅有的信息，相关偏差不会被分歧发现；若退化是偏差主导，集成不能改善。|改变估计器稳定性，所有子集合计只复用同m张双编码桥接。|
|r4_top100|大范围修正破坏原本正确的全库排序；只让旧top100内部按已有alpha2方案重排，其他位置完全保留。|旧top100漏掉的正例永远无法追回；若收益需要召回外部正例，该限制有害。|明确改变允许修正的排序范围，不声称检索更快或安全保证。|

## 五项研究5预案

|ID|核心假设与实现|最强反对解释及失败检验|与已有路线的差异|
|---|---|---|---|
|r5_consensus|两个不同目标编码器一致检索到的候选更可靠；仅用融合top64中同时属于两单编码器top64的图像，无交集回退支持。|二者可能一致犯错，且交集会损失有效多样性；细粒度域若无增益则不能视为可靠筛选。|此前只把重合作为题级开关信号，这次用于具体候选选择。|
|r5_median|算术均值被离群候选拉偏；以8步Weiszfeld几何中位数替代候选均值，其余R2读出保持。|污染可能是多数簇，几何中位数会更坚定选错；若失败则排除少数离群解释。|不是单独margin、密度权重，而是候选间距离的鲁棒位置估计。|
|r5_variance|检索候选在某些维度剧烈变化，这些维度对当前五类可能是干扰；使用候选类内对角方差收缩后，统一变换支持、候选与逐张查询。|候选方差可能反映判别特征，校正会抹掉信息；EuroSAT/DTD失败将削弱此假设。|从均值调整转到支持侧可估计的局部度量，不使用查询协方差。|
|r5_distribution|一个均值丢失类内多模态；保留原top64中16个固定rank候选，真实支持和候选的加权闭式岭回归保留分布。|伪标签误差会被回归放大；若同域也失败，原型压缩未必是瓶颈。|不同于先前logsumexp核相似度读出；明确控制伪支持总权重与截距。|
|r5_support_cv|5shot有可用于校准的真实标签；用支持内留一风险对五个融合权重软选择，再收缩到等比例基线。|25张支持仍小，留一风险估计噪声大；若EuroSAT无改善，不再盲目增加选择复杂度。|不同于开发集固定融合或无标签代理门控；不适用于1shot，无图库。|

以上候选均可在缓存上运行，主风险是研究收益不确定，工程可实现与可证伪性明确。没有一项把oracle或额外标签当正式方法。每个候选既保留均值结果，也保留对预定问题的反证价值。

## 相关工作核对（有界查新，2026-09-09）

1. Shen et al. *Towards Backward-Compatible Representation Learning*. CVPR 2020. [官方论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)。兼容检索任务已建立，但训练式BCT权限不同；本轮不把任务定义当创新。
2. Moschella et al. *Relative Representations Enable Zero-Shot Latent Space Communication*. ICLR 2023. [论文](https://arxiv.org/abs/2209.15430)。锚点相似度公共坐标是直接重叠；高斯秩残差只是对外插目标的有界变化。
3. Radenović et al. *Fine-tuning CNN Image Retrieval with No Human Annotation*. [论文](https://arxiv.org/abs/1711.02512)。检索描述子的白化是已知方向；本文用无标签旧图库收缩部分白化，不复现其监督/重建驱动变换或训练结果。
4. Iscen et al. *Efficient Diffusion on Region Manifolds*. CVPR 2017. [官方论文](https://openaccess.thecvf.com/content_cvpr_2017/html/Iscen_Efficient_Diffusion_on_CVPR_2017_paper.html)。图扩散用于检索已有充分先例；本轮扩散桥接权重，只用旧侧图且无区域特征。
5. Yadav et al. *Efficient k-NN Search with Cross-Encoders using Adaptive Multi-Round CUR Decomposition*. Findings EMNLP 2023. [论文](https://aclanthology.org/2023.findings-emnlp.544/)。强调top-k误差而非全局平均误差；本轮只限制旧top100修正，不访问候选原图或新增cross-encoder评分，不能借用其性能保证。
6. Blum and Mitchell. *Combining Labeled and Unlabeled Data with Co-Training*. COLT 1998. [作者机构记录](https://publications.ri.cmu.edu/combining-labeled-and-unlabeled-data-with-co-training)。多视图相互校验为相邻机制；CLIP/DINO并非独立视图，本轮不声称其理论假设成立，也不进行互训练。
7. Minsker. *Geometric Median and Robust Estimation in Banach Spaces*. Bernoulli 2015. [论文](https://arxiv.org/abs/1308.1334)。鲁棒聚合的基础邻域；真实图库候选相关且可能多数污染，本轮不套用独立估计器的偏差界。
8. Yang et al. *Free Lunch for Few-shot Learning: Distribution Calibration*. ICLR 2021. [论文](https://arxiv.org/abs/2101.06395)。迁移统计量用于少样本已存在；本轮没有有标签base-class统计、没有生成新样本，只分析无标签检索候选方差。
9. Bertinetto et al. *Meta-learning with Differentiable Closed-form Solvers*. ICLR 2019. [作者页面](https://robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/)。闭式岭回归已知；新增点只能是本权限下候选分布的处理和真实支持内部校准，不能宣称岭回归原创。

该检索覆盖直接兼容检索、锚点表示、图检索、稳健统计和少样本读出，结合仓库原查新与失败报告；未穷尽2026所有预印本。十项均以局部优化实验而非原创算法发布推进。标准文献结论与本轮推断分开记录。

# 2026年9月16日研究暂停与结果增补报告

按最新指令暂停实验。本报告记录9月15日恢复研究后新增的证据；9月14日以前的完整报告与原始研究历史仍在既有云端归档中。本轮没有新的最优方法，也没有形成经过确认的新论文贡献。

## 核心结果

固定比较为DTD与EuroSAT各500个五类五样本任务，均为已暴露开发任务。现有参照平均准确率91.560%，DTD91.010667%，EuroSAT92.109333%。这些数字不能解释为独立新域确认。

| 方向 | DTD (%) | EuroSAT (%) | 平均 (%) | 对照判断 |
|---|---:|---:|---:|---|
| 支持视图交叉熵 | 91.000000 | 92.090667 | 91.545333 | 未超过现有参照 |
| 条件Laplace预测 | 91.000000 | 92.085333 | 91.542667 | 未超过现有参照 |
| 视图轨道信号 | 90.842667 | 89.552000 | 90.197333 | 未超过现有参照 |
| 源训练类别协方差 | 89.792000 | 90.122667 | 89.957333 | 未超过现有参照 |
| 有序裁剪对应 | 90.997333 | 92.048000 | 91.522667 | 未超过现有参照 |

## 保留的优点、负结果与疑问

- 类别协方差：在同一模型族内，相比同先验的共享协方差提升0.288个百分点，相比独立训练的共享协方差提升0.365333个百分点，两个预设门槛均通过。该局部机制效应值得保留；但源先验学习比未训练先验下降1.090667个百分点，完整候选89.957333%明显低于参照。额外源监督必须明确，不能宣称同预算胜出。过拟合、优化过程和域差异尚未被分离。
- 有序裁剪：91.522667%比参照低0.037333个百分点，与打乱裁剪位置的平均结果仅差+0.000267个百分点。全部预设晋升条件失败，关闭这套固定读出，不再用暴露结果挑裁剪权重或正则化参数。只排除本次固定方案，不能推断所有空间模型无效。
- 重复图片误差：DTD99.347%、EuroSAT98.040%的错误发生在另一次任务中曾被正确识别的图片上。EuroSAT有92/1019个重复的相同查询/类别集合随支持样本改变而改变对错；DTD没有这样的匹配重复组。说明错误受任务背景影响，但不等于这些错误可以被新方法挽回。
- 支持视图交叉熵、条件Laplace和视图轨道信号：完整分数、预设比较和审计保留；均没有成为新最优。各自的局部效果和限制以所附原始REPORT.md为准，不将不同协议或模型族的弱对照效应拼成共同优势。

## 实验问题与审计边界

- 当前开发图像池经过反复研究；任务重采样区间仅表示条件精度，不是独立图像或新域泛化区间。
- 源训练协方差与无源标签参照的监督预算不同，必须同时保留更强参照和模型族内控制。
- 有序裁剪主结果最初缺少idea归属字段，已通过补充更正记录修复；数值未变。
- 重复图片分析首次运行因NPZ惰性读取重复解压超时；改成每域一次物化后成功，失败尝试和修复说明保留。
- 现有结果尚不足以支撑新的通用方法贡献或论文可投稿结论。后续任务条件方法只停留在待核对问题，未选择、未实验。

## 证据目录

下列每项报告连同代码、原始数组、协议和审计收进归档。之前的归档位置：https://github.com/SuRicci/my-research-fsl/releases/tag/quest-012-archive-20260914 。

- [experiments/analysis/augmentation-geometry-20260915/REPORT.md](reports/analysis/augmentation-geometry-20260915/REPORT.md)
- [experiments/analysis/calibration-class-partition-20260916/REPORT.md](reports/analysis/calibration-class-partition-20260916/REPORT.md)
- [experiments/analysis/certified-cap-screen-20260916/REPORT.md](reports/analysis/certified-cap-screen-20260916/REPORT.md)
- [experiments/analysis/class-disjoint-natural-20260916/REPORT.md](reports/analysis/class-disjoint-natural-20260916/REPORT.md)
- [experiments/analysis/class-group-spectrum-control-20260916/REPORT.md](reports/analysis/class-group-spectrum-control-20260916/REPORT.md)
- [experiments/analysis/conditional-laplace-qualification-20260916/REPORT.md](reports/analysis/conditional-laplace-qualification-20260916/REPORT.md)
- [experiments/analysis/covariance-coefficient-factorial-20260916/REPORT.md](reports/analysis/covariance-coefficient-factorial-20260916/REPORT.md)
- [experiments/analysis/cross-encoder-coupling-20260916/REPORT.md](reports/analysis/cross-encoder-coupling-20260916/REPORT.md)
- [experiments/analysis/directional-stopping-20260916/REPORT.md](reports/analysis/directional-stopping-20260916/REPORT.md)
- [experiments/analysis/five-shot-deletion-stability-20260916/REPORT.md](reports/analysis/five-shot-deletion-stability-20260916/REPORT.md)
- [experiments/analysis/fixed-covariance-composition-20260916/REPORT.md](reports/analysis/fixed-covariance-composition-20260916/REPORT.md)
- [experiments/analysis/gallery-factorial-20260916/REPORT.md](reports/analysis/gallery-factorial-20260916/REPORT.md)
- [experiments/analysis/gallery-source-scalar-20260916/REPORT.md](reports/analysis/gallery-source-scalar-20260916/REPORT.md)
- [experiments/analysis/label-allocation-20260916/REPORT.md](reports/analysis/label-allocation-20260916/REPORT.md)
- [experiments/analysis/paired-support-instability-20260916/REPORT.md](reports/analysis/paired-support-instability-20260916/REPORT.md)
- [experiments/analysis/prospective-class-group-confirmation-20260916/REPORT.md](reports/analysis/prospective-class-group-confirmation-20260916/REPORT.md)
- [experiments/analysis/repeated-image-errors-20260916/REPORT.md](reports/analysis/repeated-image-errors-20260916/REPORT.md)
- [experiments/analysis/source-original-risk-20260916/REPORT.md](reports/analysis/source-original-risk-20260916/REPORT.md)
- [experiments/analysis/source-response-repeatability-20260916/REPORT.md](reports/analysis/source-response-repeatability-20260916/REPORT.md)
- [experiments/analysis/support-origin-pairing-20260916/REPORT.md](reports/analysis/support-origin-pairing-20260916/REPORT.md)
- [experiments/main/class-covariance-control-20260916/REPORT.md](reports/main/class-covariance-control-20260916/REPORT.md)
- [experiments/main/conditional-laplace-20260916/REPORT.md](reports/main/conditional-laplace-20260916/REPORT.md)
- [experiments/main/orbit-signal-20260916/REPORT.md](reports/main/orbit-signal-20260916/REPORT.md)
- [experiments/main/ordered-crop-correspondence-20260916/REPORT.md](reports/main/ordered-crop-correspondence-20260916/REPORT.md)
- [experiments/main/support-view-ce-20260916/REPORT.md](reports/main/support-view-ce-20260916/REPORT.md)

## 暂停与恢复

研究已暂停，仅执行归档、上传核验和工作区清理。恢复需用户明确指令；先读取本报告、RESTORE.md和清理回执，再决定是否继续任务条件方法核对。不得重跑已完成的有序裁剪、协方差或误差汇总来恢复上下文。

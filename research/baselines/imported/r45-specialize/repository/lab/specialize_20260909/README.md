# 实验4/5：十项分领域改进

本目录对应用户2026-09-09授权的第四轮测试。十项候选、固定配方及领域目标见[candidates.json](candidates.json)，推理、反证条件和相关工作见[ideas.md](ideas.md)，冻结运行规则见[PLAN.md](PLAN.md)。运行根目录为52服务器`/data/liuhaoyu/r45-specialize-20260909`。

执行前在两个独立Python解释器分别运行test_r4_contracts.py和test_r5_contracts.py，避免旧研究同名src模块互相覆盖。run.sh限定物理GPU2，controller.py按候选清单逐项运行；每项单独保存日志、退出码、原始预测和summary.json。源文件SHA、顺序、参数与比较规模在第一次运行时冻结到protocol.json。中途不能直接改冻结源码后续跑。

```bash
cd /data/liuhaoyu/individual-research
bash lab/specialize_20260909/run.sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH=/data/liuhaoyu/r45-scale-20260909/vendor:$PWD \
  /data/liuhaoyu/.conda/envs/torch/bin/python lab/specialize_20260909/audit.py
```

读取旧特征与低维矩阵分解主要使用CPU；实验4图库图构建和实验5候选计算使用GPU。研究4的900个方法比较单元共享底层查询与桥接；研究5的150000次候选题评估共享部分题，均不能称为同等数量独立样本。所有结果使用已查看图像池，属于探索性复测。保留正、负结果与计算代价，局部收益不要求总体均值胜出。

最终报告、机器汇总及核验回执放在results/；完整排名、特征和大预测文件主要保存在服务器。核心包与完整产物备份见交付回执。

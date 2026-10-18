# ROBIN-20261018-01：内部 Robin 参数化原型（2026-10-18）

- Q1/Q2/Q3；**MIXED**；解析仅静态内部凝聚单调性（P，精确算术），其余为有限样本数值（N）和过松充分界（B，普通浮点）；无已完成的连续 HTC × 全频率舍入验证。
- 模型：合成 12×15 RC 网格，共 180 个温度 DOF；**12 个真实连接端口**，底部 **12 个内部 Robin 未知量**；HTC 两分区各 \(h_j\in[0.008,3]\)，对角正热容与独立弱接地，\(H_j\succeq0\)。两个内部功耗点、温度观测与 Robin 环境温度输入。
- 误差：固定 \(W=S_g(0,(0.4,0.4))\)，独立 HTC 随机持出点（种子 20261018）与有限频率集合；不得视为全盒/全频率。全阶富集仅实移位 \(\{0,0.03,0.4,5\}\)；这是**固定移位**消融，未实现自适应算法。
- 固定 ROM 72 阶，10 个持出 HTC 点：标称无灵敏度 \(1.0388\times10^{-2}\)，一阶 HTC 导数 \(3.8057\times10^{-3}\)，HTC 四角训练 \(3.6119\times10^{-3}\)，精确保留内部 Robin 迹 \(4.6456\times10^{-2}\)，均为持出 DtN sampled max。其对应 MIMO 相对误差见研究笔记。
- 三矩+一阶 HTC 导数：72 阶 / 4 次实全阶 LU / 648 RHS 列，训练有限 HTC 上数值 \(H_\infty(J)^2\approx2.0499\times10^{-3}\)，持出 DtN sampled max \(6.4776\times10^{-4}\)。四角三矩：16 次 LU / 864 RHS，训练值 \(7.3316\times10^{-4}\)，持出 \(3.9782\times10^{-4}\)。不能仅凭 LU 次数声称整体成本胜出。
- 正面 P：内部 Robin 迹中间凝聚 \(U_r(0)\succeq U(0)\) 且 \(U_r(0)\preceq(1+\delta)U(0)\) 时，对全部 \(h\ge0\)，有 \(0\preceq S_{g,r}(0,h)-S_g(0,h)\preceq\delta S_g(0,h)\)。若精确包含静态谐延拓，稳态在整个非负 HTC 域精确（精确算术）。
- 负面 B/N：直接套动态消元范数上界的放大系数约 202；粗 resolvent-Lipschitz 分箱严重松弛；内部 Robin 迹精确保留动态精度不占优。
- 复现：`python robin_followup.py --nx 12 --ny 15 --ranks 56 72 --holdout 10`、`python robin_real_hinf.py --nx 12 --ny 15 --rank 72`、`python robin_static_uniform.py`、`python robin_lipschitz_probe.py --ranks 72`（目录内，建议 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1）。原型起源于基线 a847444 的独立实验；Python 3.13 / NumPy 2.3.5 / SciPy 1.17.0。
- 优先后续：结构化 Robin 反馈型鲁棒 \(H_\infty\) 判据的**有效性和审核成本**，其次是自适应实移位 / HTC 导数方向成本权衡。不要从单例外推工程规模。

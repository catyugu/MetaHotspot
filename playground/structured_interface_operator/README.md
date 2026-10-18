# 内部 Robin 参数化动态热宏模型（当前活动研究）

本目录研究 **第三类边界条件换热系数 HTC** 对组件动态 MIMO 模型的影响，而非材料参数盒。最终连接端口仅为真实组件互连表面 `g`；Robin 换热表面归属于待消元内部自由度 `i`，**不**输出为额外连接端口。

\[
A_{ii}(s,h)=K_{ii}^{(0)}+sC_{ii}+\sum_j h_jH_{j,ii},\quad H_{j,ii}\succeq0,
\]
\[
S_g(s,h)=A_{gg}-A_{gi}A_{ii}(s,h)^{-1}A_{ig}.
\]
环境温度 \(T_{\infty,j}\) 通过 \(h_j b_jT_{\infty,j}\) 贡献输入；必须同时检查接口 DtN、功耗到温度、Robin 环境温度到温度的联合 MIMO 传递。冻结 HTC 的 LTI \(H_\infty\) 不等于时变 HTC 或逐时刻最高温度证书。

## 当前方向与证据限制

- 正实频移、同一实稀疏分解生成多阶 Krylov 矩方向；HTC 一、二阶灵敏度与 HTC 角点采样是**对照手段**，不是原创性的先验结论。
- 实 Galerkin 保留 HTC 的正半定仿射更新及正热容，前提是原热网络满足相应 SPD/PSD 假设；浮置模型另行处理。
- 低阶实状态空间的 **数值** \(H_\infty\) 残差指标独立于实频选点，用于有限 HTC 训练集合的审核。**目前的 Robin 原型采用固定实移位，不是自适应算法，也没有连续 HTC 盒 × 全频率的经过舍入验证的证书。**
- 把 Robin 迹精确保留为内部状态，只是内部凝聚对照；最终对外仍只有 `g`。它在当前同阶数动态实验中较差，不能作为默认路线。
- 稳态内部凝聚的 Loewner 单调性可得连续 \(h\ge0\) 的相对界；瞬态的简单统一界和粗 Lipschitz 包围目前太松。

原先材料参数研究及其原始代码/数字已**完整归档**到 [archived/structured_interface_operator_material_20261018](../archived/structured_interface_operator_material_20261018/README.md)。那些结果是历史研究，不代表对 Robin HTC 的验证，切勿直接引用其材料盒 BRL 数字作为当前结论。

## 复现（Python、NumPy、SciPy；单线程 BLAS）

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_followup.py --nx 12 --ny 15 --ranks 56 72 96 --holdout 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_real_hinf.py --nx 12 --ny 15 --rank 72
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_static_uniform.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_lipschitz_probe.py --ranks 72
```

本目录脚本都是与生产求解器分离的合成网格原型。无完整在线连接开销、无数万自由度正式计时，不能据此宣称工业级效率。现有结果/局限见 [实验记录](records/EXPERIMENT_2026-10-18_ROBIN.md) 和 [研究笔记](ROBIN_RESEARCH_NOTE.md)；研究问题见 [议程](RESEARCH_AGENDA.md)；[记录规范](RECORD_PROTOCOL.md)继续有效。

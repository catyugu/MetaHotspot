# 学术调研与判决：带符号 Poisson–Loewner 余量界及最终 SVD 验收

2026-10-07，agent/work，基于 HEAD d7e229bca40b94d75d332827097d6ecd86fc2317。
本轮与前轮改动保留在本地；未提交或推送。生产 `utils.py` 未修改。

**本轮得到数学上不同的余量界，而非细分、缓存或 setup 优化。**
固定同一最终 76 阶 SVD 基、同一 80 项时间展开、同一二阶参数试验、同一连续参数盒，
原标量界 `0.00212601824` 不通过，新矩阵界 `0.00192675643` 通过。
该对照没有参数包围细分；full trial/Riesz RHS 完全相同。
阈值为用户指定的最终 `2*tolerance=0.002`。

## 1. 调研定位和贡献边界

本轮读取了下列一手来源的相关正文：

| 文献 | 已有内容及本项目的区别 |
|---|---|
| Son–Stykel 2017，DOI 10.1137/15M1027097，作者 PDF，第 3–5 节 | 参数 Lyapunov、min-theta 谱下界、残差/natural energy estimates 已有理论。使用这些工具本身不构成新贡献。 |
| Gugercin–Antoulas–Beattie 2008，DOI 10.1137/060666123，Hilbert/插值/Lyapunov 相关章节 | H2 最优性已有完整框架；标准 H2 trace 目标不能直接代替相对全输入 lambda_max 目标。 |
| Garrett 2016，Harmonic functions, Poisson kernels，第 4 节 | 半平面 Poisson 核是标准事实。本轮将正积分和核比值用于时间创新 Gram 的矩阵序包围。 |
| Rettberg 等 2024，DOI 10.1007/s10444-024-10195-8，第 3 节 | 分层、auxiliary error estimates 已有理论；饱和假设不能未经验证采用。本轮完整保留 raw 证书，不使用 saturation。 |

来源链接与完整精确算术证明见 [SIGNED_POISSON_PROOF.md](../SIGNED_POISSON_PROOF.md)。
arXiv 1811.08327 的摘要也检索到；HTML 读取失败，未据此声称阅读其全文。
未检索到的同等构造不能据此断言不存在；本轮证明数学增量和适配价值，不证明发表新颖性。

## 2. 数学本质改变

旧界把每个时间残差变成标量范数，交叉项作绝对值包围。
新界保留完整 `F_j^T K(a)^{-1} F_k`，并构造对所有实际特征率都合法的 PSD 时间矩阵 M。

时间创新 Gram 的精确正积分表示是

\[
 A(\lambda)=\frac1{2\pi}\int_{\mathbb R}
 \frac\lambda{\lambda^2+\omega^2}\Psi(i\omega)^*\Psi(i\omega)\,d\omega.
\]

Poisson 核比值直接给出

\[
 A(\lambda)\preceq\max(\lambda/\mu,\mu/\lambda)A(\mu).
\]

这允许用有限几何谱子区间和 PSD 正部更新构造共同 M，严格包围**连续谱**。
它不是在有限谱点或频率点验证最大值。整个余量成为带符号的联合二次型

\[
 \Delta_F^2\le\max_\nu\lambda_{max}
 \sum_{j,k}M_{jk}F_{j,\nu}^T K(a)^{-1}F_{k,\nu}.
\]

相同控制索引内的时间、空间和输入方向均保留。PSD 凸性认证控制索引之间的全部参数。
系数缺陷与**精确末端能量**一起推导，未截掉时间尾项。
主要判决实验 `envelope_depth=0`，同一完整谱区间，不改变基或 trial 节点。

结构性优势可证明不是常数修饰：偶数 N、重复移位、相同残差方向，率区间
`[epsilon,2 epsilon]` 趋于零时，带符号界趋于零，旧标量界趋于 `sqrt(2)N`。
两者比例可以无界。N=8 的验证结果：

| epsilon | 旧标量界 | 新矩阵界 | 旧/新 |
|---:|---:|---:|---:|
| 1e-3 | 11.28338631 | 0.17787832 | 63.43 |
| 1e-5 | 11.31340445 | 0.01788753 | 632.47 |
| 1e-7 | 11.31370546 | 0.00178885 | 6,324.56 |

同尺度缩放 rates/shifts 不改变该谱包围，因此没有显式 alpha^-3 补偿。
但共同 M 仍放松每个空间谱通道的条件分布，Riesz 范数仍反映物理条件数；不是全部病态性已解决。

## 3. 同一 raw 基的判决：改善有限，但方向确实保留

Case1 矩阵重建：364 节点，raw 基 95 阶，hash 与前轮完全相同。
N=80，degree p，两轴线性盒宽系数 f；均无包围细分。

| f | p | 旧联合能量余量 | 新带符号余量 | 新/旧 |
|---:|---:|---:|---:|---:|
| 0.002 | 1 | 0.02145614 | 0.01147810 | 0.535 |
| 0.002 | 2 | 0.000541846 | 0.000392147 | 0.724 |
| 0.002 | 3 | 0.0000315761 | 0.0000261208 | 0.827 |
| 0.0002 | 1 | 0.000217927 | 0.000118723 | 0.545 |
| 0.0002 | 2 | 0.000000575295 | 0.000000426429 | 0.741 |

该改善不是数值设置变化：每个 JSON 同时记录旧 `energy_bound` 与新 `matrix_bound`，
二者共用同一试验与全部逆作用。新构造只增加 N 阶矩阵代数及交叉 Gram。
完整物理参数域仍未通过。小模型完整域 p=4、不细分时新最终界约 `0.00253183`，
也没有通过 raw 阈值 0.001。负结果保留在 `rational_chain_signed_matrix64_cover4.json`。

## 4. SVD 后 2 tau：原 stock 不通过，受动态证书约束的 SVD 基通过

原提取容差 tau=0.001。用户允许最终 SVD 后动态阈值 2 tau=0.002。
所有 future `run_rational_dynamic.py --basis-stage final` 使用 2 tau，raw 仍使用 tau。
这不是从欧氏快照 cutoff 自动推出的定理。

Case1 stock 最终基为 34 阶。中心实际参考误差 `0.0175905853`，仍为放宽阈值的 8.8 倍；
因此不能只把接受阈值改成 2 tau 就宣布 stock 最终基通过。

验证策略：固定 stock 快照和 SVD 方向，保留 uniform mode。在同一 raw 95 维模型内，
逐个增加 SVD 方向，用中心值拒绝，用连续动态证书接受。不假设误差随阶数单调。
raw 证书预算 tau，raw 到 final 证书预算 tau，通过标准分层三角界得到 final 的 2 tau。
这项 SVD 后的保护规则是应用已有分层理论，**不是本轮主要数学创新**。

在 f=0.0002 盒上，选择结果为 75 个 SVD 方向 + uniform mode，最终 76 阶：

| 量 | 数值 |
|---|---:|
| 原 stock 最终阶数 | 34 |
| raw 阶数 | 95 |
| 受动态证书约束的最终 SVD 阶数 | 76 |
| 保留末端奇异值比 | 4.167927e-6，而非原 cutoff 0.001 |
| raw 连续盒证书 | 0.000995351134 |
| raw→final 连续盒证书 | 0.000948024795 |
| 分层 final 上界 | 0.001943375928 |
| 物理全阶→final 的直接连续盒上界 | 0.001363545333 |
| 7 个独立全阶参考点的最大误差 | 0.001358406359 |

这不是证明“原 34 阶基没问题”，而是得到了另一个实际 SVD 截断后的基。
76 不是最小阶数定理；split budget 三角界可能保守。
保存的 `.npz` 包含最终/原始基、坐标、奇异值、矩阵、范围和移位，足以独立核验。

## 5. 纯数学判决：相同最终基、相同盒、相同 RHS，旧界拒绝而新界接受

保持该 76 阶基不变，N=80，p=2，均无包围细分；阈值固定 0.002。

| f | 旧标量界 | 新矩阵界 | 样本最大真实参考 | 旧 / 新判决 |
|---:|---:|---:|---:|---|
| 0.0002 | 0.00136369420 | 0.00136354533 | 0.00135840636 | 通过 / 通过 |
| 0.002 | 0.00194673930 | 0.00179703970 | 0.00135934832 | 通过 / 通过 |
| 0.0022 | 0.00212601824 | 0.00192675643 | 0.00135939844 | **拒绝 / 通过** |
| 0.0024 | 0.00233834048 | 0.00207965445 | 0.00135943821 | 拒绝 / 拒绝 |
| 0.003 | 0.00319875049 | 0.00269383887 | 0.00135949688 | 拒绝 / 拒绝 |

该 f 定义为 `a=c-f(c-range_low), b=c+f(range_high-c)`，c 为范围几何中心。
这是有效 HTC 坐标的连续盒宽系数，不是物理 HTC 的相对半宽。
f=0.0022 相对于 f=0.0002 的两轴宽度各为 11 倍。
保留相邻失败宽度，不能只展示经过选择的通过盒。
f=0.002、p=3 的新界为 0.00143101369；更宽 f=0.02、p=3 仍约 0.09813125，未通过。

## 6. 独立验证与成本

中心以正 integrand `E(t)^T C E(t)`、`T(t)^T C T(t)` 对 `[0,infinity)` 独立自适应积分。
这避免用 Q+Qr-cross 的大数相减作为唯一参考。
正积分的相对误差为 `0.00135824419919944`，代数 eigenmode Gram 为
`0.00135824419946038`，差约 `2.61e-13`。积分估计误差 `5.44e-10`；
这个积分是独立参考，不是区间证书。

SVD 选阶共检查 43 个小模型中心，只有最终 76 阶候选进入连续 compression 验收。
full RHS 与 raw-model RHS 分开报告：

| 阶段 | full RHS | 95 维 raw-model RHS | 其他 |
|---|---:|---:|---|
| stock 快照提取 | 94 | 0 | stock AMG-CG，原设置 |
| raw 的连续认证 | 8,069 | 0 | 1 次分母、198 个 trial shift/node 分解、1 次 Riesz 分解 |
| raw→final 连续认证 | 0 | 8,068 | 2 次 95 维广义谱分解 |
| 最终物理模型直接认证复核 | 8,069 | 0 | 独立于分层接受证书 |
| 单个 f=0.0022 的旧/新数学对照 | 8,069 | 0 | 同一组逆作用同时计算两种界 |

单次 SVD 保护及直接复核的 full 成本 `94+8069+8069=16,232`，远大于 stock 94 RHS。
参考点全阶特征分解、正积分、小模型特征检查另计；多个诊断 sweep 不是新方法的
提取预算。本轮没有证明端到端优于 stock，也不以矩阵代数更快作为数学进展。

修改前 20 tests passed。矩阵构造测试先因缺少模块失败，再实现后 26 passed。
small-model 谱与 SVD 嵌套测试先失败再实现，29 passed；增加有限区间抵消、共同尺度
不变性、矩阵包围细分与近相关输入核验后，最终 **32 passed**。
`git diff --check` 通过。

## 7. 复现命令

从仓库根目录，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python`。

```bash
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 80 --propagation matrix --widths .002 .0002 --degrees 1 2 3 --output playground/adaptive_bci_sampling/results/rational_case1_signed_matrix80.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage final --terms 80 --propagation matrix --widths 0 .0002 --degrees 0 2 --output playground/adaptive_bci_sampling/results/rational_case1_final_2tau_matrix.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --output playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_2tau.json
python playground/adaptive_bci_sampling/verify_guarded_svd_cells.py --basis playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_2tau.npz --output playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_wider_cells.json
python -m pytest playground/adaptive_bci_sampling/test_field_audit.py playground/adaptive_bci_sampling/test_rational_dynamic.py playground/adaptive_bci_sampling/test_matrix_innovation.py -q -p no:cacheprovider
```

## 8. 限制与下一项真正的研究问题

Case1 是矩阵重建，尚未与 native 组装逐项对照。精确算术接受证明完整，线性求解、
Poisson 矩阵正部、控制转换、谱区间和输入白化没有向外舍入，所有结果标记
`floating_point_certified=False`。最终保证仅覆盖记录中的连续局部盒，没有全 Case1 HTC 域证书。

当前共同 M 对所有空间谱通道使用一个包围，仍损失“残差在哪些空间谱区间”的条件信息。
下一项有数学价值的任务是给出可验证的谱通道条件化矩阵包围，包含通道近似余量，
同时扩大连续盒；不能仅把投影/缓存减少算作解决这一问题。
这次已经证明并实测带符号矩阵界的结构性增益，但还没有得到可发表的新颖性结论、
经济全域认证或普适动态 SVD 最小阶数算法。

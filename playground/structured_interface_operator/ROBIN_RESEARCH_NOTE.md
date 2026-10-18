# MetaHotspot — 内部 Robin 参数化动态 MIMO 宏模型：纠偏与首轮可复现实验

- 日期：2026-10-18
- 研究基线：`catyugu/MetaHotspot`, branch `agent/experiment1018`, SHA `a847444713e98dd22cdcd9df9e118962444c60af`
- 工作位置：独立本地数值原型（未对 GitHub 仓库写入、未提交、未推送）
- 证据类型：数学推导（精确算术）+ 有限参数/频率的普通浮点实验。**不存在已经完成的连续 HTC × 全频率的可验证数值证书。**
- 主导需求：物理外部连接端口全集 `g` 保留；内部 Robin 换热面 `b` **不**作为外部连接端口；明确内部功耗与温度观测及 Robin 环境温度输入。

## 1. 纠正研究对象

标准线性热 RC 半离散模型，令 `h_j` 为某个 Robin 面区域的非负换热系数：

\[
(K^{(0)}+s C+\sum_{j=1}^p h_j H_j)x
= B_g u_g+B_p p+\sum_j h_j b_j T_{\infty,j},\qquad H_j\succeq0.
\]

`H_j` 局部支撑于待消元的 Robin 边界区域，`b_j` 是环境温度驱动的空间载荷。区分 `h`（改变算子的参数）与 `T_infty`（输入）；如果 `h(t)` 本身随时间变化，冻结参数 LTI 的 `H_inf` 合同不自动覆盖时变闭环。

仅将真正互连表面定义为对外物理端口 `g`；内部未知量 `i` 含 Robin 边界。对第三类边界（与 `g` 不相交的原型），

\[
A_{ii}(s,h)=K_{ii}^{(0)}+s C_{ii}+\sum_jh_jH_{j,ii},\quad
S_g(s,h)=A_{gg}-A_{gi}A_{ii}(s,h)^{-1}A_{ig}.
\]

需要联合检验：`S_g`、内部功耗与测温、`T_infty` 驱动通道、连接后响应。Robin 边界不是接口参数 `E`，更不能被扔到组件外面以省略训练。

## 2. 本轮的四种候选（全部对外只有 g）

1. **nominal_real**：固定标称 `h=(0.4,0.4)`，四个非负实移位形成共享 Galerkin 空间。
2. **derivative_1/2**：复用上述实矩阵 LU，以 `dX/dh_j=-A_{ii}^{-1}H_jX` 构造一、二阶 HTC 参数灵敏度；与频率矩 Krylov 富集可组合，已知先例，不作为新颖性。
3. **corner_real**：四角 HTC 训练，仍使用实数移位；角点训练只生成候选，**绝不作为连续 HTC 认证**。
4. **internal_trace_exact**：仅作为结构性对照：算法内部精确保留 Robin 面的未知温度，先对深层内部自由度消元，再在**内部**消元 Robin 迹变量；最终输出接口仍然只有 `g`。这一方案带来边界未知量数量 `q` 的状态开销，不可能免费。

以上方法使用实 Galerkin，投影后的 Robin 刚度仍为正半定，且热容保持正定；在满足原模型导热矩阵正定的范围中可保持热学被动性。原型尚未覆盖浮置网络的常数零模态。

## 3. 实验模型与合同

合成不均匀二维 RC 网格、对角正热容、顶部 `m=nx` 个**真实**连接端口、底部 `q=nx` 个待消元 Robin 未知量、左侧少量固定弱冷却以保证 `h=0` 时的 SPD。底部两半边各一个独立 HTC 参数，`h in [0.008,3]^2`；两个内部功耗点、两个内部测温点、两个底部平均温度观测、两个 Robin 环境温度输入。**外部没有增加任何连接端口**。

两种规模：180 自由度、m=q=12；480 自由度、m=q=24。所有原型共享固定 `W=S_g(0,h_nom)>0`，谱范数接口误差 `||Lw^-1(Sr-S)Lw^-T||_2`。联合 MIMO 误差取所列温度输出/输入传递矩阵相对谱范数。后者不代替接口合同。

10 或 12 个未参与训练的对数随机 HTC 点、4 个角点压力样本及每点 10 个离散频率；有限集合最大值不能称作 HTC 连续盒/全频率上确界。种子 20261018；NumPy 2.3.5、SciPy 1.17.0，单线程 BLAS。

### 3.1 固定总状态预算，非自适应四实移位（moments=1）

| 全阶 n, 外部端口 m, ROM r | 方案 | 实全阶 LU / RHS 列 | 独立 HTC/频率最大 DtN 误差 | 独立 HTC/频率最大相对 MIMO 误差 |
|---|---|---:|---:|---:|
| 180, 12, 72 | nominal_real | 4 / 72 | 1.0388e-2 | 1.2378e-2 |
| 180, 12, 72 | derivative_1 | 4 / 216 | 3.8057e-3 | 2.1756e-3 |
| 180, 12, 72 | corner_real | 16 / 288 | 3.6119e-3 | 2.0456e-3 |
| 180, 12, 72 | internal_trace_exact | 4 / 112 | 4.6456e-2 | 1.3416e-2 |
| 480, 24, 128 | nominal_real | 4 / 120 | 2.6429e-2 | 2.7818e-2 |
| 480, 24, 128 | derivative_1 | 4 / 360 | 1.5573e-2 | 4.4666e-3 |
| 480, 24, 128 | corner_real | 16 / 480 | 1.5750e-2 | 4.4952e-3 |
| 480, 24, 128 | internal_trace_exact | 4 / 208 | 1.2973e-1 | 2.1788e-2 |

**MIXED：** HTC 灵敏度在同阶预算下接近角点法，可节省全阶 LU，但 RHS 增多。完整内部 Robin 迹保留的参数无关性在动态误差上并未抵消额外状态开销，暂时不宜成为主路线。所有离线 Gram、SVD、证书审核、连接/在线费用尚未公平累计，不能据此声称端到端提速。

### 3.2 正实频移多阶矩 + HTC 灵敏度 + 独立小系统数值 Hinf

同一 180/12/r72，**四个固定实移位** `0,0.03,0.4,5`（只是本轮消融参数，不是自适应选点或最终方法）；训练 HTC 四角 + 标称，持出独立 HTC 10 点。Hinf 在阶数 r 的实状态空间上数值 Hamiltonian 检查，不用复数全阶富集。

| 训练策略 | 频率矩 | HTC 导数阶 | LU / 实 RHS 列 | 训练 HTC 最大数值 `||J||_Hinf^2` | 持出 HTC/频率最大 DtN 误差 |
|---|---:|---:|---:|---:|---:|
| 标称 h | 1 | 0 | 4 / 72 | 3.3123e-2 | 1.0388e-2 |
| 标称 h | 3 | 0 | 4 / 216 | 3.1792e-2 | 1.0240e-2 |
| 标称 h | 1 | 1 | 4 / 216 | 9.4601e-3 | 3.8057e-3 |
| 标称 h | 3 | 1 | 4 / 648 | **2.0499e-3** | **6.4776e-4** |
| 四角 h | 3 | 0 | 16 / 864 | 7.3316e-4 | 3.9782e-4 |

**POSITIVE（局部数值）**：对于广泛 HTC 改变，单纯增加频率矩方向效果有限，HTC 方向至关重要；一阶 HTC 响应与三阶实频矩组合在此例非常有效，并以更少的全阶实 LU 获得接近四角的精度。**NEGATIVE / NOT PROVED**：四角三矩仍有更低残差；单次 LU 节省并不能代替离线总成本；固定实移位的消融结果不保证自适应选择也有此效果。

残差 LTI 状态空间与直接 residual 构造在数个独立 HTC/频率点上相对差异最大约 7.9e-12；采样检查了 `||S_r-S||_W<=||J||^2`，最大浮点违反约 4.7e-16。Hamiltonian 结果与稠密频扫相近，但**没有舍入认证，也没有覆盖所有 HTC**。

## 4. 数学结构结论：静态可以很漂亮，动态尚未攻克

为研究目的临时分块 `i=b union z`，b 是 **内部 Robin 迹变量**，z 为剩余内部自由度。定义独立于 HTC 的中间凝聚矩阵

\[
U(s)=A_{[g,b],[g,b]}-A_{[g,b],z}A_{zz}(s)^{-1}A_{z,[g,b]}.
\]

最终 DCTM 端口仅 `g`，在内部消元 b 得到

\[
S_g(s,h)=U_{gg}(s)-U_{gb}(s)[U_{bb}(s)+H_b(h)]^{-1}U_{bg}(s).
\]

这不是把 b 出口暴露到外部；b 只是内部消元坐标。

**PROOF / STATIC。** 如果 `U(0)>0`，其实 Galerkin 近似 `Ur(0)>=U(0)`。令

\[
\delta=\lambda_{\max}\left(U(0)^{-1/2}[U_r(0)-U(0)]U(0)^{-1/2}\right),
\]

则由 Loewner 序与 Schur 补单调性，**对任意** `H_b(h)>=0` 都有

\[
0\preceq S_{g,r}(0,h)-S_g(0,h)\preceq\delta S_g(0,h).
\]

这是真正连续 `h>=0` 的**静态相对算子界**，不依赖 HTC 网格和训练角点。若 `V_z` 包含 `K_zz^-1 K_z,[g,b]` 的静态谐延拓方向，`U_r(0)=U(0)`，故所有 `h>=0` 下的接口稳态 DtN **精确**；本次数字浮点误差约 1e-15。代价是保留 q 个 b 内部未知量，动态精度仍可能较差。浮置零模态与能量半范数需要另推导。

**PROOF / DYNAMIC SUFFICIENT（宽松）。** 对本轮对角热容及 SPD `K0`，令 `alpha=lambda_min(U_bb(0))>0`，`M=||K_bg||+||K_zz^-1/2K_zb|| ||K_zz^-1/2K_zg||`。利用被动 DtN 的 Hermitian part 大于等于其 DC 取值，对所有 `h>=0`、`omega in R` 有

\[
\|S_{g,r}(i\omega,h)-S_g(i\omega,h)\|_2
\le \underbrace{[1+(M/\alpha)^2]}_{\Gamma}\,\|U_r(i\omega)-U(i\omega)\|_2.
\]

这一不等式使参数域**理论上消失**，可结合对 `U` 的一份全频率残差上界。不过 180 DOF 原型的 `Gamma≈201.93`，而有限点上的误差放大比不到 1，界过于保守，不能作为实用容差控制方案。

**NEGATIVE / PARAMETER LIPSCHITZ。** 对直接 embedded Galerkin 的 HTC 仿射残差
`J(s,h)=D+C(h)(sI-A(h))^-1B`，`A,C` 皆仿射、`B,D` 与 h 无关，得到全频率导数范数上界

\[
\|\partial_{h_j}J\|_{H_\infty}\le
\|B\|\left(\frac{\|C_j\|}{\alpha_B}+
\frac{\max_{h\in B}\|C(h)\|\,\|A_j\|}{\alpha_B^2}\right),
\quad\alpha_B=\lambda_{\min}[-A(h_{\min,B})].
\]

这是完整 HTC 子盒上的精确算术充分导数界，但 `alpha_B` 可能很小。在 180/12/r72 一阶导数模型，对 `[0.008,3]^2` 等分 8×8 后，该宽松 Lipschitz 扩张项仍达到 O(1e4)，而有限点 `Hinf(J)` 只有约 0.1。因此，朴素 resolvent-Lipschitz 二分**明显失败**。脚本使用采样中心值仅作诊断，绝不将其与 Lipschitz 增量之和误标为真正参数域上界。

## 5. 研究路线决定建议（等待讨论，不替用户直接改合同）

**推荐主线 A：直接 embedded Robin-parametric Galerkin。** 保留真实 g；在线矩阵 `Kr(h)=Kr0+sum h_j Hjr`；功耗、观测与 `h_j T_infty,j` 输入统一进入 MIMO；以正实 shift、多阶频矩、适度 HTC 灵敏度构造共享 V。用独立的固定 h 低阶 `H_inf(J)^2` 来指导和停止有限参数富集，并发展 HTC 结构化连续域判据。

**对照 B：基于内部 Robin 迹的无参数凝聚。** 在完整 q 可负担时研究其静态 h-uniform 证明，动态证书和局部/层次边界状态压缩；当前同阶数动态误差为负面，不能成为默认生产方案。

**核心未解决问题：** 对正半定、边界局部支撑的 Robin 更新，能否借助 dissipative feedback / robust positive-real / IQC 等结构，在不保留 q 个边界未知量、也不依赖昂贵的高次参数多项式 BRL 的前提下，得到与数值 `H_inf` 同量级的连续 HTC 区间 × 全频率上界？目前无证明或有效实例，不应宣称已经解决。

先在几百维模型上比较同阶数和同容差时完整离线成本（RHS/LU、稠密 Gram、Hamiltonian、连续审核）与连接 MIMO 误差，再扩展至数万自由度。高维 HTC `p` 会使全角点训练 `2^p` 爆炸，而一阶灵敏度按 `p` 扩张；需要局部/分区激励策略，但这些复杂度结论是一般计数，不是生产性能证明。

## 6. 文献边界

- Hernández-Becerro, Spescha, Wegener, “Model order reduction of thermo-mechanical models with parametric convective boundary conditions: focus on machine tools,” *Computational Mechanics* 67 (2021), 167–184, online 2020, DOI: https://doi.org/10.1007/s00466-020-01926-x 。**全文可读**；HTC bilinearization / Krylov basis extension / 理论有限频域误差是已有工作，不能作为本项目发明。
- Bruns, Benner, “Parametric model order reduction of thermal models using the bilinear interpolatory rational Krylov algorithm,” 2014, DOI: https://doi.org/10.1080/13873954.2014.924534 。**读摘要和引言**；涉及 HTC 和热接触等多参数，BIRKA 无条件不保稳定性，故本项目真实热 Galerkin 的 SPD 结构值得保留。
- Codecasa et al., “Partition-based approach to parametric dynamic compact thermal modeling,” *Microelectronics Reliability* 79 (2017), 361–370, DOI https://doi.org/10.1016/j.microrel.2017.06.059 。**读摘要/可见引言，没有查到全文算法**；声称用户可选精度和紧误差保证，必须在主张新颖前核对保证的具体参数、端口、范数及成本。

## 7. 复现

将脚本放在同一目录，Python 3.13、NumPy、SciPy：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_followup.py --nx 12 --ny 15 --ranks 56 72 96 --holdout 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_followup.py --nx 24 --ny 20 --ranks 96 128 160 --holdout 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_real_hinf.py --nx 12 --ny 15 --rank 72
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_static_uniform.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robin_lipschitz_probe.py --ranks 72
```

对应源码：`robin_model.py`（共同热 RC 模型和对照）、`robin_followup.py`、`robin_real_hinf.py`、`robin_static_uniform.py`、`robin_lipschitz_probe.py`。不是仓库实际生产求解器；不对工业规模、正式效率或连续 HTC 验证做声称。
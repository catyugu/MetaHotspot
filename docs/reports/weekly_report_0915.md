# weekly_report_0915

## 非共形接口连接

两个热模型连接时不要求两侧原始 face 一一对应，而是在两侧界面网格的并集上构造 **common patches**。对任一侧 $S$，原始边界 face $j$ 到物理界面的半单元热导为

$$
g_{S,j}=\frac{k_{S,j}A_{S,j}}{d_{S,j}}.
$$

若公共面片 $i$ 落在原始 face $j(i)$ 内，面积为 $A_{if,i}$，按面积比例 $\xi_{S,i}=A_{if,i}/A_{S,j(i)}$ 分配后，有

$$
h_{S,i}=\xi_{S,i}(E_Sg_S)_i
       =\frac{k_{S,j(i)}A_{if,i}}{d_{S,j(i)}}.
$$

若 ROM 一侧温度场写成 $T_S\approx V_Sq_S$，令 $R_S=E_SV_S$、$H_S=\operatorname{diag}(h_S)$。公共面温度 $T_\Gamma$ 只满足代数热流平衡，因此可以精确消去。定义

$$
H_\Gamma=(H_L^{-1}+H_R^{-1})^{-1},
$$

最终连接只需加入

$$
\boxed{
\begin{bmatrix}
R_L^TH_\Gamma R_L & -R_L^TH_\Gamma R_R\\
-R_R^TH_\Gamma R_L & R_R^TH_\Gamma R_R
\end{bmatrix}}
$$

即可。common patches 只负责非共形积分，不作为额外全局自由度；连接矩阵保持对称、守恒。

## Extended-FANTASTIC：冻结 closing SVD 与容差

本周继续核对 FANTASTIC / BCI CTM 文献与当前代码，并明确区分“文献方法探索”和“已经反复验证的项目算法基线”。相关文献包括：

- Codecasa et al., **“FAst Novel Thermal Analysis Simulation Tool for Integrated Circuits (FANTASTIC),”** THERMINIC 2014；
- Codecasa et al., **“Matrix Reduction Tool for Creating Boundary Condition Independent Dynamic Compact Thermal Models,”** THERMINIC 2015；
- Codecasa et al., **“Connecting MOR-based Boundary Condition Independent Compact Thermal Models,”** THERMINIC 2017；
- Codecasa et al., **“Versatile MOR-based Boundary Condition Independent Compact Thermal Models with Multiple Heat Sources,”** *Microelectronics Reliability* 87 (2018), 194–205；
- Codecasa, d’Alessandro, Bornoff, **“Galerkin’s Projection Framework for BCI CTMs—Part I: Extended FANTASTIC Approach,”** *IEEE TCPMT* 11(11), 1792–1803, 2021；
- Codecasa et al., **“Boundary Condition Independent Compact Thermal Models Enhanced by Contour Elements,”** THERMINIC 2023。

此前为研究 2021 Algorithm 1 的另一种实现解释，引入了 `build_parametric_basis_literature()`，把 closing SVD 从项目原有的 normalized exact-response snapshots 改成了 source projection-space columns。该改动同时显著改变了最终 ROM order，因此它不能作为对现有算法的无条件“修正”。项目的 closing SVD 与 extraction tolerance 已经过反复验证，本轮把它们作为**硬约束冻结**：删除 `build_parametric_basis_literature()`，所有实验重新使用 `metahotspot.macromodel.utils.build_parametric_basis()`，并恢复 simple EROM 的固定容差 $10^{-3}$。whole-PoP 继续使用论文对齐的固定容差 $10^{-2}$；两处均不再通过调容差或 closing SVD 来追求更低阶数。

项目标准 closing SVD 的行为保持不变：收集 residual test 失败后得到的 exact response snapshots，对每列归一化后做 SVD，以既定 relative singular-value cutoff 截断，再显式保留 h-free BCI 系统的 uniform-temperature null mode。全阶精确响应继续使用 AMG-preconditioned CG 求解。

### 标准 contour-element 实现

2023 contour elements 的标准顺序仍保持为：**先提取 BCI CTM，再压缩 boundary temperature / heat-flux representation**，不把 contour polynomial 当成额外内部 ROM state。一个矩形面上使用总次数不超过 $p$ 的二维 $L^2$ 正交多项式，系数数目为

$$
\frac{(p+1)(p+2)}{2}.
$$

对 FVM 上分片常数的 modal boundary trace $v(r)$，论文式 (15) 为

$$
\widehat V_b=\int_\Gamma\psi_b^T(r)v(r)\,dA,
$$

并由式 (16) 恢复边界温度、式 (17) 的转置映射传递热流。实现使用 Legendre product basis 和解析矩形积分。本轮独立 Gauss quadrature 检查得到积分最大差异 $7.63\times10^{-17}$，连续 $L^2$ Gram 最大误差 $1.33\times10^{-15}$。

按 2023 论文给出的四个 side $p=2$、bottom $p=4$、top $p=10$，系数总数按其公式为 $4\times6+15+66=105$；论文正文报告 $110\times42$，但没有给出额外 5 个系数的来源，因此当前实现仍遵循公式和明确给出的阶数。

## 2023 whole-PoP：按冻结算法重新实验

实验仍直接把**整个 PoP**作为一个 BCI ROM 提取对象。由于论文没有公开足够的商业模型几何和材料数据，不能逐单元复刻其 561,408-DoF FloTHERM 模型；当前使用 562,176-DoF structured-FVM surrogate，对齐论文公开的降阶问题：两个独立 die sources、四个侧面 + bottom + top 共六个外表面、side/bottom/top 三个独立 HTC 参数，以及相同 HTC 范围和 contour polynomial degrees。

固定论文容差 $10^{-2}$、20 residual-probe rounds 下，`build_parametric_basis()` 共接受 63 个 exact responses；closing SVD 保留 13 个 snapshot modes，再加入 uniform null mode，最终得到 **14 阶** whole-PoP ROM。最大 accepted residual 为 $8.606\times10^{-3}$，满足固定 extraction tolerance。该 14 阶结果来自完整 562,176-DoF whole-PoP，而不是此前的 bottom-only 12-state 问题。

full boundary trace 的稳态结果为：

| case | order | global $L_\infty$ | volume $L_2$ | die1 junction | die2 junction |
| --- | ---: | ---: | ---: | ---: | ---: |
| paper Fig.2 BC | 14 | 1.401% | 2.064% | 0.105% | 0.357% |
| paper Fig.4 BC | 14 | 2.015% | 2.428% | 0.352% | 0.462% |
| range low corner | 14 | 0.081% | 0.021% | 0.008% | 0.009% |
| range high corner | 14 | 3.266% | 3.616% | 0.816% | 0.979% |

独立 source/shift transfer validation 在 $s=0,0.1,10\;\mathrm{s}^{-1}$ 上分别激励两个 source；in-range 最大 global $L_\infty$ 误差为 **5.471%**，发生在 `range_high`、source 2、$s=0$。因此当前最重要的事实不是“14 阶非常小”，而是：在不改变 closing SVD 和 tolerance 的约束下，residual acceptance 已满足，但一些 full-field holdout error 仍明显高于 1%。下一步优化应针对训练覆盖、参数采样或接口表示等允许改变的部分，而不是改 closing SVD/tolerance。

论文固定 contour degrees 在该 surrogate 上仍不足。四个 side 的 boundary-trace projection error 约 13.75%，bottom 为 61.43%，top 为 16.28%；in-range contour steady error 最大 **16.464%**，source/shift transfer 最大 **17.951%**。bottom surface 是当前最明显的 boundary representation bottleneck。后续如果研究 contour elements，应在冻结 interior extraction 算法的前提下，单独研究 per-surface adaptive degree / subdivision，并把 boundary projection error 与系统误差作为独立指标。

## simple EROM：固定 $10^{-3}$、同一 closing SVD 重新实验

`simple_erom_case1` 继续执行“extract once, reuse everywhere”。同一个 copper cube 只提取一次，所有 attachment cases 复用完全相同的 ROM。训练空间包含一个内部 physical source，以及一个训练用的 `z-` **constant interface heat-flux direction**，用于覆盖附件模型向 ROM 反向注热；该 interface direction 只参与 basis training，不作为物理 source 导出。

本轮取消把 tolerance 当作压阶参数。固定项目基线容差 $10^{-3}$ 时，共得到 19 个 exact response snapshots；closing SVD 保留 10 个 modes，加 uniform null mode 后最终 ROM order 为 **11**。五个 attachment cases 都复用同一个 11-state ROM，恰好与冻结 FloTHERM EROM baseline 的 11 阶相同。

| case | FloTHERM / MHS order | steady global: FloTHERM / MHS | transient global: FloTHERM / MHS | MHS junction |
| --- | ---: | ---: | ---: | ---: |
| baseline_copper | 11 / 11 | **0.742%** / 0.778% | 1.777% / **0.563%** | 0.080% |
| bottom_htc_strong | 11 / 11 | 2.765% / **0.835%** | 4.022% / **0.787%** | 0.037% |
| external_source_200w | 11 / 11 | 4.151% / **0.646%** | 2.426% / **0.411%** | 0.086% |
| all_stress | 11 / 11 | 3.542% / **0.611%** | 2.276% / **0.473%** | 0.074% |
| layered_extreme_source | 11 / 11 | 3.231% / **0.718%** | 2.089% / **0.590%** | 0.079% |

因此修正后的可比结论是：**在相同 11 阶下**，MetaHotspot 的 steady global error 在 `baseline_copper` 略高于 FloTHERM（0.778% vs 0.742%），但在其余四个 stress cases 明显更低；transient global error 在五个 case 中均更低，且 MHS junction error 全部低于 0.086%。这比此前通过改 closing SVD / tolerance 得到的 18-state 对比更符合当前算法约束。

## 求解器与验证

本轮没有回退已经确认的求解器修正。full-order thermal solves、whole-PoP validation solves，以及连接后的 simple EROM steady/BDF1 solves 继续使用 **AMG-preconditioned CG**；连接系统的固定矩阵复用 Ruge-Stüben AMG hierarchy/V-cycle preconditioner，瞬态每一步用前一步作为 CG warm start。修正路径不使用 `spla.splu` / `spla.spsolve`。

最终 GitHub Actions run `34993111326` 在 Ubuntu 24.04 / Python 3.12 上完成。验证 commit 为 `54b0860ce29aeab517005af8b45489a581746ca4`，artifact 为 `10406776457`。rollback/reference check、contour quadrature test、562,176-DoF whole-PoP、Release/Ninja build、93/93 repository tests、五个 reusable simple-EROM cases、Python `compileall`、direct-sparse-solver rejection check 与 `git diff --check` 均通过。精简结果同步到 `playground/connecting_mor_models/baseline/metahotspot/`。

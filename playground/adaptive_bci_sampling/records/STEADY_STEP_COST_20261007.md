# 最新稳态/阶跃目标及提取成本比较

日期 2026-10-07。用户只要求稳态 `2*epsilon`、阶跃瞬态 `2*sqrt(epsilon)`，并确认使用全场能量范数、所有输入组合、所有时刻。默认 `epsilon=0.001`，稳态阈值 0.002，阶跃阈值 0.0632455532。生产提取器未修改。

## 准确的目标

设 `K_h=K+sum h_i H_i`、零初值、阶跃输入 `G w`：

\[
X_h=K_h^{-1}G,\quad X_{h,V}=V K_{h,V}^{-1}V^TG,
\]
\[
S_h(t)=(I-e^{-C^{-1}K_ht})K_h^{-1}G,
\quad S_{h,V}(t)=V(I-e^{-C_V^{-1}K_{h,V}t})K_{h,V}^{-1}V^TG.
\]

验收为

\[
\sup_{h,w}\frac{\|(X_h-X_{h,V})w\|_{K_h}}{\|X_hw\|_{K_h}}\le2\epsilon,
\qquad
\sup_{h,t>0,w}\frac{\|(S_h(t)-S_{h,V}(t))w\|_C}{\|S_h(t)w\|_C}\le2\sqrt\epsilon.
\]

输入取独立列空间，`t=0` 通过右极限解释；不使用 0/0 或任意分母 regularization。两个平方阈值是 `4*epsilon^2` 和 `4*epsilon`。此前冲激 `L2_C` 积分误差与新的点时刻阶跃相对误差是不同对象；没有将前者的证书直接换算为后者。

本轮范围是 364 单元 Case1，3 个提取种子，相同有效 HTC 局部盒 `f=0.0002`，中心 `c=sqrt(l*u)`，盒为 `[c-f*(c-l),c+f*(u-c)]`。完整训练域仍未通过。数据保留所有拒绝结果。

## 新目标允许更低的最终阶数

沿用 tolerance=0.001 的 stock 快照及 SVD 方向，按稳态条件重新验收最终截断，包含常数模态：

| 种子 | stock 最终阶数 | 此前冲激验收最终阶数 | 新目标最终阶数 | 连续 HTC 盒稳态上界 |
|---:|---:|---:|---:|---:|
| 20260805 | 34 | 76 | 48 | 0.001898662645 |
| 20260806 | 34 | 80 | 48 | 0.001846207253 |
| 20261007 | 34 | 80 | 53 | 0.001971287537 |

stock 34 阶基在中心的真实全输入 K-energy 稳态误差分别为 `0.01021773,0.00851271,0.00956356`，已经超过 0.002。它们的中心全时刻阶跃证书通过 0.06324555，但不能因此忽略稳态失败。新阶数仅是这批固定 SVD 方向的首次验收结果，不是最小子空间阶数证明。

新基在 low、center、high 和固定独立种子生成的 4 个内部点上，均取得每个点的**所有时刻**阶跃证书。证书最大上界分别约 `0.06323785,0.06311807,0.06314481`。这些上界刻意围绕用户阈值停止区间细化，不能将它们当作实际最坏误差估计。

**尚缺少阶跃的连续 HTC 盒证书。** 七个点的全时刻保证不是整个参数盒的保证。本轮结果不能宣称两个目标均已在连续 HTC 域完整认证。

## 单次交付成本比较

一次 RHS 表示一个完整模型线性系统的单列右手端逆作用。所有选阶/停止过程中使用的全阶逆作用均计入；谱分解另列，不能据 RHS 数推导墙钟加速。

| 种子 | 原 stock tau=0.001 提取 RHS | 新 SVD 验收提取 RHS | 选阶 RHS | 稳态证书 RHS | 新路线逆作用总计 |
|---:|---:|---:|---:|---:|---:|
| 20260805 | 94 | 94 | 4 | 36 | 134 |
| 20260806 | 97 | 97 | 4 | 36 | 137 |
| 20261007 | 100 | 100 | 4 | 36 | 140 |

所以新界/新验收**没有减少原提取本身的 94/97/100 次求解**。和这组未通过稳态目标的 stock 基直接比较，增加了 40 次逆作用；这不是等精度胜出结论。

为得到可比的通过对象，还对原 stock 算法预设内部容差 `[epsilon/2,epsilon/5,epsilon/10,epsilon/20]`，从宽至严测试，首次满足同一局部连续稳态证书及同样七点全时刻阶跃证书即停止：

| 种子 | 首次通过的 stock 内部容差 | stock 最终阶数 | stock 提取 RHS | 加共同 4+36 后 | 新路线总逆作用 | 减少比例 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260805 | 0.0001 | 49 | 150 | 190 | 134 | 29.5% |
| 20260806 | 0.00005 | 53 | 176 | 216 | 137 | 36.6% |
| 20261007 | 0.0001 | 49 | 153 | 193 | 140 | 27.5% |

对照稳态证书上界分别约 `0.00171932708,0.00182613709,0.00153668769`（完整精确读数在 JSON）。三个对照的七个全时刻阶跃证书也均通过。就**提取 RHS 本身**而言，新路线相对这组已通过对照减少约 `37.3%,44.9%,34.6%`。但对照不是所有可能 stock 内部容差中的最优结果；比较不能升级为一般提取最优性结论。

对照容差不是免费得到的。上述 preset sweep 含失败提取的探索 RHS 分别是 `397,562,394`；表格是已知内部容差后的单次交付成本。若从原 tau=0.001 的失败提取开始计探索，还需加 `94,97,100`。原提取的验证 matvec 数、AMG setup、SVD、QR 与约化代数均在 JSON 的 stock stats 保留，不以最终阶数代替这些成本。

阶跃认证路线当前是小规模稠密谱参考：候选最终基在七个 HTC 点各付一次全阶广义谱分解，共 **7 次 full eigensolve**，另有所有时间包围的小矩阵运算。新路线另报告一次 stock 中心谱检查，这是外部对照，不用于新基验收。两组通过对象付同样的七点谱认证成本，因此上表 RHS 优势成立于共同谱检查之外；尚未证明大规模端到端经济优势。三组新实验并行运行，墙钟不宜用作严格速度比较。

此前冲激方案单次 raw 证书 8069、最终直接复认证 8069 个全阶 RHS，合计加提取为 `16232,16235,16238`，还未含压缩模型证书和外部参考。新目标确实使大量这种冲激认证工作不再必要；这是目标变更的收益，不能归因于带符号矩阵界的新数学增益。

## 连续稳态证书的依据

对每个参数轴取端点，在约化模型中求解稳态，再构造唯一张量线性插值多项式 `q(h)`。它的残差 `R_q=G-K_h Vq(h)` 在每轴次数至多 2，精确转换为 Bernstein 控制矩阵 `R_nu`。Galerkin 最优性、逆的 Loewner 序、PSD 二次型凸性给出

\[
E_h^TK_hE_h\preceq R_q(h)^TK_h^{-1}R_q(h)
\preceq\sum_\nu B_\nu(h)R_\nu^TK_{low}^{-1}R_\nu.
\]

分母 `G^TX_h` 不小于 high 角点约化转移 `G^TV K_{high,V}^{-1} V^TG`。用这个共同 SPD 分母求每个控制 Gram 的最大广义特征值，即为全输入稳态平方误差界。两维、4 输入共 `3^2*4=36` 个完整 RHS、一项 sparse factorization。四个 trial 节点全部是约化求解，没有额外 full RHS。

选阶使用一个中心完整参考 `K_c^{-1}G`，4 RHS；逐项拒绝超限阶数后才运行连续盒证书。固定训练快照没有新增。这里是已有 Riesz–Bernstein 数学的目标化使用，没有声称新的提取数学定理或新颖性。

## 固定 HTC 的全时刻阶跃证书依据

在 `C` 白化后的完整正交谱坐标，令 `lambda,U` 为完整谱，`r,W` 为恢复的约化谱，`b=U^TG`、`br=W^TG`、`D=U^TCW`。完整及约化阶跃坐标是

\[
F(t)=\operatorname{diag}(f_{\lambda_j}(t))b,\quad
R(t)=D\operatorname{diag}(f_{r_j}(t))br,\quad
f_\lambda(t)=(1-e^{-\lambda t})/\lambda.
\]

`F(t)^TF(t)` 随时间 Loewner 单调增大。对区间 `[a,b]`，白化 `Q(a)=F(a)^TF(a)`，得到矩阵 `Z` 满足 `Z^T Q(a) Z=I`。若 `m=(a+b)/2`，则整个时间区间的相对误差不超过

\[
\|(F(m)-R(m))Z\|_2+
\tfrac{b-a}{2}\big(\|\operatorname{diag}(e^{-\lambda a})bZ\|_F+
\|\operatorname{diag}(e^{-ra})brZ\|_F\big).
\]

这是均值积分和导数矩阵范数上界，不是时间采样验收。未通过区间二分；真实中点误差超限则返回拒绝见证。开始 `[0,t0]` 改用 `F(t)/t`，其导数范数由 `lambda*b/2` 包围，分母用 `F(t0)/t0`；无 0/0。尾部 `[T,infinity)` 使用实际稳态误差加两个齐次指数尾，分母使用 `Q(T)`。因此初段、全部中间区间、无限尾段共同覆盖每一个时刻。所有输入组合保留完整 Gram，未降为 trace。

默认区间预算只是诊断资源上限，耗尽返回 unresolved/rejected 而非接受。精确算术证明覆盖时间；full eigensolve 和 inverse actions 均未作外向舍入包围，始终 `floating_point_certified=False`。

## 验证与复现

修改前旧测试 34 passed。新增三项测试先因缺少 `steady_step_audit` 模块失败，再实现：阶跃初始/中间/末端覆盖、缺失输入拒绝、连续稳态盒控制。最终 37 passed，`git diff --check` 通过。Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python
python playground/adaptive_bci_sampling/steady_step_audit.py --seed 20260805 --output playground/adaptive_bci_sampling/results/steady_step_seed20260805.json
python playground/adaptive_bci_sampling/steady_step_audit.py --seed 20260806 --output playground/adaptive_bci_sampling/results/steady_step_seed20260806.json
python playground/adaptive_bci_sampling/steady_step_audit.py --seed 20261007 --output playground/adaptive_bci_sampling/results/steady_step_seed20261007.json
python playground/adaptive_bci_sampling/steady_step_audit.py --stock-sweep --output playground/adaptive_bci_sampling/results/steady_step_stock_tighter.json
python -m pytest playground/adaptive_bci_sampling/test_field_audit.py playground/adaptive_bci_sampling/test_rational_dynamic.py playground/adaptive_bci_sampling/test_matrix_innovation.py playground/adaptive_bci_sampling/test_steady_step_audit.py -q -p no:cacheprovider
```

剩余主问题应按新目标聚焦：在保持输入/空间方向结构的条件下，构造无需稠密全阶谱分解的连续 HTC、全时刻阶跃包围，再以相同验收条件比较总提取加认证成本。

# 确定性 BCI 参数采样与连续参数盒证书

本目录当前保留的是一套**确定性、可复现、带连续参数盒证明**的仿射 HTC 采样与
提取流程，以及它在原始两组 Case 1 上与 stock Extended-FANTASTIC 随机提取器的
受控比较。历史探索记录（包括已判定无效的方案及其原因）移入 `records/`。

代码（全部英文标识符与注释）：

```text
zolotarev.py                       有限区间 Zolotarev 规则 + 每群谱区间 Krylov 包络
residual_certificate.py            A(h_min)-Riesz 逐点残差证书（供贪心选点使用）
certified_box.py                   连续参数盒证书（本目录的核心工具）
deterministic_design.py            确定性选点、快照组装、盒合法频率计划
exact_error.py                     精确参数映射与误差见证（全阶对照）
certify_extraction.py              驱动：设计 + stock 基线 + 证书 + 全阶验证
bench_matrix_cell_certificate.py   单元矩阵证书的对照检查（违反/细化/紧度）
bench_box_branch_and_bound.py      无富化的盒分支定界与叶细化成本
bench_pareto_budget.py             (参数点数, SVD cutoff) 的提取预算与认证缺陷
bench_dynamic_bridge_toy.py        动态桥的人工问题否证
bench_solver_cost.py               全阶求解成本（算子规模、端口数、库）
bench_extraction_time.py           提取计时与 AMG-CG 逆作用对比
bench_stock_time.py                stock 提取器的时间基线
test_certified_sampling.py         语义测试（证书、Zolotarev、谱区间、计数）
test_box_frequency_plan.py         盒合法频率计划的回归测试
test_inexact_moment_theory.py      inexact-moment 桥的语义测试（人工小系统）
records/                           正面结论 4 份 + 失败路线总表 NEGATIVE_RESULTS.md
```

复现（仓库根目录，2.5 mm 两组 Case 1）。`--steady-cells 32` 意味着 1024 个证书单元、
五个基底各扫一遍，加上 40 个参数 x 3 个 `dt` 的全阶验证，单机约 25 分钟；把
`--steady-cells` 降到 8 只需约 2 分钟（代价是逐项归一界变松，见第 2 节注）：

```text
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python playground/adaptive_bci_sampling/certify_extraction.py 2.5 \
  --steady-cells 32 --steady-order 3 --certificate-blocks 4 --greedy-maximum 3
```

```text
cd playground/adaptive_bci_sampling
PYTHONPATH=<repo>/python:. python -m unittest test_certified_sampling -v
```

---

## 1. 问题与基线

被提取对象是仿射 HTC 族 `A(p) = K + s*C + sum_i p_i H_i`（`H_i` 半正定，
`0 < p_low <= p <= p_high`，四个共址源/结温端口）。基线是 stock
Extended-FANTASTIC（`metahotspot.macromodel.utils.build_parametric_basis`）：

* 每个源、每条频移按 MPMM 椭圆匹配点求解；下一个 HTC 参数按对数均匀**随机**抽取，
  连续 `probe_rounds` 次残差检查通过后推进；
* 全部精确响应做**列归一化 SVD**（相对截断 `1e-3`），再补常向量。

它的两个缺陷都是结构性的：快照集依赖随机种子；有限次随机探针通过不等于连续参数域
上的最坏情形结论。实测尾部（已退役记录的旧计划测量）：三个参数点时十个种子给出
`0.0047% / 0.094% / 4.40%`（最小/中位/最大）的结温误差；当前计划下的两个种子见第 2 节。

## 2. 结果摘要（两次驱动运行）

两次运行的共同设置：40 个验证参数（4 个物理 HTC 角点 + 36 个固定种子对数均匀参数），
40 步 BDF1，`dt = 5 / 50 / 500 s`，SVD cutoff `1e-3`，**全部基底使用同一个盒合法频率
计划**（`box_spectral_interval`）。逐项误差按**同一参数**的精确稳态结温传递归一化。
证书设置：`--steady-order 3 --certificate-blocks 4`，5 mm 用 256 个对数单元、
2.5 mm 用 1024 个，三阶 jet、4x4 锚块，**不增加任何提取阶段的全阶求解**。

2.5 mm（9072 单元，14 条频移 + DC）：

```text
基底                        全阶求解  ROM 阶  证书(对角)  证书(逐项)  证书(绝对)  step(dt=50)  steady
确定性 3 点（本方法）           180     47     1.430e-05   7.937e-03   1.491e-05   1.201e-03   4.837e-04
stock, seed 20260805          152     43     1.480e-05   6.808e-03   1.542e-05   2.441e-03   8.724e-04
stock, seed 7                 161     42     2.034e-05   7.989e-03   2.120e-05   3.094e-03   1.330e-03
stock, seed 20260805（旧计划） 126     42     4.737e-05   1.334e-02   4.967e-05   1.757e-03   6.092e-03
stock, seed 7（旧计划）        127     42     6.222e-05   1.513e-02   6.525e-05   1.924e-03   6.770e-03
```

5 mm（1320 单元，13 条频移 + DC，证书 256 单元）：

```text
基底                        全阶求解  ROM 阶  证书(对角)  证书(逐项)  证书(绝对)  step(dt=50)  steady
确定性 3 点（本方法）           168     42     2.278e-04   1.480e-01   2.387e-04   5.773e-04   2.873e-04
stock, seed 20260805          137     36     2.322e-04   1.493e-01   2.433e-04   1.785e-03   1.068e-03
stock, seed 7                 138     37     2.292e-04   1.483e-01   2.401e-04   1.410e-03   7.605e-04
stock, seed 20260805（旧计划） 116     35     2.362e-04   1.524e-01   2.475e-04   1.262e-03   2.723e-03
stock, seed 7（旧计划）        118     36     2.337e-04   1.513e-01   2.449e-04   1.680e-03   4.697e-03
```

（“旧计划”= 修复前的裸 K 谱区间计划，见 `records/NEGATIVE_RESULTS.md` 的 A8。它的
稳态列因此比盒合法计划差约一个数量级，这正是不修计划时随机提取器在低 HTC 角上的
实际表现。）

结论：

1. **在该验证协议下最坏误差更低**：这里的“最坏”指 40 点验证集上、`dt = 5 / 50 / 500`、
   40 步 BDF1 协议内**观测到的最大值**，不是整个 HTC 盒的最坏情形，也不是 vendor 的
   Hankel / 时空能量指标。按此口径，确定性 3 点的最坏步进项是 stock 两个种子的
   1/2 与 1/2.6（2.5 mm）、1/3.1 与 1/2.4（5 mm），最坏稳态项为 1/1.8 与 1/2.8
   （2.5 mm）、1/3.7 与 1/2.6（5 mm）。证书列同向更紧或持平。
   **本表不支持“达标”**：2.5 mm 确定性设计的步进项 `1.201e-03` 仍高于 1e-3，表中没有任何
   一行达到 1e-3 动态目标；结论只是“优于这两个 stock 种子”。
2. **全阶 RHS 求解数（full-order RHS solves）不再更少**：当前设计对每条频移都用全部已选参数（`plan x points` 的完整
   张量），2.5 mm 需 180 次对 stock 的 152 / 161，5 mm 需 168 对 137 / 138。这里比较的是
   RHS 求解除数（full-order RHS solves）；分解/setup、缓存复用与 RHS 求解是三个不同的
   成本对象，不要用“求解次数”笼统概括。
   早先“少 27% 求解”的结论属于旧计划加“只对低频移铺开全部点”的旧设计，已在本次收束
   中作废（见 3.1 与 `records/NEGATIVE_RESULTS.md` 的 B19--B23）。要省求解数必须回到
   那条已被否掉的低频面规则。
3. **首次给出连续参数盒保证**：对**盒内每一点**，交付基底的稳态结温传递误差不超过表中
   数值；同一套证书套在基线基底上也成立。基线本身无法给出任何这类结论。
4. 证书是**交付后的事后验证**：与基底来源、SVD 截断、ROM 阶数无关；可以在截断阈值、
   参数点集合之间做取舍，而不产生一次额外提取求解。

**注（两个表的列不可横向比较）**：逐项归一界是最弱的一种归一，它随单元数下降；5 mm
那两行只用了 256 个单元，2.5 mm 用 1024 个，所以 5 mm 的 `1.48e-01` 与 2.5 mm 的
`7.94e-03` 不是同一件事。对角归一界与绝对界给的是紧量级，它们在同一单元数下可比。

## 3. 新方法

### 3.1 确定性选点

三个阶段，全程无随机数：

1. `zolotarev_seed`：对每个 HTC 群，用 Krylov 极值特征值迭代给出该坐标的广义谱区间
   `spectrum(A_minus, H_i)` 的**外包**，再用有限区间 Zolotarev 规则取一个内点。该点
   是一维核 `1/(lambda + p)` 的极小极大有理插值节点，误差有闭式界
   `4*exp(-n*pi^2/log(16*gamma))`。
2. `certified_greedy_points`：在确定性的 41x41 对数网格上，按 `A(h_min)`-Riesz
   残差**证书**取最大者作为下一个参数（弱贪心）。每个候选点的得分都是该点传递误差的
   上界，停止条件因此是确定性的有限候选陈述，而不是“随机探针连续通过”。
3. `build_basis`：每端口沿用与 stock 相同的 MPMM 椭圆频移（**盒合法**计划，13/14 条），
   在全部已选参数上补 `s = 0`（稳态）端点。快照集是 `(plan, points)` 的**完整张量** ——
   每条频移都用全部已选点，不区分低频与高频，也没有 `dt` 或任何调用者尺度进入提取。
   最后做**未经改动**的列归一化 SVD（`1e-3`）+ 常向量。稳态端点与贪心评分共用缓存，
   因此不产生重复分解。

2.5 mm 上选出的三点为有效坐标 `(346.03, 14.53)`、`(1.00, 234.38)`、`(9285.71, 0.996)`，
   选点证书停在 `6.662e-03`（5 mm 为 `(334.84, 10.39)`、`(1.00, 118.58)`、
   `(9285.71, 0.992)`，停在 `4.125e-03`；两次都是第 3 点后触发预算停止）。
   2.5 mm 每端口 14 条频移加 DC，故 `4 端口 x 15 x 3 点 = 180` 次全阶求解（5 mm 是
   `4 x 14 x 3 = 168`）。

### 3.2 整盒证书：命题

固定实频移 `s >= 0`、交付基底 `V`（列正交，来源不限）、盒 `[p_low, p_high]`。记

```text
A(p)   = K + s*C + sum_i p_i H_i
X(p)   = A(p)^-1 G
X_V(p) = V (V^T A(p) V)^-1 V^T G
Y(p)   = G^T X(p),   Y_V(p) = G^T X_V(p)
```

则对**盒内每个 p** 与每个端口对 (a,b)：

```text
|Y_ab(p) - Y_V,ab(p)| <= B_ab := sqrt(D_a D_b)
D_a  = max over Bernstein nodes of  [ block^T Gram(anchor) block ]_aa
```

其中 `Gram(anchor) = Z^T A(anchor)^-1 Z` 是残差张成空间
`Z = [G, (K+sC)V, H_1 V, ..., H_d V]`（列数 `(d+2)m + k`）在锚点处的 Riesz Gram。
相对形式再除以**同一参数**精确稳态传递的下界；由 M-矩阵单调性
`dY_ab/dp_k = -x_a^T H_k x_b <= 0`，单元上角点的精确传递即该下界。

### 3.3 证明

1. **Galerkin 最优性**：`X_V(p)` 是 `X(p)` 在 `range(V)` 内的 `A(p)`-能量投影，故对
   *任意*试探系数矩阵 `q(p)`，`r_q(p) = G - A(p) V q(p)` 满足
   `Y(p) - Y_V(p) = r(p)^T A(p)^-1 r(p) <= r_q(p)^T A(p)^-1 r_q(p)`；且
   `Y - Y_V` 半正定，故非对角项被 `sqrt(对角_a * 对角_b)` 控制。
2. **Loewner 单调性**：`H_i` 半正定、`p` 不低于锚点，故 `A(p) >= A(anchor)`、
   `A(p)^-1 <= A(anchor)^-1`，于是加权残差范数被锚点 Riesz 范数上控。
3. **可计算化**：取 `q(p)` 为**小系统**
   `(V^T A(p) V) a = V^T G` 在单元中心的 Taylor jet。于是 `r_q` 是多项式，其分块系数
   落在固定张成空间 `Z` 内；多项式在单元上的张量 Bernstein 系数是凸组合，因此 Gram
   二次型在该点的值不超过系数逐个取值中的最大者——这一步给出**整单元**（而非单点）的界。
4. **归一化**：由 3.2 的单调性取单元上角点精确传递作分母，即为同参数相对误差上界。

全部步骤只需一次锚点稀疏分解与 `span_columns` 次稀疏回代；单元上的评估是小系统稠密
代数，与模型规模无关。把单元数、jet 阶数提高不产生任何新的全阶求解。

### 3.4 成本与实测紧度

* 成本（本次两次运行实测）：2.5 mm 每个基底 16 个锚 Gram + 1024 个角点解，证书耗时
  `173--226 s`；5 mm 的 256 单元为 `5.5--5.7 s`。证书的求解全部落在**固定算子**上，
  与提取阶段的 AMG-CG 逆作用是两条不同的路径（见第 6 节第 6 条）。
* **上界性质经审计**（历史运行，本次收束未复跑：5 mm、256 单元、三阶 jet、每单元 25 个
  随机内点，共 6400 个全阶对照解）：没有任何单元出现 `证书 < 实测`；全盒最大实测绝对
  误差 `5.531e-03` 与其所在单元的证书 `5.622e-03` 相差约 2%。单元级比值可以到数百倍
  （该单元自身误差极小），逐点比值则只有 1.1--5.7（2.5 mm，10 余个分布点）。
  同一检查的轻量版本是 `certify_extraction.py --audit-cells 8 --audit-samples 2`；
  任何 `证书 < 实测` 都会直接报错。
* **细化与紧度**（`records/MATRIX_CELL_CERTIFICATE.md`，5 mm、设计基底、`s = 0`；
  该记录用的是旧计划）：16 单元/轴、3 阶 jet、块锚下界为 `1.8396e-04`，650 点精确盒
  最大值 `1.8248e-04`，只高 0.8%；本证书的量是平方量，换算到状态即认证的最坏相对
  A-能量状态误差 `1.356e-02` 对真实 `1.351e-02`。同一套证书在 2/4/8/16 单元/轴下
  逐级收紧（局部锚 2 阶：`7.435e+01 -> 2.817e+00 -> 5.860e-02`），且**所有配置的
  Loewner 违反数为 0**。细化两个方向不可互换：固定分割升阶每阶约省 4 倍（8 单元/轴），
  细化分割收益更大，但每个自建锚的单元要付一次 Riesz Gram（5 mm 上 130 次稀疏求解）。
* 证书对基底有区分力，但幅度不大：本次两次运行里确定性基底的对角界只比 stock 紧
  2--30%（5 mm `2.278e-04` 对 `2.322e-04 / 2.292e-04`；2.5 mm `1.430e-05` 对
  `1.480e-05 / 2.034e-05`）。早先“基线松 2.5--3.3 倍”的说法来自旧计划与不同的
  单元/锚配置，已作废。
* `shift = 1/dt` 时同一套证书覆盖 BDF1 递推所用的算子族（`test_shift_certificate_
  covers_the_shifted_family` 在玩具族上验证 `bound >= 实测`）。

## 4. 已验证无效的方案

30 余条失败路线（每条含判据式数字、判定类别与复现命令）收束在
`records/NEGATIVE_RESULTS.md`，不在本文件重复。按判定分类，历史上被否掉的是：

* **界太松**（不等式成立，数值不可能驱动 1e-3 判断）：全局矩阵分式 Bernstein 与均匀
  Neumann 残差证书、单一全局谱标量的廉价盒残差界、BDF1 输出残差界（`K_min^-1` 全局
  Gram）、把锚定 Riesz 论证搬到频率轴、先验闭式点集的全纯保证、张量 Zolotarev 的直接
  计数。
* **被反例证伪**（前提在实测数据上直接失效）：只用四个 HTC 角点；加密确定性采样应带来
  单调改善；切角规则（单角点、三切角、条件 Zolotarev 边缘布点）；场残差 greedy 加密
  候选网格或将种子加密到 2x2；在未改动的交付基上事后加接受证书（SVD 截断按快照方差而
  非方程残差排序）。
* **成本不可行**：Taylor/Bernstein 局部盒证书、单元 Chebyshev/Neumann 细分、约化
  Neumann--Bernstein 残差盒、固定 Zolotarev 4x4 张量、active-set oracle 的真实 Case 1
  验证（`118.10 s` 对 stock `1.12 s`）、1 mm 的稀疏直接 LU 谱准备。
* **测量更正**（原数字撤回，不是方法失败）：`2.61e-2` first-term failure（LU cache 键只
  按 shift 值）、`semigroup_box_bounds`、“一个 Gramian 同时给出两个 all-input 指标”。
* **仍开放、不是被否证**：先验闭式采样的方案选择（见第 6 节第 7 条），以及频率轴
  （Hankel、脉冲能量）的整盒证书。

## 5. 复现与验证

* **31 个语义测试全部通过（0.49 s）**，分三个模块：
  `test_certified_sampling.py`（Bernstein 包络、Zolotarev 闭式界与节点、每群谱区间外包、
  逐点残差证书、单元界上控实测误差、分母单调下界、细化收敛、平移算子族覆盖、贪心可
  复现、求解计数）、`test_box_frequency_plan.py`（盒合法计划的 Loewner 包围与 5 mm
  回归）、`test_inexact_moment_theory.py`（人工小系统上的 inexact-moment 恒等式）。
* **本次两次驱动运行**（第 2 节的数字来源）：

```text
# 5 mm：13 条频移 + DC，256 个证书单元
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python playground/adaptive_bci_sampling/certify_extraction.py 5 \
  --steady-cells 8 --steady-order 3 --certificate-blocks 4 --greedy-maximum 3 --output <path>.json

# 2.5 mm：14 条频移 + DC，1024 个证书单元（约 25 分钟）
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python playground/adaptive_bci_sampling/certify_extraction.py 2.5 \
  --steady-cells 32 --steady-order 3 --certificate-blocks 4 --greedy-maximum 3 --output <path>.json
```

```text
# 全盒审计（每单元 samples^d 个随机内点，任何 证书 < 实测 都会直接报错）
PYTHONPATH=python python playground/adaptive_bci_sampling/certify_extraction.py 5 --steady-cells 8 --steady-order 2 --audit-cells 8 --audit-samples 2 --skip-stock
```

生成 JSON 全部写在仓库外（`--output`；本目录 `.gitignore` 只用于防误提交）。
测试运行方式：

```text
cd playground/adaptive_bci_sampling
PYTHONPATH=<repo>/python:. python -m unittest discover -s . -p "test_*.py"
```

## 6. 局限与未决问题

1. **未认证的量**：40 步 BDF1 轨迹误差与频率轴（Hankel、脉冲能量）只有实测；
   证书覆盖的是给定实频移（`s = 0`，以及 `s = 1/dt` 的算子族）的稳态传递。
2. **浮点**：所有不等式在实数域精确成立；稀疏分解、Gram 与稠密求解是普通浮点，
   报告中显式给出 `floating_point_certified=False`。要做成计算机辅助证明需要
   外向舍入的区间线性代数。
3. **逐项归一化仍偏松**：弱耦合项的自身分母很小，逐项相对界在 2.5 mm/1024 单元
   为 `7.937e-03`（5 mm/256 单元更松，见第 2 节注）；对角归一与绝对量已到
   `1.430e-05 / 1.491e-05`。
4. **选点最优性**：种子点有单变量极小极大依据，后续点只是确定性贪心，没有
   多维最优性证明。
5. **未做**：跨网格（1 mm）证书、更多 HTC 分组下的证书、把证书闭入截断阈值搜索
   （用证书挑选最小合格 ROM 阶数）；以及 2.5 mm 上 1024 个证书单元带来的 25 分钟
   墙钟（降低 `--steady-cells` 是最直接的取舍，代价是逐项归一界变松）。
6. **求解策略（本次未改动的成本项）**：提取与全阶参考使用 AMG 预条件 CG
   （`rtol = 1e-10`；算子 `K + s*C + sum_i p_i H_i` 对称正定，因此不需要 GMRES）。
   1 mm 上的对照说明这条路线的必要性：同一份谱准备，稀疏直接 LU 要 `101.03 s` /
   `4.37 GB`，AMG-CG 只要 `18.91 s` / `668 MB`，两种方法的节点相对差 `1.2e-13` /
   `1.6e-13`、代表点上的误差差最大 `9.4e-12`（`records/NEGATIVE_RESULTS.md` B24）。
   但以下路径仍在用稀疏直接分解：`certified_box.py` 的每角点分母、
   `exact_error` 的参数映射、`deterministic_design` 的选点与最小算子。
   5 mm 上直接分解其实更便宜（每点约 `4 ms` 对 CG 的 `16 ms`，实测），
   2.5 mm 与 1 mm 上则相反——2.5 mm 证书的 `173--226 s` 几乎全部来自每角点一次分解。
   把这批固定算子求解统一到 AMG-CG（必要时按规模选择）是明确的下一步；
   记录不把迭代逆声明为正式认证的谱包围，那需要单独的余项论证。

7. **开放候选（不是被否证的路线）：先验闭式参数采样**。参数只通过边界单元的对角扰动进入
   （`H_k = diag(a_k)`），所以 Woodbury 在**一个**参数处的 `m_b + n_src` 次解就精确给出整个参数
   族（不是拟合，残差就是求解器精度）；沿任意射线整个参数向量只通过单个标量进入，解是简单极点
   之和，射线极点求值器可用。实测（2.5 mm）：场误差 `5.86e-07`、结温相对误差 `5.10e-08`、最难角
   `cond(M) = 2.46e+04`；探针中最差 `cond(M) = 2.92e+13` 处场误差仍只有 `9.50e-08`（病态是良性
   的，而且是被检查过而不是被假设的）；极点级数对全阶解 `6.84e-07`。未达成的是**采样方案本身**：
   在结温向量上计数无用（只有 `n_src` 行，任何容差下都饱和）；只用从参考点出发的对角射线张成的
   span 会停在约 2% 结温误差上不再下降；未加权的 Gram 会随网格细化抬高计数（要用求和为一的正交
   权重，加权后 15/25/35 点每轴的计数才一致）。缺口的数学形式见 `THEORY.md` 的 P1「Robin 解流形
   的 n-width 衰减」。

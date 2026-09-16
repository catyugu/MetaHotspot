# weekly_report_0915

## 非共形接口连接

- 两个热模型连接时，不要求两侧原始 face 一一对应，而是在两侧界面网格的并集上构造 **common patches**。
- 对任一侧 $S$，原始边界 face $j$ 到物理界面的半单元热导为

$$
g_{S,j}=\frac{k_{S,j}A_{S,j}}{d_{S,j}}.
$$

- 若公共面片 $i$ 落在原始 face $j(i)$ 内，面积为 $A_{if,i}$，按面积比例 $\xi_{S,i}=A_{if,i}/A_{S,j(i)}$ 分配，则

$$
h_{S,i}=\xi_{S,i}(E_Sg_S)_i
       =\frac{k_{S,j(i)}A_{if,i}}{d_{S,j(i)}}.
$$

- 若 ROM 一侧温度场写成 $T_S\approx V_Sq_S$，令 $R_S=E_SV_S$、$H_S=\operatorname{diag}(h_S)$。公共面温度只满足代数热流平衡，因此可以消去。定义

$$
H_\Gamma=(H_L^{-1}+H_R^{-1})^{-1},
$$

则连接项为

$$
\boxed{
\begin{bmatrix}
R_L^TH_\Gamma R_L & -R_L^TH_\Gamma R_R\\
-R_R^TH_\Gamma R_L & R_R^TH_\Gamma R_R
\end{bmatrix}}
$$

- common patches 只负责非共形积分，不作为额外全局自由度，因此连接矩阵仍保持对称和守恒。

## FANTASTIC / BCI CTM 路线校正

- 本周继续核对 FANTASTIC、可连接 BCI CTM、Extended FANTASTIC 和 contour elements 相关工作。主要参考包括 2014/2015 FANTASTIC、2017/2018 可连接 BCI CTM、2021 Extended FANTASTIC，以及 2023 contour elements。
- 之前尝试过把 closing SVD 改成 projection-space columns 的解释，这会明显改变 ROM order，也没有足够依据把它当成现有方法的直接替代。本周回到之前已经反复验证的做法：对 accepted exact-response snapshots 逐列归一化后做 closing SVD，并显式保留无 HTC 系统的 uniform-temperature mode。
- simple EROM 继续使用 $10^{-3}$ tolerance。whole-PoP 则把项目验证基线收紧到 $10^{-3}$；需要强调的是，2023 论文报告的是 $10^{-2}$，这里的 $10^{-3}$ 是为了当前 surrogate 上的精度而采用的项目基线，并不是说论文使用了这个值。
- 全阶响应求解继续使用 AMG-preconditioned CG。

## 2023 whole-PoP

- 仍然把整个 PoP 作为一个 BCI ROM 提取对象，而不是把某一个表面或局部区域单独降阶。
- surrogate 为 **562,176 DoF**，含两个独立 die heat sources，外边界为四个 side、bottom、top 共六个表面；四个 side 共用一个 HTC 参数，bottom 和 top 各自独立，因此共有三个 HTC 参数。
- HTC 范围仍与论文公开范围一致：side 0.1–200、bottom 0.1–1000、top 0.1–10000 W/(m²K)。
- 在 $10^{-3}$ tolerance、20 次 residual probe 下，共得到 105 个 exact responses；closing SVD 保留 23 个 modes，再加 uniform mode，最终 ROM order 为 **24**。最大 accepted residual 为 $9.58\times10^{-4}$。

当前 extraction baseline 的 full boundary trace 稳态结果为：

| 工况 | global $L_\infty$ | volume $L_2$ | die1 junction | die2 junction |
| --- | ---: | ---: | ---: | ---: |
| paper Fig.2 BC | 0.350% | 0.309% | 0.0063% | 0.0114% |
| paper Fig.4 BC | 0.418% | 0.528% | 0.0247% | 0.0330% |
| range low corner | 0.031% | 0.0057% | 0.0027% | 0.0027% |
| range high corner | 0.533% | 0.757% | 0.0175% | 0.0196% |
| paper Fig.3 extrapolation | 0.949% | 1.136% | 0.0719% | 0.0916% |

- 训练范围内最坏稳态 global $L_\infty$ error 为 **0.533%**，出现在 range-high corner。
- 对两个 source 分别在 $s=0,0.1,10\;\mathrm{s}^{-1}$ 做独立 transfer holdout，训练范围内最坏 global $L_\infty$ error 为 **0.513%**。
- 对比之前 $10^{-2}$ 的试验，ROM order 从 14 增加到 24，训练范围内最坏稳态误差从 3.266% 降到 0.533%，transfer holdout 从 5.471% 降到 0.513%。
- 这些结果说明 24 阶 response space 对目前测试的内部 source、HTC 参数范围和若干频移响应已经足够准确，但这并不等价于它已经覆盖了模型连接后可能出现的大部分边界驱动响应。后者仍然是需要单独验证的提取问题。

## contour elements

- contour elements 仍按论文的定义处理：先提取 BCI CTM，再压缩 boundary temperature / heat-flux representation，不把 contour polynomial 当作额外内部 ROM state。
- 一个矩形面上使用总次数不超过 $p$ 的二维多项式，系数数目为

$$
\frac{(p+1)(p+2)}{2}.
$$

- 按论文给出的四个 side $p=2$、bottom $p=4$、top $p=10$，系数总数为

$$
4\times 6+15+66=105.
$$

论文正文写成 $110\times42$，但没有给出额外 5 个系数的来源，因此这里仍按明确给出的阶数和计数公式处理。

### 边界离散复核

- 原 whole-PoP 试验把 Robin 项按 $hA$ 直接作用在 boundary cell trace 上。进一步按有限体积半单元热导和零质量 boundary node 消元重新计算后，固定同一个 24-state interior basis，full boundary trace 的训练范围内最坏稳态误差为 **0.503%**，transfer holdout 最坏为 **0.481%**。
- 同样采用半单元 / boundary-node formulation 后，论文固定 contour degrees 的最坏稳态误差仍为 **23.735%**，transfer holdout 最坏为 **24.552%**。因此此前约 24% 的异常并不是由 $hA$ 边界离散方式造成的。
- 之前报告的逐面 boundary-trace projection error 只能作为表示空间的诊断量，不能直接当成系统误差指标。例如 side trace 的投影误差并不小，但在实际系统响应中单独压缩 side 几乎不改变误差。因此后续应以 connected response / transfer error 为主要判据。

### 划分与阶数敏感性

保持同一个 24-state interior basis，并采用半单元 / boundary-node formulation，只改变 bottom 的表示方式；side 仍用 $p=2$，top 仍用 $p=10$：

| boundary representation | 表示规模 | in-range steady max | transfer max |
| --- | ---: | ---: | ---: |
| full boundary trace | 41,856 | **0.503%** | **0.481%** |
| 论文固定 degrees：bottom 单块 $p=4$ | 105 | 23.735% | 24.552% |
| bottom 单块 $p=16$ | 243 | 0.710% | 0.720% |
| bottom 单块 $p=24$ | 415 | 0.671% | 0.689% |
| bottom $5\times5$ 几何对齐分区，每块 $p=1$ | 165 | 0.824% | 0.897% |
| bottom $5\times5$ 几何对齐分区，每块 $p=2$ | 240 | 0.669% | 0.689% |

- bottom 单块 polynomial 随阶数提高会平滑收敛到 full-trace 结果，说明 contour 投影、积分和温度 / 热流映射本身没有表现出明显的基础实现错误。
- 根据 surrogate 内部 $\pm4.5$ mm、$\pm8$ mm 几何特征对 bottom 做分区后，局部低阶 polynomial 也能把系统误差恢复到约 0.7–0.9%。
- 但这反过来说明当前 contour-element 方案依赖于模型特定的先验：不同模型可能需要不同的 surface partition 和 polynomial degree。虽然它可以有效压缩边界表示，但如果需要人工根据几何和响应特征调节划分与阶数，就不是一个令人满意的通用 BCI boundary representation 方案。

## simple EROM

- simple EROM 继续遵守“提取一次，所有外接工况复用同一个 ROM”的原则。
- 训练空间包含一个内部 physical source，以及一个 constant interface heat-flux direction，用来覆盖外接模型向 ROM 反向注热的响应；后者只参与 basis training，不作为物理 source 使用。
- tolerance 固定为 $10^{-3}$。共得到 19 个 exact responses，closing SVD 保留 10 个 modes，再加 uniform mode，最终 order 为 **11**，和 FloTHERM EROM 的 11 阶相同。

| 工况 | FloTHERM / MHS order | steady global: FloTHERM / MHS | transient global: FloTHERM / MHS | MHS junction |
| --- | ---: | ---: | ---: | ---: |
| baseline copper | 11 / 11 | **0.742%** / 0.778% | 1.777% / **0.563%** | 0.080% |
| strong bottom HTC | 11 / 11 | 2.765% / **0.835%** | 4.022% / **0.787%** | 0.037% |
| 200 W external source | 11 / 11 | 4.151% / **0.646%** | 2.426% / **0.411%** | 0.086% |
| combined stress | 11 / 11 | 3.542% / **0.611%** | 2.276% / **0.473%** | 0.074% |
| layered extreme + source | 11 / 11 | 3.231% / **0.718%** | 2.089% / **0.590%** | 0.079% |

- 在相同 11 阶下，MHS 的 baseline-copper steady global error 略高于 FloTHERM，其余四个 stress cases 均更低。
- transient global error 在五个工况中均低于 FloTHERM，MHS junction error 全部低于 0.086%。
- constant interface heat-flux direction 的训练说明，可以在提取阶段主动加入边界激励来扩大 response space 对外接工况的覆盖；但单一常数方向只适用于这个简单例子，尚不能解决一般三维边界响应的覆盖问题。

## 求解器

- full-order response、whole-PoP validation、连接后的 steady system 和 BDF1 transient system 均使用 **AMG-preconditioned CG**。
- steady 问题复用同一个 AMG hierarchy；固定时间步的 transient 问题复用 $K+C/\Delta t$ 的 AMG hierarchy，并用前一步解作为 CG warm start。
- 目前没有观察到需要回退到稀疏直接法的理由。

## 当前结论

- whole-PoP 在 $10^{-3}$ 项目基线下得到 24 阶 ROM；对目前测试的内部 source、HTC 参数范围和 transfer holdout，full-trace error 约为 0.5%，说明当前 interior response space 在这些工况下已经比较稳定。
- contour elements 的大误差已经确认主要来自 boundary partition / polynomial order 与当前模型不匹配，而不是 contour 公式或边界离散的基础错误。几何对齐分区可以把误差降回 1% 以下，但它依赖特定模型的几何与响应先验，因此并不是理想的通用方案。
- 更关键的问题转移到了**提取阶段的 boundary-response coverage**：需要设法让内部 response basis 在不依赖具体外接模型的前提下，覆盖大部分常规连接工况可能激发的边界温度 / 热流响应。
- 后续应优先调研如何在 FANTASTIC / BCI CTM 提取过程中加入代表性的边界激励、port response 或自适应 enrichment，并用 connected-system / transfer error 判断覆盖是否充分；在此基础上再研究紧凑、自动的 boundary representation，而不是继续手工调 contour partition 和 degree。

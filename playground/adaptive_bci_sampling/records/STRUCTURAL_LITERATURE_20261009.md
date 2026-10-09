# 不依赖提高参数阶数的结构性路线：文献核对与研究判决

日期：2026-10-09。基线：`bd5dba4b2850f9a0d87058526ab6f7288539305f`。
本记录是文献与源码核对，没有新增 ROM 性能实验，也没有证明新的全域尾项定理。

## 本次改变的判断

1. “把参数放在边界反馈中，整体求解而不逐阶展开”已有直接先行工作，不能作为新颖性主张。
2. “再降阶误差系统，用第二残差补足遗漏尾项”也已有直接先行工作。当前原型在连续域、全时间和真实 CG 缺陷方面的实现进展，不等于该思想首次提出。
3. 提高多项式阶数收益有限，并不自动说明共同空间维数必须很大；参数系数逼近和空间可压缩性是两个问题。
4. 下一步应先研究高秩 Robin 边界下的输入驱动反馈可达集合，再判断是否有值得推导的、全域有效的谱尾项。参数维数小不等于边界扰动秩小。

## 本项目验收契约

系统 `C x' + K(h)x = G u`，C SPD，K(h)=K0+ΣhiHi 对目标盒 SPD，Hi PSD。
零初值、固定 HTC、任意有符号常输入组合。对指定 SVD 前空间 V：

- 稳态全场 K-energy 相对误差 ≤ 0.001。
- 每个 t>0 的阶跃全场 C-energy 相对误差 ≤ sqrt(0.001)。
- 整个连续参数盒，不能由有限训练点或平均参数误差替代。
- 正式规模全阶求解使用 AMG–CG，包含实际求解缺陷；不采用全阶稠密谱分解。

当前模型仍为 Case1 matrix-only reconstruction，有效 Robin 域的原生物理 HTC 映射尚未独立验证。现有浮点评估不等于向外舍入机器证书。

## 文献逐项核对

### L1. Sampling-free 的参数反馈：最直接的先行工作

Christopher Beattie, Serkan Gugercin, Zoran Tomljanović,
*Sampling-free model reduction of systems with low-rank parameterization*,
Advances in Computational Mathematics 46, 83 (2020).
[发表版](https://doi.org/10.1007/s10444-020-09825-8)，
[作者预印本](https://arxiv.org/abs/1912.11382)。

发表版摘要与作者预印本第 2 节核对：将低秩参数依赖拆成四个非参数子系统和参数反馈；Theorem 1 从子系统误差及反馈放大项组成参数化误差界；Theorem 2 在反馈子系统保正实、其余子系统稳定等条件下保证全参数稳定性。故“不采样参数、参数反馈精确保留”已有方法。

适用障碍：低秩变化是重要成本条件。预印本明确指出高秩变化会产生大量端口。其传递函数范数界也不能直接替代本项目逐时刻相对全场阶跃契约；同一固定 Galerkin V 与分别降阶后连接子系统的实现并不自动等价。

更早先行工作：U. Baur, C. Beattie, P. Benner,
*Mapping parameters across system boundaries: parameterized model reduction with low rank variability in dynamics*, PAMM 14(1), 19–22 (2014)。本次从 L1 第 2.4 节与参考文献核对其关系，没有独立获取该短文全文。

### L2. 谱尾项与共同端口空间

Kathrin Smetana, Anthony T. Patera,
*Optimal Local Approximation Spaces for Component-Based Static Condensation Procedures*,
SIAM Journal on Scientific Computing 38(5), A3318–A3356 (2016).
[DOI](https://doi.org/10.1137/15M1009603)，
[MIT 存档](https://dspace.mit.edu/entities/publication/98c5582d-4c97-4e6e-9f02-2bea41a29a29)。

核对出版页和 MIT 作者存档摘要：由紧传递算子及其伴随构造最优端口空间；参数 spectral greedy 构造共同空间。摘要明确将参数验收表述为丰富训练集。不能据此推断整个连续 HTC 盒已获认证。

Julia Schleuss, Kathrin Smetana,
*Optimal Local Approximation Spaces for Parabolic Problems*,
Multiscale Modeling & Simulation (2022).
[DOI](https://doi.org/10.1137/20M1384294)，
[作者预印本](https://arxiv.org/abs/2012.02759)。

核对全文 Theorem 3.4 / 式(3.4)：局部空间时间传递算子的最优误差为 sqrt(λ[n+1])。全局误差用局部 L2(H1) 误差控制 graph norm，非齐次源项需另行处理；Remark 5.3 说明数据项和范数限制。这个谱尾项针对定义完备的算子，不能替换成快照 SVD 尾能量。有限时段、局部范数与本项目 t∈(0,∞)、全场相对范数仍有距离。

### L3. 低空间秩与参数多项式阶数可以脱钩

Markus Bachmayr, Albert Cohen,
*Kolmogorov widths and low-rank approximations of parametric elliptic PDEs*.
[作者全文](https://arxiv.org/abs/1502.03117)。

核对第 2 节及第 4 节：给出低秩与多项式逼近表现不同的例子；特定分片扩散系数可通过骨架上的 Steklov–Poincaré 算子得到更快 n-width 衰减，且依赖几何。不能把该结论泛化成任意 Robin 动态系统的快速衰减定理。它支持把空间构造和参数系数表示分开研究。

Albert Cohen, Ronald DeVore,
*Kolmogorov widths under holomorphic mappings*, IMA Journal of Numerical Analysis 36(1), 1–12 (2016).
[作者全文](https://arxiv.org/abs/1502.06795)。

核对全文引言和主结论：解析解映射可传递 n-width 衰减信息，但有前提和速率损失。此类存在性/渐近结论不能提供当前宽参数域和指定容差下足够紧的可计算尾项。

### L4. 共同 Gramian 与真正的确定性截断定理

M. Petreczky, R. Wisniewski, J. Leth,
*Balanced truncation for linear switched systems*, Nonlinear Analysis: Hybrid Systems (2013).
[作者全文](https://arxiv.org/abs/1302.0221)。

核对式(9)–(10)、Theorem 6：共同可控/可观测 Gramian 不等式和相应平衡截断给出 L2 输入输出误差界 2Σ discarded σ。对仿射参数、固定 P/Q、固定 B/输出算子，不等式对参数仿射，验证全部盒顶点即可覆盖盒；这是矩阵仿射凸性推论，不是说任意参数依赖 Gramian 只查顶点即可。

限制：共同 Gramian 要满足完整不等式，采样/插值近似不能自动继承；标准算法所得 ROM 未必是本项目指定的 Galerkin ROM。常输入在无限时间上不是 L2，有限时间输入范数随 sqrt(T) 增长，L2 定理不直接给逐时刻相对阶跃保证。大规模共同 Gramian 的紧度和完整尾谱认证也是成本问题。

Igor Pontes Duff, Sara Grundel, Peter Benner,
*New Gramians for Linear Switched Systems: Reachability, Observability, and Model Reduction*.
[作者全文](https://arxiv.org/abs/1806.00406)。

核对 Definition 1、Theorem 3 和 Theorem 4：递归核编码参数/模式作用生成的可达空间，误差界另需 Assumption 1 等条件。上一条讨论的“参数作用链”与这些已有机制高度相关；不能仅更换名称宣称创新。

### L5. 双线性化已有热模型应用，但不要偷换误差契约

Angelika Bruns, Peter Benner,
*Parametric model order reduction of thermal models using the bilinear interpolatory rational Krylov algorithm*,
Mathematical and Computer Modelling of Dynamical Systems 21(2), 103–129 (2015).
[DOI](https://doi.org/10.1080/13873954.2014.924534)。

出版信息与摘要核对，全文获取失败；本次另读 Benner 的
[作者报告](https://csc.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-MOR4MEMS-KIT-2015.pdf)，
其中明确展示参数系统双线性化和热模型应用。不据此声称该论文证明了本项目所需的连续域阶跃界。

Martin Redmann,
*Type II Balanced Truncation for Deterministic Bilinear Control Systems*,
SIAM Journal on Control and Optimization (2018).
[DOI](https://doi.org/10.1137/17M1147962)，[作者全文](https://arxiv.org/abs/1709.05655)。

核对式(9)–(12)、Theorem 4.1 / Corollary 4.2：控制幅值进入 Gramian 条件，有界控制下给有限时间 L2 输出误差和尾谱界。该框架有可利用的确定性理论，但泛化的双线性控制界未直接利用“HTC 固定、参数作用纯耗散”。宽 HTC 幅值与任意物理功率缩放必须分别处理；不能把二者一并当有界控制后声称满足原契约。

### L6. 参数依赖 Lyapunov 及辅助误差系统

Nguyen Thanh Son, Tatjana Stykel,
*Solving Parameter-Dependent Lyapunov Equations Using the Reduced Basis Method with Application to Parametric Model Order Reduction*,
SIAM Journal on Matrix Analysis and Applications 38(2), 478–504 (2017).
[DOI](https://doi.org/10.1137/15M1027097)。

本次核对出版摘要：仿射依赖、min-theta、后验估计与 greedy 构造参数平衡截断。未独立取得全文，故不把它的后验估计解释为共同 Gramian 的全域 Loewner 上包络。

Johannes Rettberg, Dominik Wittwar, Patrick Buchfink, Robin Herkert, Jörg Fehr, Bernard Haasdonk,
*Improved a posteriori Error Bounds for Reduced port-Hamiltonian Systems* (2024).
[DOI](https://doi.org/10.1007/s10444-024-10195-8)，[作者全文](https://arxiv.org/abs/2303.17329)。

核对第 3.2–3.4 节：辅助系统近似真实误差，以第二残差补足遗漏；层次差值单独作界需难以验证的 saturation 假设，保留残差的加法界则不靠该假设。与本项目辅助误差框架接近，因此“辅助误差系统本身”不是研究空白。其有限时段残差积分界与本项目连续参数全时间认证并不相同。

### L7. 有理逼近的连续谱界：工具而非直接解答

Stefano Massei, Leonardo Robol,
*Rational Krylov for Stieltjes matrix functions: convergence and pole selection*,
BIT Numerical Mathematics (2021; online 2020).
[DOI](https://doi.org/10.1007/s10543-020-00826-z)，[作者全文](https://arxiv.org/abs/1908.02032)。

核对 Corollaries 3.14–3.18：固定 SPD 算子的移位预解式、指数及 Stieltjes 函数有连续移位/时间界与理论极点选择。可用来研究固定内部扩散算子的动态截断。但 HTC 使算子本身变化，且 Hi 通常不与扩散算子交换；固定 A 的结果不等于共同 HTC 空间定理。理论极点代替经验移位本身也不是足够的研究贡献。

## 由源码核对排除“两个参数即秩二”的误解

`case1_system.py` 中 Hi 是正面积权重乘边界指示形成的对角矩阵。
故 rank(Hi) 等于该组受作用单元数，而不是独立 HTC 参数数。两组边界支撑不相交，联合秩相加。

按 `model_case1.py` 原有 Case1Config 顶点生成规则和 DIES 掩码做几何计数：

| mesh mm | 自由度 | 网格形状 | 顶部 Hi 秩 | 底部 Hi 秩 | 联合边界秩 |
|---|---:|---|---:|---:|---:|
| 10 | 364 | 7×13×4 | 4 | 91 | 95 |
| 1.5 | 47,085 | 43×73×15 | 196 | 3,139 | 3,335 |
| 1 | 122,400 | 60×102×20 | 400 | 6,120 | 6,520 |

这是源码确定的几何/秩核对，零全阶 RHS，不是原生组装验证，也不是新的性能实验。
本机直接导入重建函数因缺 PyAMG 失败；随后用 Python AST 仅载入原配置常量及 Case1Config，按原掩码计数，未替换求解器。

## 可以直接写出的结构，和仍需研究的定理

以下为本项目算子的代数推论，不主张新颖性。
取原目标盒下端作为参考 K*=K(h_min)，令 Q 为全部 Robin 支撑上的坐标嵌入。
由于 Hi 对角，可写

K(h)=K*+Q Δ(h) Qᵀ，Δ(h)≥0。

参考算子用正下端，避免原 K0 的 Neumann 常数零模使基系统在 s=0 不稳定。
定义 R(s)=(sC+K*)^-1，S(s)=QᵀR(s)Q。
零初值的精确状态传递为

X(s,h)=[R(s)G−R(s)Q Δ^(1/2)
        (I+Δ^(1/2)S(s)Δ^(1/2))^-1 Δ^(1/2)QᵀR(s)G]U(s)。

该形式在 Δ 部分为零时仍可用，不需要 Δ^-1。参数作用被整体保留，不需参数 Taylor 阶数。
R/S 的耗散与正实结构支持研究保结构反馈；但稳定不等于误差小，反馈放大项仍要证明。

此处 Q 有 3335/6520 列，直接把所有边界方向当自由输入可能过于昂贵和保守。
真正的问题是：能否利用 QᵀR(s)G 以及反馈作用，认证一个更小的边界可达空间，
使被舍弃方向在所有 h、所有 t 和所有常输入 u 下的贡献都有确定性上界？
“G 只有四列”不保证闭环边界轨迹始终位于某个固定四维空间。

另一个关键障碍：端口传递谱通常基于所有边界数据，且扩散平滑性依赖源区、目标区间的距离。
本项目要求全场（含边界单元），不能套用只针对远离边界内部区域的谱衰减结论。

## 下一阶段只做三个判决，不预先宣布新方法

1. **结构判决**：小模型比较任意边界激励的传递谱、真实输入驱动闭环边界集合的宽度、以及原误差系统的对应集合。
   扩大参数范围但不提高参数多项式阶数，检查所需方向数及遗漏贡献；采样结果只用于假设筛选。
   若仅内部区域低秩、含边界的全场不低秩，应直接报告此障碍。
2. **证明判决**：尝试把一个完整传递算子谱尾项，或经完整不等式认证的可达椭球尾项，通过耗散反馈组合成全域全时间界。
   必须补齐输入驱动集合对反馈作用的遗漏控制、近零时间相对分母、非零稳态、以及指定 Galerkin ROM 的实际误差。
   不能靠训练集最坏值、层次 saturation 经验、H∞/Hankel 标签或低秩因子残差小来跳过它们。
3. **规模判决**：仅当小模型证明判决有实质通过时，到 47k/122k 检查保留维数、全阶 RHS、AMG setup、时间、内存与认证效果。
   不把扩大局部盒、提高插值阶数或缩减缓存开销本身称作数学创新。

潜在贡献的准确说法：**高秩、耗散 Robin 边界下，输入驱动的反馈可达压缩及其连续域逐时刻全场相对误差尾项**。
这个组合在本次核对的文献中没有直接完成；这不是证明文献中不存在其他先行工作，也不是声称我们已完成该定理。
现有联合余项与辅助系统仅保留为认证支撑，直到结构判决给出更具体的主线。

## 检索与证据范围

检索包含 parameter mapping / sampling-free low-rank parameterization / optimal port transfer spectrum /
parabolic local approximation / generalized Gramians / bilinear thermal PMOR / deterministic type-II BT /
holomorphic n-widths / Stieltjes rational Krylov / auxiliary error systems。
核心判断来自作者论文全文或出版社/机构摘要；只有摘要的工作已逐项标明。
2026 年 LFR/离散时间新工作的检索摘要也出现，但本次未取得全文，不据其宣称满足本项目连续时间契约。
网页抓取时间不作为出版时间。没有更改生产提取器或实验算法，没有运行/添加 playground 单元测试。

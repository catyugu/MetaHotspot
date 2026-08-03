# weekly_report_0720

## 一、代码改进

- 对架构进行一定调整和整合，整理原先零碎模块，做成明确的建模 - 编译 - 求解 - 后处理几步架构。
- 添加了 C 语言的 API，允许用代码进行建模和查询，便于后续利用 Python 等其他脚本语言开展脚本化建模和算法实验。

## 二、宏模型/降阶模型相关技术调研

### 静态凝聚类的方法

> S. Lan, M. Tang and J. Mao, Domain Decomposition and Reduction Method for Efficient Thermal Simulation and Design of 2.5D Heterogeneous Integration.
> Shunxiang Lan, Min Tang, Liang Chen, and Junfa Mao. Thermal Resistance Network Derivative (TREND) Model for Efficient Thermal Simulation and Design of ICs and Packages.

线性的 Neumann to Dirichlet 模型。类似于在界面处加一个额外面 DOF，然后做静态凝聚（Static Condensation），或者说动力学里所谓 Guyan 缩减。经典静态凝聚直接从离散矩阵构造 Schur 补；而这些文章中通过逐端口施加单位热流、测量接口温升，数值辨识出非核心区域的接口响应矩阵（文中所谓 T，其实就是舒尔补的逆）。但是无论如何，构建代价比较大（至少要 N 次求解）。而且最致命的问题是，这样产生的接口矩阵是稠密的，也就意味着接口单元数量如果较大，对求解器会比较讨厌。（显然我们会希望稠密子块不要太大）

优点：

- 无精度损失
- 加载便利
- 对线性问题，可自然扩展之使其可用于瞬态问题

缺点：

- 构建代价高
- 若宏模型内部有非线性，则会失效
- 不是 BCI

### （改进的）多点矩匹配算法 Multi-Point Moment Matching

这个方法是

> L. Codecasa, V. d'Alessandro, A. Magnani, N. Rinaldi and P. J. Zampardi, "Fast novel thermal analysis simulation tool for integrated circuits (FANTASTIC)"

采用的方法。原论文的手法是基于有限元，稍加改造自然也可用于 FVM。MPMM 利用的事实是：端口负载导致的动态热扩散模式通常是由少数几个主要模式决定。它们对应几个主要时间常数，因此只需要捕获主要热动态模式。文章中使用的是一种改进的 MPMM 方案。

MPMM 本身技术性的地方主要在于：

- 使用温升而非原始温度数值，削除零频模态的影响
- 使用了一套自适应的方法，根据相对误差容限估计需要的频点数量，并且能给出准确的频率采样点取值
- 频点都是正实数，系统矩阵良态，便于 CG 等手段求解

改进的地方：

- 基本 MPMM 方法使用整个系统的全局最小、最大广义特征值。改进方法对每个端口分别估计两个特征值，决定要算的频点。最后通过 SVD 消除不同端口和频点间的冗余
- 逐频点增量式构造基空间，映回去作为下一次全阶解的初始猜测，让之后的求解收敛更快
- 将频点按从大到小排序逐个计算，因为通常较大频率下求解更为容易（对角占优）

优点：

- 端口自由度少时，计算量小，压缩度高
- 甚至可以恢复出被降阶的模型内部的温度分布
- 天然适配瞬态场景
- 适当修改后，可以做到 BCI

> 可见于工作:
> L. Codecasa, A novel approach for generating boundary condition independent compact dynamic thermal networks of packages
> D. Lou, S. Weiland, Parametric model order reduction for large-scale and complex thermal systems

缺点：

- 预处理依然有点贵（需求解的次数=总频点数量，但是逐步构造子空间时，由于初猜比较优，迭代求解较快）
- 有损（不过损失可控）
- 需要对热源做拉氏变换，带来额外的计算代价和精度损失

### Affine Parametric ROM

可见于：

> P. Benner, S. Gugercin, K. Willcox, "A Survey of Projection-Based Model Reduction Methods for Parametric Dynamical Systems," SIAM Review, 2015.
> B. Peherstorfer, K. Willcox, M. Gunzburger, "Survey of Multifidelity Methods in Uncertainty Propagation, Inference, and Optimization," SIAM Review, 2018.

考虑线性热问题：

$$
C(\mu)\dot{x}+K(\mu)x=f(\mu)
$$

其中 $\mu$ 表示参数，例如：

- 对流系数 ($h$)
- 接触热阻 ($R_c$)
- 材料导热率 ($k$)
- 环境温度 ($T_\infty$)

Affine ROM 假设：

$$
\begin{align}
K(\mu)&=K_0+\sum_{i=1}^{p}\theta_i(\mu)K_i \\
C(\mu)&=C_0+\sum_i\theta_i(\mu)C_i
\end{align}
$$

这样离线预计算：

$$

K_{ri}=V^TK_iV

$$

在线只需直接计算：

$$

K_r(\mu) = K_{r0} - \sum_i\theta_i(\mu)K_{ri}

$$

例如对于对流换热系数 h 有（$f_h$ 指只在边界条件面上为 1 的 0-1 向量）：

$$
(K_0​+h K_h​)T=Q + hf_h​T_{\infty}​
$$

则 $\theta_0 = 1, \theta_1 = h$，这样我们只需要对 $K_0$ 和 $K_h$ 分别进行降阶存储，即可做到边界无关的方法，而不需要完整建模 BC 所在端口。

根据公开文档推测，flotherm 使用的基本上就是所谓 FANTASTIC 的增强版 MPMM 配合上 Affine Parametric ROM。比较简易的一个实现是在多个参数—频率组合点构造并合并相应子空间。进一步，则可能要用混合矩匹配，但其算法复杂度和计算代价都比较大，超出了现在可讨论的范畴。

## 三、遇到的问题

- 发现网格较不均匀（相邻单元尺度差别较大），或者网格单元不同尺度上比例比较大时，产生比较显著的离散误差。为此自适应层加密恐怕是必需的。
- 部分工况下用标准的 Green-Gauss 梯度恢复来做节点温度恢复，结果甚至不如反距离权重加权平均。
- Flotherm 使用尚有困难，建模逻辑和其他仿真软件不太相同且资料比较少。跟着教程做或者用模板建立的模型可以计算收敛，但是自行建立的简单模型常常无法收敛，而且无法精确知晓和控制其实际使用的控制方程和边界条件，还需要进一步熟悉。

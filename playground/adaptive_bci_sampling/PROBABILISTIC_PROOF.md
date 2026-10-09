# 参数残差算子的随机尾项认证

本阶段允许概率认证，但明确区分容差 `tau`、HTC 超限风险 `eta` 和证书失败概率 `delta`。
方法认证 SVD 前的固定 Galerkin 空间。生产 `utils.py` 不变。
当前实现的通过条件是**稳态** K-energy；不是全时间阶跃双认证。
数学定理使用精确算术，实验使用普通浮点和实际 AMG-CG 残差，尚无 outward rounding。

## 1. 两种概率命题

设连续盒为 D，抽样测度 nu 为有效 Robin 系数的独立 log-uniform 分布。
所需量词不能混淆：

1. 同时全域：`P_sketch{for every h in D, error(h) <= tau} >= 1-delta`。
2. 连续分布风险：`P_offline{nu{h:error(h)>tau} <= eta} >= 1-delta`。

随机参数验证直接得到第二种。置信度再高，也不能排除一个极小的坏参数区域。
第一种需要额外的连续最大值上界，或同时覆盖所有残差的算子不等式及其验收。
本文给出同时有效的随机**界函数**；将界函数在有限随机 HTC 留出集验收，得到第二种命题。
`--cover-cells` 尝试整个盒的确定性界函数包络，不能用其未完成的叶子宣称第一种验收。

## 2. 固定空间中的精确残差表示

令

$$K(h)=K_0+\sum_{i=1}^p h_iH_i,\qquad C\succ0,\qquad K_*:=K(h_{\min})\succ0,$$

其中 $H_i\succeq0$，所以 $K(h)\succeq K_*$。
固定空间 $V$ 的稳态解与残差为

$$y_j(h)=(V^TK(h)V)^{-1}V^Tg_j,\qquad
r_j(h)=g_j-K(h)Vy_j(h).$$

取任意正的缩放 $b_i,b_s$，定义

$$R_j=[g_j,-K_0V,-b_1H_1V,\ldots,-b_pH_pV,-b_sCV],$$

$$c_j(h)=[1,y_j(h)^T,(h_1/b_1)y_j(h)^T,\ldots,(h_p/b_p)y_j(h)^T,0]^T.$$

于是 $r_j(h)=R_jc_j(h)$ 对连续域上每个 h 精确成立。
最后的 CV 块允许同一恒等式表示动态残差，但不自动提供动态认证。
若 $C=\operatorname{diag}(c)$，作**不截断**的经济 QR：

$$C^{-1/2}R_j=Q_jT_j,\qquad Q_j^TQ_j=I.$$

当列数超过 n 时，$Q_j$ 为 n 阶正交矩阵，$T_j$ 为矩形矩阵。
不因训练快照的奇异值小而丢弃残差方向。

从训练 HTC 点形成

$$Z_j(:,\ell)=T_jc_j(h_\ell)/\sqrt{g_j^TVy_j(h_\ell)},\quad
S_j=Z_jZ_j^T/N_{\rm fit},$$

用完整正定正则化平方根 $W_j$ 作为系数度量。它只影响界的松紧，不能作为证据。
令 $z_j(h)=W_j^{-1}T_jc_j(h)$，则

$$r_j(h)=C^{1/2}Q_jW_jz_j(h).$$

每个输入的度量单独拟合，不用混合输入的平均响应掩盖困难输入。
候选提取可共享 V；训练验收逐输入计算规范化残差并取最大值。
任意有符号输入组合由后续矩阵不等式处理，不用逐列通过代替组合通过。

## 3. 保留误差方向，再认证未知尾项

定义固定算子

$$A_j=K_*^{-1/2}C^{1/2}Q_jW_j,\qquad
A=[A_1,\ldots,A_q].$$

先用独立的训练高斯系数进行 k 次 $K_*$ 求解，得到共同逆像空间 X。
使用实际计算的 X 构造 $E$，使 $E^TK_*E=I$。
**X 不需要是精确求解**；这里只需要一个固定的实际子空间。
令 $U=K_*^{1/2}E$，$P=UU^T$，待认证的尾项算子是

$$A_{\rm tail}=(I-P)[A_1,\ldots,A_q].$$

这是系数空间的块算子。各输入的 $W_j$ 保持分离；它不是一个混合输入的 POD。
一个联合尾项证书比 q 个分别认证的尾项只需更少的求解，且免除了未知尾项的额外 q 因子。

对冻结的 $A_{\rm tail}$ 抽 m 个**新的**标准高斯向量 $\omega_\ell$。
取

$$L=\sqrt{2/\pi}\,\delta_s^{-1/m}\max_{1\le\ell\le m}
\|A_{\rm tail}\omega_\ell\|_2.$$

则

$$\Pr\{\|A_{\rm tail}\|_2\le L\}\ge1-\delta_s.$$

**证明。** 固定最大右奇异向量 v。对任意标准高斯向量，
$\|A_{\rm tail}\omega\|\ge\|A_{\rm tail}\|\,|v^T\omega|$。
标准正态密度最大值为 $1/\sqrt{2\pi}$，故
$\Pr\{|v^T\omega|<a\}\le\sqrt{2/\pi}a$。
取 $a=\sqrt{\pi/2}\delta_s^{1/m}$，m 个独立探针均小于 a 的概率不超过 $\delta_s$。
该事件控制整个算子，而非若干 HTC 点；没有对不可数参数点作 union bound。证毕。

这就是 Halko–Martinsson–Tropp 的标准谱范数见证思想，不是新概率定理。
其价值是认证 RHS 数不随残差空间维数 d 增长。界可能因尾项有效秩变大而松弛，
绝不能由 RHS 数不依赖 d 推断其效用也不依赖 d。

### 3.1 不需要生成 $K_*^{-1/2}$

第 ell 个探针的物理右端为

$$b_\ell=(I-K_*EE^T)\sum_j C^{1/2}Q_jW_j\omega_{j,\ell}.$$

解 $K_*x_\ell=b_\ell$ 后，其精确能量给出
$\|A_{\rm tail}\omega_\ell\|=\sqrt{b_\ell^TK_*^{-1}b_\ell}$。
所有见证共用一个 AMG setup。
若只得到实际近似 $\widetilde x$，令 $d=b-K_*\widetilde x$。
经实际矩阵乘法验证的 $K_*\succeq\alpha C$ 给出

$$\sqrt{b^TK_*^{-1}b}
\le\sqrt{\widetilde x^TK_*\widetilde x}
+\|C^{-1/2}d\|/\sqrt\alpha.$$

实现采用这一上界，而不是把 CG 标称容差当作精确解。
M-matrix 情况下，用正向量 w 和实际 $K_*w$ 的 Collatz 商验证 alpha。
验证 alpha、QR 和正交化的浮点舍入仍需 interval arithmetic 才能形成机器层严格证书。

## 4. 任意有符号输入的稳态能量上界

记 $R(h)=[r_1(h),\ldots,r_q(h)]$，$p_j(h)=E^Tr_j(h)$，
$P_r(h)=[p_1(h),\ldots,p_q(h)]$。
在第 3 节的**单个联合事件**上，对任意输入 a 有

$$a^TR(h)^TK_*^{-1}R(h)a
\le a^T B(h)a,$$

$$B(h)=P_r(h)^TP_r(h)
+L^2\operatorname{diag}(\|z_1(h)\|^2,\ldots,\|z_q(h)\|^2).$$

**证明。** 已保留部分与未知尾项在 $K_*$ 能量中正交。保留部分的能量是
$\|P_r(h)a\|^2$，不能再用 q 倍逐列范数代替它。
未知尾项对应块系数 $[z_1a_1;\ldots;z_qa_q]$，其能量最多为
$L^2\sum_j\|z_j\|^2a_j^2$。证毕。

记完整稳态响应 X 和 ROM 响应 $X_V=V(V^TK(h)V)^{-1}V^TG$，误差为 F。
Galerkin 正交性给出

$$F^TK(h)F=R(h)^TK(h)^{-1}R(h)\preceq B(h),$$

$$X^TK(h)X=X_V^TK(h)X_V+F^TK(h)F.$$

设 $Q_V(h)=X_V^TK(h)X_V\succ0$，
$\lambda(h)=\lambda_{\max}(B(h),Q_V(h))$，则

$$\sup_{a\ne0}\frac{\|F(h)a\|_{K(h)}}{\|X(h)a\|_{K(h)}}
\le U(h):=\sqrt{\frac{\lambda(h)}{1+\lambda(h)}}.$$

这里的 U 是**整个连续盒上同时有效的随机界函数**，并且处理全部有符号输入。
在线只有小型仿射矩阵求解和系数矩阵运算；没有每个 HTC 点上的 FOM 求解。
若分别认证每个 $A_j$，须分配各自失败预算并对未知尾项加 q 倍对角包络。
实现保留 `--witness-mode separate` 作为失败对照。

## 5. 连续分布上的风险验收

冻结 V、各 $W_j$、E 和 L，再从独立流抽 N 个 HTC 点。
定义 $I(h)=1_{U(h)>\tau}$。若 N 点全部为零，则

$$N\ge\left\lceil\frac{\log\delta_v}{\log(1-\eta)}\right\rceil$$

保证以至少 $1-\delta_s-\delta_v$ 的概率，真实误差在 nu 下的超限概率最多为 eta。
**证明。** 条件于冻结的随机证书，若界函数的超限概率大于 eta，
N 次零失败概率最多为 $(1-\eta)^N\le\delta_v$。
在算子证书事件上，真实误差超限集包含于界函数超限集。两次失败预算相加即可。
有 k 个失败时用一侧 Clopper–Pearson 上界；不能因“多数通过”而宣称 eta 验收。

默认预算各为 delta/2；例如 eta=0.01、delta=1e-6 需要 N=1444。
置信度和 99% 参数覆盖是两回事。nu 是有效 Robin 参数的 log-uniform，
不是原物理 HTC 分布，更不是随时间变化的 HTC。

多次重试或按留出集调参需要新留出集和可求和预算，例如
$\delta_k=\delta/(k(k+1))$。探索阶段的各次成功不得合并成一次 delta 证书。
如选择最终配置后重新认证，必须使用此前未观察的见证和留出流。

## 6. 整个连续盒的可选稳态包络

在盒中心 h_c，记 $A_c=V^TK(h_c)V$，半径为 d_i，

$$\rho=\sum_i d_i\|A_c^{-1/2}(V^TH_iV)A_c^{-1/2}\|_2.$$

若 $\rho<1$，则
$\|A_c^{1/2}y_j(h)\|\le\|A_c^{1/2}y_j(h_c)\|/(1-\rho)$。
残差满足精确恒等式

$$r_j(h)=r_j(h_c)+\sum_i(h_i-h_{c,i})
[K(h_c)VA_c^{-1}(V^TH_iV)-H_iV]y_j(h).$$

将方括号映入已认证的系数度量即可用三角不等式上界整个盒中的残差。
ROM 响应能量有单调下界 $Q_V(h)\succeq Q_V(h_{\rm upper})$。
对不能通过的盒二分；所有叶子通过才得到同时全域稳态验收。
代码的 cell cap 明确返回 `passed=false` 和未完成盒数。
这个包络仍可能非常松，不是本阶段默认的概率验收路径。
`--correction` 变体没有实现该连续盒包络，禁止与 cover 同用。

## 7. 动态延伸的正确边界

零初值阶跃的 ROM 满足 $V^TCV\dot y+V^TK(h)Vy=V^TG$。
其动态残差

$$r(t,h)=G-K(h)Vy(t,h)-CV\dot y(t,h)$$

仍在第 2 节固定残差空间内。冻结的随机算子事件因此可同时用于所有 h 和所有 t。
但**稳态风险通过不蕴含阶跃风险通过**。
误差方程和 Young 不等式给出对每个输入 a：

$$\frac{d}{dt}\|e(t)a\|_C^2+\alpha\|e(t)a\|_C^2
\le a^TB(t,h)a,$$

$$e(t,h)^TCe(t,h)\preceq\int_0^t e^{-\alpha(t-s)}B(s,h)\,ds.$$

ROM 的 $y,\dot y$ 是有限维指数函数，右端可以在小系统中求积分。
不过残差强迫在初始时刻一般不为零，直接套该能量不等式会产生 $O(\sqrt t)$ 上界，
无法和真实 $O(t)$ 阶跃响应比较。必须结合

$$\|e(t)a\|_C\le\int_0^t e^{-\alpha(t-s)}\|C^{-1/2}r(s,h)a\|\,ds$$

处理 $t\downarrow0$，并认证完整时间轴上的相对误差及其真实响应分母。
有限时间审计、MPMM 的匹配点理论、Hankel 范数或稳态证书均不能跳过此步骤。
本阶段保留这个可证延伸，不把尚未完成的全部时间风险验收写成算法成功。

## 8. 算法与成本

1. 对每个输入沿用 stock 频率区间和 MPMM 移位计划；额外加入稳态 s=0。
2. 用含角点的训练池，对每个输入分别计算相对残差。共享 V 时取最大输入/参数/移位误差，
   只对被选中的输入求解并加入响应。训练池只是选择 V。
3. 以 QR 合并响应和常数方向，得到 SVD 前空间；不应用截断 SVD。
4. 分别拟合每个输入的残差系数度量；构造 k 阶共同逆像误差方向。
5. 用独立 m 个联合尾项见证，包含实际 CG 缺陷。一个 K_* setup。
6. 冻结所有对象，进行独立连续测度风险验收，或者完整盒包络验收。
7. 失败只得到拒绝结论；若改变 V 或度量，重新使用独立预算进行认证。

联合模式的认证 RHS 是 `1+k+m`，其中 1 是 Collatz 向量。
分别认证模式是 `1+k+q*m`。全部构造求解必须加回提取成本。
QR、系数度量、池搜索和 N 次 ROM 查询并不是免费操作。比 stock 少 RHS 不等于更快。
小模型独立谱范数 oracle 和 FOM 精度审计不属于提取成本，但分别报告。
正式模型不做稠密 FOM 谱分解，也不以 splu 代替 AMG-CG。

### 8.1 被拒绝的修正残差变体

`--correction` 在 E 中解实际 K(h) 的误差 Galerkin 方程，记 z 和
$\widetilde r=r-K(h)Ez$。有精确分解

$$r^TK(h)^{-1}r=(E^Tr)^T(E^TK(h)E)^{-1}(E^Tr)
+\widetilde r^TK(h)^{-1}\widetilde r.$$

其残差空间需增加 $K_0E,H_iE$；新的完整系数度量及独立尾项见证重新构造。
它不依赖 saturation 假设，但扩大的系数空间可能使随机界更差。
小模型实验已发现这一问题，故保留为显式 opt-in 判决，不作默认改进。

## 9. 先行工作与新颖性边界

- [Smetana–Zahm–Patera, 2019](https://arxiv.org/abs/1807.10489)：随机残差/对偶误差估计，
  指定概率下接近 1 的 effectivity 和很多参数查询的控制；不能把有限查询集变成整个连续域。
- [Balabanov–Nouy, Part I](https://arxiv.org/abs/1803.02602)：仿射残差空间的随机嵌入、
  Galerkin 和误差估计。这直接排除了“发现残差在有限维空间”作为创新的说法。
- [Balabanov–Nouy, Part II](https://arxiv.org/abs/1910.14378)：最小残差、字典空间、
  sketch 的后验认证；必须比较其方法，而不是只比较几个工程实现。
- [Balabanov–Nouy, preconditioners](https://arxiv.org/abs/2104.12177)：参数逆算子的插值和随机算子认证；
  本阶段的逆像误差空间与此紧密相关。
- [Halko–Martinsson–Tropp, 2011](https://arxiv.org/abs/0909.4061)：谱范数见证、随机 range finder，
  上文维数无关的 m 次认证来自这个标准工具。
- [Alamo–Tempo–Luque–Ramirez, 2015](https://arxiv.org/abs/1304.0678)：随机分析、
  Sequential Probabilistic Validation 和样本复杂度。独立风险验证不是新理论。
- [Lou–Weiland thermal pMOR](https://arxiv.org/abs/1803.05240)：热模型参数化 Krylov 与误差分析，
  是物理模型对照之一。

可继续研究的组合是：逐输入度量、共同逆像误差方向、一次联合未知尾项算子证书、
不混淆连续分布风险和同时全域保证的完整提取验收流程。
这目前只是可验证的候选方法，尚不足以宣称学术创新或更快、更鲁棒的完整动态提取器。

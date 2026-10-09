> 历史归档：保留当时的目标、结果与局限；“最新”等措辞仅指当时状态。当前验收和研究入口见 [当前 README](../../README.md)。

# 正交有理时间展开的连续参数单元界

日期：2026-10-07。基于 `agent/work` 的 `d7e229b`。

本文件给出**精确算术**下的构造与证明。`rational_dynamic.py` 是浮点原型，
没有区间包围；所有输出保留 `floating_point_certified=False`。
这是一种固定 Galerkin 基的动态验收方法，不是已经优于 stock 的提取器。

本文保留历史冲激积分目标的完整推导。最新用户验收为稳态 `2 epsilon`
与全时刻阶跃 `2 sqrt(epsilon)`，见 [STEADY_STEP_PROOF.md](STEADY_STEP_PROOF.md)。

## 1. 假设与目标

沿用 THEORY.md：

\[
K(h)=K_0+\sum_{i=1}^d h_iH_i,\quad H_i\succeq0,\quad
C\succ0,\quad K(a)\succ0,\quad h\in[a,b].
\]

矩阵实对称，`V` 满列秩，源矩阵 `G` 输入列独立。写
`Cr=V^T C V`、`Kr(h)=V^T K(h)V`、`Gr=V^T G`。
目标为全场、全输入组合的零初值 impulse 相对误差，积分区间 `[0,infinity)`：

\[
\epsilon_{imp}(h;V)^2=\lambda_{max}(J(h),Q(h)),\qquad
Q(h)=\tfrac12G^TK(h)^{-1}G.
\]

不保证节点最大误差、任意时变输入的相对误差、时变 HTC、非对称流热算子。
有效仿射系数的盒可以包围物理 HTC 的单调映射，但不覆盖未纳入模型的物理误差。

## 2. 时间系数与精确尾项：无需完整 Lyapunov 方程

任取同一列正实移 `sigma_0,...,sigma_{N-1}`，每个参数点定义

\[
q_0=G,\quad x_k=\sqrt{2\sigma_k}(K(h)+\sigma_k C)^{-1}q_k,
\quad q_{k+1}=q_k-\sqrt{2\sigma_k}Cx_k.
\]

对约化系统完全相同地定义 `qr_0=Gr`、`yr_k`、`qr_{k+1}`，使用 `Kr,Cr`。
全场系数缺陷是 `e_k=x_k-V yr_k`。这对应实 Takenaka–Malmquist 时间函数
（重复 shift 时退化为 Laguerre 系统）的系数；不需要依赖这个名称来证明下列恒等式。

质量白化后设 `B=C^{-1/2}K C^{-1/2}`、`g=C^{-1/2}G`，并令

\[
r_N(z)=\prod_{k=0}^{N-1}\frac{z-\sigma_k}{z+\sigma_k}.
\]

标量 Cauchy 核有**精确**递推：

\[
\frac1{x+y}=\frac{2\sigma}{(x+\sigma)(y+\sigma)}
 +\frac{(x-\sigma)(y-\sigma)}{(x+\sigma)(y+\sigma)}\frac1{x+y}.
\]

逐步展开，代入全阶与约化的谱分解，可同时计算 full/full、reduced/reduced 和
full/reduced 三种时间积分。于是

\[
J(h)=\sum_{k=0}^{N-1}e_k^TCe_k+J_N(h),\qquad J_N(h)\succeq0.
\]

`J_N` 恰为以下两条剩余轨迹之差的时间 Gram：全阶轨迹以 `C^{-1}q_N` 为初值，
约化轨迹以 `Cr^{-1}qr_N` 为初值并通过 `V` 提升。这也证明有限系数前缀给出真实
`J` 的 Loewner 下界。全阶与约化的尾能量分别是

\[
Q_N=\tfrac12q_N^TK(h)^{-1}q_N,\qquad
Qr_N=\tfrac12qr_N^TKr(h)^{-1}qr_N.
\]

后式用的是 `Kr^{-1}`，不是 `Cr^{-1}`。两条尾轨迹的 Hilbert 空间三角不等式
给出尾误差范数不超过两条尾能量范数之和。没有截断时间、人工频段或未计入的有理骨架误差。

## 3. 用多项式试验函数包围真正的系数

取任意全阶矩阵多项式 `xhat_k(h)` 和约化矩阵多项式 `yhat_k(h)`。原型通过参数
Chebyshev–Lobatto 节点上的直接系数响应插值产生它们；插值节点本身不承担认证。

从 `qhat_0=G` 按 `qhat_{k+1}=qhat_k-sqrt(2 sigma_k) C xhat_k` 更新，定义

\[
f_k=\widehat q_k-(K(h)+\sigma_kC)\widehat x_k/\sqrt{2\sigma_k}.
\]

约化侧同样定义 `fr_k`。全阶量都是双重（dual）坐标的 RHS，而非未经说明的白化状态。
由多项式精确乘法计算 `f_k`：若试验每轴次数 `p`，残差以每轴次数 `p+1` 表示，
不可丢弃乘法产生的高阶系数。

单元统一分母取

\[
D=\tfrac12G^TK(b)^{-1}G\succ0.
\]

因为 `K(h)<=K(b)`，`Q(h)>=D`。原型对 `D` 做输入空间白化，即以下矩阵范数
都作用于 `D^{-1/2}` 变换后的输入。全输入组合仍然保留，不是逐源列最大值。
也可以采用 `D=Gr^T Kr(b)^{-1}Gr/2`，但它较松，且下面的一致性结论必须改成
相对于该下界分母有严格余量。**原型默认使用全阶精确分母，计入 m 次 RHS。**

写归一化残差的 Bernstein 表示

\[
f_k(h)D^{-1/2}=\sum_\nu B_\nu(h)F_{k,\nu},\quad
\eta_k=\max_\nu\sqrt{\lambda_{max}(F_{k,\nu}^TK(a)^{-1}F_{k,\nu})}.
\]

由 Bernstein 非负分拆、范数凸性及逆的 Loewner 序，
`||f_k D^{-1/2}||_{K(h)^{-1},op}<=eta_k`。约化侧用 `Kr(a)^{-1}` 得到 `etar_k`。
控制矩阵不是节点值，也不是对矩阵逐元取最大。

## 4. 没有全局谱下界的递推缺陷界

令 `delta q_k=q_k-qhat_k`，初始为零。直接减去两个递推得

\[
\delta x_k=\sqrt{2\sigma_k}(K+\sigma_kC)^{-1}(\delta q_k+f_k),
\]
\[
\delta q_{k+1}=
[I-2\sigma_kC(K+\sigma_k C)^{-1}]\delta q_k
-2\sigma_kC(K+\sigma_k C)^{-1}f_k.
\]

在 `K^{-1}` dual 范数下，白化谱计算给出：

\[
\|I-2\sigma C(K+\sigma C)^{-1}\|_{K^{-1}\to K^{-1}}\le1,
\]
\[
\|2\sigma C(K+\sigma C)^{-1}\|_{K^{-1}\to K^{-1}}\le2,
\]
\[
\|\sqrt{2\sigma}(K+\sigma C)^{-1}\|_{K^{-1}\to C}
=\max_{\lambda\in spec(B)}\frac{\sqrt{2\sigma\lambda}}{\lambda+\sigma}
\le1/\sqrt2.
\]

因此 `u_0=0`、`u_{k+1}=u_k+2 eta_k` 可包围 `delta q_k`，
`||delta x_k D^{-1/2}||_{C,op}<=(u_k+eta_k)/sqrt2`。
约化侧有相同的 `ur_k`。这三个常数不需要计算 `lambda_min(B)`，没有
`alpha^{-3}` 的换算链。但残差方向仍要通过 `K(a)^{-1}` 度量，**并非误差界与
病态性无关**；参数盒很宽时这个锚点度量也会松。

若另有合法统一谱区间 `[alpha,beta]`，可以进一步用

\[
\rho_k=\max\{|(\alpha-\sigma_k)/(\alpha+\sigma_k)|,
                  |(\beta-\sigma_k)/(\beta+\sigma_k)|\},
\quad d_k=2\sigma_k/(\alpha+\sigma_k),
\]

和 `c_k=max_[alpha,beta] sqrt(2 sigma_k lambda)/(lambda+sigma_k)`，改成
`u_{k+1}=rho_k u_k+d_k eta_k`，系数缺陷界改成 `c_k(u_k+eta_k)`。
Galerkin Ritz 谱位于全阶谱区间内，约化侧也可使用这组常数。

原型仅在 `C` 对角正、`K(a),K(b)` 为对称 Z 矩阵的条件下自动构造区间：
任选正向量 `v`，Collatz–Wielandt 给出
`alpha=min_i (K(a)v)_i/(C_ii v_i)`；取 `v=K(a)^{-1}C 1`，计 1 次 RHS。
**实际用的是重新计算的 `K(a)v`，不假设数值解精确。**
上界取 `beta=max_i sum_j |K(b)_ij|/C_ii`。
适用条件不满足就退回 `rho=1,d=2,c=1/sqrt2`，不使用未认证 Ritz 下界。
谱辅助只改善常数；基本接受定理不依赖它。

## 5. 连续单元接受定理

将试验系数缺陷 `ehat_k=xhat_k-V yhat_k` 提升到同一 Bernstein 次数，控制矩阵记
`E_{k,nu}`（同样按 `D` 归一化），定义

\[
M=\max_\nu\sqrt{\lambda_{max}(\sum_kE_{k,\nu}^TCE_{k,\nu})},
\quad A=\sqrt{\sum_k[c_k(u_k+\eta_k+ur_k+\eta r_k)]^2}.
\]

`M` 包围试验缺陷的**整段系数栈**范数：先堆叠所有 k，再用 Bernstein 范数凸性。
不能仅看某个 k 或某个输入列。`A` 来自真正与试验系数栈之间的三角不等式。

对最后的试验 dual RHS 以相同方法定义 `tn`、`trn`：
`tn>=||qhat_N D^{-1/2}||_{K(h)^{-1},op}`，约化侧同理。
精确尾项能量及第 4 节递推缺陷给出

\[
\boxed{\ \sup_{h\in[a,b]}\epsilon_{imp}(h;V)
\le M+A+\frac{tn+u_N+trn+ur_N}{\sqrt2}.\ }
\]

右端 `<=tau` 才接受该连续单元。四项分别是主项、递推余量、全阶尾项、约化尾项。
它涵盖全部输入组合、全部状态及完整时间轴。数值验证点不是证明中的前提。

退化为单点、使用精确 `D=Q(h)` 时，还有
`epsilon_imp(h)>=max(0,M-A)`，因为真实有限前缀 Gram 不超过 `J`。
这可用于区分“证书太松”和“固定基真的不满足目标”。

## 6. 精确算术下的有限认证存在性与公平搜索

假设整个紧参数盒上真实误差有严格余量：`sup epsilon_imp<tau`。
重复任意一个有限正 shift 周期时，因为全阶与约化的谱统一处于某个紧正区间内，
`max_spec |r_N|` 一致趋零。因此两条真实尾项一致趋零。这里谱区间只用于存在性证明，
不要求算法知道它。

固定一个足够长的 N，系数响应是参数的光滑函数。将各单元缩小，取其中心常量
系数试验（`p=0` 已足够），残差 Bernstein 控制趋零；有限步递推余量也趋零。
主项 Bernstein 上界与真实前缀范数之差趋零；`D=Q(b)` 与参数点真实分母之差
也趋零。因此存在有限的时间项数和有限单元细分深度，使全部单元通过。

一个可证明有限终止的搜索方式是公平枚举“时间周期数、空间细分深度”这两个整数对，
每个有限对最终都得到检查；一旦某个覆盖全盒的有限分割通过就停止。
**这没有证明任意混合升阶/加 N/细分策略都会终止。** 尤其 `N` 增大同时加深余量
递推，而误差界不必单调。当前 runner 只实现固定设置的判决实验和有限覆盖，不实现
高效的公平搜索提取器，也不宣称找到了采样最优方法。

## 7. 数值边界与成本

全阶多项式试验每个节点需 `N*m` 次 RHS；原型还有每个时间项、每个残差控制、
每个输入列的 `K(a)^{-1}` 作用，以及尾项、精确分母和可选 Collatz 向量。
全阶 trial/Riesz 使用稀疏 LU，与 stock AMG-CG 提取墙钟不作同后端比较。
RHS、稀疏分解数分别记录；约化计算不算全阶 RHS，但其成本仍存在。
独立 dense eigenmode 参考是 n 阶谱分解，不伪装成免费验证或折算为少数 RHS。

浮点原型没有证明线性解、Cholesky、Gram、插值变换、特征值与 Collatz 商的舍入包围。
Bernstein 在高阶下的数值条件也要考虑。独立测试支持实现正确性，不能替代这些数值证明。
原型成功只表示解析连续盒构造在浮点下通过，不能写成完全机器验证的证书。

## 8. 与已有工作的关系

- Mi, Qian, Wan (2012), *A fast adaptive model reduction method based on
  Takenaka–Malmquist systems*, Systems & Control Letters 61:223–230。
  https://www.sciencedirect.com/science/article/pii/S0167691111002817
  正交有理时间基不是本项目的创新；本次取得出版社检索摘要与作者 PDF 检索片段，
  全文下载被阻，故没有据其全文宣称某个具体公式或与本文定理等价。
- Wolf, Panzer, *The ADI iteration for Lyapunov equations implicitly performs
  H2 pseudo-optimal model order reduction*, arXiv:1309.3985v2。
  https://arxiv.org/html/1309.3985v2
  本次读取全文。ADI、rational Krylov、低秩 Lyapunov 残差的关系已有理论；本文系数
  递推与这些构件密切相关，不应以“新有理递推”作为论文贡献。
- Feng, Chellappa, Benner (2024), *A posteriori error estimation for model
  order reduction of parametric systems*, DOI 10.1186/s40323-024-00260-8。
  https://link.springer.com/article/10.1186/s40323-024-00260-8
  本次读取全文。文献区分 rigorously bounded quantities 与更紧但不严格的 estimators，
  说明离散矩阵层面与任意降阶构造结合的研究背景。
- Feng, Antoulas, Benner (2017), *Some a posteriori error bounds for
  reduced-order modelling of (non-)parametrized linear systems*。
  https://www.numdam.org/item/10.1051/m2an/2017014.pdf
  已读取正文；其 transfer-function/output 指标不能直接替换本项目 all-input
  relative full-state impulse 指标。

本次贡献限于：适配当前目标的正系数前缀/精确尾项恒等式，以及 affine HTC 的
Bernstein dual-residual 连续单元包围。是否构成可发表新颖性、是否比现有证书便宜，
尚未成立。实验结果见 records/RATIONAL_DYNAMIC_EXPERIMENT_20261007.md。

## 9. 第二个判决原型：耦合缺陷递推（实验更差）

`propagation_model=coupled` 尝试保留 full/reduced 残差抵消。设
`L=CV Cr^{-1}`、`R(h)=K(h)V-L Kr(h)`、`d_k=q_k-L qr_k`。
精确系数缺陷满足

\[
e_k=(K+\sigma_k C)^{-1}(\sqrt{2\sigma_k}d_k-R(h)yr_k),
\quad d_{k+1}=d_k-\sqrt{2\sigma_k}Ce_k.
\]

多项式试验的耦合残差正好是 `fE_k=f_k-L fr_k`。
两条递推相减后，误差 forcing 还包含 `-R(h)(yr_k-yhat_k)/sqrt(2 sigma_k)`。
这个项不能因为 `fE_k` 较小而省略。

原型用全单元上界

\[
\kappa_R=\max_{corners}\|K(a)^{-1/2}R(h)Cr^{-1/2}\|,
\quad a^r_k=c_k(ur_k+\eta r_k),
\quad \eta^E_k=\max_{controls}\|fE_{k,\nu}\|_{K(a)^{-1},op}.
\]

`R(h)` 仿射，故角点矩阵范数界合法。更新
`z_k=etaE_k+kappa_R ar_k/sqrt(2 sigma_k)`，
`uE_{k+1}=rho_k uE_k+d_k z_k`，系数缺陷余量为 `c_k(uE_k+z_k)`。
这里更新系数 `d_k` 与 dual RHS 的命名在公式中重名；实现使用 `feedback`，没有覆盖状态。

全阶尾源误差满足 `delta q_N=delta d_N+L delta qr_N`，用

\[
\kappa_L=\|K(a)^{-1/2}L Kr(b)^{1/2}\|
\]

得到 `||delta q_N||_{K(h)^{-1},op}<=uE_N+kappa_L ur_N`。
这是因为 `K(h)^{-1}<=K(a)^{-1}` 且 `Kr(h)^{-1}>=Kr(b)^{-1}`。
因此第 5 节接受界仍成立，只需替换系数递推余量和全阶尾项的 tube。

这证明该实现的上界是合法的；**没有证明标量化的 kappa_R 补偿够紧**。
实测 Case1 根盒 p=3 约 `1.02e9`，比独立递推的 `2.20e5` 更差。
它再次显示：先把约化误差方向变成标量，再乘耦合算子全范数，会严重丢失结构。
目前只保留为有证明、可复现的负结果，不作为默认路线。

## 10. 优化：联合前缀与尾项的创新能量界

`propagation_model=energy` 不把每步 dual tube 重复累加到后续每个系数。
固定一个参数值并在广义特征坐标中，令 `lambda` 为一个正特征率，令
`z_k`、`f_k` 分别是以 `K^{-1}` 度量归一化的源误差和试验残差。
第 4 节的强迫递推逐模态变成

\[
 z_{k+1}=t_k z_k-b_k f_k,\qquad
 \delta x_k=g_k(z_k+f_k),\quad z_0=0,
\]
\[
 t_k=\frac{\lambda-\sigma_k}{\lambda+\sigma_k},\quad
 b_k=\frac{2\sigma_k}{\lambda+\sigma_k},\quad
 g_k=\frac{\sqrt{2\sigma_k\lambda}}{\lambda+\sigma_k}.
\]

输出取**全部系数缺陷与末端能量的联合向量**
`(delta x_0,...,delta x_{N-1},z_N/sqrt(2))`。
设此映射的列 Gram 为 `A(lambda)`，则精确地

\[
 A_{jj}=b_j,\qquad
 A_{jk}=-\frac{b_jb_k}{2}\prod_{\ell=j+1}^{k-1}t_\ell
 \quad(j<k).
\]

证明：一次创新在第 j 步输出 `g_j f`，并留下源 `-b_j f`。
后续齐次系数平方和加末端能量，按 Parseval 恰为 `b_j^2 f^2/2`；
因此列平方和 `g_j^2+b_j^2/2=b_j`。两列在第 k 步开始重叠。
前一列到该步的源为 `-b_j prod(t_{j+1:k-1}) f`，记为 `z`。
两列从此步开始的交叉内积等于
`(g_k^2-t_k*b_k/2) z*f_k=(b_k/2) z*f_k`，给出非对角项。
该恒等式与后续截断长度无关，原因正是末端能量仍然保留。
测试用显式强迫递推的完整映射核验 Gram，包括正负 Cayley 因子。

若全单元谱位于 `[alpha,beta]`，令

\[
 \bar b_k=\frac{2\sigma_k}{\alpha+\sigma_k},\qquad
 \rho_k=\max_{\lambda\in\{\alpha,\beta\}}
 \left|\frac{\lambda-\sigma_k}{\lambda+\sigma_k}\right|.
\]

对每步残差的全输入算子范数上界 `eta_k`，函数演算和 Cauchy–Schwarz 给出

\[
 \mathcal I(\eta)^2\le
 \sum_j\bar b_j\eta_j^2+
 \sum_{j<k}\bar b_j\bar b_k\eta_j\eta_k
 \prod_{\ell=j+1}^{k-1}\rho_\ell.
\]

这里右侧定义一个合法的联合能量余量上界；实现还取其与
`sum_j sqrt(bar b_j)*eta_j` 的最小值，后者是逐创新输出的三角界。
没有谱区间时可用 `bar b=2,rho=1`，仍合法。
注意：实际 Gram 保留符号，**当前包围使用绝对值谱上界，尚未保留完整空间方向**。
不能把这个改进称为解决了第 9 节的方向损失问题。

为连接到动态误差，构造试验响应：多项式前缀加上试验末端源的精确尾响应。
第 2 节正交展开保证其误差范数恰是上述联合向量的范数；没有遗漏时间尾项。
全阶与约化响应的试验差可由 `M+T_F_trial+T_R_trial` 包围，故

\[
 B_{energy}=M+T_{F,trial}+T_{R,trial}
             +\mathcal I(\eta^F)+\mathcal I(\eta^R)
\]

是连续单元、全部输入、无限时间的合法界。约化谱属于全谱区间，故同一谱界可用于两者。
实现同时保留独立递推界，选择二者的最小值，保证不因切换模式而变松。
`propagation` 在 energy 被选中时是联合创新余量，`full_tail/reduced_tail` 只含试验尾项，
不要再把旧递推的末端 tube 加一遍。

## 11. 优化：参数包围细分与局部输入度量

`envelope_depth=d` 将每个参数轴均匀二分 d 次，只限制**已经构造好的多项式试验**。
de Casteljau 算法精确地给出各子盒的 Bernstein 控制向量；它不是新的插值或采样验收。
对子盒 `S=[a_S,b_S]`：

1. 子盒上的残差、多项式前缀和末端源均使用其精确限制；因此原有证明逐盒成立。
2. 仍使用原盒下角 `K(a)^{-1}` 的 Riesz 度量和原盒谱区间。因为
   `K(h)>=K(a)`，这个度量在每个子盒仍有效。
3. 分母改用 `D_S=Q(b_S)`，其对所有 `h in S` 都满足 `D_S<=Q(h)`。
   reduced 分母模式则使用 `0.5 Gr^T Kr(b_S)^{-1} Gr`，仍为合法下界。
4. 设原分母 `D=L L^T`、子盒分母 `D_S=L_S L_S^T`。
   原控制矩阵右乘 `S_input=L^T L_S^{-T}` 即得到子盒输入白化后的控制矩阵。
   因为 `G L^{-T} S_input=G L_S^{-T}`，所有输入组合仍被正确认证。

逆作用可随限制线性传播：若原控制 `F_nu` 已有 `A_nu=K(a)^{-1}F_nu`，
de Casteljau 对 `F`、`A` 作同样线性组合，则 `A_restricted=K(a)^{-1}F_restricted`。
白化右乘也与左侧逆作用交换。因此每个细分控制的 dual Gram
`F_restricted^T A_restricted` 是精确的；**不需要新增 Riesz 或时间系数求解**。
本实现保留向量及其逆作用，未按数值秩截断，也未遗漏压缩残差。
新增 full RHS 仅用于子盒分母，每个非原上角的子盒上角需要 m 个 RHS，
全域均匀细分需要 `m*(2^(d*parameter_dimension)-1)` 个。

计算各子盒 `B_energy(S)` 后取最大值，即为原盒的合法覆盖界；再与未细分界取最小值。
这个优化收紧多项式凸包和单调分母造成的损失，未改变固定 ROM 基、trial 节点或模型。
主项、余量、尾项报告为达到最大合成界的同一个子盒的分量。
原盒 residual maxima、`energy_bound` 和 `independent_bound` 另留作比较。
浮点逆作用、Cholesky、de Casteljau 和 Gram 都仍没有向外舍入；严格浮点认证仍未完成。

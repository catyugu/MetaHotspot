> 历史归档：保留当时的目标、结果与局限；“最新”等措辞仅指当时状态。当前验收和研究入口见 [当前 README](../../README.md)。

# 稳态 2 ε 与全时刻阶跃 2√ε：定义、构造及证明

本文对应最新用户验收，实现为 `steady_step_audit.py`。所有定理均为精确算术结论；
实际 inverse actions、稠密谱分解和 Gram 运算尚无外向舍入包围。
成本与判决见 实验记录（本地记录：`records/STEADY_STEP_COST_20261007.md`）。

## 1. 模型、输入空间与范数

设实对称矩阵满足

\[
K(h)=K_0+\sum_{i=1}^d h_iH_i,\quad H_i\succeq0,
\quad C\succ0,\quad K(a)\succ0,\quad h\in[a,b].
\]

`V` 为固定满列秩最终基；输入矩阵 `G` 列独立。输入列冗余时须先对 `ker(G)`
取商，不能随意给分母加 regularization。定义 `K_V=V^T K V`、`C_V=V^T C V`、`G_V=V^T G`。
全部场都是相对环境温度的温升，初值为零。稳态和阶跃为

\[
X=K(h)^{-1}G,\quad X_V=V K_V^{-1}G_V,
\]
\[
S(t)=(I-e^{-C^{-1}K(h)t})X,\quad
S_V(t)=V(I-e^{-C_V^{-1}K_Vt})K_V^{-1}G_V.
\]

验收对象是所有固定输入组合 `w` 到整个温度场的映射：

\[
e_s(h)=\sup_{w\ne0}\frac{\|(X-X_V)w\|_{K(h)}}{\|Xw\|_{K(h)}},
\qquad
e_t(h,t)=\sup_{w\ne0}\frac{\|(S(t)-S_V(t))w\|_C}{\|S(t)w\|_C}.
\]

要求 `sup_h e_s<=2ε` 与 `sup_{h,t>0}e_t<=2√ε`，平方阈值分别 `4ε²,4ε`。
`t=0` 两场均为零，以右极限解释相对误差。不是逐节点峰值、任意时变功率、时变 HTC，
也不是标准 H2 trace 指标。冲激 L2 积分相对误差并不自动提供上述点时刻相对误差界。

## 2. 稳态 Galerkin 恒等式与任意试验的上界

写 `E=X-X_V`、`Q=G^T K^{-1}G=X^T K X`。Galerkin 正交性 `V^T K E=0` 给出

\[
E^T K E=G^T E=G^T X-G^T X_V\succeq0,
\qquad e_s^2=\lambda_{\max}(E^T K E,Q).
\]

对任何矩阵试验 `q(h)`，令 `R_q=G-K(h)Vq(h)`。任意 `w` 的 Galerkin 投影最优性给出

\[
\|Ew\|_K^2\le\|(X-Vq)w\|_K^2=w^T R_q^T K(h)^{-1}R_qw.
\]

因此矩阵序 `E^T K E ≼ R_q^T K(h)^{-1}R_q` 成立，不需要残差逐元非负。

## 3. 连续 HTC 稳态证书

在各轴两个端点求解小型约化稳态，构成张量线性插值多项式 `q(h)`。
它只用约化求解；正当性不要求节点上已经满足误差阈值。
由于 `K(h)` 仿射，残差每轴次数至多 2，可精确写为

\[
R_q(h)=\sum_\nu B_\nu(h)R_\nu,
\quad B_\nu(h)\ge0,\quad\sum_\nu B_\nu(h)=1.
\]

这里 `R_ν` 是多项式 Bernstein 控制矩阵，不是插值节点残差。由 `K(h)≽K(a)`，
对任意输入 `w`，逆矩阵序与平方范数凸性依次给出

\[
w^TE^TKEw\le w^TR_q^TK(a)^{-1}R_qw
\le\sum_\nu B_\nu(h)w^TR_\nu^TK(a)^{-1}R_\nu w.
\]

令

\[
D=G_V^T K_V(b)^{-1}G_V.
\]

由完整 SPD Galerkin 逆作用的投影恒等式，`G^T K(b)^{-1}G≽D`；又逆的单调性给出
`Q(h)≽Q(b)≽D`。如果 `D` 不正定，拒绝这个相对界或改用合法的完整分母，不填补其核空间。
当 `D≻0` 时，设

\[
B_s^2=\max_\nu\lambda_{\max}(R_\nu^TK(a)^{-1}R_\nu,D).
\]

每项控制 Gram 均不大于 `B_s² D`，故其凸组合不大于 `B_s² Q(h)`，证明

\[
\boxed{\sup_{h\in[a,b]}e_s(h)\le B_s.}
\]

二维、四输入、每轴二阶残差需要 `3²×4=36` 个 Riesz RHS；四个 trial 节点无需完整求解。
这是已有 Riesz–Bernstein 构造针对新目标的使用，不声称新的发表级定理。

## 4. 固定 HTC 的谱坐标表示

以下固定 `h`。取完整及约化广义谱

\[
K U=C U\Lambda,\quad U^T C U=I,
\qquad K_V U_r=C_V U_r\Lambda_r,\quad U_r^TC_V U_r=I.
\]

设 `W=V U_r`、`b=U^TG`、`b_r=W^TG`、`D_c=U^TCW`，于是 `D_c^TD_c=I`。
以完整谱坐标表示完整和约化阶跃：

\[
F(t)=\operatorname{diag}(f_{\lambda_j}(t))b,
\quad R(t)=D_c\operatorname{diag}(f_{r_j}(t))b_r,
\quad f_\lambda(t)=\frac{1-e^{-\lambda t}}\lambda.
\]

因为 `U` 是完整 C-正交基，实际相对 C-energy 误差等于

\[
e_t(t)^2=\lambda_{\max}((F-R)^T(F-R),Q_t(t)),
\quad Q_t(t)=F(t)^TF(t).
\]

对 `t>0`，`f_λ(t)>0`，输入独立性给出 `Q_t(t)≻0`。每个 `f_λ(t)` 随 t 单调增加，
故 `Q_t(t)` 随 t Loewner 单调增加。这只用于完整分母，不要求误差自身随时间单调。

## 5. 中间时间区间的导数包围

在 `0<a<b` 上，设 `m=(a+b)/2`，选择 `Z` 满足 `Z^T Q_t(a) Z=I`。
对任意 `t∈[a,b]`，分母序 `Q_t(t)≽Q_t(a)` 给出

\[
e_t(t)\le\|(F(t)-R(t))Z\|_2.
\]

导数为

\[
F'(t)=\operatorname{diag}(e^{-\lambda t})b,
\quad R'(t)=D_c\operatorname{diag}(e^{-r t})b_r.
\]

`D_c` 列正交，且各指数衰减，因此整个区间都有

\[
\|(F'-R')Z\|_2\le
\|\operatorname{diag}(e^{-\lambda a})bZ\|_F+
\|\operatorname{diag}(e^{-r a})b_rZ\|_F=L_{a,b}.
\]

由积分基本定理、谱范数三角不等式，得到

\[
\boxed{\sup_{t\in[a,b]}e_t(t)\le
\|(F(m)-R(m))Z\|_2+\frac{b-a}{2}L_{a,b}.}
\]

这是两个端点之间全部时间的上界。中点真误差可作为拒绝见证，但仅当上式通过才接受区间。

## 6. 零时刻邻域的包围

用 `H(t)=F(t)/t` 和 `H_r(t)=R(t)/t` 连续延拓至零。由

\[
f_\lambda(t)/t=\int_0^1 e^{-\lambda st}\,ds
\]

得 `H(0)=b`、`H_r(0)=D_c b_r`。因完整对角因子随 t 递减，
`H(t)^T H(t)≽H(t_0)^T H(t_0)` 对 `0≤t≤t0` 成立。取 `Z0` 白化后者。
又

\[
\left|\frac{d}{dt}(f_\lambda(t)/t)\right|
\le\int_0^1\lambda s\,ds=\lambda/2,
\]

故得到

\[
\boxed{\sup_{0\le t\le t_0} e_t(t)\le
\|(b-D_cb_r)Z_0\|_2+
\frac{t_0}{2}(\|\Lambda b Z_0\|_F+\|\Lambda_r b_r Z_0\|_F).}
\]

误差右极限为 `||(b-D_c b_r)Z_b||2`，其中 `Z_b` 白化 `b^Tb`。
若该真实极限超过阈值，可直接拒绝；实现没有在 `t=0` 计算 0/0。

## 7. 无穷时间尾部的包围

令 `F∞=Λ⁻¹ b`、`R∞=D_c Λ_r⁻¹ b_r`。在 `t≥T` 上以 `ZT` 白化 `Q_t(T)`。
完整与约化指数尾分别有

\[
\|(F(t)-F_\infty)Z_T\|_2\le e^{-\lambda_{min}T}\|F_\infty Z_T\|_F,
\]
\[
\|(R(t)-R_\infty)Z_T\|_2\le e^{-r_{min}T}\|\Lambda_r^{-1}b_rZ_T\|_F.
\]

再次使用分母单调性及三角不等式，得到整个 `[T,∞)` 的上界

\[
\boxed{B_\infty=\|(F_\infty-R_\infty)Z_T\|_2+
e^{-\lambda_{min}T}\|F_\infty Z_T\|_F+
e^{-r_{min}T}\|\Lambda_r^{-1}b_rZ_T\|_F.}
\]

注意这里的极限是 C-energy 稳态误差，与第 2 节 K-energy 稳态目标不同，不能互相代换。

## 8. 全时刻证书与参数覆盖边界

初段、有限个接受的中间区间、无穷尾段覆盖 `[0,∞)`。
其上界最大值 `Bt` 因而证明 **固定 h** 下 `sup_t e_t(h,t)≤Bt`。
如果发现真实初始/末端/中点超限，则拒绝；区间资源上限耗尽只可返回未解决，不能接受。

这个命题不需要误差随时间单调，不靠时间采样验收，也不需要冲激→阶跃换算。
它当前需要完整稠密谱分解；没有建立大型稀疏模型上的经济证书。
在七个 h 点重复此命题，只认证七条全部时间轨迹；它不包围未审计的 HTC 点。
因此联合目标 `sup_{h,t}` 的连续参数证明仍未完成，即使稳态部分已对整个局部盒通过。

## 9. 最终基、选择与接受规则

设 `W` 为 stock 快照及常数模态的数值满秩正交基，`U_s` 为归一化快照的 SVD 方向。
将保留的 `U_s[:, :k]` 与常数模态放入 W 坐标中两次正交化，得到 `T_k`，交付 `V_k=W T_k`。
最终稳态证书和阶跃证书都直接针对 `V_k`，不能沿用 raw W 的保证。

从 stock cutoff 的保留数起逐项检查，不对动态误差随阶数单调作假设。
完整中心稳态读数仅拒绝候选，接受稳态必须用第 3 节的连续盒证书。
接受固定参数的阶跃必须用第 5–7 节的全时间包围。
即使这两个阶段通过，仍应将 `step_certified_continuous_htc` 保持为 false，直到另有参数包围。

所有构造都可在精确算术证明下复核；数值阶段的外向包围与论文新颖性判断是独立未完成问题。

## Follow-up: continuous local HTC step coverage

The earlier missing parameter-domain bridge is now supplied by
[AFFINE_STEP_BRIDGE_PROOF.md](AFFINE_STEP_BRIDGE_PROOF.md), with results in
records/AFFINE_STEP_BRIDGE_20261007.md（本地记录：`records/AFFINE_STEP_BRIDGE_20261007.md`）.
Three final SVD bases pass the all-time step threshold throughout the narrower
stated cell, via a relative quadratic-form perturbation theorem. This does not
extend the claim to the full original HTC box. The original fixed-parameter
proof above remains the prototype's center temporal certificate.

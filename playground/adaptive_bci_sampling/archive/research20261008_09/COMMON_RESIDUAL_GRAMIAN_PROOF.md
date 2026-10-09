# 共同输入残差 Gramian：连续域尾项定理与判决门槛

日期：2026-10-09。基于 `52e1870`。这是一个可以判决的证书族，
不是已经胜出的新提取器。普通浮点大模型实验不等于机器严格认证。

## 1. 对象和假设

原系统及被验收的固定空间为

\[
C\dot x+K(h)x=Gu,\quad \dot z+A(h)z=Bu,\quad x(0)=z(0)=0,
\]

其中 \(V^TCV=I\)、\(A(h)=V^TK(h)V\)、\(B=V^TG\)、
\(K(h)=K_0+\sum_i h_iH_i\)，\(H_i\succeq0\)，且参数盒下端给出
\(K_\ell\preceq K(h)\preceq K_u\)。首先假设 \(G=CVB\)；第 7 节处理真实包含缺陷。
定义

\[
D(h)=K(h)V-CVA(h),\quad Q(h)=D(h)^TK_\ell^{-1}D(h),
\]

并取 \(K_\ell\succeq\alpha C\)、\(A_\ell\succeq\beta I\)，
\(0<\mu<\beta\)。这些量均与时间无关。残差不是任意约化系数方向的范数：
证书接受量始终是 \(B^TPB\)，保留了输入激发方向。

寻找一个共同 \(P\succeq0\)，使每个盒顶点满足

\[
A(h_v)P+PA(h_v)-2\mu P\succeq Q(h_v). \tag{1}
\]

因为 D 仿射且 \(D^TK_\ell^{-1}D\) 矩阵凸，顶点的 (1) 蕴含整个连续盒上的 (1)。
证明：写 \(D(h)=\sum_v\theta_vD_v\)，\(\sum\theta_v=1\)，则
\(\sum_v\theta_vD_v^TK_\ell^{-1}D_v-D(h)^TK_\ell^{-1}D(h)
=\sum_v\theta_v(D_v-D(h))^TK_\ell^{-1}(D_v-D(h))\succeq0\)。
左侧储能表达式对参数仿射。这不需要参数采样、特征向量导数或谱间隙假设。

## 2. 输入驱动的共同动态尾项

对任何固定参数及输入 u，令

\[
v(t)=e^{-A(h)t}Bu,\qquad g(t)=D(h)v(t).
\]

沿轨迹对 \(e^{2\mu t}v(t)^TPv(t)\) 求导并使用 (1)，得到

\[
\int_0^\infty e^{2\mu t}\|g(t)\|_{K_\ell^{-1}}^2dt
\le u^TB^TPBu. \tag{2}
\]

这里的 g 是被舍弃动态方向的真实冲激残差，而不是训练快照的 POD 能量。
共同可达族是 \(\{D(h)e^{-A(h)t}B:h\in\mathcal H,t\ge0\}\)；
P 给出其输入加权储能包围。它并没有构造一个新的低秩全阶空间 W，
也没有把 P 的小特征值自动等同于可丢弃的全阶误差方向。

设 \(p=\lambda_{\max}(B^TPB)\)。以下均为对每个 u 的界，故覆盖任意有符号常值输入组合。
真实误差 \(e=x-Vz\) 满足

\[
C\dot e+K(h)e=-D(h)z,\quad e(0)=0. \tag{3}
\]

## 3. 两个时间分支

能量法对任意 \(C\dot w+K(h)w=r\) 给出

\[
\|w(t)\|_C^2\le\|w(0)\|_C^2+
\int_0^t\|r(s)\|_{K(h)^{-1}}^2ds. \tag{4}
\]

由 \(K(h)^{-1}\preceq K_\ell^{-1}\)，以及
\(D z(s)=\int_0^s g(v)dv\)，Cauchy–Schwarz 和积分换序给出

\[
\int_0^t\Big\|\int_0^s g(v)dv\Big\|_{K_\ell^{-1}}^2ds
\le {t^2\over2}\int_0^t\|g(v)\|_{K_\ell^{-1}}^2dv.
\]

所以短时间分支是

\[
\|e(t)\|_C\le {t\over\sqrt2}\sqrt{u^TB^TPBu}. \tag{5}
\]

为了避免稳态残差无限积分，令
\(r_\infty=-D A^{-1}Bu\)、\(e_\infty=K(h)^{-1}r_\infty\)。
由 (2) 的指数加权 Cauchy–Schwarz，

\[
\|r_\infty\|_{K_\ell^{-1}}^2\le {u^TB^TPBu\over2\mu}.
\]

进一步 \(\|e_\infty\|_C^2\le u^TB^TPBu/(2\mu\alpha)\)。
减去稳态后，(3) 的驱动变为 \(\pm\int_t^\infty g(s)ds\)。再次应用加权 Cauchy–Schwarz：

\[
\int_0^\infty\Big\|\int_t^\infty g(s)ds\Big\|_{K_\ell^{-1}}^2dt
\le {u^TB^TPBu\over4\mu^2}. \tag{6}
\]

证明可先用
\(\|\int_t^\infty g\|^2\le e^{-2\mu t}(2\mu)^{-1}
\int_t^\infty e^{2\mu s}\|g(s)\|^2ds\)，再积分换序。
(4) 及三角不等式于是给出长时间分支

\[
\|e(t)\|_C\le L\sqrt{u^TB^TPBu},\quad
L={1\over\sqrt{2\mu\alpha}}+
\sqrt{{1\over2\mu\alpha}+{1\over4\mu^2}}. \tag{7}
\]

(5)、(7) 在同一参数盒内对所有正时间同时成立。
注意 L 仍含慢衰减尺度：输入驱动性本身不保证相对误差的转换足够紧。

## 4. 相对阶跃与稳态

通过可逆输入变换可取 \(B^TB=I\)。一般情形保留 \(\|B\|\) 与 \(\sigma_{\min}(B)\)。
记 \(f_t(a)=(1-e^{-at})/a\)。由标量谱演算
\(f_t(A)\succeq t(I+tA)^{-1}\)，及逆的 Loewner 单调性，

\[
B^Tf_t(A(h))B\succeq R(t):=tB^T(I+tA_u)^{-1}B.
\]

不能直接将这个不等式平方；正确做法是对每个 u 使用
\(u^TB^Tz\le\|Bu\|\|z\|\)。所以

\[
\|z(t)\|\ge d(t)\|u\|,\quad
d(t)={\lambda_{\min}(R(t))\over\|B\|}. \tag{8}
\]

d(t) 单调增加，而 d(t)/t 单调减少。
结合 (5)、(7)，误差/ROM 比值的整个时间轴上界在交点
\(T=\sqrt2L\) 达到：

\[
q={L\sqrt p\over d(\sqrt2L)},\qquad
{\|e(t)\|_C\over\|x(t)\|_C}\le{q\over1-q}\quad(q<1). \tag{9}
\]

这是上界包围的最大值，非真实误差最大值；无需时间网格或截断无限尾端。

令 \(m_\infty=\lambda_{\min}(B^TA_u^{-1}B)\)。稳态 ROM 能量下界为
\(\|Vz_\infty\|_{K(h)}\ge\sqrt{m_\infty}\|u\|\)，而
\(\|e_\infty\|_{K(h)}\le\sqrt{p/(2\mu)}\|u\|\)。
Galerkin K 正交性给出

\[
s=\sqrt{{p\over2\mu m_\infty}},\qquad
{\|e_\infty\|_{K(h)}\over\|x_\infty\|_{K(h)}}
\le {s\over\sqrt{1+s^2}}. \tag{10}
\]

双验收依然是阶跃≤\(\sqrt{0.001}\)、稳态≤0.001，未放宽 SVD 前容差。

## 5. 不必反复调 SDP：整个标量化证书族的必要门槛

在每个顶点解零指数权重的观测 Lyapunov 方程

\[
A_vP_v+P_vA_v=Q_v.
\]

任意满足 (1) 的共同 P 都满足 \(P\succeq P_v\)：
对差的 Lyapunov 方程，其右侧为正半定，用稳定半群积分即可证明。
因此 \(p\ge p_0=\max_v\lambda_{\max}(B^TP_vB)\)。
因 \(0<\mu<\beta\)、\(L(\mu)\ge L(\beta)\) 及
\(d(t)\le m_\infty/\|B\|\)，(9) 的上界本身必满足

\[
q\ge {\sqrt{p_0}L(\beta)\|B\|\over m_\infty}. \tag{11}
\]

若 (11) 已超过 \(\sqrt\epsilon/(1+\sqrt\epsilon)\)，
**这个固定 V 的整个单 P、单 μ、标量 p 包围族无法通过**。
这不是原 ROM 真误差的下界；也不排除矩阵输入比值、分时间储能、
分速率储能、参数依赖 P 或新 V。脚本使用 \(\sigma_{\min}(B)\le\|B\|\)
代替分子中的 \(\|B\|\)，仍是有效、略弱的必要门槛。

CG 解只提供 Q 的区间时，使用第 6 节的 Q 下界解 Lyapunov 方程即可。
这个下界可能不正定，但 Lyapunov 差的右端仍正半定，因此论证不变。
浮点 Lyapunov 解和特征值仍需舍入包络，才能把数值失败判决升级为机器证明。

## 6. 真求解缺陷和复杂度

一次共同 \(K_\ell\) AMG 构造，对 D 的仿射系数解
\(Y_i\approx K_\ell^{-1}D_i\)。写顶点 \(D=K_\ell Y+E\)，则

\[
Q=Y^TK_\ell Y+Y^TE+E^TY+E^TK_\ell^{-1}E.
\]

前面三项构成变分下界 \(Q^-\)，最后一项正半定，并且
\(E^TK_\ell^{-1}E\preceq \|C^{-1/2}E\|^2 I/\alpha\)。
所以有明确上下界 \(Q^-\preceq Q\preceq Q^+\)，不假定 CG 精确求解。
用 Q+ 找 P；用 Q− 计算必要门槛。

若 SDP 输出存在负 PSD 主元或 LMI 缺陷，可先给 P 加最小必要标量 I，
再以 \(\delta/[2(\beta-\mu)]I\) 修复剩余 LMI 缺陷。
这只是精确算术的修复公式；数值最小特征值本身未经 outward rounding。
本原型不把 SDP 的 `optimal` 状态当作严格证书。

d 个参数、r 阶 ROM：全阶认证为 \(1+(d+1)r\) RHS，其中 1 是
Collatz 衰减下界的正向量求解。小型计算为 2^d 个 Lyapunov 方程或 LMI。
因此即使成功，它也尚未降低旧方法的全阶逆作用数量。
`--storage gate` 给出解析可行的各向同性 P 上界，同时计算 (11)，不执行无意义 SDP。

## 7. 输入包含缺陷和精确 toy

浮点 C 正交/输入包含有缺陷时，保留
\(R_0=G-CVB\)、\(r_0=\|C^{-1/2}R_0\|\)。其额外阶跃绝对误差
≤\(r_0 f_t(\alpha)\|u\|\)。用保守分母
\(\sigma_{\min}(B)f_t(a)\)、\(a=\lambda_{\max}(A_u)\)，相对 ROM
额外项≤\(r_0\max(1,a/\alpha)/\sigma_{\min}(B)\)。稳态能量额外项为
\(r_0/\sqrt{\alpha m_\infty}\)。这些项在脚本中没有删掉。
这仍不覆盖全部点积/正交化/特征值的浮点误差。

toy 使用 JSON 中明确给出的 4 状态 grounded thermal M-matrix，两个非负输入、
两个与 K0 非对易的对角 Robin 参数，盒为 [0,1]^2，耦合 c=1/10000。
V=(e1,e2)，\(K_\ell\succeq I\)、\(A(h)\succeq I\)、\(A(h)\preceq4I\)。
\(D^TK_\ell^{-1}D\preceq c^2I\)，故取 \(P=c^2I\)、\(\mu=1/4\)
得到精确 LMI 余量≥c²/2。L<4，保守有理分母得到 q≤17c。
阶跃真相对界≤17/9983，稳态真相对平方界≤8c²/(1+8c²)。
双验收比较在 Fraction 上完成，不涉及参数或时间采样。

这个 toy 只证明定理可用；不能拿来声称相对旧 toy 更紧或已解决 Case1。

## 8. 另一条容易误入的支线：静态零起点二次储能

考虑任意固定参数的稳定增广系统
\(\dot y=-Ly+bu\)，y 包含真状态和 ROM 状态，u 为常值；
要证明 \(y^TJy\le0\)，其中 J 是“误差平方减允许的 ROM 平方”。
若试图用一个与时间无关的二次储能 \([y;u]^TP[y;u]\) 同时满足：

1. 储能支配目标 \(P\succeq\operatorname{diag}(J,0)\)；
2. 所有初始 \([0;u]\) 的储能非正；
3. 沿增广系统储能不增；

那么这个证书实际上要求 **整个可达线性包** 上 \(y^TJy\le0\)，
而不只是常值输入产生的阶跃轨迹上。证明如下。
1 和 2 迫使 \(P_{uu}=0\)，PSD 块的零对角又迫使 \(P_{yu}=0\)。
3 的零 u-u 块迫使 \(P_{yy}b=0\)，并有
\(Q=L^TP_{yy}+P_{yy}L\succeq0\)。L 稳定意味着
\(P_{yy}=\int_0^\infty e^{-L^Ts}Qe^{-Ls}ds\succeq0\)。
由 \(b^TP_{yy}b=0\)，连续非负被积函数必须恒零，故
\(Qe^{-Ls}b=0\)；同一积分公式再给出 \(P_{yy}e^{-Lt}b=0\)。
因此 Pyy 在 \(\operatorname{span}\{e^{-Lt}b:t\ge0\}\) 上恒零。
由于 \(P_{yy}\succeq J\)，J 在这个整个线性空间上非正。

若增广真/ROM 对的可达线性包已经是整个状态空间，这要求 J 全空间非正，
通常不可能：单独取非零真状态、零 ROM 状态，误差输出就为正。
这并非真实阶跃误差反例，而是上述静态单储能证书的结构障碍。
时间依赖储能、非单调储能、显式时间基函数或参数/输入流形约束不在此排除范围内。
此支线不应再靠扩大 SDP 维数解决。

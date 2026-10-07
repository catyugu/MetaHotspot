# 带符号创新 Gram 的 Poisson–Loewner 连续谱包围

本证明在精确算术下成立。它改进的是动态余量的数学包围，不改变 trial 节点、参数
细分或解算器。实现 `matrix_innovation.py`、`rational_dynamic.py` 尚未作向外舍入。
基础正交有理展开、精确尾项和 Bernstein 试验见 `RATIONAL_DYNAMIC_PROOF.md`。

这是历史 STATE-IMPULSE 目标的证明。最新稳态 `2 epsilon`、全时刻阶跃
`2 sqrt(epsilon)` 目标另见 [STEADY_STEP_PROOF.md](STEADY_STEP_PROOF.md)，
不能由本文的冲激积分相对误差直接推出。

## 1. 假设与目标

`C>0`，`K(h)=K0+sum h_i H_i>0`，`H_i>=0`，参数盒 `h in [a,b]`，固定满列秩 V。
`G` 的源能量 Gram `Q(h)=0.5 G^T K(h)^{-1}G` 正定。目标为

\[
 \epsilon(h;V)=\|u\mapsto T_h(\cdot)u-T_V(\cdot)u\|_
 {Q(h)\to L^2([0,\infty);C)}.
\]

这是全部输入向量到无限时间全部温度状态的算子范数；不等于标准 H2 trace 指标，
也不是任意时间功率输入的相对诱导范数。
假设已知合法广义谱区间 `[alpha,beta]`，覆盖盒内 `C^{-1/2}K(h)C^{-1/2}`。
Galerkin 约化谱也在其中。当前物理模型用 Collatz 下界和行和上界；小型 raw 模型
可用盒角点的完整广义特征值（原型额外报告两次稠密分解），均只在精确算术意义下合法。

## 2. 强迫创新映射的精确 Gram

固定正时间移位 `sigma_0,...,sigma_{N-1}`。对于多项式试验系数 `xhat_k(h)`，
`qhat_0=G`、`qhat_{k+1}=qhat_k-sqrt(2 sigma_k) C xhat_k`，残差为

\[
 F_k=qhat_k-(K+\sigma_k C)xhat_k/\sqrt{2\sigma_k}.
\]

实际与试验之间的缺陷满足

\[
 \delta x_k=\sqrt{2\sigma_k}(K+\sigma_k C)^{-1}(\delta q_k+F_k),\quad
 \delta q_{k+1}=\delta q_k-\sqrt{2\sigma_k}C\delta x_k,\quad\delta q_0=0.
\]

将 C 白化并以广义特征向量对角化。单个特征率 `lambda>0` 下，以 `K^{-1}` 度量
归一化的源误差和残差记为 z、f；系数输出以 C 度量归一化。则

\[
 z_{k+1}=t_k z_k-b_k f_k,\quad \delta x_k=g_k(z_k+f_k),
\quad t_k=\frac{\lambda-\sigma_k}{\lambda+\sigma_k},\quad
 b_k=\frac{2\sigma_k}{\lambda+\sigma_k},\quad
 g_k=\frac{\sqrt{2\sigma_k\lambda}}{\lambda+\sigma_k}.
\]

全部系数缺陷和精确末端能量组成输出
`(delta x_0,...,delta x_{N-1},z_N/sqrt(2))`。
`RATIONAL_DYNAMIC_PROOF.md` 第 10 节的望远镜能量恒等式给出其 N×N Gram

\[
 A_{jj}(\lambda)=b_j,\quad
 A_{jk}(\lambda)=-\tfrac12 b_jb_k\prod_{\ell=j+1}^{k-1}t_\ell\quad(j<k).
 \tag{1}
\]

不能把非对角项替换成绝对值后再声称保留了方向。本轮保留 (1) 的符号。

## 3. Poisson 表示与相对谱包围

定义全通函数 `B_j(s)=prod_{ell<j}(s-sigma_ell)/(s+sigma_ell)`，

\[
 \psi_j(s)=\frac{2\sigma_j}{s+\sigma_j}B_j(s).
\]

单个 f_j 的系数输出及末端尾响应重构为

\[
 \delta\mathcal T_j(s)=\frac{\sqrt\lambda}{s+\lambda}\psi_j(s)f_j.
\]

证明：第 j 步产生系数 `sqrt(2 sigma_j lambda)/(lambda+sigma_j) f_j`，随后
尾源为 `-2 sigma_j/(lambda+sigma_j) f_j`。两部分合成为

\[
 \frac{\sqrt\lambda\,2\sigma_j}{\lambda+\sigma_j}
 \frac{B_j(s)}{s+\sigma_j}
 \left(1-\frac{s-\sigma_j}{s+\lambda}\right)f_j
 =\frac{\sqrt\lambda\,\psi_j(s)}{s+\lambda}f_j.
\]

后续齐次展开及其精确末端响应只是这个尾响应的正交展开，没有引入截断误差。
由连续时间 Parseval 对每个实输入向量，得到

\[
 A(\lambda)=\frac1{2\pi}\int_{-\infty}^{\infty}
 \frac{\lambda}{\lambda^2+\omega^2}
 \Psi(i\omega)^*\Psi(i\omega)\,d\omega,
 \quad \Psi=(\psi_0,...,\psi_{N-1}). \tag{2}
\]

积分的 Hermitian 矩阵非负；对称频率积分为实对称。被积函数可积，且 (2) 包括
末端能量，因此与 (1) 完全相同。

对任意 `lambda,mu>0`，Poisson 核的比值满足

\[
 \frac{\lambda/(\lambda^2+\omega^2)}{\mu/(\mu^2+\omega^2)}
 =\frac\lambda\mu\frac{\mu^2+\omega^2}{\lambda^2+\omega^2}
 \le\max(\lambda/\mu,\mu/\lambda).
\]

乘以非负矩阵并积分，得到关键的**矩阵序**不等式

\[
 A(\lambda)\preceq\max(\lambda/\mu,\mu/\lambda)A(\mu). \tag{3}
\]

把合法谱区间分成几何子区间 `[l_i,u_i]`，令 `mu_i=sqrt(l_i u_i)`、
`r_i=sqrt(u_i/l_i)`。因此整个子区间都有 `A(lambda)<=r_i A(mu_i)`。
这里中心矩阵不是谱采样验收；(3) 显式包围中心之间所有特征率。
原型取每个子区间宽比最多 2。可采用其他固定宽比，证明不依赖其调参。

令 `M_1=r_1 A(mu_1)`；逐次用

\[
 M_{i+1}=M_i+(r_{i+1}A(\mu_{i+1})-M_i)_+ \tag{4}
\]

构造共同上界。对于对称 D，`D_+>=0` 且 `D_+-D>=0`，故更新同时支配旧 M
及新中心包围矩阵。因此最终 `M>=A(lambda)` 对整个谱区间成立，并且 `M>=0`。
正序和反序各自产生合法 M，按 trace 选择其中一个仍合法；不能对两个 M 作逐元素最小值。
没有谱区间时，由 `A>=0`、`trace A=sum b_j<=2N` 可用 `M=2N I`。

共同尺度变化 `lambda,sigma -> c lambda,c sigma` 不改变 A 或 M，故此谱包围
没有显式 `alpha^{-3}` 的补偿因子。Riesz 范数本身仍可能反映实际物理条件数，
不能声称消除了全部病态性。共同 M 也未保留每个空间谱通道的独立分布。

## 4. 从连续谱到连续 HTC 的带符号余量界

谱函数演算将第 2 节的联合缺陷能量 Gram 写成按全阶特征率分块的 `A(lambda)`。
对任何输入 u，(3)–(4) 给出

\[
 \|\delta\mathcal T\,u\|_{L^2(C)}^2
 \le\sum_{j,k}M_{jk}\,u^T F_j^T K(h)^{-1}F_k u.
\]

因为 `M>=0`，`K(h)^{-1}<=K(a)^{-1}` 可以在此**整个块二次型**中合法替换：
若 `M=L L^T`，该二次型是各列 `sum_j L_{j ell}F_j u` 的 dual 能量之和。
逐项替换带符号交叉内积而不使用整体 PSD 结构则是不合法的。
故

\[
 S_F(h)=\sum_{j,k}M_{jk}F_j(h)^T K(a)^{-1}F_k(h) \tag{5}
\]

是一个 PSD 输入 Gram 上界。

将已由分母下界 D 白化的所有 F_j 写为**同一参数控制索引**的 Bernstein 多项式
`F_j(h)=sum_nu w_nu(h) F_{j,nu}`，`w_nu>=0,sum w_nu=1`。
PSD 二次型的凸性给出

\[
 \Delta_F^2\le\max_\nu\lambda_{max}
 \left(\sum_{j,k}M_{jk}F_{j,\nu}^T K(a)^{-1}F_{k,\nu}\right). \tag{6}
\]

这保留同一控制索引内，所有时间步、空间向量、输入方向之间的完整交叉 Gram。
不是每步单独求最大后再相加。约化系统以 `Kr(a)^{-1}` 同理得到 `Delta_R`。
按前轮合法的试验主项 M_trial 和精确试验尾项，最终

\[
 \epsilon(h;V)\le M_{trial}+T_{F,trial}+T_{R,trial}+\Delta_F+\Delta_R. \tag{7}
\]

实现还与前轮合法标量界取最小值，避免共同 M 较松时退化。
已有参数细分可与 (6) 组合，但本轮的数学判决实验固定 `envelope_depth=0`。
因此实验改善不归因于参数细分或新增 full RHS。

## 5. 与标量包围存在本质差异

取偶数 N、全部 `sigma_j=1`，所有残差向量方向相同且 f_j=1。
当 `lambda->0`，`b_j->2,t_j->-1`，因此

\[
 A(\lambda)\to2 ss^T,\qquad s_j=(-1)^j,\qquad s^T\mathbf1=0.
\]

在有限连续区间 `[epsilon,2 epsilon]` 上，取单个 Poisson 包围
`M=sqrt(2) A(sqrt(2) epsilon)`，则 `1^T M 1->0`。
相反，前轮绝对值谱二次包围和逐创新三角界均趋于 `sqrt(2)N`。
两种界的比值因此可以无界增长；这不是通过更换实现或提高采样分辨率取得的常数加速。

N=8 的复现实验：epsilon=1e-5 时旧界 11.31340445，新界 0.01788753，约 632 倍；
epsilon=1e-7 时约 6,325 倍。所有区间都包含连续特征率。
反例只证明本轮相对于旧标量界的结构性优势，不证明发表新颖性或对所有模型的优势。

## 6. 最终 SVD 基与 2 tau 预算

用户指定：提取 tolerance 为 tau，最终降阶并 SVD 后的动态接受阈值可以为 2 tau。
这只是新的验收标准，不能由原欧氏快照奇异值截止推出一般的 2 tau 动态定理。

设 W 为已取得完整动态证书的 raw 基，T 为其坐标内的 SVD 截断基，V=WT。
raw 内层模型为 `(W^T K W,W^T C W,W^T G)`，V 的 Galerkin 动态与该模型再投影 T
完全相同。源能量 `Q_W(h)<=Q(h)`，由 SPD Galerkin 逆作用下界成立。
因此，若盒内全阶到 raw 的证书为 B_W，raw 到 final 以 Q_W 归一化的证书为 B_comp，
则相同输入、相同物理场范数下的三角不等式给出

\[
 \epsilon(h;WT)\le B_W+B_{comp}. \tag{8}
\]

分别使用不超过 tau 的预算即可得到最终不超过 2 tau。
这是标准分层误差三角界，本轮不以它作为原创数学贡献；不使用 saturation 假设。

原型 `svd_dynamic_guard.py` 从 stock 已保留的 SVD 方向数开始，按阶数顺序检查。
中心精确动态误差只用于拒绝，接受必须有连续盒证书 (7)。
不假设 Galerkin 动态误差随空间阶数单调；没有以二分查找隐藏该假设。
候选基始终以 raw 坐标构造，包含 uniform mode。预算规则不是最小阶数定理。
最后再对物理全阶与最终 V 直接构造独立证书，验证 (8) 的实际结果。

## 7. 文献边界

- Son–Stykel (2017), DOI 10.1137/15M1027097，作者 PDF：
  https://scwww.math.uni-augsburg.de/~stykel/Publications/SonStykel17-SIMAX-38-2.pdf
  读取正文第 3–5 节。参数 Lyapunov、min-theta、Euclidean/natural energy residual
  estimates 是已有工具，不能以使用 Lyapunov 或残差作为原创性。
- Gugercin–Antoulas–Beattie (2008), DOI 10.1137/060666123：
  https://vtechworks.lib.vt.edu/bitstream/10919/48145/1/060666123.pdf
  读取 Hilbert 空间、插值与 Lyapunov 条件的关系。该 H2 优化框架不是本项目
  相对全输入 lambda_max 指标的自动保证。
- Garrett (2016), *Harmonic functions, Poisson kernels*，第 4 节：
  https://www-users.cse.umn.edu/~garrett/m/complex/notes_2014-15/08c_harmonic.pdf
  上半平面 Poisson 核是标准事实；(3) 的核比值比较可直接代数证明。
- Rettberg et al. (2024), DOI 10.1007/s10444-024-10195-8：
  https://link.springer.com/content/pdf/10.1007/s10444-024-10195-8.pdf
  读取第 3 节。hierarchical/auxiliary error bounds 已有理论；不能把 (8) 或
  最终阶数的证书验收冒充新概念。

本轮的明确增量是：把 (2) 的正积分结构转为连续谱矩阵序包围，并由 (5)–(6)
保留时间、场和输入方向，适配当前连续参数目标。是否有同等文献构造、是否可发表，
尚不能由此次检索排除，需要进一步系统查重。严格浮点认证和经济全域验证仍未完成。

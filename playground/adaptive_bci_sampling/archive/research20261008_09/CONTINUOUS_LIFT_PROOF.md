# 静态残差提升：连续参数域、任意常输入、全部正时间

本轮研究从时间积分转向矩阵函数的相对 divided difference。以下是自包含推导，
不宣称定理或组合的文献首次性。原型给出小型热系统的精确算术连续域证书；
当前 Case1 基底未通过，不能称为解决成本问题的新算法。

## 1. 契约与假设

\(C\succ0\)，\(K(h)=K_0+\sum_i h_iH_i\succ0\)，\(H_i\succeq0\)，
\(h\) 是盒内任意**固定**参数。\(V^TCV=I\)，固定 SVD 前空间，不随参数或时间改变。
\(A(h)=V^TK(h)V\)，\(B=V^TG\)，\(D(h)=K(h)V-CVA(h)\)。
先假设 \(G=CVB\)，即输入方向已包含在空间中。

真解和 Galerkin 解为

\[
C\dot x+Kx=Gu,\quad \dot z+Az=Bu,\quad x(0)=z(0)=0,
\qquad x_r=Vz .
\]

\(u\) 是任意常向量，包括带符号的输入组合。不是只检查单独输入列。
验收仍为稳态相对 K 范数 \(\epsilon=10^{-3}\)，每个 \(t>0\) 的相对 C 范数
\(\sqrt\epsilon\)。若对应真解为零，使用不除以零的范数不等式解释。

## 2. 全时间相对误差引理

令 \(\widehat K=C^{-1/2}KC^{-1/2}\)，\(U=C^{1/2}V\)，
\(\widehat D=C^{-1/2}D=\widehat KU-UA\)，
\(f_t(a)=(1-e^{-ta})/a\)。定义静态提升

\[
X=C^{1/2}K^{-1}D=\widehat K^{-1}\widehat D.
\]

**对所有 \(t>0\)，有**

\[
\|x(t)-Vz(t)\|_C\le \|X\|_F\,\|z(t)\|_2. \tag{1}
\]

证明：分别正交对角化 \(\widehat K\) 和 \(A\)，特征值记为 \(\lambda_i,a_j>0\)。
只用于理论证明；算法不计算全阶特征分解。设

\[
M_t=(f_t(\widehat K)U-Uf_t(A))f_t(A)^{-1}.
\]

在这两个特征基中，

\[
(M_t)_{ij}=\frac{f_t(\lambda_i)/f_t(a_j)-1}{\lambda_i-a_j}
\widehat D_{ij}.
\]

\(f_t\) 单调递减，而 \(a f_t(a)=1-e^{-ta}\) 单调递增。
若 \(\lambda\ge a\)，则 \(a/\lambda\le f_t(\lambda)/f_t(a)\le1\)；
若 \(\lambda\le a\)，反向区间同理。因此

\[
\left|\frac{f_t(\lambda)/f_t(a)-1}{\lambda-a}\right|\le\lambda^{-1}.
\]

重复特征值用连续极限；\(|f_t'(\lambda)/f_t(\lambda)|\le1/\lambda\)。
于是逐元素平方求和得到 \(\|M_t\|_F\le\|X\|_F\)，而
\(C^{1/2}(x-Vz)=M_tz\)，证明 (1)。

若盒内 \(\|X(h)\|_F\le\rho<1\)，三角不等式给出

\[
\forall h\in\mathcal H,\ \forall t>0,\ \forall u:\quad
\frac{\|x-Vz\|_C}{\|x\|_C}\le\frac\rho{1-\rho}. \tag{2}
\]

这里不需要早时间段、有限时间网格、无穷尾段、辅助误差 ROM 或时间积分。
Frobenius 范数保留了左右特征基之间的非对易性。
**不能仅凭上述逐元素界把 (1) 中的 Frobenius 范数换成谱范数。**
本轮未证明这个特殊 Schur 乘子的谱范数收缩性。

输入包含不精确时，设 \(R_0=G-CVB\)，\(\sigma=\sigma_{\min}(B)>0\)，
\(K(h)\succeq\alpha C\)，\(A(h)\preceq a_+I\)。额外误差为
\(f_t(\widehat K)C^{-1/2}R_0u\)。利用
\(\|z(t)\|\ge f_t(a_+)\sigma\|u\|\) 及
\(f_t(\alpha)/f_t(a_+)\le\max(1,a_+/\alpha)\)，可在 \(\rho\) 中加入

\[
\eta_0=\frac{\|C^{-1/2}R_0\|_2}{\sigma}
\max(1,a_+/\alpha). \tag{3}
\]

浮点 C 正交误差需另外严格控制；Case1 本轮只报告其大小，不作舍入认证。

## 3. 一次全阶算子构造覆盖整个连续盒

令 \(K_-=K(h_{\min})\)，有 \(K(h)\succeq K_-\succeq\alpha C\)。
矩阵逆的 Loewner 单调性和 \(C\preceq K(h)/\alpha\) 给出

\[
\|X(h)\|_F^2
=\operatorname{tr}(D^TK(h)^{-1}CK(h)^{-1}D)
\le\frac1\alpha\operatorname{tr}(D(h)^TK_-^{-1}D(h)). \tag{4}
\]

\(D(h)\) 仿射，所以右侧是凸二次函数。
盒中任意点是顶点的凸组合，凸性保证其值不超过最大顶点值。
这不是“真实误差最大值在顶点”的假设。

\[
\rho^2=\frac1\alpha\max_{v\in\operatorname{vertices}(\mathcal H)}
\operatorname{tr}(D(v)^TK_-^{-1}D(v)) \tag{5}
\]

给出 (2) 的**连续全域充分条件**。计算 \(K_-^{-1}[D_0,D_1,\ldots,D_d]\)
后，顶点只操作 Gram，精确求解成本最多 \((d+1)r\) RHS、一个全阶算子构造；
用 Collatz 求 \(\alpha\) 时加一个 RHS。不需要为每个参数盒另解一套系统。
若某 \(D_i=0\)，可省相应 RHS；toy 中 D 常量，只有两个 RHS。

这是认证阶段成本，不包含构造 V 的成本，也不是证明其比已有输入驱动辅助法便宜。
(4) 会丢失 C 方向信息；(1) 又要求控制全部 r 个系数方向。
两者都是明确的潜在保守性，不能用时间采样替代。

## 4. 同一连续域的稳态条件

仍假设输入被精确包含。稳态误差 \(e_s=-K^{-1}Dz_s\)，且
\(V^TKe_s=0\)。因此

\[
\|x_s\|_K^2=z_s^TAz_s+e_s^TKe_s.
\]

对每个盒顶点验证小矩阵

\[
D(v)^TK_-^{-1}D(v)\preceq\tau^2 A(v) \tag{6}
\]

即可推出盒内全部 h 上该不等式：左侧矩阵凸，右侧仿射。
结合 \(K(h)^{-1}\preceq K_-^{-1}\) 得到

\[
\frac{\|e_s\|_K}{\|x_s\|_K}\le
\frac\tau{\sqrt{1+\tau^2}}. \tag{7}
\]

toy 采用 \(\tau=1/1000\)，用精确有理 LDL 正主元验证 (6)，
所以 (7) 严格小于 0.001。阶跃验收把 (2) 的有理上界平方后与 1/1000 比较。
**两项验收都不依赖参数或时间采样，也没有浮点 PSD 判决。**

## 5. 迭代解缺陷的数学界

在 Klo 下，\(K_-Y=D-R\)，则

\[
\|K_-^{-1/2}D\|_F\le
\|K_-^{1/2}Y\|_F+\alpha^{-1/2}\|C^{-1/2}R\|_F.
\]

点参数下直接界为

\[
\|C^{1/2}K^{-1}D\|_F\le
\|C^{1/2}Y\|_F+\alpha^{-1}\|C^{-1/2}R\|_F.
\]

Case1 驱动包含真实 AMG–CG 缺陷项，不把 tol 当作误差上界；但点积、范数、
正交性、Collatz 和特征值没有 outward rounding，标记 `floating_point_certified=False`。
这与 toy 的 Fraction 整数/有理算术证书严格区分。

## 6. 为什么不能只提升 m 个稳态输入误差

令 C=I，G=e1，

\[
K_0=\begin{pmatrix}71/50&-1/5&-9/10\\-1/5&2&-1\\-9/10&-1&3\end{pmatrix},
\quad H=\operatorname{diag}(1,0,0),\quad h\in[0,1],
\quad V=\begin{pmatrix}1&0\\0&3/5\\0&4/5\end{pmatrix}.
\]

这是 SPD grounded thermal M-matrix，输入非负，Robin 项对角非负，V 固定正交。
对所有 h，真稳态解都是 \([1,3/10,2/5]^T/(1+h)\in\operatorname{range}(V)\)。
所以 \(D(h)A(h)^{-1}B=0\)：全部稳态输入误差解恒零。
但是 \(e''(0)=-D(h)B\)，且

\[
\|D(h)B\|^2=361/2500>0
\]

与 h 无关。由解析性，瞬态误差不是恒零。
此外在 h=1,t=1，原型对矩阵阶跃 Taylor 多项式做到 60 次，
利用 SPD 半群收缩获得余项 \(M^{60}/61!\)，其中 M 是精确有理的行和范数上界。
使用正交补方向 \(w=[0,-4/5,3/5]^T\) 得到误差下界，真解范数用有理平方根上界包住；
最终在有理数上验证 \(\mathrm{lower}^2>10^{-3}\mathrm{upper}^2\)。
因此不仅排除了“零稳态误差意味着零瞬态误差”，还严格证明了该点违反本项目阶跃门槛。
采样结果另记，不参与证明。

这个反例只否定**仅由稳态输入误差决定瞬态保证**的捷径。
它不否定误差动力学近似低维，也不否定带严格遗漏尾项控制的输入可达近似。

## 7. 本轮文献边界

- [Henrion 等，参数依赖 Lyapunov 函数](https://homepages.laas.fr/henrion/Papers/polylyap.pdf)：
  冻结连续参数的多项式 Lyapunov、鲁棒稳定性和 LMI 层级已有理论；
  鲁棒稳定性本身不是这里的每时刻全场相对误差。
- [O'Connor–Grepl，Offline Error Bounds for the Reduced Basis Method](https://www.igpm.rwth-aachen.de/Download/reports/grepl/OG_OfflineBounds_Preprint.pdf)：
  用灵敏度信息覆盖整个连续参数域已有先例，主要推导为椭圆问题。
  不能把“离散训练集外也认证”本身当创新。
- [Rettberg 等，2023，ALP 误差界](https://arxiv.org/abs/2303.17329)：
  辅助误差系统已有研究；前轮改进点态数值效果不能替代共同连续域证书。
- [Jawecki–Auzinger–Koch，2019，指数与 phi 函数可计算误差界](https://doi.org/10.1007/s10543-019-00771-6)：
  divided difference、缺陷与 phi 函数误差已有工具；本轮的相对阶跃静态提升式
  给出直接推导，尚未完成足以支持首次性主张的文献排查。
- [Jarlebring–Rubensson，arXiv:1206.1762v2](https://arxiv.org/abs/1206.1762)：
  官方页明确撤稿，Lemma 2 不成立。不能引用其摘要当作谱范数 Schur 收缩定理。
  本轮只使用可逐元素平方验证的 Frobenius 不等式。

应继续研究的具体对象是输入可达的、跨参数共同有效的近似，而非整个 ROM 系数空间
的不变子空间缺陷。若要替代 (1)，必须证明结构化 Schur 乘子或参数化可达储能界，
并给出被截断方向的确定性上界及相应离线成本；这是未完成的研究问题，不是现成算法。

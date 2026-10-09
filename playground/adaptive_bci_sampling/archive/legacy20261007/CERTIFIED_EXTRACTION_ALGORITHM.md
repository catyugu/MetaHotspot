> 历史归档：保留当时的目标、结果与局限；“最新”等措辞仅指当时状态。当前验收和研究入口见 [当前 README](../../README.md)。

# 认证驱动的仿射有理提取：从原始算子到降阶算子

这是完整可执行的研究提取器，不再是对随机基底的外部审计脚本。
入口为 `certified_extraction.extract_certified_rom`。当前实现采用稀疏直接
快照求解和稠密谱认证，已验证局部连续 HTC 域；尚不能宣称完整原始
HTC 域或大规模工业系统上的成本优势。

## 1. 输入、输出与误差契约

输入原始算子

\[
C\dot x+K(h)x=Gu,\quad x(0)=0,\qquad K(h)=K_0+\sum_{j=1}^d h_jH_j,
\quad h\in\mathcal H=\prod_j[h_j^-,h_j^+].
\]

要求实对称矩阵，\(C\succ0\)，\(H_j\succeq0\)，\(K(h^-)\succ0\)，
以及输入矩阵 \(G\) 满列秩。\(K_0\) 本身可以半正定。参数是固定的仿射
Robin 系数，不能将物理 HTC 到有效 Robin 系数的非线性转换隐去。

输出为 \(V,K_{0,r},C_r,G_r,H_{j,r}\)，其中

\[
K_{0,r}=V^TK_0V,\quad C_r=V^TCV,\quad G_r=V^TG,\quad H_{j,r}=V^TH_jV.
\]

`operator(h)` 返回 \(K_r(h)\)，`steady(h,u)` 和 `step(h,u,times)` 返回
重构到原始全部节点的温升。`save/load` 存储/加载降阶算子和重构基底，
不需要原始矩阵或快照。

**原始全阶系统只认证 SVD 前空间** \(V_0\)：

\[
\sup_{h,u}\frac{\|x_\infty-x_{0,\infty}\|_{K(h)}}{\|x_\infty\|_{K(h)}}
\le\varepsilon,\qquad
\sup_{h,t>0,u}\frac{\|x-x_0\|_C}{\|x\|_C}\le\sqrt\varepsilon.
\]

SVD 后的目标为 \(2\varepsilon\) 和 \(2\sqrt\varepsilon\)。为证明这个目标，
仅在已提取的 \(r_0\) 阶系统中检查压缩误差，不重新调用原始全阶证书。
所有输入组合均参与算子范数认证。参数在一次响应期间固定。

## 2. 初始化：保护零时间与稳态响应

在域的算术中心 \(h_c\) 求解

\[
B_0=C^{-1}G,\qquad X_c=K(h_c)^{-1}G.
\]

以 \([B_0,X_c,\mathbf1]\) 构造初始快照矩阵。所有列归一化后，用带主元
QR 得到完整独立空间；只在机器精度级别处理线性依赖，不按
\(\varepsilon\) 丢弃方向。

这样做具有两个数学性质：

1. \(C^{-1}G\subset\operatorname{range}(V_0)\)，故
   \(V_0C_{0,r}^{-1}G_{0,r}=C^{-1}G\)，初始阶跃斜率对所有输入精确。
2. \(K(h_c)^{-1}G\subset\operatorname{range}(V_0)\)，故中心稳态对所有输入精确。

常数方向保留保守传导算子的均匀温度模态。质量求解的 RHS 与热系统
求解的 RHS 分开计数。

## 3. 未压缩空间认证驱动循环

维护覆盖**完整请求域**的单元列表，初始只有根单元。按固定顺序选择
首个未通过的单元，使用以下判据；不随机接受，也不默默缩小请求域。

### 3.1 参数几何是否允许当前充分条件通过

计算中心全阶谱，并缓存该单元的全阶参数变化界 \(d_F\)。只随基底
更新降阶参数变化界 \(d_R\)。其构造见
[AFFINE_STEP_BRIDGE_PROOF.md](AFFINE_STEP_BRIDGE_PROOF.md)：相对二次型
角点半径、Duhamel 短/长时间界，以及覆盖整个时间轴的白化区间界。

对 \(a=\sqrt\varepsilon\)，中心时间证书可使用的预算为

\[
b_c=\frac{a(1-d_F)-d_F-d_R}{1+d_R}.
\]

若 \(d_F\ge1\)、半径不满足条件或 \(b_c\le0\)，沿相对二次型变化贡献
最大的参数轴二分单元。两个子单元完整替代父单元，保持原请求域。
贡献指标是

\[
(h_j^+-h_j^-)\,
\|K_c^{-1/2}H_jK_c^{-1/2}\|_2.
\]

若到达单元预算，返回 `unresolved`。

### 3.2 时间误差不足时生成有理响应块

在中心使用已缓存的全阶谱和当前降阶模型，在固定的 96 个对数时间点
计算全输入组合的相对 C-能量误差。**这些点只决定如何补充空间，不用于接受。**

若采样峰值超过 \(0.65b_c\)，取其时间 \(t_*\)，选择正移位

\[
\sigma=1/t_*,\qquad (K(h_c)+\sigma C)X_\sigma=G.
\]

全部输入一起求解一个响应块并加入快照空间。一个因子分解对应 \(m\)
个 RHS。若采样足够小，调用解析全时间中心证书，内部阈值为
\(0.95b_c\)；若其不能通过，也使用峰值时间补充有理块。

中心证书包括近零解析界、有限时间区间导数包络和无限尾界。因此接受
仍覆盖全部 \(t>0\)，而非上述 96 个时间点。

### 3.3 稳态证书不足时生成 Riesz 校正方向

在单元角点求降阶稳态，作多线性插值 \(q(h)\)，构造二次张量
Bernstein 残差控制矩阵 \(R_\nu\)。令

\[
D=G_r^TK_r(h^+)^{-1}G_r,
\quad M_\nu=R_\nu^TK(h^-)^{-1}R_\nu.
\]

稳态界为

\[
s=\sqrt{\max_\nu\lambda_{\max}(M_\nu,D)}.
\]

若 \(s>\varepsilon\)，设 \(Z^TDZ=I\)，取最坏控制矩阵的最大特征向量
\(v\)，加入方向

\[
p=K(h^-)^{-1}R_{\nu_*}Zv.
\]

此 Riesz 作用已在证书的批量求解中得到，因此添加该方向**不增加新的
原始系统逆作用**。它针对的是证书的最坏控制方向，不声称它是实际
参数误差的全局最大点。嵌套空间的真实稳态 Galerkin 误差不会增加，
但 Bernstein 上界不一定单调，因此每次仍重新检查证书。

### 3.4 停止规则

每个单元同时满足

\[
s\le\varepsilon,\qquad
\frac{e_c+d_F+(1+e_c)d_R}{1-d_F}\le\sqrt\varepsilon
\]

才允许停止全阶提取。任何空间变化后重新检查所有单元，不沿用旧的
时间界。中心谱及单元下端 Riesz 因子可以复用，计数只计实际新增工作。

快照数、原始空间阶数、参数单元数均有显式预算。预算耗尽或没有新的
独立方向时返回未解决状态。具有这些预算时程序有限停止；不声称任意
系统在给定预算内一定能得到有用的低阶模型。

## 4. 约束 SVD：先保护响应，再压缩其余方向

在 \(V_0\) 坐标中，保护空间包含：

- 原始降阶模型的初始斜率 \(C_{0,r}^{-1}G_{0,r}\)；
- 常数方向；
- 每个全阶认证单元中心的原始降阶稳态方向。

令该空间的正交基为 \(P\)，归一化快照的坐标矩阵为 \(Y\)。对

\[
Y_\perp=(I-PP^T)Y
\]

进行 SVD。候选压缩空间为

\[
T_k=\operatorname{orth}[P,U_{\perp,1:k}],\qquad V=V_0T_k.
\]

从残余奇异值大于 \(\varepsilon\|Y\|_2\) 的方向数开始，逐步增加 \(k\)。
在固定保护空间与固定剩余阶数下，该选择最小化快照 Frobenius 投影残差：
它等于残余 SVD 尾部奇异值平方之和。**这只是候选选择性质，不是响应误差证明。**

## 5. 仅在原始降阶模型上认证压缩误差

将 \(V_0\) 对应的 \(r_0\) 阶算子当作源系统，\(T_k\) 当作其 Galerkin
基底，重复同一连续参数证书，要求

\[
s_{\mathrm{comp}}\le\varepsilon,\qquad
e_{\mathrm{comp}}\le\frac{\sqrt\varepsilon}{1+\sqrt\varepsilon}.
\]

若参数预算不足，允许只在这个 \(r_0\) 阶系统上细分认证单元；若中心
时间预算不足但采样误差仍小于压缩目标的一半，也尝试细分。采样仅指导
选择，最终所有小模型单元都必须解析通过。小模型单元预算为 64。

压缩检查的域始终覆盖原始认证域，不能把失败单元删掉。若某阶数失败，
增加保留方向数；不假定动态误差随阶数单调。保留全部原始独立方向时，
压缩是恒等映射，压缩误差精确为零。

### 最终稳态误差证明

因为 \(\operatorname{range}(V)\subset\operatorname{range}(V_0)\)，
Galerkin 正交性给出逐输入的 Pythagoras 恒等式

\[
\|x_\infty-x_{V,\infty}\|_K^2
=\|x_\infty-x_{0,\infty}\|_K^2
 +\|x_{0,\infty}-x_{V,\infty}\|_K^2.
\]

并且 \(\|x_{0,\infty}\|_K\le\|x_\infty\|_K\)。因此

\[
s_{\mathrm{final}}\le
\sqrt{s_0^2+s_{\mathrm{comp}}^2}\le\sqrt2\varepsilon<2\varepsilon.
\]

### 最终阶跃误差证明

逐时刻三角不等式和
\(\|x_0\|_C\le(1+e_0)\|x\|_C\) 给出

\[
e_{\mathrm{final}}\le e_0+(1+e_0)e_{\mathrm{comp}}
\le\sqrt\varepsilon+(1+\sqrt\varepsilon)
\frac{\sqrt\varepsilon}{1+\sqrt\varepsilon}
=2\sqrt\varepsilon.
\]

直接把压缩预算也设成 \(\sqrt\varepsilon\) 会多出 \(\varepsilon\)，因此
不能宣称恰好两倍。程序采用上式的严格预算。压缩细分单元上的界与包含
它的原始全阶认证单元上的界组合，逐个输出最终证书。

## 6. 导出、稳定性与调用

\(C_r\succ0\)，且整个域上 \(K_r(h)\succ0\)，所以导出的固定参数系统
稳定。仿射结构和所有原始温度节点的重构映射均保留。

```python
from certified_extraction import extract_certified_rom, ExtractedROM

rom = extract_certified_rom(K0, C, G, H, h_ranges, epsilon=1e-3,
                            max_enrichments=64, max_cells=16)
if not rom.report['final_certified']:
    raise RuntimeError(rom.report['reason'])
rom.save('playground/adaptive_bci_sampling/results/model_operators.npz')
loaded = ExtractedROM.load('playground/adaptive_bci_sampling/results/model_operators.npz')
x_steady = loaded.steady(h, u)
x_step = loaded.step(h, u, times)
```

设置 `PYTHONPATH=python:playground/adaptive_bci_sampling`。归档只有
\(V,K_{0,r},C_r,G_r,H_{j,r}\) 和 JSON 报告，没有原始系统矩阵或快照。
诊断失败时也可以返回候选算子，但 `final_certified=false`，不能作为满足
目标的模型使用。

## 7. 实践限制与比较口径

当前谱证书需要全阶稠密特征分解及矩阵范数；输入检查也包含 PSD 和
Cholesky 检查。这些成本都在比较中披露。大系统的可扩展证书尚未实现。

比较必须使用相同原始算子、同一参数域与同一误差目标。包括原始随机
算法、仅更换为相同稀疏直接求解器的随机对照，以及收紧随机提取容差后
使用相同原始空间证书/约束 SVD 的同目标对照。逆作用减少不能自动转述
成总时间加速，失败提取/调参成本和认证成本分别计数。

数学证明按精确算术成立；实现用普通双精度，不提供外向舍入的机器
证书。没有文献优先权、最小阶数或全域高效覆盖的主张。

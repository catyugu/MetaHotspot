# 随机挑战中的证书／响应联合任务分配

本阶段不引入共享响应空间。各输入保留自己独立的快照和残差校正。
`affine_decay.py` 的正见证仅计算衰减下界，不加入任何输入或最终 ROM 的空间。
完整全时间包围仍使用 `RANDOM_TASK_PROOF.md` 的 scalar 版本。
本文件证明新增参数衰减函数、诊断分流和自适应风险接受规则；不预先断言成本胜出。

## 1. 连续参数衰减函数

假设 $C=diag(c)>0$，$K$ 对称、非对角项非正，所有 $H_i$ 对角非负。
对任意严格正向量 $z$，定义

$$D_z(h)=diag\left((A(h)z)_k/z_k\right),\qquad
\ell_z(h)=\min_k \frac{(A(h)z)_k}{c_kz_k}.$$

由 ground-state 恒等式，任意 $x$ 满足

$$x^T(A(h)-D_z(h))x=
\sum_{k<l}-A_{kl}z_kz_l(x_k/z_k-x_l/z_l)^2\ge0.$$

于是 $A(h)\succeq\ell_z(h)C$，即使某些参数下 $D_z(h)$ 有负项，
这一不等式依然有效。保留已验证的整个盒下界 $\alpha_0>0$，任意
有限正见证集定义

$$\underline\alpha(h)=\max\left(\alpha_0,\max_{z\in\mathcal Z}\ell_z(h)\right)
\le\lambda_{min}(A(h),C).$$

每一见证的比值是仿射函数：

$$\frac{(A(h)z)_k}{c_kz_k}=
\frac{(Kz)_k}{c_kz_k}+\sum_i h_i\frac{(H_i)_{kk}}{c_k}.$$

不用拟合、插值、邻域假设或忽略遗漏方向。对所有参数同时成立，
适用于任意维数的这个对角 Robin 参数族。若只有边界行受参数影响，
内点可以精确压缩为一个最小截距，查询只访问边界系数与每个见证的截距。
见证来自一次 AMG-CG 求解 $A(h_*)z=c$；有效性只依赖有限正向量和
实际 $Kz$ 产品，不假设 CG 精确。全部见证 RHS 和时间计入提取账本。

原残差双范数仍用共同 $D_0\preceq A(h)$。能量不等式可以同时利用
$A(h)\succeq\underline\alpha(h)C$ 与 $A(h)^{-1}\preceq D_0^{-1}$：

$$\frac{d}{dt}\|E(t)u\|_C^2\le
-\underline\alpha(h)\|E(t)u\|_C^2+\|r(t)u\|_{D_0^{-1}}^2.$$

所以只替换时间传播的标量衰减率，不需为每个参数重新构造残差加权 QR。
稳态证书、初始段、分母及解析区间余项都不改变，无穷尾段使用同一新衰减率。
这里控制了整个状态空间的遗漏方向，但仍是统一状态方向包围；
不是已构造了严格低维误差动力学或方向分辨的半群。

## 2. 严格的单调性诊断与任务分流

固定空间、参数、时间分割及残差区间上界，对一个区间 $d>0$，
scalar 传播函数为

$$\Phi_C(a,b,R)=e^{-ad}b+\frac{1-e^{-ad}}aR,$$

$$\Phi_D(a,b,R)=\sqrt{e^{-ad}b^2+\frac{1-e^{-ad}}aR^2}.$$

$e^{-ad}$ 与 $(1-e^{-ad})/a=\int_0^d e^{-as}ds$ 对 $a$ 单调不增。
故两个函数对衰减率不增，对进入区间的误差界不减；它们的最小值
也保持这两种单调性。逐区间归纳，有限段界对 $a$ 不增。
尾段的 $R/\sqrt a$ 同样不增，初始段不依赖 $a$。因此完整 scalar
证书对合法衰减率单调不增。

Galerkin 最小 Ritz 值满足

$$\lambda_{min}(A(h),C)\le\lambda_{min}(V^TA(h)V,V^TCV)=:\overline\alpha_V(h).$$

将它代入证书只能作**乐观诊断**，绝不可用来接受。如果连这个上界
都不能让同一 scalar 证书通过，则任何合法衰减下界也不能单独修复
这个空间在该参数的这份证书。该结论不表示真实 ROM 误差一定超限。
如果乐观诊断通过，只表示购买新正见证可能有用，不保证新见证充分紧。

算法在独立挑战拒绝后分流：

1. 稳态拒绝：执行原先同输入稳态残差富集。
2. 时间拒绝、乐观诊断也失败，或正见证预算用尽：同输入动态残差富集。
3. 时间拒绝、乐观诊断通过、还有预算：付一次 RHS 购买正见证，更新
   $\underline\alpha$。冻结新的证书，另开独立 HTC 风险流。

默认新选项预算为零，复现上一阶段。新实验显式使用
`--certificate-mode scalar --decay-budget 12`。正见证购买最多 12 RHS，
不对参数进行无限贪心最大化，也不按参数维数形成指数角点网格。
由于矩阵版的终点上界选择没有在本阶段证明同样单调性，新分流仅启用 scalar。

发现仍用固定四提案随机 tournament／随机角点混合；每四个同输入
残差反馈 RHS 后接受新的独立全域挑战。不把有限训练池停滞作为接受。
不能据此证明所有问题都迅速脱困，或高参数维数时固定预算一定充分。

## 3. 自适应证书的风险证明

第 $k$ 次尝试在历史条件下冻结响应空间和所有正见证。取新独立
$h_1,\ldots,h_{N_k}\sim\mu$，其中

$$\delta_k=\delta/[k(k+1)],\qquad
N_k=\left\lceil\frac{\log\delta_k}{\log(1-\rho)}\right\rceil.$$

若该证书拒绝集的质量超过 $\rho$，零拒绝概率不超过 $\delta_k$。
真实联合稳态／全部时间误差超限集是证书拒绝集的子集，所以同一
结论覆盖真实超限风险。对所有自适应空间和证书尝试用条件概率及
union bound，错误接受概率不超过 $\sum_k\delta_k=\delta$。

挑战失败后追加的见证依赖该挑战；不能重新使用已经检查的样本
宣布接受。实现开启新的 epoch、新的独立流和新的 $\delta_k$。
空间未变时可复用残差 QR，但计入查询、诊断、见证及认证开销。
新购买证书不是事后、循环外的装饰；它直接改变循环里的任务分配与接受。

## 4. 先例和创新边界

Collatz–Wielandt／M-matrix 正向量特征值界已有完整理论，不是本阶段
新发现。参数 coercivity 下界的 offline-online 构造和 successive
constraint 方法也已有先例：

- Huynh, Rozza, Sen, Patera (2007),
  https://www.numdam.org/articles/10.1016/j.crma.2007.09.019/
- Glück, Mironchenko, Stability criteria for positive semigroups (2024), §6,
  https://link.springer.com/article/10.1007/s00028-024-01044-8

本阶段可检验的算法设计点是：用全时间证书的严格单调性与 Ritz
**上界**作不可能修复诊断，在新独立随机挑战中分配有限的
“买证书／买同输入响应方向”预算。是否有学术创新及更高效率，
须看同保证对照和正式规模，不能仅凭组合组件或提高下界命名创新。

所有证明为精确算术，代码普通浮点、无外向舍入。Case1 为矩阵重构，
native_validated=false。只对 SVD 前基底认证；稳态容差不放宽，时间容差
为其平方根。没有共享响应快照、没有新增单元测试套件。


## 5. 允许少量拒绝点的随时有效证据

旧零失败验收是有效但保守的检验：目标容许正质量的拒绝集，遇到
第一个拒绝点就购买新方向会把部分统计问题误处理成空间问题。
新选项 `--risk-test mixture` 在每个冻结 epoch 内保持证书不变。
设 $X_j=1$ 表示第 $j$ 个独立 HTC 的证书拒绝，真实拒绝质量为 $p$。
检验原假设 $p\ge\rho$，固定四个备择 $q\in\{0,\rho/16,\rho/4,\rho/2\}$。

$$L_n(q)=\prod_{j=1}^n
(q/\rho)^{X_j}((1-q)/(1-\rho))^{1-X_j},\qquad
L_n=\tfrac14\sum_q L_n(q).$$

在原假设下，每个因子的条件期望

$$p q/\rho+(1-p)(1-q)/(1-\rho)\le1.$$

所以每个 $L_n(q)$ 及其固定混合都是初值一的非负超鞅。
$q=0$ 分量出现一次拒绝后资本为零，不作 $0^0$ 或对数无穷相减。
由 Ville 不等式，

$$P_{p\ge\rho}\{\exists n:L_n\ge1/\delta_k\}\le\delta_k.$$

跨过这个阈值才接受；这允许拒绝点存在，同时严格保留同一个风险
目标。全部时间和输入组合仍由每个 HTC 的解析证书内部覆盖。
不是对时间抽样，也不是仅用经验超限比例接受。

计算预算 $N_{cap}=\lceil2N_{zero}\rceil$。每次可以继续收集证据，或
在即使剩余点全部通过也无法在预算内越过阈值时提前拒绝。最大未来
资本的计算用当前每个分量资本乘上其全部好点增益，仍在混合前计算。
该提前拒绝不增加错误接受概率；失败后回到第 2 节的证书／响应分流。
接受样本量是自适应停止时刻，不能再套固定样本零失败公式。
跨 epoch 仍使用 $\delta_k=\delta/[k(k+1)]$，条件于历史后用 Ville，
最后 union bound 给出总错误接受概率不超过 $\delta$。

资本测试与 time-uniform confidence sequences／betting 方法已有先例：
Howard, Ramdas, McAuliffe, Sekhon, https://arxiv.org/abs/1810.08240 ；
Waudby-Smith, Ramdas, https://arxiv.org/abs/2010.09686 。
这些统计工具不是本项目发明。这里研究的是证据、证书和响应富集的
循环内任务分配及真实成本。更多 HTC 查询可能抵消 RHS 节省，必须计时。

## 6. 保留图边的全秩逆下算子

令 $z_0$ 是第一次下端辅助求解的正向量，$D_0=diag(A_-z_0/z_0)$。
对每条无向边 $i<j$，$w_{ij}=-(A_-)_{ij}z_iz_j\ge0$，记

$$q_{ij}=e_i/z_i-e_j/z_j,\qquad
A_-=D_0+\sum_{i<j}w_{ij}q_{ij}q_{ij}^T.$$

任意保留边集 $F$ 定义

$$B=D_0+\sum_{(i,j)\in F}w_{ij}q_{ij}q_{ij}^T.$$

于是对整个 HTC 盒都有

$$0\prec D_0\preceq B\preceq A_-\preceq A(h),\qquad
A(h)^{-1}\preceq B^{-1}\preceq D_0^{-1}.$$

选择分块内边作为 $F$，$B$ 是全秩 SPD 分块算子。各块独立 Cholesky
给出精确算术下的白化 $L_B^{-1}R$，其平方 Gram 为 $R^TB^{-1}R$。
把原证书的 $D_0^{-1}$ 加权 QR 改为这个白化 QR，得到严格的稳态和
全时间误差界；每一个状态方向都包含在 $B^{-1}$ 中，没有遗漏低秩尾项。
不将普通 block Jacobi 当下算子：普通主子块仍保留跨块边的对角贡献，
一般不能证明它低于原算子。本构造只保留完整的 ground-state 边项。

对 scalar 证书，同一个固定空间的每一残差 Taylor 项和列范数余项
在新对偶范数下不增。所有能量传播界、稳态界和尾段界因此不增；
$C^{-1}$ 收缩补界不变。新图界在精确算术下不会使同一空间的 scalar
证书更弱。包含更多边的嵌套分块也保持此序关系。

Case1 原型每块覆盖完整竖向柱，水平每边 2 个网格单元；这是一项
实验性的结构选择，不是从验证数据拟合的尾项。一般对称 M-matrix
可用任意固定图分区，同一证明成立。局部因子化和残差白化必须计时，
不将本地三角解当免费；它们不构造完整模型响应或新增全局 FOM RHS。
块大小 $b$ 下，一次因子化约 $O(nb^2)$，每个冻结空间的加权残差块
白化约 $O(nbrp)$，原有加权 QR 成本仍在。大块虽严格更紧，但可能更贵。

删除图边的 PSD 比较和 support-graph preconditioning 是已知思想：
Boman, Hendrickson (2003), https://doi.org/10.1137/S0895479801390637 。
本原型将其用于全时间相对风险证书的残差对偶范数；不能将图比较
本身命名为学术创新。必须看正式规模、同保证随机对照及完整墙钟。


## 7. 正半定的粗 Robin 下更新

只在原图下算子上增加能证明为下算子的 Robin 项。令 $Q_i$ 是边界
自由度上的欧氏正交列，$F_i=H_i^{1/2}Q_i$，则
$0\preceq F_iF_i^T\preceq H_i$。因此

$$B(h)=B_0+\sum_i(h_i-a_i)F_iF_i^T\preceq A(h),\qquad B(h)\succ0.$$

未保留的边界方向不是被忽略的误差尾项：$H_i-F_iF_i^T\succeq0$
是显式算子不等式，原全秩 $B_0$ 仍控制这些方向。原型用有限个
二维余弦模式及竖向投影的输入支撑掩码形成 $Q_i$，不求任何输入
响应；这不是跨输入共享快照。每个参数的模式数固定，不形成参数网格。

若 $B_0=L_0L_0^T$，将所有 $L_0^{-1}F_i$ 联合 QR 为 $QT$，
$Q^TQ=I$，$\Delta(h)$ 是按参数重复的非负增量对角阵。对任意残差
$R$，记 $Y=L_0^{-1}R$，则

$$R^TB(h)^{-1}R=
Y^T(I-QQ^T)Y+(Q^TY)^T(I+T\Delta(h)T^T)^{-1}(Q^TY).$$

实现对白化残差正交余部作 QR，保留其全部 R 因子，再对小矩阵
$I+T\Delta T^T$ 作 Cholesky；无需相减两个几乎相等的误差 Gram。
每个 HTC 查询只更新一个小三角因子和原有约化动力学。
该精确恒等式不是独立创新，相关低秩逆更新已有 Woodbury 理论。
新增部件是否值得保留由正式成本判决决定，不能仅凭下界更紧宣布胜出。

## 8. 具有完整空间保证的二部图 modified LDL 下算子

图块下界丢掉块间的热传导方向，可能在细网格上变松。另一个原型
保留全局稀疏三角结构，而不构造辅助响应空间。令
$Z=diag(z_0)$，$S=ZA_-Z$；其非对角项非正且行和
$\Gamma_i=z_i(A_-z)_i>0$。假设原图为二部图，Case1 的三维近邻网格满足它。

第 $i$ 个消元步，pivot $d_i>0$，向后邻居列为 $a_j\le0$。
精确 Schur 补生成邻居间的边，权重 $a_ja_k/d_i\ge0$。二部图保证
原结构内没有这些邻居间边；原型丢弃每个**完整**新拉普拉斯边项，
故近似 Schur 补低于精确 Schur 补。保留原稀疏非对角项，更新

$$d_j\leftarrow d_j-a_j\sum_k a_k/d_i.$$

更新后的行和为 $\Gamma_j-a_j\Gamma_i/d_i>0$，所以每一步仍是
严格 grounded M-matrix；全部 pivot 为正，因子存在。以每一步精确
消元的三角 congruence 归纳，得到

$$A_-\succeq B_{MIC}:=Z^{-1}L\operatorname{diag}(d)L^TZ^{-1}\succ0.$$

事实上 $B_{MIC}\succeq D_0$。对余下问题归纳假设其下因子覆盖
对角行和。令 $w=-a\ge0$、$\sigma=\sum_jw_j$、$d_i=\Gamma_i+\sigma$。
本步下因子与原对角行和之差的 Schur 补为

$$\frac{\Gamma_i}{d_i}\left(diag(w)-ww^T/\sigma\right)\succeq0.$$

它是加权方差矩阵；$\sigma=0$ 的孤点单独成立。逆序归纳完成
$D_0\preceq B_{MIC}\preceq A_-\preceq A(h)$ 的证明。MIC 与分块图
下界之间没有在此证明互相的序关系，但它们都不弱于原纯对角逆界。

白化只需

$$\operatorname{diag}(d)^{-1/2}L^{-1}ZR.$$

这是完整状态尺寸的稀疏三角应用，不能当免费；账本记录应用的
RHS 列数和耗时，与原始 $A(h)+sC$ 的 AMG-CG RHS 分开。复杂度
约 $O(nnz(L)rp)$；初始因子对本二部图结构为线性规模，不要求完整
模型稠密 Cholesky 或原系统逆像辅助库。可叠加第 7 节的正 Robin 下更新。

modified incomplete Cholesky 及行和补偿已有先例，不是本项目发明：
Gustafsson, A class of first order factorization methods, BIT 18 (1978), 142–156；
Gustafsson, On modified incomplete cholesky factorization methods (1979),
https://doi.org/10.1002/nme.1620140803 。本阶段证明的是这里实际实现的
二部图、正向量缩放和丢弃完整边构造的下算子性质，并在完整时间风险
循环中检验其效益，不把一般 IC 因子自动当作经过证明的逆上界。

# 随机任务发现、同输入残差反馈与全时间联合风险认证

实现：`random_task_extraction.py`、`diagonal_time_certificate.py`。
本文件证明接受规则；**不证明提取成本必然优于随机法**。
阶段判决及复现见 `RANDOM_TASK_GATE_20261010.md`。

## 1. 问题与量词

固定 HTC，零初值，任意有符号常数输入组合。设

$$C=\operatorname{diag}(c)>0,\quad
 A(h)=K_0+\sum_{i=1}^p h_iH_i,\quad h\in[a,b],$$

其中 $H_i\succeq0$，$A_-=A(a)$ 是对称正定 M-matrix。
输入矩阵 $G$ 列独立。所有认证直接针对 SVD 前固定空间 $V$。
令 $X=A(h)^{-1}G$，$X_V=V(V^TAV)^{-1}V^TG$，$S(t)$、$S_V(t)$
分别为完整和 Galerkin 阶跃响应矩阵。

$$e_s(h)=\sup_{w\ne0}\frac{\|(X-X_V)w\|_{A(h)}}{\|Xw\|_{A(h)}},
\qquad e_t(h)=\sup_{t>0,w\ne0}
\frac{\|(S(t)-S_V(t))w\|_C}{\|S(t)w\|_C}.$$

目标是对各维独立 log-uniform 分布 $\mu$，以至少 $1-\delta$ 的
置信度保证

$$\mu\{h:e_s(h)>\epsilon\ \text{或}\ e_t(h)>\sqrt\epsilon\}\le\rho.$$

这里风险只放在 HTC 参数上；**时间和输入组合在风险事件内已取全部**。
因此不同于对 $(h,t)$ 联合采样后声称“几乎所有时间”通过。
它仍不是整盒每个参数都满足容差，也不覆盖随时间变化的 HTC 或功率。

## 2. 一次辅助求解给出的确定性对偶界

求一个正向量 $z$；实现使用一次 AMG-CG 求解 $A_-z=c$。
实际计算后检查 $z>0$、$A_-z>0$，定义

$$D=\operatorname{diag}\left(\frac{(A_-z)_i}{z_i}\right),
\qquad \alpha=\min_i D_{ii}/c_i>0.$$

无需假设这次 CG 是精确求解。对任意向量 $x$，直接展开有

$$x^T(A_--D)x=
 \sum_{i<j}(-A_{-,ij})z_i z_j
 \left(\frac{x_i}{z_i}-\frac{x_j}{z_j}\right)^2\ge0.$$

所以对原始整个连续盒同时有

$$A(h)\succeq A_-\succeq D\succeq\alpha C,
\qquad A(h)^{-1}\preceq D^{-1}.$$

辅助求解缺陷已经通过实际乘积 $A_-z$ 留在 $D$ 中；不是用未核查的
Ritz 最小值当衰减下界。这个对角界可以很松，且不适用于不满足
M-matrix 条件的一般混合有限元系统。它不是共同低秩响应空间。

## 3. 稳态任意输入证书

取 $V^TCV=I$，$F=V^TG$、$A_r=V^TAV$，令

$$R_s=G-AV A_r^{-1}F,\quad Q=F^TA_r^{-1}F,
\quad B=R_s^TD^{-1}R_s.$$

Galerkin 正交给出真实误差能量 Gram
$E_s=R_s^TA^{-1}R_s\preceq B$，完整分母 Gram 为 $Q+E_s$。
若 $Q>0$，令 $\ell=\lambda_{\max}(B,Q)$，则对任意 $w$

$$\frac{w^TE_sw}{w^T(Q+E_s)w}
\le \frac{\ell}{1+\ell},\qquad
U_s(h)=\sqrt{\ell/(1+\ell)}.$$

不对逐源相对误差取最大来替代任意输入抵消组合。

## 4. 直接认证阶跃残差，无完整时间步进

将约化矩阵对角化 $A_rU=U\Lambda$，$W=VU$、$B_r=W^TG$，
$f_\lambda(t)=(1-e^{-\lambda t})/\lambda$。约化阶跃为
$S_V(t)=W f_\Lambda(t)B_r$。
实际强迫残差满足精确恒等式

$$r(t)=G-CS_V'(t)-AS_V(t)
 =R_0+J f_\Lambda(t)B_r,
\quad R_0=G-CW B_r,\quad J=CW\Lambda-AW.$$

初始斜率 $C^{-1}G$ 放入各自输入空间，无需额外 RHS；理论上 $R_0=0$。
实现不把算得的 $R_0$ 清零。

完整误差 $E(t)=S(t)-S_V(t)$ 满足 $CE'+AE=r$，$E(0)=0$。
对任意固定输入 $w$，收缩半群和 Young 不等式分别给出

$$\frac{d}{dt}\|Ew\|_C\le-\alpha\|Ew\|_C+\|r w\|_{C^{-1}},$$

$$\frac{d}{dt}\|Ew\|_C^2
\le-\alpha\|Ew\|_C^2+\|r w\|_{D^{-1}}^2.$$

第二式来自 $-2e^TAe+2e^Tr\le-e^TAe+r^TA^{-1}r$。
两个合法界取小，避免早期单用能量界产生 $\sqrt t/t$ 的假奇点。

用 $Z$ 白化 $G^TC^{-1}G$，即 $Z^TG^TC^{-1}GZ=I$。
下文所有残差矩阵右乘同一个 $Z$，误差界是 $E(t)Z$ 的算子范数。
因 $W^TCW=I$，

$$d(t)=\sigma_{\min}(f_\Lambda(t)B_r Z)
\le\inf_{\|v\|=1}\|S_V(t)Zv\|_C.$$

实际上等号成立。$S_V(t)^TCS_V(t)$ 随 $t$ Loewner 单调增加，
所以 $d(t)$ 单调不减。当 $\|E(t)Z\|_C\le b(t)<d(t)$，三角不等式给出

$$e_t(h,t)\le\frac{b(t)}{d(t)-b(t)}.$$

这一步处理的是完整响应分母，不将约化响应直接冒充真实分母。

## 5. 初段、解析时间区间、无穷尾段

### 初段 $[0,t_0]$

令 $r_0=\|C^{-1/2}R_0Z\|_2$，
$L_0=\sum_j\|C^{-1/2}J_j\|_2\|(B_rZ)_{j,:}\|_2$。
由 $|f_\lambda(t)|\le t$，$\|E(t)Z\|_C\le tr_0+t^2L_0/2$。
又 $f_\lambda(t)/t$ 单调递减，所以整个初段都有

$$\frac{\|E(t)Z\|_C}{d(t)}
\le\frac{t_0r_0+t_0^2L_0/2}{d(t_0)}.$$

这同时覆盖相对误差的零时刻右极限，不计算 $0/0$。

### 中间区间 $[a,b]$

令 $m=(a+b)/2$，$\Delta=(b-a)/2$。分别取 $M=C$ 或 $D$。
残差前三阶 Taylor 项加第四阶余项给出对区间上**每个**时间的界

$$\|M^{-1/2}r(t)Z\|_2\le
 \sum_{k=0}^3\frac{\Delta^k}{k!}
 \|M^{-1/2}r^{(k)}(m)Z\|_2
 +\frac{\Delta^4}{24}\sum_j
 \|M^{-1/2}J_j\|_2\|(B_rZ)_{j,:}\|_2\lambda_j^3e^{-\lambda_j a}
 =q_M.$$

因为 $r^{(k)}(t)=J(-\Lambda)^{k-1}e^{-\Lambda t}B_r$ 对 $k\ge1$ 成立，
余项完全来自约化谱；不需要完整谱，也不是把中点当时间采样验收。
保留每一 Taylor 项中的带符号矩阵乘积，只在余项中作列范数三角界。

若起点误差界为 $b_a$，令 $v=(1-e^{-\alpha(b-a)})/\alpha$，则

$$b_b=\min\left\{e^{-\alpha(b-a)}b_a+v q_C,
\sqrt{e^{-\alpha(b-a)}b_a^2+v q_D^2}\right\}.$$

整个区间的误差不超过 $\max(b_a,b_b)$。证明：两条标量包络从同一
$b_a$ 起步，分别单调趋向自己的平衡值；若任一包络递减，它在整个
区间不超过 $b_a$；否则两条都递增，各自不超过自己的末端值。
因此整个区间的 $b/d$ 比值不超过 $\max(b_a,b_b)/d(a)$。

### 无穷尾段 $[T,\infty)$

稳态残差 $R_s=R_0+J\Lambda^{-1}B_r$，而

$$\|D^{-1/2}r(t)Z\|_2\le
\|D^{-1/2}R_sZ\|_2+
\sum_j\|D^{-1/2}J_j\|_2\|(B_rZ)_{j,:}\|_2 e^{-\lambda_jT}/\lambda_j
=q_\infty.$$

能量不等式给出整个尾段误差不超过
$\max(b_T,q_\infty/\sqrt\alpha)$，分母至少 $d(T)$。
实现取 $T=40/\lambda_{r,\min}$，**没有丢弃**指数尾项。
初段、所有中间区间、无穷尾段的最大 $b/d$ 为 $\eta$。
当 $\eta<1$ 时 $U_t(h)=\eta/(1-\eta)$ 上界全部时间与全部输入组合。

## 6. 全阶成本与约化查询

基更新后的残差算子只含

$$\mathcal R=[G-CVF,\ CV A_{r,0}-K_0V,
\ CV A_{r,1}-H_1V,\ldots,CV A_{r,p}-H_pV].$$

对 $C^{-1/2}\mathcal R$、$D^{-1/2}\mathcal R$ 作稳定 QR，只保留三角
因子。每个新 HTC 的时间界只需约化谱、小矩阵乘法和奇异值。
不存在每个参数、每个时间的 FOM 逆作用。
但 QR 更新成本约 $O(n((p+1)r+m)^2)$，查询还涉及 $O(r^3)$ 约化谱和
时间区间数；**一次辅助 RHS 并不意味着认证墙钟成本免费**。
实现当前在每次接受尝试重建 QR，并完整计时。

## 7. 从原始算子到交付算子的提取循环

1. 各输入只拥有自己的空间 $V_j$；加入自己的 $C^{-1}g_j$ 初始斜率。
   用原生产算法的移位计划作发现支持集，报告共同谱计划成本。
2. 每轮只提出固定数目的新 HTC：以概率 $1-\theta$ 从 $\mu$ 抽样，
   以概率 $\theta$ 抽取各坐标独立 Bernoulli 的角点。对每个候选仅在
   一个输入的有限移位集上作约化残差检查，取小批量中的最大者富集。
   没有固定池、连续优化、参数网格或 $2^p$ 角点搜索。
3. 在检查轮临时拼接各 $V_j$ 与常数方向构成交付空间 $V$，认证上述
   **实际联合 Galerkin 模型**。末尾拼接是原算法允许的交付动作，不是
   用共享空间训练某一输入。
4. 稳态界失败：在同一参数计算各自局部稳态残差，选对角对偶能量
   相对指标最大的输入，解 $A(h)z_j=r_{s,j}$，只加入其自己的空间。
   精确算术下 $x_{V_j}+z_j=A^{-1}g_j$；因此这一方向直接修补当前
   稳态任务，而不是在其他频率反复提取同一个原输入。
5. 全时间界失败：第一版取首个未通过的解析区间所给时间 $\tau$。当前
   默认在已访问区间中最大化强迫贡献代理
   $q_{D,k}^2(b_k-a_k)e^{-\alpha(T-b_k)}$，取其中心时间为 $\tau$；
   $T$ 是首个失败时间。该代理受能量卷积启发，不参与通过判定，也
   不是已计算的真实误差贡献。`--time-feedback failure` 保留第一版。
   在所选 $\tau$ 计算每个
   输入自己的局部动态残差 $r_j(\tau)$，按 $C^{-1}$ 相对残差选择输入，
   解 $(A(h)+\tau^{-1}C)z_j=r_j(\tau)$，只加入其自己的空间。
   这是方向选择启发式；动态误差没有嵌套单调性或必降定理，随后仍
   必须重新认证。每四个反馈 RHS 重新检查。
6. 候选冻结后作本节 8 的独立随机接受检查。一旦出现失败立即返回
   步骤 4/5；验收就在提取循环内，而非循环外的装饰。
7. 接受后交付 $V^TK_0V$、$V^TCV$、$V^TH_iV$、$V^TG$。
   本阶段不做 SVD 截断；若以后截断，必须重新认证实际最终空间。

不证明阶段 5 的代价总优于原输入快照；这由正式模型和消融比较判决。
资源耗尽、依赖富集、非正下界均不得返回通过。

## 8. 自适应循环中的合法风险预算

第 $k$ 次接受尝试分配 $\delta_k=\delta/[k(k+1)]$；总和为 $\delta$。
该轮空间、对角界和全部时间算法冻结，随后用新独立 $\mu$ 样本查询
联合界 $U(h)=\max\{U_s(h)/\epsilon,U_t(h)/\sqrt\epsilon\}$。
默认不要求角点通过：连续分布中有限角点的质量为零，不能将整盒式
额外准入混入分布风险目标。`--accept-corners` 保留第一版的额外拒绝
检查（高维最多 16 个随机角点），不产生角点覆盖声明。

取

$$N_k=\left\lceil\frac{\log\delta_k}{\log(1-\rho)}\right\rceil.$$

所有 $N_k$ 个随机检查均通过才接受。若冻结候选的真实联合超限风险
大于 $\rho$，证书失败集合的概率至少为 $\rho$，所以条件于历史的
错误接受概率不超过 $(1-\rho)^{N_k}\le\delta_k$。
对全部自适应轮次作并集界，得到第 1 节的联合风险结论。
被拒绝的随机参数可用于后续富集，但不能复用其余样本作为新轮接受数据。

在随机发现阶段，固定提案预算与 $\mu$ 密度下界避免局部优化器独占搜索：对任意正质量
参数集合 $B$，单个发现提案命中它的概率至少 $(1-\theta)\mu(B)$。
反馈阶段本身可以连续作局部校正，不声称每个富集 RHS 都独立全域抽样。
默认每四个反馈 RHS 后重新从整个 $\mu$ 抽取独立的联合风险挑战，
没有某一个角点长期阻塞随机挑战。每轮至少一个新 $\mu$ 检查参数，
对固定正质量集合 $B$，预先独立生成的前 $k$ 轮首样本都遗漏它的概率
为 $(1-\mu(B))^k$；“实际执行到 $k$ 轮且这些样本都遗漏”的联合概率
不超过这个值，不将条件于自适应停止的样本误当无条件独立样本。
更关键的是，每次停止都要接受独立的全时间联合风险挑战，不能把固定
池通过或局部残差停滞包装成连续域通过。
**这是防止不安全停止的定理，不是所有问题上快速终止或最优复杂度定理。**

## 9. 数值与文献边界

证明为精确算术。普通浮点 QR、特征值、实际乘积尚未使用外向舍入；
因此不能声称机器级严格认证。Case1 是矩阵重构，native_validated=false。
辅助求解和富集均为 AMG-CG；只有小模型独立审计允许完整谱分解。

- [Grepl–Patera 2005](https://numdam.org/articles/10.1051/m2an:2005006/)：
  仿射抛物问题的残差认证、参数时间自适应与离线在线分离已有先行工作。
- [Cohen–Dahmen–DeVore–Nichols 2020](https://www.numdam.org/articles/10.1051/m2an/2020004/)：
  随机训练集替代昂贵离散网格已有理论，不能把小批量随机最大值当新思想。
- [Billaud-Friess 等 2023，第 3 节及附录 A](https://arxiv.org/html/2304.08784v3)：
  PAC bandit 与概率弱贪心已有分析；本文没有复制其全域准最优 maximizer 结论。
- [Hesthaven–Stamm–Zhang 2014](https://www.numdam.org/item/M2AN_2014__48_1_259_0/)：
  高维贪心搜索加速已有 saturation 型方案，本实现不用 saturation 假设。
- [Druskin–Simoncini，自适应有理 Krylov，正文第 2 节](https://www.dm.unibo.it/~simoncin/ratkr5.pdf)：
  自适应移位、残差相关的有理空间构造已有先行方法；不能把可变移位
  或减少原输入快照数独立当新颖性。
- [Botchev–Grimm–Hochbruck，正文第 4–5 节](https://na.math.kit.edu/download/papers/BotchevGrimmHochbruck12.pdf)：
  矩阵指数的误差 ODE、半群残差界、Richardson 校正和重启已有分析。
  因此本轮的时间残差校正本身不能独立宣称原创。
- [Druskin–Simoncini–Zaslavsky，正文第 3 节](https://www.dm.unibo.it/~simoncin/tangadapt6.pdf)：
  MIMO 有理空间的方向和移位自适应已有方法，也有 AMG-PCG 规模比较。
  本原型保持各源独立，没有将其共享切向空间直接移植作训练器。

对角 ground-state 恒等式、半群能量界、Taylor 余项及序贯概率并集界都
是标准工具。可以进一步审查的具体贡献候选，是一次辅助求解、保留残差
抵消的全时间相对界、同输入时间残差校正及循环内联合风险接受的组合。
本轮不凭“未检索到完全相同标题”宣称学术新颖性或普遍成本优势。
## 第二版：保留输入方向的矩阵时间包围

标量化可制造任意大的输入条件数损失。取误差上界 Gram
$B=\operatorname{diag}(\eta^2 L^2,\eta^2)$、响应 Gram
$P=\operatorname{diag}(L^2,1)$，广义相对界为 $\eta$，
独立最大分子／最小分母却给出 $\eta L$。第二版保留输入维数
$p\times p$ 的误差 Gram。此例解释结构损失，不证明新包围处处更紧。

在区间 $[a,b]$，令 $m=(a+b)/2,d=(b-a)/2$。加权残差矩阵在
$m$ 的三阶 Taylor 多项式的四个 Bernstein 控制矩阵是

$$
\begin{aligned}
U_0&=r-dr_1+d^2r_2/2-d^3r_3/6,\\
U_1&=r-dr_1/3-d^2r_2/6+d^3r_3/6,\\
U_2&=r+dr_1/3-d^2r_2/6-d^3r_3/6,\\
U_3&=r+dr_1+d^2r_2/2+d^3r_3/6,
\end{aligned}
$$

其中 $r_k$ 是第 $k$ 阶导数。Bernstein 权重非负且和为一，故对每个
输入向量 $u$，凸性给出 $\|r_{poly}(t)u\|^2\le\max_k\|U_ku\|^2$。
从 $M=U_0^TU_0$ 开始，依次令
$M\leftarrow M+(U_k^TU_k-M)_+$；对称正部保证保留以前的
Loewner 上界并覆盖新的控制 Gram。四阶余项算子范数 $e$ 使用
下文同一解析余项，绝不丢掉。令 $s=\sqrt{\operatorname{tr}M}$，
矩阵 Young 不等式给出

$$r(t)^Tr(t)\preceq(1+e/s)M+(e^2+es)I=:\widehat M.$$

$s=0$ 时用 $e^2I$。分别得到 $C^{-1}$ 和 $D^{-1}$ 权重包围。
白化输入下真实误差 Gram $B_E=Z^TE^TCEZ$ 满足
$B_E'\preceq-\alpha B_E+\widehat M_D$，因为能量不等式对所有输入
方向同时成立。若 $B_E(a)\preceq B$，终点上界为
$B_d=e^{-\alpha(b-a)}B+(1-e^{-\alpha(b-a)})\widehat M_D/\alpha$。
区间内上界曲线是 $B$ 和 $B_d$ 的凸组合。
响应 Gram $P(t)=F^Tf_\Lambda(t)^2F$ 单调增加，整个区间
相对约化响应的平方界不超过

$$\max\{\lambda_{max}(B,P(a)),\lambda_{max}(B_d,P(a))\}.$$

同时计算有效的标量 $C$ 收缩界；区间可以取两种有效相对界较小值。
终点只选择一个完整有效上界：$B_d$ 或标量收缩产生的 $B_c=b_1^2I$，
按它们相对于 $P(b)$ 的最大广义特征值选一个。不能逐元素取最小。
初始段沿用误差除以 $t$ 的界和响应除以 $t$ 的单调性；无穷尾段对
稳态残差 Gram 加上保留指数余项的 Young 包围 $\widehat M_\infty$，
使用 $B(T)$ 与 $\widehat M_\infty/\alpha$ 两个端点和分母 $P(T)$。
最后仍由 $\eta/(1-\eta)$ 转成相对真实响应界，风险证明不变。

`--certificate-mode matrix` 是本实验入口默认，`scalar` 复现第一版。
输入 Gram 的保留没有消除共同衰减率的状态方向损失，也没有消除
高参数维数的仿射 QR 成本。普通浮点仍没有外向舍入。

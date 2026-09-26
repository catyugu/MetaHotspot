# 理论：命题与证明

本文件把 `adaptive_bci_sampling` 依赖的数学陈述逐条写出，**给出完整证明**，并明确区分三类状态：

* **已证**：下面给出完整证明，只依赖标出的假设。
* **已引用**：依赖外部定理，给出出处与本文件用到的形式。
* **未证**：论文缺口，第 9 节列出并说明难在哪。

另需分清两件事：本文件证明的是**数学陈述**；代码是否实现这些陈述，靠 `certify_extraction.py` 的全盒审计与单元测试来核对，那是数值核验，不构成证明。两者不可互相替代。

---

## 0. 记号与假设

单体热传导 + Robin 边界，有限体积离散后

```text
A(p, s) = K + s C + sum_{i=1..d} p_i H_i,     p in P = prod_i [p_i^-, p_i^+],  s >= 0,
```

`K = K^T` 是内部导热矩阵，`C` 是集中热容（对角、正），`H_i = B_i B_i^T` 是第 `i` 组
Robin 边界项。单调性一节额外需要：

* **(H1)** `K_ij <= 0`（`i != j`），`K = K^T`，`K psd`；
* **(H2)** `H_i = diag(a_i)`，`a_i >= 0`，`H_i` 非负半定（对一般 FEM 的边界质量阵写成
  `H_i = B_i B_i^T` 即可，不变式照旧）；
* **(H3)** `G >= 0`（源形状非负）；
* **(H4)** 存在 `p^- > 0` 使 `A(p^-, s)` 非奇异，即 Robin 项钉住了常数模态。

记 `A_ref(s) = A(p^-, s)`、`V` 为交付基底（`n x m`，列正交）、
`X_V = V (V^T A(p,s) V)^-1 V^T G`、`Y = G^T A^-1 G`、`Y_V = G^T X_V`、
`R(p,s) = G - A(p,s) X_V`、`e = A^-1 G - X_V`。

---

## 1. 命题 1（固定频移的 Galerkin 输出误差恒等式）—— 已证

**陈述.** 对任意 SPD 的 `A` 与任意 `V`，

```text
(Y - Y_V)_ab = R_a^T A^-1 R_b = e_a^T A e_b,
```

右端是精确等式而非上界，并且在 `A` 内积下 Cauchy--Schwarz 给出

```text
abs((Y - Y_V)_ab) <= sqrt(Delta_aa Delta_bb),   Delta_ab = R_a^T A^-1 R_b.
```

**证明.** `R^T A^-1 R = (G - A X_V)^T A^-1 (G - A X_V)
= G^T A^-1 G - G^T X_V - X_V^T G + X_V^T A X_V`。第三、四项给出
`X_V^T A X_V = G^T V (V^T A V)^-1 V^T G = G^T X_V = Y_V`，故
`R^T A^-1 R = Y - 2Y_V + Y_V = Y - Y_V`。又 `e = A^-1 R`，所以
`e_a^T A e_b = R_a^T A^-1 R_b`。PSD 性与 Cauchy--Schwarz 立得。证毕。

**推论.** `Y - Y_V` 是 `4 x 4`（或 `k x k`）半正定矩阵，`Delta` 也是；因此
对角项 `Delta_aa` 单独就是端口 `a` 的精确输出误差。

---

## 2. 命题 2（Woodbury 精确参数误差映射）—— 已证

**陈述.** 设 `A_ref = A(p^-, s)`，活动指标集 `J = {j : delta_j > 0}`，
`delta_j = sum_i (p_i - p_i^-) a_{i,j}`，`B_J` 为对应列。则

```text
A(p,s)^-1 = A_ref^-1 - A_ref^-1 B_J (D_J^-1 + B_J^T A_ref^-1 B_J)^-1 B_J^T A_ref^-1,
D_J = diag(delta_J),
```

并且 `A(p,s) V` 落在**与 p 无关**的张成空间

```text
Z = [ G, (K + sC)V, H_1 V, ..., H_d V ]
```

中，残差系数为 `zeta = [ e_a ; -q ; -p_1 q ; ... ; -p_d q ]`，`q = (V^T A(p,s)V)^-1 V^T G`。
于是

```text
Delta_ab(p,s) = zeta_a(p)^T [ Z^T A(p,s)^-1 Z ] zeta_b(p),
```

方括号由两张预算表 `Z^T A_ref^-1 Z`、`Z^T A_ref^-1 B` 经稠密代数得到。

**证明.** 第一式是 Sherman--Morrison--Woodbury 恒等式
`(A_ref + B_J D_J B_J^T)^-1 = A_ref^-1 - A_ref^-1 B_J (D_J^-1 + B_J^T A_ref^-1 B_J)^-1 B_J^T A_ref^-1`
的直接应用（`D_J` 可逆，因为活动集上 `delta_j > 0`）。第二式：展开
`A(p,s)V = (K + sC)V + sum_i p_i H_i V`，而
`R_a = G_a - A(p,s) V q_a`，故 `zeta` 如上。第三式由命题 1 代入即得。证毕。

**注.** 非活动列的贡献已经含在 `A_ref` 里（`p_i = p_i^-` 时该列不扰动），因此该式是
精确的，不是近似。

---

## 3. 命题 3（HTC 族是对称 M-matrix 族，传递单调）—— 已证

**陈述.** 在 (H1)--(H4) 下，对每个 `p in P`：

1. `A(p,s)` 是**非奇异对称 M-matrix**，因而 `A(p,s)^-1 >= 0`（逐元素）；
2. `x_a = A(p,s)^-1 G_a >= 0`；
3. `dY_ab/dp_k = -x_a^T H_k x_b <= 0`，**逐元素**。

**证明.** (1) `A(p,s) = K + sC + sum_i p_i H_i`，其中 `sC + sum_i p_i H_i` 是对角非负，
故 `A` 的每个非对角元等于 `K` 的对应元，由 (H1) 非正，即 `A` 是 Z-matrix。`A` 对称且由
(H4) 非奇异；SPD 矩阵的全部主子式为正（Sylvester），故 `A` 是 P-matrix。Z-matrix 与
P-matrix 的矩阵即非奇异 M-matrix，而**非奇异 M-matrix 的特征刻画正是
`A^-1 >= 0` 逐元素成立**。SPD 部分由 (H1)(H2) 与 `s >= 0` 直接给出。
(2) 由 (H3) 与 (1)，`x_a = A^-1 G_a` 是非负矩阵乘非负向量。
(3) 对 `A x_a = G_a` 求导得 `A (dx_a/dp_k) = -H_k x_a`，故 `dx_a/dp_k = -A^-1 H_k x_a`。
又 `Y_ab = G_a^T A^-1 G_b = x_a^T G_b`，而 `G_b` 与 `p` 无关（求导只有一项，没有转置项、
也没有因子 2），所以
`dY_ab/dp_k = (dx_a/dp_k)^T G_b = -x_a^T H_k A^-1 G_b = -x_a^T H_k x_b`。
由 (H2) `H_k = diag(a_k)` 与 (2) 的 `x >= 0`，
`x_a^T H_k x_b = sum_j (a_k)_j x_{a,j} x_{b,j} >= 0`。证毕。

**为什么必须写成 M-matrix 假设.** 只用 `H_k psd` 只能得到对角情形的
`x_a^T H_k x_a >= 0`，**不能**推出 `a != b` 的 `x_a^T H_k x_b >= 0`；那一步完全依赖
(1) 给出的逆的非负性。对一般 FEM 离散（边界质量阵非对角）(H1) 未必成立，此时必须
声明该假设，或只对自项使用单调分母、互项改用别的归一化。

**假设在本模型上已核对**（2.5 mm，`n = 9072`，`H_0` 支撑 64 单元、`H_1` 支撑 1008 单元）：

```text
K 非对角元最大値            0.000e+00      -> (H1) 的符号条件成立
K 的 (K - K^T) 非零元个数   0              -> 对称
H_i 非对角非零元个数        0 / 0          -> (H2)
H_i 对角元下界              >= 0           -> (H2)
G 的最小元                  0.000e+00      -> (H3)，非负
A(p^-)^-1 G 的最小元        1.385e+02 > 0  -> 逆的逐元素非负性
```

---

## 4. 命题 4（Gamma-夹逼 ⇒ 残差贪心是弱贪心）—— 已证

**陈述.** 令 `A_- = A(p^-, s)`、`Gamma = max_i p_i^+ / p_i^-`。则 `Gamma >= 1` 且

```text
A(p,s)^-1 <= A_-^-1 <= Gamma A(p,s)^-1      (Loewner),
```

从而对任意残差 `R`，以 `eta^2 = R^T A_-^-1 R` 作代理量有

```text
norm(e)_{A(p,s)} <= eta <= sqrt(Gamma) norm(e)_{A(p,s)},
```

并且以固定范数 `norm(.)_{A_-}` 为环境空间时，代理量与原误差双边等价：

```text
norm(e)_{A_-} <= eta <= Gamma norm(e)_{A_-}.
```

于是该贪心是弱贪心，weakness constant 可取 `gamma = 1 / Gamma = min_i p_i^- / p_i^+`。

**证明.** `p_i <= Gamma p_i^-` 逐分量成立，且 `K psd`、`Gamma >= 1`，故
`A(p,s) = K + sC + sum p_i H_i <= Gamma K + sC Gamma + Gamma sum p_i^- H_i`。
需要注意 `sC <= Gamma sC` 也成立，所以
`A(p,s) <= Gamma A(p^-, s) = Gamma A_-`。另一侧 `A_- <= A(p,s)` 由 `p >= p^-`。
取逆并用 Loewner 单调性得第一式。对第一式两边作 `R^T (.) R`：
左端 `R^T A(p)^-1 R = norm(e)^2_{A(p)}`，中项 `eta^2`，右端 `Gamma norm(e)^2_{A(p)}`，
开方即得第二式。第三式：由 `A_- <= A(p) <= Gamma A_-` 有
`norm(e)^2_{A_-} <= norm(e)^2_{A(p)} <= Gamma norm(e)^2_{A_-}`，与第二式复合即得。证毕。

**引用（weak greedy 的速率传递）.** 有了 `gamma` 与固定环境空间的代理量等价，即可用
DeVore--Petrova--Wojtaszczyk, *Constr. Approx.* 37(3):455--466 (2013), Thm 3.2 与
Cor. 3.3：`eps_{2n}(F) <= 2 gamma^-1 d_n(F)`，即弱贪心以常数 `2/gamma` 继承
Kolmogorov `n`-width 速率。

**这条定理在本问题上目前是空的，原因必须写明.** 本模型的
`Gamma = max_i p_i^+ / p_i^- = 9285.79`，故定理给出的常数为 `2/Gamma^-1 = 1.9e4`，
在 `1e-3` 目标上不构成任何有用陈述。而实测的双边比只有

```text
eta^2 / norm(e)^2_{A(p)}  in [1.107, 9.102]     预测 [1, 9285.79]
eta   / norm(e)_{A(p)}    in [1.052, 3.017]     预测 [1, 96.36]
```

即真实等价常数约 `1--3`，比定理保守值小三个数量级。**要让这条定理非空，必须把全局锚点
换成单元局部锚点**：按对数划分把 `P` 分成每轴 `k` 份，单元内
`Gamma_alpha = 10^{(log10 p^+ - log10 p^-)/k}`，`k = 8` 时 `Gamma_alpha = 3.16`，
`2/gamma_alpha = 6.3`；`k = 16` 时 `Gamma_alpha = 1.78`。所以
**branch-and-bound / 单元局部证书不是实现细节，而是让速率定理非空的前提**。

---

## 5. 命题 5（锚定 Riesz 单元证书是有效上界）—— 已证

**陈述.** 设单元 `Q = [low, high]`，`a = low`。以任意**多项式**试验
`q(.) : Q -> R^{m x k}` 构造岭系数 `zeta(.)`（对多项式 `q`，`zeta` 是多项式），令
`S = Z^T A(a)^-1 Z`。则对每个 `p in Q`，

```text
Delta(p) <= zeta(p)^T S zeta(p) <= max_{nu in Bernstein 节点} zeta(nu)^T S zeta(nu),
```

右端可算，且 `abs((Y - Y_V)_ab(p)) <= sqrt(Delta_aa(p) Delta_bb(p))`。

**证明.** 第一步是 Galerkin 最优性：真系数 `q_*(p)` 使残差能量最小，故对任意试验
`Delta(p) = zeta_*(p)^T (Z^T A(p)^-1 Z) zeta_*(p) <= zeta(p)^T (Z^T A(p)^-1 Z) zeta(p)`。
第二步是 Loewner：`p >= a` 逐分量给出 `A(p) >= A(a)`，故 `A(p)^-1 <= A(a)^-1`，
于是 `Z^T A(p)^-1 Z <= S`，代入即得。第三步是凸性：`zeta` 多项式，其张量 Bernstein 系数是
非负组合，`xi^T S xi` 在 `xi` 上凸（`S psd`），故二次型在单元上的上确界被 Bernstein
节点值控制；`sweep` 正是在这些节点上取极大。最后一步是命题 1 的 Cauchy--Schwarz。证毕。

**推论（为什么审计比较的是绝对量）.** 上式的逐项相对量需要除以 `abs(Y_V,ab)`，而弱耦合
项的自身分母可以任意小，所以逐项相对界必然松；`steady_absolute_bound` 与
`steady_relative_bound` 是两个不同的量，**不可互相比较**（文档曾在此处出过错，已修）。

---

## 6. 命题 6（种子位置不影响正确性）—— 已证

**陈述.** 交付基底 `V` 与其证书只依赖：(i) 被选中的参数点集合，(ii) 快照与压缩规则。
种子规则只影响 (i) 中选了哪些点。因此把 `coordinate_spectral_enclosures` 的
`eigsh + (1 +- safety)` 换成任何别的（哪怕是错的）区间，都不会使任何证书失效。

**证明.** 命题 5 的界对**任意** `V` 成立，与 `V` 如何得到无关；`V` 由 (i)(ii) 决定，
而种子只进入选点过程。故证书的有效性与种子规则无关。证毕。

**结论.** 该函数只应称为 *spectral estimate for seed placement*，不能称为
"certified spectral enclosure"：`eigsh` 的 Ritz 值配合人为 safety factor 不构成
一侧特征值界（无谱隙信息时，Ritz 残差只能说明附近存在特征值，不能给出
`lambda_min >= lambda_lower` 这类不等式）。若要把它写进定理，必须换成 Temple/Kato
型围道、惯性计数（Sturm 序列 + 分解）或 Gershgorin 型解析界；就当前用途（只定位种子）
而言，降格为启发式即可，正确性由命题 6 兜底。

---

## 7. 命题 7（成本比较的充要条件）—— 已证

**陈述.** 设单次算子 setup 成本 `T_s`、单次右端迭代成本 `T_i`；记频率数 `N_s`、参数点数
`N_p`、源端口数 `N_g`。本方法 `N_op = N_s N_p`、`N_rhs = N_g N_s N_p`；stock 每个
`(端口, 频移, 探针)` 都换一个算子，故 `N_op = N_rhs = S`。于是

```text
T_ours < T_stock   <=>   N_s N_p ( T_s + N_g T_i )  <  S ( T_s + T_i )
                   <=>   T_s / T_i  >  ( S - N_s N_p N_g ) / ( N_s N_p - S ).
```

**证明.** 成本模型 `T = N_op T_s + N_rhs T_i` 代入并整理即得；右端分母为负时不等式方向
取反。证毕。

**实测常数**（单线程，见 §5.1）：

```text
1 mm : T_s = 0.16 s, T_i(单右端 CG) = 0.03--0.18 s  -> T_s/T_i 约 1--5
2.5 mm: T_s = 0.06 s, T_i 约 4e-3 s                   -> T_s/T_i 约 15
```

代入 `N_s N_p = 45`、`S = 159`、`N_g = 4`：阈值
`(159 - 180) / (45 - 159) = 0.184`，而实测 `T_s / T_i = 0.16 / 0.093 = 1.72 > 0.184`，
故求解预算上成立（实测 24.0 s 对 62.6 s，2.6 倍）。

**这个模型不含非求解开销，而 1 mm 上它恰好占主导。** 把模型补成
`T = N_op T_s + N_rhs T_i + T_other`：本方法的 `T_other = 95.2 s`（`plan` 4.8 s +
`selection` 90.1 s，见 §5.1），stock 侧没有对应项，于是

```text
T_ours < T_stock  <=>  N_s N_p (T_s + N_g T_i) + T_other  <  S (T_s + T_i),
```

1 mm 上左端 `23.9 + 95.2 = 119.1` s 对右端 `62.6` s，不成立——这正是实测的 1.9 倍慢。
所以成本命题成立与否由 `T_other` 决定，而 `T_other` 全部来自 selection。

---

## 8. 引理 8（resolvent telescoping）—— 已证

**陈述.** 令 `p^(0) = q`、`p^(i) = (p_1, ..., p_i, q_{i+1}, ..., q_d)`、`p^(d) = p`。则

```text
A(p)^-1 - A(q)^-1 = sum_{i=1..d} (p_i - q_i) A(p^(i-1))^-1 H_i A(p^(i))^-1.
```

**证明.** 由 `A(p^(i)) - A(p^(i-1)) = (p_i - q_i) H_i` 与恒等式
`X^-1 - Y^-1 = X^-1 (Y - X) Y^-1` 逐项相加即得。证毕。

**用途与边界.** 该式把多参数误差精确拆成坐标分解，可作为坐标方向误差的起点。但它
**不**给出任何一张成的秩界，因此不能据此声称"Zolotarev 种子在多参数下最优"。种子目前
只能定位为 *coordinatewise minimax deterministic initialization*。

---

## 9. 未证（论文缺口，按优先级）

1. **参数一致的动态界（P0，核心缺口）.** 现有全部已证结论都在固定实频移下，"频率方向的
   FANTASTIC 指数收敛 + 参数方向的 certified 逼近"两者之间**没有桥**。可行的分解是两层：
   对每个 `p` 先取该参数点自身的理想空间 `W(p) = span{(A(p) + sigma_j C)^-1 G}`（原
   FANTASTIC 的 `O(eps_f)` 覆盖 `FOM(p) -> W(p)`），再证
   `max_j dist_{A(p,sigma_j)}((A(p)+sigma_j C)^-1 G, V) <= eps_p`；然后把"矩快照被
   `eps_p` 扰动"的 moment-matching 稳定性常数 `C_rat` 界出来，得到形如
   `sup_p norm(Z_p - Z_{V,p}) / norm(Z_p) <= 2 eps_f + C_rat eps_p` 以及时空能量版本的
   `2 sqrt(eps_f) + C_E sqrt(eps_p)`。**这一步没有做**，`C_rat` 需要 Zolotarev
   skeleton / 有理插值的稳定常数，我没有推导。
2. **Robin 局部参数的 `n`-width 定理（P1）.** Bachmayr--Cohen, *Math. Comp.* (2017),
   Cor. 4.2 的强指数结果建立在分段常数 **diffusion** 几何上，而这里的参数是 Robin
   **边界**算子 `a_p(u,v) = a_0(u,v) + sum_i p_i int_{Gamma_i} u v dS`，不是同一问题。
   §3.6.1 已把该引用收窄为背景。要真正拥有 `e^{-cn}` 必须专门证明 disjoint Robin
   patches 的 solution manifold 宽度衰减（可能的路线：`A_ref^-1 B` 的奇异值几何衰减，
   即椭圆算子的低秩结构，但要自己写）。
3. **完整 BCI 覆盖（P1）.** 现在严格认证的是共址传递 `G^T A^-1 G`（junction）。BCI CTM
   还需要边界温度/热流的可嵌入响应。自然扩展是把 `G` 扩成 `F = [G, B_Gamma]`，共址性质
   保持，命题 1 的 PSD 恒等式不变。
4. **交付前 SVD（P0，结构上已处理）.** 压缩会改变空间，故"贪心的 raw span 证书"不等于
   "交付 ROM 的证书"。本目录的做法是让**所有证书都针对最终的 `V`**（命题 5 对任意 `V`
   成立），因此结构上已闭环；需要补的是显式写明"截断失败则加严或保留更多奇异向量"的
   回退规则，以及不再试图给 `1e-3` 截断赋予理论意义。
5. **连续停止（P0，结构上可闭合）.** 当前 `certified_greedy_points` 的选点在 `41 x 41`
   粗网格上，严格陈述只能是"该有限候选集上取最大证书点"。闭合方式是按命题 5 把停止条件
   写成连续盒陈述（`sup_{p in P} eta(V_n, p) <= eps_p`），即"确定性提议 + 连续接受证书"；
   更强的是把命题 5 的单元界做成 branch-and-bound。两者都还没做。
6. **`d >= 3`（P1）.** 全部算例是 `d = 2`；`41^d` 候选在 `d = 3` 已是 68921，`d = 4` 约
   2.8e6。需要自适应划分或 B&B 替代全张量网格。

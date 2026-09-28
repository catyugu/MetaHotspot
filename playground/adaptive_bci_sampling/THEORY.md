# 理论：命题与证明

本文件把 `adaptive_bci_sampling` 依赖的数学陈述逐条写出，并区分三类状态：

* **已证**：下面给出完整证明，只依赖标出的假设。
* **已引用**：依赖外部定理，给出出处与本文件用到的形式。
* **未证**：真实缺口，第 2 节列出并说明缺在哪一步。

本文件证明的是**数学陈述**；代码是否实现这些陈述，靠 `certify_extraction.py` 的整盒审计与
单元测试核对，那是数值核验，不构成证明。两者不可互相替代。

本节的全部命题已经过逐条外部审读，并按审读结论收紧陈述：命题 1 补满列秩与 Loewner 矩阵
形式；命题 2 定义选择子并消除符号冲突；命题 3 删除“一般 FEM 不变”的说法；命题 4 拆成
估计量等价性（已证）与 rate transfer（缺前提）；命题 5 把极值控制从节点值改为 Bernstein
系数；引理 8 修正符号。

---

## 0. 记号与假设

单体热传导 + Robin 边界，有限体积离散后

```text
A(p, s) = K + s C + sum_{i=1..d} p_i H_i,    p in P = prod_i [p_i^-, p_i^+],    s >= 0.
```

全程假设：

```text
K = K^T >= 0,   C = C^T > 0,   H_i = H_i^T >= 0,   A(p^-, 0) > 0.
```

于是对每个 `p in P`、`s >= 0` 有 `A(p, s) >= A(p^-, 0) > 0`，因此 `A(p,s)` 与
`V^T A(p,s) V`（`V` 满列秩）都正定，后文所有逆与 Galerkin 约化算子都合法。

```text
A_-   = A(p^-, s)
X     = A(p,s)^-1 G
Q     = (V^T A(p,s) V)^-1 V^T G
X_V   = V Q
Y     = G^T A(p,s)^-1 G
Y_V   = G^T X_V
R     = G - A(p,s) X_V
e     = X - X_V
Delta = Y - Y_V
```

只有用到**逐元正性**或**互端口单调性**时才额外假设（下面称 M-matrix 假设）：

```text
K_ij <= 0 (i != j),   H_i = diag(a_i) 且 a_i >= 0 逐元,   G >= 0 逐元.
```

这三条对 Galerkin 恒等式与 Loewner 阶论证都不是必需的。注意一般 FEM 的边界质量阵只保证
`H_i >= 0`，其非对角元可以非零：那足以支撑 Loewner 单调性，但**不足以**支撑命题 3 的逐元
传递单调性，所以前者不能替后者作保。

### 0.1 研究作用域与误差口径（先读）

**研究目标是端口传递族** `Z(s; mu) = B^T (s M + K + sum_j mu_j H_j)^-1 B` 在**系统范数**下的
逼近 —— 即 FANTASTIC 式的输入-输出保证（冲激响应 / `H2`、Hankel）—— 代价指标是最小化
`N_FOM := N_RHS`（全阶逆作用次数）。**全场温度保真不是研究目标**，只是证明过程中的辅助工具。

同址恒等式

```text
Z - Z_V = E^T A E = R^T A^-1 R >= 0
```

说明端口认证可以比状态认证宽松得多：`lambda_max(Z - Z_V, Z)` 同时等于某个状态 A-能量相对
误差的**平方**。这个状态解释只作为 lemma / 证明装置保留，**不得**当作 headline 指标、接受
判据或问题难度度量。

作用域标签（本文件与 `records/NEGATIVE_RESULTS.md` 共用）：

```text
PORT-SYSTEM   H2 / Hankel / 冲激响应系统范数     研究目标
PORT-FIXED-S  单个实频移上的 Z(s; mu)            当前整盒证书的对象
PORT-STEP     采样 step-response 度量             实测诊断，未认证
STATE-AUX     全场 / A-能量 / 状态范数            仅证明装置与诊断
COST          N_FOM / N_op / 墙钟 / 内存          代价指标
```

`N_FOM := N_RHS`（全阶逆作用次数）是主优化目标；`N_op`（不同全阶算子 / setup 次数）、墙钟与
内存单独报告，不要用一个“求解次数”笼统概括三者。**fixed-real-shift 证书与 system-norm 证书
绝不能混称“动态证书”**：前者是本文件的命题 1/5/5M，后者是本节的 P0，两者之间目前没有定理。

按这个作用域，两个旧主理论对象已删除：**state-energy 作为共同接受目标**（原 P0 的第二条
定理）与 **Robin 解流形 n-width 逼近复杂度**（原 P1）。前者不是研究目标；后者研究的是状态
Hilbert 空间中的逼近复杂度 `d_n(M)`、`M = {A(p,s)^-1 G w}`，已经不是本问题需要的复杂度对象
（现在需要的是“达到端口系统范数要求所需的信息量”，见 P0 与 P1 BCI 端口嵌入定理）。

---

## 1. 命题

### 命题 1（Galerkin 输出误差恒等式）—— 已证

**陈述.** 对任意 `A = A^T > 0`、任意满列秩 `V`、任意 `G`，

```text
Y - Y_V = R^T A^-1 R = e^T A e >= 0          (Loewner 阶),
(Y - Y_V)_ab = R_a^T A^-1 R_b = e_a^T A e_b,
abs((Y - Y_V)_ab) <= sqrt( (Y - Y_V)_aa (Y - Y_V)_bb ).
```

**证明.** Galerkin 正交性给出 `V^T R = 0`。展开

```text
R^T A^-1 R = G^T A^-1 G - G^T X_V - X_V^T G + X_V^T A X_V,
X_V^T A X_V = G^T V (V^T A V)^-1 V^T G = G^T X_V = Y_V,
```

故 `R^T A^-1 R = Y - Y_V`。又 `e = A^-1 R`，故 `e^T A e = R^T A^-1 R`。半正定矩阵的逐元
Cauchy--Schwarz 给出最后一式。证毕。

**推论.** `Delta` 是半正定矩阵，`Delta_aa` 单独就是端口 `a` 的精确输出误差；这是矩阵
恒等式，后文命题 5M 直接引用它，不需重新逐元讨论。

---

### 命题 2（Woodbury 精确参数映射）—— 已证

**陈述.** 额外假设每个 `H_i` 对角：`H_i = diag(a_i)`、`a_i >= 0`。固定 `s`，记
`A_ref = A(p^-, s)`，并定义逐元增量与**选择子**

```text
delta_j = sum_i (p_i - p_i^-) (a_i)_j,     J = { j : delta_j > 0 },
E_J     = [ e_j ]_{j in J}   (单位坐标列),  D_J = diag(delta_j, j in J).
```

则 `A(p,s) = A_ref + E_J D_J E_J^T`，且

```text
A(p,s)^-1 = A_ref^-1 - A_ref^-1 E_J (D_J^-1 + E_J^T A_ref^-1 E_J)^-1 E_J^T A_ref^-1.
```

残差落在**与 `p` 无关**的张成空间 `Z = [G, (K + sC)V, H_1 V, ..., H_d V]` 中：设 `u_a` 为
`R^k` 第 `a` 个坐标向量、`q_a` 为 `Q` 第 `a` 列，则

```text
R_a = Z zeta_a(p),   zeta_a(p) = [ u_a ; -q_a ; -p_1 q_a ; ... ; -p_d q_a ],
Delta_ab(p,s) = zeta_a(p)^T [ Z^T A(p,s)^-1 Z ] zeta_b(p).
```

方括号由**三张锚点预算表**

```text
Z^T A_ref^-1 Z,      Z^T A_ref^-1 E_J,      E_J^T A_ref^-1 E_J
```

经稠密代数得到（Woodbury 修正项需要第三张表，仅前两张不够）。

**证明.** `H_i` 对角给出 `A(p,s) - A_ref = diag(delta) = E_J D_J E_J^T`；逆公式即
Sherman--Morrison--Woodbury（`D_J` 在活动集上正、可逆）。又 `A(p,s)V =
(K + sC)V + sum_i p_i H_i V`，故 `R_a = G u_a - A(p,s) V q_a = Z zeta_a(p)`。代回命题 1
即得精确（非近似）公式：`p_i = p_i^-` 的非活动列不扰动，其贡献已含在 `A_ref` 里。证毕。

---

### 命题 3（对称 M-matrix 族与传递单调性）—— 已证

**陈述.** 在 M-matrix 假设下，对每个 `p in P`、`s >= 0`：

1. `A(p,s)` 是非奇异对称 M-matrix，因而 `A(p,s)^-1 >= 0` 逐元；
2. `x_a = A(p,s)^-1 G_a >= 0` 逐元；
3. `d Y_ab / d p_k = -x_a^T H_k x_b <= 0` 逐元（全部端口对）。

对角项 `Y_aa` 的单调性只需要 `H_k >= 0`，不需要 M-matrix 假设。

**证明.** 由全局假设 `A(p,s)` 对称正定；其非对角元等于 `K` 的对应元，故非正，即
`A(p,s)` 是对称正定 Z-matrix，从而是非奇异 M-matrix，其逆逐元非负（这是非奇异
M-matrix 的特征刻画）。由 `G_a >= 0` 得 `x_a = A^-1 G_a >= 0`。对 `A x_b = G_b` 求导得
`d x_b / d p_k = -A^-1 H_k x_b`，而 `G_b` 与 `p` 无关，故

```text
d Y_ab / d p_k = d (G_a^T x_b) / d p_k = -x_a^T H_k x_b.
```

由 `H_k = diag(a_k)`、`a_k >= 0` 与 `x_a, x_b` 逐元非负得 `x_a^T H_k x_b >= 0`；`a = b`
时仅用 `H_k >= 0` 即得。证毕。

**为什么必须写清这一步.** 只用 `H_k >= 0` 只能得到自项的 `x_a^T H_k x_a >= 0`，
**推不出** `a != b` 的 `x_a^T H_k x_b >= 0`：后者完全依赖第 1 条给出的逆的逐元非负性。
所以本节把 M-matrix 假设与“数值上观察到 `A^-1 G >= 0`”分开写：数值观察不是逆正性的证明。

**假设在本仓库模型上的核对（FVM 对角 Robin 情形）.** `K` 非对角元非正、`H_i` 对角非负、
`G` 非负，三条都成立；`A(p^-)^-1 G` 的最小元为正。对一般 FEM 边界质量阵，`H_i` 的非对角
元可以非零，此时 `A` 不再是 Z-matrix，第 1、3 条必须重新验证或放弃。

---

### 命题 4（锚定残差估计量与 Galerkin 误差的等价性）—— 已证

**陈述.** 记 `Gamma = max_i p_i^+ / p_i^- >= 1`。对每个 `p in P`、`s >= 0`，

```text
A_- <= A(p,s) <= Gamma A_-,      A(p,s)^-1 <= A_-^-1 <= Gamma A(p,s)^-1,
```

于是以 `eta^2 = R^T A_-^-1 R` 为代理量，

```text
norm(e)_{A(p)} <= eta <= sqrt(Gamma) norm(e)_{A(p)},
norm(e)_{A_-}  <= eta <= Gamma norm(e)_{A_-}.
```

**证明.** 由 `p_i <= Gamma p_i^-`、`Gamma >= 1`、`K >= 0` 与 `C > 0`：

```text
K + sC + sum_i p_i H_i <= Gamma K + Gamma sC + Gamma sum_i p_i^- H_i = Gamma A_-,
```

另一侧 `A_- <= A(p,s)` 由 `p >= p^-`。取逆并用 Loewner 单调性得第一式。由 `R = A(p,s) e`
有 `R^T A(p,s)^-1 R = norm(e)_{A(p)}^2`，代入即得第二式；第三式由
`norm(e)_{A_-} <= norm(e)_{A(p)} <= sqrt(Gamma) norm(e)_{A_-}` 与第二式复合得到。证毕。

---

### 命题 4b（代理贪心 ⇒ 固定 Hilbert 空间弱贪心）—— 缺前提 `[STATE-AUX]`

**作用域.** 本命题的逼近误差 `d_-` 是 `A_-`-能量范数下的**状态**逼近误差，它的消费者是
state-space 复杂度（速率）问题；该问题已按 0.1 的作用域退出主理论线。命题本身仍然正确，
保留为设计依据（它给出的“单元局部锚点才让速率常数非空”正是 B&B 用单元局部 Riesz 锚点的
理由），但**不得**用它为端口系统范数的接受判据作保。

**陈述.** 对单个响应向量 `x(p) = A(p,s)^-1 g`，记固定空间 `V` 在 `A_-` 范数下的最佳逼近误差

```text
d_-(p, V) = inf_{v in V} norm(x(p) - v)_{A_-}.
```

则对每个 `p`

```text
d_-(p, V) <= eta(p, V) <= Gamma d_-(p, V).
```

于是若参数点由精确极大化选出（`eta(p_n, V_n) = sup_{p in P} eta(p, V_n)`），则该步满足固定
Hilbert 空间中的弱贪心条件，weakness constant 至少为

```text
gamma = 1 / Gamma = min_i p_i^- / p_i^+.
```

**证明.** 下界：`x_V(p) in V` 是 `d_-` 的合法竞争者，故 `d_- <= norm(x - x_V)_{A_-} <= eta`
（末步用命题 4 第三式）。上界：由命题 4 第三式 `eta <= Gamma norm(e)_{A_-}`，而
`A_- <= A(p)` 与 `A(p)`-Galerkin 最优性给出

```text
norm(e)_{A_-}^2 = e^T A_- e <= e^T A(p) e <= inf_{v in V} norm(x - v)_{A(p)}^2
               <= inf_{v in V} norm(x - v)_{A_-}^2 = d_-^2,
```

故 `eta <= Gamma d_-`。证毕。（此式已在随机 SPD 族上数值核对：`Gamma = 100` 时 24 个
`(端口, 参数)` 样本上 `d_- <= eta <= Gamma d_-` 全部成立，实测 `eta / d_-` 落在
`1.77 .. 40.7`。）

**尚未具备的前提.** 要把外部 weak-greedy 速率定理用到本仓库算法上，还缺以下**算法性**前提，
它们不是代数步骤：

* 连续解流形（含矩阵值评分时 MIMO 的方向变量 `w`）必须先精确定义；
* 实际选点过程必须在该流形上取极大，否则要给出显式的近似极大化因子；
* 有限 `41^d` 候选网格不等于 `P` 上的连续极大化。

**这条定理在本问题上是空的，原因必须写明.** 本模型的 `Gamma = max_i p_i^+ / p_i^- =
9285.79`，故 `gamma = 1.08e-4`，速率常数 `2 / gamma` 约 `1.9e4`，在 `1e-3` 目标上不构成
任何有用陈述；而实测的两侧等价常数只有约 1--3。**要让速率定理非空，必须把全局锚点换成
单元局部锚点**：按对数划分把 `P` 分成每轴 `k` 份，单元内
`Gamma_alpha = 10^((log10 p^+ - log10 p^-) / k)`，`k = 8` 时 `Gamma_alpha = 3.16`、
`2 / gamma_alpha = 6.3`；`k = 16` 时 `Gamma_alpha = 1.78`。所以 **branch-and-bound 与
单元局部证书不是实现细节，而是让速率定理非空的前提**。

---

### 命题 5（锚定 Riesz--Bernstein 单元证书）—— 已证

**陈述.** 设单元 `Q = [a, b]`、锚点 `a`。取任意**多项式**约化系数 `q(p)`，记
`R_q(p) = G - A(p,s) V q(p)`，并设所选张量 Bernstein 阶数足以**精确表示**该多项式（写成
`R_q(p) = Z zeta(p)` 后，即该阶数足以精确表示向量/矩阵多项式 `zeta`，不足时用升阶补齐）。
记 `S_a = Z^T A(a,s)^-1 Z >= 0`，`zeta(xi) = sum_nu B_nu(xi) c_nu` 为其精确张量 Bernstein
表示，`B_nu(xi) >= 0`、`sum_nu B_nu(xi) = 1`，`c_nu` 是 Bernstein **系数（控制点）**。则对
每个 `p in Q`

```text
Delta(p) = Y(p) - Y_V(p) <= zeta(p)^T S_a zeta(p) <= max_nu c_nu^T S_a c_nu,
```

右端可算，且 `abs((Y - Y_V)_ab(p)) <= sqrt(Delta_aa(p) Delta_bb(p))`。

**证明.** 第一步是 Galerkin 最优性：真系数使残差能量最小，故对任意多项式试验 `q`，
`Delta(p) <= R_q(p)^T A(p,s)^-1 R_q(p)`。第二步是 Loewner：`p >= a` 逐分量给出
`A(p,s) >= A(a,s)`、`A(p,s)^-1 <= A(a,s)^-1`，代入得 `Delta(p) <= zeta(p)^T S_a zeta(p)`。
第三步是凸性：`zeta(xi)` 是其 Bernstein **控制点**的凸组合，而 `z -> z^T S_a z` 在 `S_a >= 0`
时凸，故

```text
zeta(xi)^T S_a zeta(xi) <= sum_nu B_nu(xi) c_nu^T S_a c_nu <= max_nu c_nu^T S_a c_nu.
```

**注意这一步控制极值的是 Bernstein 系数，不是多项式在插值节点上的取值**；把节点值当作上界
是错的。最后一步是命题 1 的 Cauchy--Schwarz。证毕。

**推论（为什么审计比较的是绝对量）.** 逐项相对界需要除以 `abs(Y_V,ab)`，而弱耦合项自身分母
可以任意小，所以逐项相对界必然松；`steady_absolute_bound` 与 `steady_relative_bound` 是两个
不同的量，**不可互相比较**（文档曾在此处出过错，已修）。

---

### 推论 5M（矩阵型单元证书）—— 已证

**陈述.** 设多项式残差试验的矩阵系数表示为 `R_q(p) = Z Xi(p)`、
`Xi(xi) = sum_nu B_nu(xi) C_nu`（精确张量 Bernstein 表示）。则

```text
E(p) = Y(p) - Y_V(p) <= Xi(p)^T S_a Xi(p) <= sum_nu B_nu(xi) C_nu^T S_a C_nu   (Loewner 阶),
```

因此任何逐个控制矩阵 `C_nu^T S_a C_nu` 的 `U_Q` 都给出有效界：`E(p) <= U_Q`。

分母可以只用 reduced-size 量：设 `b` 为 `Q` 的上角点。由 Loewner 单调性 `Y(p) >= Y(b)`，由
命题 1 `Y(b) - Y_V(b) >= 0`，故 `Y(p) >= Y(b) >= Y_V(b)`，于是

```text
lambda_max(E(p), Y(p)) <= lambda_max(U_Q, Y_V(b)).
```

若另有独立的角点缺陷界 `E(b) <= delta_b^2 Y(b)`、`delta_b < 1`，则
`Y_V(b) = Y(b) - E(b) >= (1 - delta_b^2) Y(b)`，故把精确分母 `Y(b)` 换成 `Y_V(b)` 最多把
广义特征值界放大 `1 / (1 - delta_b^2)`。

**证明.** 第一式是命题 5 的矩阵版。`X -> X^T S_a X` 的矩阵凸性来自

```text
theta X^T S_a X + (1-theta) Y^T S_a Y - (theta X + (1-theta) Y)^T S_a (theta X + (1-theta) Y)
= theta (1-theta) (X - Y)^T S_a (X - Y) >= 0.
```

分母部分是参数 Loewner 单调性与命题 1 的直接推论。证毕。

**这是连续停止语句的实现对象.** branch-and-bound 的严格停止语句是
`forall Q: U_Q(V) <= tau_moment^2`；`U_Q` 的正确性只依赖本推论，不需要在每个叶子上做全阶
求解（逐叶子全阶求解会把证书自己变成大量 full-order inverse action）。

---

### 命题 6（仅用于种子的启发式不影响后验认证）—— 已证

**陈述.** 设某个谱估计只用于选取种子参数点，不出现在：频移计划、任何证书常数、任何声称的
谱区间、任何动态误差定理中。则把它换成任何别的规则只改变**产出哪个** `V`；命题 1 与命题 5
的后验证书对最终 `V` 依然成立，与 `V` 如何得到无关。

**证明.** 命题 1 与命题 5 对任意满列秩 `V` 成立，不使用 `V` 的构造历史。证毕。

**结论.** `coordinate_spectral_enclosures` 只应称为 *spectral estimate for seed
placement*，不能称为 “certified spectral enclosure”：`eigsh` 的 Ritz 值配合人为 safety
factor 不构成一侧特征值界。就当前用途（只定位种子）而言，降格为启发式即可，正确性由命题 6
兜底。频移计划则**不**受本命题保护：它必须真正覆盖 `K + sum_i h_i H_i` 的谱（见第 3 节
已关闭项）。

---

### 命题 7（二项求解成本模型的精确盈亏条件）—— 已证

**陈述.** 设单次算子 setup/分解成本 `T_s`、单次右端增量成本 `T_i`，成本模型
`T = N_op T_s + N_rhs T_i`。记 `A = N_s N_p`（频移数乘参数点数）、`N_g` 为源端口数、`S` 为
stock 的求解次数。本方法 `N_op = A`、`N_rhs = A N_g`；stock 每个 `(端口, 频移, 探针)` 都换
一个算子，故 `N_op = N_rhs = S`。记 `r = T_s / T_i > 0`，则

```text
T_ours < T_stock   <=>   (A - S) r < S - A N_g,
```

这个形式不需要除法，对 `A - S` 的任何符号都成立。（除以 `T_i` 的写法只在分母不为零且符号
已知时等价，不能直接写成单个分式。）

若只有本方法存在非求解开销 `T_other`，则精确条件改为

```text
A (T_s + N_g T_i) + T_other < S (T_s + T_i).
```

**证明.** 直接代入：`A T_s + A N_g T_i < S T_s + S T_i`；两边除以正数 `T_i` 并整理即得。
带 `T_other` 的版本只多一项。证毕。

**这条模型不含非求解开销，而 1 mm 上它恰好占主导.** `T_other` 全部来自 selection
（候选打分与残差证书的稠密代数，外加一次 minimum 算子的分解），实测数字见 README 的成本
一节：在 1 mm 上求解预算上成立（约 2.6 倍），端到端反而不成立。因此“成本命题是否成立”由
`T_other` 决定。

---

### 引理 8（resolvent 坐标伸缩恒等式）—— 已证

**陈述.** 令 `p^(0) = q`、`p^(i) = (p_1, ..., p_i, q_{i+1}, ..., q_d)`、`p^(d) = p`。则

```text
A(p)^-1 - A(q)^-1 = - sum_{i=1..d} (p_i - q_i) A(p^(i-1))^-1 H_i A(p^(i))^-1.
```

**证明.** 每一步 `A(p^(i)) - A(p^(i-1)) = (p_i - q_i) H_i`；用
`X^-1 - Y^-1 = X^-1 (Y - X) Y^-1`（取 `X = A(p^(i))`、`Y = A(p^(i-1))`）得

```text
A(p^(i))^-1 - A(p^(i-1))^-1 = -(p_i - q_i) A(p^(i))^-1 H_i A(p^(i-1))^-1,
```

对称性允许交换两个因子的次序。对 `i` 求和后左端从 `A(q)^-1` 伸缩到 `A(p)^-1`，故右端带负号。
证毕。

**用途与边界.** 该式把多参数误差精确拆成坐标分解，可作为坐标方向误差的起点，但它**不**给出
任何张成的秩界，因此不能据此声称“Zolotarev 种子在多参数下最优”：种子目前只能定位为
*coordinatewise minimax deterministic initialization*。

---

## 2. 未证的数学缺口

七项旧清单里的其余项目或者已闭合、或者属于工程实现、或者已是负结果，都不要放进本节（见
第 3 节）。真正剩下的缺口是三个。

### P0 动态传递定理（Dynamic transference theorem）

**要证的陈述.** 记 `B(h) = C^-1/2 K(h) C^-1/2`、`b = C^-1/2 G`。连续盒证书控制的是若干
matching 频移上 positive-real resolvent 的缺陷

```text
delta_j(h)   (对 (B(h) + sigma_j I)^-1 b),      delta_star = max_j sup_h delta_j(h).
```

需要一条**端口系统范数**定理：

```text
Hankel:      E_Hankel(V) <= gamma_H(Sigma) + C_H(Sigma) delta_star,
```

它必须满足 vendor 的 `||Delta Z||_Hankel < 2 eps` 目标，故可接受的 matching 容差是
`delta_req = delta_req,H`。**不要把它合并成一个 `Phi_Sigma`。**

`[STATE-AUX]` 历史上这里并列了第二条定理
`E_state(V) <= gamma_S(Sigma) + C_S(Sigma) delta_star`。它已按 0.1 的作用域删除：全场温度
保真不是研究目标，状态半群只作为**证明中间量**出现（
`(B + t I)^-1 -> functional calculus / Laplace--Stieltjes -> exp(-B tau) ->` 冲激响应与
Hankel 核），不进入接受判据，也不再是 `delta_req` 的第二项。**“状态逼近不够好”绝不能作为
端口 ROM 的否决理由。**

**当前已知的部分.** 精确 matching 点骨架已经给出一个 positive-real 逼近项 `rho_Sigma`
（向量重构分析另给出稳定因子 `K_vec`），存活的结构是

```text
delta(t; V) <= gamma_Sigma(t) + K_vec(t) delta_star,     sup_t gamma_Sigma(t) = rho_Sigma.
```

玩具反例证明“只依赖 matching 缺陷的一般二次桥”是错的，存活路线是**对 `delta_star` 线性**。

**缺的那一步.** `rho_Sigma` 与 `K_vec` 控制的是 positive-real resolvent 逼近，它们还不是
`C_H / gamma_H`。缺的是一个从 positive-real resolvent 控制到**端口**时域系统范数（冲激响应
矩阵的 `H2` / Hankel）的传递定理。自然路线是

```text
(B + t I)^-1  ->  functional calculus / Laplace--Stieltjes 表示  ->  exp(-B tau)
              ->  impulse-response 与 Hankel 核,
```

它保留 SPD、自伴、共址、Stieltjes 结构。虚轴不是首选路线：`B + i omega I` 非 SPD，matching
点证书用的 Galerkin 能量极小化论证在虚轴上不直接成立，那里只作 diagnostic。

**判据.** 只有给出显式可算常数、并在整个 HTC 盒上一致地满足端口 vendor 不等式（`||Delta Z||_Hankel < 2 eps`）才算闭合。

### P0/P1 B&B 证书一致性与有限终止

**要证的陈述.** 设 `U_Q(V)` 是叶子 `Q` 上的严格矩阵证书。若真实被认证量有**严格余量**
`sup_{p in P} delta(p; V) < tau`，则给定的自适应细分策略必须在有限步后终止于
`forall Q: U_Q(V) <= tau^2`。

**当前已知的部分.** 接受正确性已闭合：只要每个叶子满足 `U_Q <= tau^2`，整盒就满足同一容差，
所以现有实现给出的停止条件在它停止时是可靠的。

**缺的那一步.** 缺一致性定理，例如在**实际**的“局部参数细分 + 多项式升阶 + block 锚点细化

* 必要时按 shift 升级 Gram”组合下证明

```text
diam(Q) -> 0  =>  U_Q(V) - sup_{p in Q} delta(p; V)^2 -> 0.
```

一个永久共享的 shift-free 公共 Gram 会有不随划分消失的大 shift 锚点地板，因此对“永不升级该
锚点”的策略无法证明有限终止。

**判据.** 在给定的细分策略下，严格真余量蕴含有限步认证终止。

### P1 BCI 边界端口嵌入定理

**要证的陈述.** 证明已交付 CTM 所需的全部 BCI 边界变量（结温、边界温度、边界热流自由度）可
表示为一个**固定、与参数无关的共址**端口算子 `F`，使得所需 BCI 误差陈述由
`F^T A(p,s)^-1 F` 的认证推出。

**当前已知的部分.** 对任意固定的共址端口矩阵 `F`，命题 1 原样适用：

```text
F^T A^-1 F - F^T X_V = R_F^T A^-1 R_F >= 0,
```

即共址对称端口族的 Galerkin 误差恒等式**已经闭合**，这里不缺新的 Galerkin 定理。

**缺的那一步.** 缺的是建模/嵌入陈述：确认 BCI 耦合实际需要哪些边界端口变量；证明它们承认
所需的固定共址实现；证明该实现的认证蕴含所要求的边界温度与热流精度。

**判据.** BCI 耦合变量被映射为一个与参数无关的已认证端口系统，且该认证对交付的边界误差
度量有精确蕴含。

---

## 3. 已关闭项（不要再写进未证清单）

* **前置 SVD 回退**：作为 certified post-processing 已闭合。对 `W_n` 做压缩得 `V_r`，对
  `V_r` 重跑同一证书，不通过就增大 `r`（必要时对 rank 二分）直到通过；满秩仍不通过才回到
  富集。需要明确指出的是：单调的是**真误差**；每轮用**新生成** trial 算出的 Bernstein/Riesz
  上界本身没有自动单调性，把上一轮 trial 作为候选一直 carry forward 即可恢复
  `U_Q^{n+1} <= U_Q^n`（纯 reduced algebra，目前未实现）。
  **关闭的作用域**：关闭的是**压缩-传递的理论迁移问题** —— 不需要从 raw 空间证书推导压缩
  误差定理，直接对最终交付的 `V` 重跑证书即可。**未闭合**的是动态接受阈值 `tau_moment`，
  它取决于 P0 动态传递定理，不能写成“vendor 保证已解决”。
* **`d >= 3`**：不是数学正确性缺口。本文件的命题对有限 `d` 与维数无关，问题只是
  tensor-product 候选网格与 Bernstein 复杂度（`41^3 = 68921`，`41^4` 约 `2.8e6`）。
* **频移计划的谱区间**：数学结构与工程缺陷部分已闭合。Robin 盒谱包围 `lambda_min(K_-)`、
  `lambda_max(K_+)` 由 Loewner 序给出（`K_- <= K(h) <= K_+` 把全盒谱覆盖归约为两个端点
  广义特征问题），已进生产路径（`box_spectral_interval`），并有回归测试固化。
  **未闭合**的是浮点下的一侧特征值**严格**包围：生产路径用迭代特征求解器取端点，返回的是
  数值近似而不是带余项的严格单侧界，因此“浮点实现也严格 certified”还缺一个 residual /
  inertia 辅助的一侧谱认证。这个缺口很小但不是零，不要把它写成已经解决。
* **inexact-moment 后向误差桥**：作为交付 ROM 的路线已关闭；兼容条件 `R = R X^+ X` 在交付
  基底上严重失败，pre-SVD 情形又因 `delta / lambda_min >> 1` 使扰动论证失效。证据见
  `records/NEGATIVE_RESULTS.md`。
* **逐 shift 加性 defect 传播**：已关闭；修正 LU cache 测量 bug 之后，该界仍只是松的上估
  （`sum_k c_k / g_defect` 落在 1.3..25，稳定高估而非双向漂移），不能驱动停止。
* **单一全局谱标量的廉价盒残差界**：已关闭；不等式成立但对决定误差的点 effectivity 达
  `2.6e4 .. 2.1e5`，量级上不可能驱动贪心接受。
* 失败路线的完整证据、测量更正与被取代的实验都属于 `records/NEGATIVE_RESULTS.md`，不属于
  本文件的未证清单。

---

## 4. 引用的外部定理与文献定位

**外部定理（已引用）**

* **weak greedy 的速率传递**：DeVore, Petrova, Wojtaszczyk, *Constr. Approx.*
  37(3):455--466 (2013), Thm 3.2 与 Cor. 3.3（arXiv:1204.2290）：
  `eps_{2n}(F) <= 2 gamma^-1 d_n(F)`；Binev 等, *SIAM J. Math. Anal.* 43(3):1457--1472
  (2011), DOI `10.1137/100795772` 给出同一结论并附常数。同文 Theorem 4.1 指出：**由样本
  元素张成的子空间一般不可能达到 n-width 速率**，所以不存在 `eps_n <= C d_n` 形式的结论。
  调用前必须核对命题 4b 的算法前提（连续流形、取极大方式）。`[STATE-AUX]` 这条速率传递
  服务的是 state-space 复杂度问题，已按 0.1 的作用域退出主理论线；保留为设计依据，不作为
  端口系统范数的接受判据。
* **估计量驱动的贪心的速率条件**：Buffa, Maday, Patera, Prud'homme, Turinici,
  *ESAIM M2AN* 46(3):595--603 (2012), DOI `10.1051/m2an/2011056`, Thm 3.1：达到指数速率要求
  n-width 衰减率 `beta > log(1 + M / alpha_coer)`——**coercivity 下界越差，要求的衰减率越高**。
  这是把固定 `A(h_min)`-Riesz 换成参数相关 Riesz 的理论动机，而不是经验改进。
* **能量范数下界是恒等式**：Prud'homme 等, *ESAIM M2AN* 36(5):747--771 (2002) 与 Yano,
  *SIAM J. Sci. Comput.* 40(1):A388--A420 (2018) 给出
  `norm(u - u_N)_{A(p)} = norm(r)_{A(p)^-1}`；命题 2 的精确映射算的正是右端，故其
  effectivity 是 1，不是待定常数。
* **频率方向的一参数定理**：Massei, Robol, *BIT Numer. Math.* 61 (2021), DOI
  `10.1007/s10543-020-00826-z`, Corollary 3.18 对 `(K, C)` 的 Cauchy--Stieltjes 函数给出
  `norm(f(A)v - x_l)_2 <= 8 f(a) norm(v)_2 rho_{[a,4b]}^l`。椭圆频移计划正是这条定理的实现。

**文献定位（三层，按可信度）**

* FANTASTIC 系**从未给出连续 HTC 盒上的证明界**：已发表的是采样流程（MPMM 椭圆频移 +
  HTC 随机对数均匀抽取 + 残差探针 + 列归一化 SVD）与有限验证集上的相对误差百分比。因此
  本目录的整盒陈述是新的。
* 与这里结构上最接近的已发表确定性做法不是采样：Dong, Griffo, Wang, *IEEE TPEL*
  35(8):8550--8558 (2020), DOI `10.1109/TPEL.2020.2965248`，处理同样的
  `C T' + (K + sum_i h_i K_i) T = F Q`（`K_i` 对角、支集互不相交），用确定性均值参数 Krylov
  展开，给出准确度比较但**不给** HTC 盒上的界。
* 命题 2 所依赖的 Woodbury 恒等式最接近的先行工作是 Beattie, Gugercin, Tomljanović,
  arXiv:1912.11382 (2019), *Adv. Comput. Math.* 46:17 (2020)：它把同一恒等式用于**构造**无
  采样降阶模型（要求 `k x k` 子系统可降阶），本目录把同一恒等式用于**精确评估**已交付基底的
  整盒误差（不要求可降阶，代价是 `m_b + Q` 次回代）。检索到的一手材料里，两边都没有把该
  恒等式放进采样循环内去评估认证界。

**取证限制.** IEEE TCPMT 2021、THERMINIC/SPI/SEMI-THERM 论文集、Springer LNCSE 第 17 章、
Siemens 的 BCI-ROM 验证/最佳实践 PDF 均为付费或授权受限，上述涉及它们的判断只依据摘要与
元数据。

# 全场传递算子的目标、恒等式与未证问题

本文件区分已证的固定实移代数、明确定义的全场动态目标与尚未建立的动态保证。
本次数学核对见 [MetaHotspot 项目对话](https://chatgpt.com/g/g-p-6aa80d54d2608191adb78637c1211232/c/6abcaa95-3260-83ee-a2f5-3efec6022d6a)。
历史端口测量与旧 vendor 常数只留在 `records/FAILURE_ARCHIVE.md`。

## 0. 假设与验收对象

**2026-10-07 最新用户目标：** 最终 SVD 基只要求稳态全场 K-energy 相对误差
不超过 `2*epsilon`，阶跃全场 C-energy 相对误差在所有时刻不超过
`2*sqrt(epsilon)`，均覆盖所有固定输入组合。输入阶跃、零初值，`t=0` 用右极限。
平方 Gram 阈值分别是 `4*epsilon^2` 与 `4*epsilon`。
此前的 STATE-IMPULSE 是独立历史研究目标，其积分证书不能直接替代此阶跃点时刻目标。
最新成本与目标复验见 [稳态/阶跃与提取成本](records/STEADY_STEP_COST_20261007.md)。

```text
K(h) = K + sum_i h_i H_i,       h in P = product_i [h_i^-, h_i^+],
A(h,s) = K(h) + s C,           s >= 0,
K = K^T >= 0, C = C^T > 0, H_i = H_i^T >= 0, K(h^-) > 0.
```

`V` 满列秩，`G` 是固定功率输入矩阵。`G` 有冗余列时，以下广义特征值均在输入空间对
`ker(G)` 取商后理解；数值审计假设输入列独立，不用任意分母 regularization 掩盖秩亏。
目标是 `X(h,s)=A(h,s)^-1G` 到**全部温度自由度**的映射，不是某个读出量。

### 固定实移：STATE-FIXED-S

```text
X_V = V (V^T A V)^-1 V^T G,    E = X - X_V,
epsilon_fix(h,s;V) = sup_{w: Xw != 0} ||Ew||_A / ||Xw||_A,
epsilon_fix(P,Sigma;V) = max_{s in Sigma} sup_{h in P} epsilon_fix(h,s;V).
```

`||z||_A^2=z^TAz`。有限实移集合 `Sigma` 上通过不等于整个 `s>=0` 轴通过。
`D_fix=epsilon_fix^2` 是代码中平方缺陷的含义。场容差 `tau` 必须与 `sqrt(D_fix)` 比较。
节点最大误差、普通 L2、热流和非共址读出要另行定义与验证，不能自动替换该范数。

### 全场动态：STATE-IMPULSE

对 `C x'(t)+K(h)x(t)=G u(t)`，零初值、`u(t)=w delta_0(t)`，定义

```text
T_h(t) = exp(-C^-1 K(h) t) C^-1 G,
C_V = V^T C V, K_V(h) = V^T K(h) V, G_V = V^T G,
T_h,V(t) = V exp(-C_V^-1 K_V(h) t) C_V^-1 G_V,
||z||_{L2_C}^2 = integral_0^infinity z(t)^T C z(t) dt,
epsilon_imp(h;V) = sup_{w: T_h(.)w != 0}
                  ||(T_h-T_h,V)w||_{L2_C} / ||T_h w||_{L2_C},
epsilon_imp(P;V) = sup_{h in P} epsilon_imp(h;V).
```

这是独立定义的全场动态范数，不宣称等于未取得精确定义的厂商 energy norm。
它覆盖所有固定 impulse 输入组合；任意时变输入下的诱导误差还需要独立的卷积/系统范数论证。
同一个用户容差可以分别要求 `epsilon_fix<=tau` 和 `epsilon_imp<=tau`；二者之间没有已证换算。
结温、端口 H2/Hankel、名义功率 step 与有限时刻场恢复均是补充诊断。

## 1. 固定实移恒等式与证书（已证）

### 命题 1：全输入场能量误差的精确读数

令 `Y=G^T X`、`Y_V=G^T X_V`、`R=G-A X_V`。Galerkin 正交性给出 `V^T A E=0`，故

```text
Delta = Y-Y_V = G^T E = (X_V+E)^T A E = E^T A E = R^T A^-1 R >= 0.
```

对任意 `w`，`||Ew||_A^2=w^T Delta w`，`||Xw||_A^2=w^T Yw`，于是
`D_fix=lambda_max(Delta,Y)`。这是全场、全输入组合的相对能量误差平方；共址输出只是读出它的
小矩阵。`|Delta_ab|<=sqrt(Delta_aa Delta_bb)` 来自半正定性，残差矩阵本身不必非负。

该推导对复频率非 SPD 算子、非共址读出或时间递推的变化 RHS 不直接适用。

### 命题 2：对角 Robin 族的 Woodbury 映射

额外假设 `H_i=diag(a_i)`、`a_i>=0`。固定 `s`，令 `A_ref=A(h^-,s)`，活动对角增量为 `D_J>0`，
对应坐标选择矩阵为 `B_J`，则

```text
A(h,s)^-1 = A_ref^-1
            - A_ref^-1 B_J (D_J^-1+B_J^T A_ref^-1 B_J)^-1 B_J^T A_ref^-1.
Z = [G, (K+sC)V, H_1 V, ..., H_d V],
R = Z Xi(h), Xi(h) = [I; -Q; -h_1 Q; ...; -h_d Q], Q=(V^TAV)^-1V^TG.
Delta(h,s)=Xi(h)^T [Z^T A(h,s)^-1 Z] Xi(h).
```

第一式是 Woodbury 恒等式，第二式由展开 `A(h,s)V` 得到。需要**三张**表：
`Z^T A_ref^-1 Z`、`Z^T A_ref^-1 B_J`、`B_J^T A_ref^-1 B_J`。
精确代数不等于廉价提取：全活动边界表需要边界列数级的逆作用与稠密存储。

### 命题 4/4b：锚定残差与固定 Hilbert 空间距离

HTC 下界严格为正时，令 `Gamma=max_i h_i^+/h_i^-`、`A_-=A(h^-,s)`。
由半正定项逐项比较得 `A_-<=A<=Gamma A_-`，逆序为 `A^-1<=A_-^-1<=Gamma A^-1`。
对单列误差 `e` 和残差 `r=Ae`，定义 `eta^2=r^T A_-^-1 r`，则

```text
||e||_A <= eta <= sqrt(Gamma) ||e||_A.
```

设 `d_-=inf_{v in range(V)}||x-v||_{A_-}`，正确证明是

```text
d_- <= ||e||_{A_-} <= ||e||_A <= eta,
eta <= sqrt(Gamma)||e||_A
    = sqrt(Gamma) inf_v ||x-v||_A
    <= Gamma inf_v ||x-v||_{A_-} = Gamma d_-.
```

最后一步使用 **`A<=Gamma A_-`**；旧版本从 `A_-<=A` 推出相反方向的不等式是错误的。
常数 `Gamma` 在这些假设下不能改善：`A_-=I`、`A=diag(1,Gamma)`、`V=span(e_1)`、
`x=e_2` 给出 `d_-=1`、`eta=Gamma`。精确极大化绝对 `eta` 可给弱贪心常数 `1/Gamma`；
有限候选网格、相对矩阵评分与近似极大化需要额外前提，不能直接宣称 n-width 速率或采样最优。

### 命题 5/5M：Riesz–Bernstein 单元界

设 HTC 单元 `[a,b]`、锚点 `a`，任意多项式约化试验 `q(h)` 的残差为 `R_q=Z Xi(h)`。
以足够阶数**精确表示**残差系数：`Xi(xi)=sum_nu B_nu(xi) C_nu`，其中 Bernstein 基函数非负、
和为 1，令 `S_a=Z^T A(a,s)^-1 Z`。Galerkin 最优性、逆的 Loewner 序与矩阵凸性依次给出

```text
Delta(h,s) <= R_q^T A(h,s)^-1 R_q
           <= Xi(h)^T S_a Xi(h)
           <= sum_nu B_nu(xi) C_nu^T S_a C_nu.
```

矩阵凸性由 `theta(1-theta)(X-Y)^T S_a(X-Y)>=0` 得证。控制极值的是 Bernstein 控制矩阵，
不能以插值节点值替代；也不能把各矩阵逐元取最大当作 Loewner 上界。
在任意 SPD 分母 `D_b=Y_V(b)` 下，定义

```text
u_Q = max_nu lambda_max(C_nu^T S_a C_nu, D_b).
```

每个控制矩阵均 `<=u_Q D_b`；又 `Y(h)>=Y(b)>=Y_V(b)=D_b`，故
`D_fix(h,s)<=u_Q`。`u_Q<=tau^2` 才认证该单元的场容差。分母秩亏时必须拒绝有限相对界或另取
合法分母。此矩阵证明无需 M-matrix 逐元正性；逐项端口相对界则另需其读出与正性条件。

接受正确性对任意最终交付 `V` 成立，与选点方法无关。压缩后须重新认证最终基，不能继承 raw
快照空间的证书。多项式升阶、单元细分与 Gram 重建都需计费；上界不保证随每轮新 trial 单调。

## 2. 全场动态范数的 Gram 表达（已证定义关系）

白化 `B=C^-1/2 K(h) C^-1/2>0`、`b=C^-1/2 G`，则 `C^1/2 T_h(t)=exp(-Bt)b`。因此

```text
Q(h) = integral_0^infinity T_h(t)^T C T_h(t) dt
     = b^T (2B)^-1 b = (1/2) G^T K(h)^-1 G,
J(h;V) = integral_0^infinity (T_h-T_h,V)^T C (T_h-T_h,V) dt,
epsilon_imp(h;V)^2 = lambda_max(J(h;V), Q(h)).
```

指数稳定性保证积分收敛。最后一式由时间-空间范数定义的 Rayleigh 商得出。
`trace(J)/trace(Q)` 只测输入列的总能量比，不能代替所有线性组合的最坏广义特征值。
`(1/2)G^T C^-1G` 不是这里的动态分母。正交有理展开与精确尾项、
Poisson–Loewner 连续 HTC 盒证书已实现为研究原型，见
[基础动态证明](RATIONAL_DYNAMIC_PROOF.md) 与 [带符号矩阵证明](SIGNED_POISSON_PROOF.md)。
这些冲激证书不直接认证最新的阶跃目标；后者见 [稳态与阶跃证明](STEADY_STEP_PROOF.md)。

## 3. 尚未证明的主问题

**P0：固定实移全场误差到全场 impulse 的传递。** 需要可计算、连续 HTC 盒一致的保证，并且
包含有限有理骨架本身的逼近误差，不能只有 matching defect。二维反例已经足够：
`C=I`、`B=diag(1,2)`、`b=(1,1)^T`、`V=span((B+sigma I)^-1b)`，则该实移误差为零，
但 `exp(-Bt)b=(exp(-t),exp(-2t))^T` 不可能始终在同一条直线上，所以动态场误差非零。
旧端口二次桥反例属于历史端口范数，不能当成本文件新定义的全场 impulse 反例。

**谱包围。** 裸 `K` 的源驱动谱估计不是整个 Robin 族的认证谱区间。若 P0 使用统一
`alpha I<=C^-1/2K(h)C^-1/2<=beta I`，需认证盒角点的一侧谱界；普通 Ritz 值或人为 safety
factor 不够。固定实移后验界本身不依赖选移规则，动态定理则可能依赖。

**连续动态认证与有限终止。** 2026-10-07 的
[正交有理时间原型](RATIONAL_DYNAMIC_PROOF.md) 给出固定基 `epsilon_imp(P;V)` 的
精确算术单元上界、精确时间尾项和公平搜索下的有限覆盖存在性；它直接认证动态系数，
没有证明 P0 的有限 matching 点传递。其认证逆作用成本不可接受，浮点运算未区间包围，
尚不能采用为新提取器。高效实际细分/升阶/锚点升级策略的上界一致收敛仍需证明。
后续联合创新能量界和精确参数包围细分，已在同一小模型固定基上把完整域认证 RHS
减少约 72%，并扩大 Case1 重建的可接受局部单元；这不改变最终 stock 基超限的事实，
也没有形成经济的提取加认证方法，详见 [优化记录](records/RATIONAL_DYNAMIC_OPTIMIZATION_20261007.md)。
随后 [带符号 Poisson–Loewner 界](SIGNED_POISSON_PROOF.md) 在余量中保留完整
时间/空间/输入交叉 Gram；有相对旧标量界无界增益的例子，同一 Case1 最终 76 阶
SVD 基、同一盒/试验/RHS 下，旧界 0.002126 拒绝、新界 0.001927 接受。
用户指定最终 SVD 动态阈值为 2 tau（Gram 阈值 4 tau^2）；嵌套路线把 raw 和截断各分配 tau。
该分预算只是充分条件，raw 超过 tau 不意味着最终 2 tau 目标失败。
后续 1200 单元复验中 raw 上界 0.001614 被分预算路线拒绝，但直接认证
101 阶最终 SVD 基得到 0.001762 < 0.002，独立正时间积分为 0.001757。
364/1200 单元重建矩阵与 C++ 原生装配相对差异不超过 1.3e-15，
详见 [进一步确认](records/SIGNED_POISSON_CONFIRMATION_20261007.md)。
这不推出标准 cutoff 的一般动态保证；stock 34 阶基在此放宽下仍超限。
76 阶为固定 SVD 方向中的证书验收结果，只取得局部连续盒保证，
详见 [数学判决与最终 SVD 实验](records/SIGNED_POISSON_EXPERIMENT_20261007.md)。
永久共享 `s=0` Gram 可能在大 `s` 存在不消失的地板，故仅细分 HTC 不足。

**边界耦合与任意功率。** 当前 `G` 的响应保证不覆盖未训练的附接热流输入。
完整 MOR 方法论须明确附接所需输入空间、场读出与热流验收；状态 impulse 误差也不自动给出
任意功率历程的相对系统范数保证。

## 4. 成本与研究纪律

新提取方法仅以容差与 HTC 范围为控制参数；不采用独立 cutoff、候选网格大小、种子数、
人为频段分割或富化轮数来调成绩。线性解精度等数值误差预算须从容差与分析推导。

```text
N_FOM = N_snapshot + N_selection_inverse + N_stopping/certificate_inverse,
N_total_audit = N_FOM + N_external_reference.
```

一次完整右端逆作用计一次，源列、边界列与 Riesz 列均须计费；谱 matvec、AMG setup、CG 迭代、
墙钟和内存分别记录。认证不是免费开销；降低快照数不等于提取性能改善。
比较应使用同一场目标、物理 HTC 盒、独立 holdout 与数值误差预算，并报告最终阶数。
只有满足全场误差约束且端到端性能优于 Extended BCI FANTASTIC，才可采用候选方法。
当前目录完成目标清理和审计，没有宣称已经得到该方法。

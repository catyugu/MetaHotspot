# Taylor jet 参数盒证书：针对最终 SVD ROM 的结温误差

这是一次数学路线及可行性核查，不是已经达到整个 HTC 范围 `1e-3` 保证的声明。特别地，当前实现给出 **BDF1 的前 40 步、全部 16 个热源至结温传递、同一参数的精确稳态传递归一化**的单个连续参数盒上界。可增加同一个锚点的 Taylor 阶数，以计算换取更大的合格参数盒；无随机测试点，也没有改动提取器的 `1e-3` SVD 截断。

## 一般性与假设

设 `A(p)=K+sum_i p_i H_i` 对参数盒处处对称正定，`H_i` 半正定，`C` 半正定，`G` 含任意多个输入/共址输出。令 `V` 为**已交付**的、线性无关的 SVD 后基底；这里对 `V` 的来源、被丢弃的奇异值和选择的采样点**不作假设**。时间步长 `dt>0`，`D=C/dt`，`B(p)=A(p)+D`。全阶和 Galerkin 递推为

```text
B(p) X_n(p)   = G   + D X_(n-1)(p),    X_0=0;
B_V(p) a_n(p) = V^T G + D_V a_(n-1)(p), a_0=0;
Y_n=G^T X_n, Y_n^V=G^T V a_n.
```

不要求离散矩阵为 M 矩阵，`G` 的元素也可以带符号。证明针对空间和时间均已离散的有限维模型；迭代求解、浮点 Cholesky/LU 和舍入误差尚须另行验证，才可称为严格数值认证。

## 整盒二阶上界

令参数盒 `[l,u]` 的中心为 `c=(l+u)/2`，半宽为 `h=(u-l)/2`，`delta=p-c`，`E(delta)=sum_i delta_i H_i`。只在 `c` 求一次原模型的时间轨迹和各坐标灵敏度：

```text
B(c) Z_(i,n) = D Z_(i,n-1) - H_i X_n(c),    Z_(i,0)=0.
```

同样计算最终 ROM 的灵敏度 `z_(i,n)`。则 `Y_n(p)` 的一阶项恰好是 `sum_i delta_i G^T Z_(i,n)`。设 `R_n(p)=X_n(p)-X_n(c)-sum_i delta_i Z_(i,n)`；把两个递推相减，得到**精确恒等式**

```text
B(p) R_n(p) = D R_(n-1)(p)
              - sum_(i,j) delta_i delta_j H_i Z_(j,n),   R_0=0.
```

因为 `B(p)>=B(l)>D`，`B(p)^(-1)D` 在 `B(p)` 能量范数中严格收缩，取不超过 `1` 的统一收缩上界便得到

```text
|G_a^T R_(b,n)(p)| <= ||G_a||_(B(l)^-1)
  * sum_(k=1)^n sum_(i,j) h_i h_j
                        ||H_i Z_(j,k),b||_(B(l)^-1).
```

将完全相同的推导用于缩减矩阵 `V^T K V, V^T C V, V^T H_i V, V^T G`，得到 ROM 的余项上界 `T^V_(ab,n)`。若 `e_(ab,n)(c)=Y_(ab,n)(c)-Y^V_(ab,n)(c)` 且 `g_(i,ab,n)` 是全阶与 ROM 的一阶导数之差，则整个参数盒上

```text
|Y_(ab,n)(p)-Y^V_(ab,n)(p)|
 <= |e_(ab,n)(c)| + sum_i h_i |g_(i,ab,n)|
    + T^F_(ab,n) + T^V_(ab,n).                     (1)
```

稳态归一化也可整盒控制。置 `X=A(c)^(-1)G`，`A_l=A(l)`，则 resolvent 恒等式给出

```text
Y_inf(p) = Y_inf(c) - X^T E(delta) X
           + (E(delta) X)^T A(p)^(-1) (E(delta) X).
```

最后一项的对角元至多为 `sum_(i,j) h_i h_j |(H_i X_a)^T A_l^(-1) (H_j X_a)|`，记为 `q_a`；非对角元绝对值至多 `sqrt(q_a q_b)`。据此，

```text
|Y_inf,ab(p)| >= |Y_inf,ab(c)|
                  - sum_i h_i |(X^T H_i X)_ab|
                  - sqrt(q_a q_b) =: d_ab.        (2)
```

只要 `d_ab>0`，将 (1) 除以 `d_ab` 就是**所有参数、所有指定时间步、指定端口对**同指标的相对误差上界。如果 `d_ab<=0`，该盒待细分，绝不可把它当作通过。稳态误差本身用相同的 resolvent 恒等式分别包络全阶和最终 ROM。

方程 (1) 的余项随盒半宽为 `O(||h||^2)`；但一阶误差项可能是 `O(||h||)`。因此一般情形下**不能**许诺全域只用 `O(epsilon^(-d/2))` 个盒，更不能从单变量 Zolotarev 的指数界推断多参数的无维数依赖保证。

## 任意阶与高维的延拓

上式是一阶 Taylor jet 加二阶余项。更一般地，令多重指标 `alpha in N^d`，`T_(0,n)=X_n(c)`；各系数由同一个锚点的算子递推：

```text
B(c) T_(alpha,n) = D T_(alpha,n-1)
                   - sum_(i: alpha_i>0) H_i T_(alpha-e_i,n).
```

对任意向下封闭的指标集合 `I`，构造 `P_(I,n)(delta)=sum_(alpha in I) delta^alpha T_(alpha,n)`；**精确**代数余项满足

```text
B(p) [X_n(p)-P_(I,n)(delta)]
 = D [X_(n-1)(p)-P_(I,n-1)(delta)]
   - sum_(alpha in I, i: alpha+e_i not in I)
                           delta^(alpha+e_i) H_i T_(alpha,n).
```

所以只需用 `B(l)^(-1)` 的范数及三角不等式，便能给出任意 `d`、任意向下封闭稀疏 jet 的**边界残差上界**。已实现的选项为总次数 `|alpha|<=r`；其余项严格为 `O(||h||^(r+1))`，需要的中心灵敏度右端数量为 `binomial(d+r,r)`，不需要 `d` 维参数张量采样。下一步可根据**已认证边界残差中各多重指标的贡献**按需增添指标，再在证书下降不足时细分参数盒。此规则可处理不对易的 `H_i`；但 `binomial(d+r,r)` 仍可能很大，未经额外各向异性/低有效维数假设，不存在免费的高维无维数保证。

一个严谨的选择准则是比较候选新指标带来的**计算工作量与已证明误差上界的下降**；若加入后界不降，仍可细分。这个准则还没有在整个 Case 1 参数域实现或测量，不能把小盒结果误称为已完成的全域自适应算法。

## 不需要经验阈值的选点和停止规则

1. 用既有 Zolotarev 点初始化快照；它只影响初始效率，不是保证成立的条件。照旧运行 `1e-3` 快照 SVD。
2. 对当前**最终基底**的每个参数盒计算 (1)–(2)。盒中心的全阶轨迹也给出此盒的真实误差下界。全盒上界 `< epsilon` 的盒直接退休。
3. 若中心的真实同指标误差 `>= epsilon`，就在该中心增加相应参数/频率的快照，**按原 SVD 规则重新压缩**，然后重新审核受影响的盒。否则可先提升同一个锚点的 Taylor jet 阶数或沿证书贡献最大的参数方向二分该盒；仅当重新计算得到更小的有效上界时接受提升。优先处理相对上界最大的盒。
4. 所有盒均退休才报告离散问题的全参数盒、给定 BDF1 时间窗保证；若 SVD 后的误差不能降到 `epsilon`，明确报告未认证，不能无限细分掩盖此事实。

这是一种 **输出导向的确定性分支定界**：中心误差决定是否需要增加快照，导数及已证明的余项决定细分。存在正的全域误差裕量且稳态分母处处不为零时，随着细分，区间上、下界趋向同一真实最坏值。然而原 `1e-3` SVD 本身不约束所有端口的相对结温误差，因此不能证明任意目标误差都可通过增加快照达到；全域复杂度也可能随参数维度指数增长。

## 原始两组 Case 1 的实测紧度

2.5 mm、9072 单元；**同一批** 92 个快照，最终 47 阶，原来的 `1e-3` SVD。以物理 HTC `(100,100)` 对应的有效参数为中心；半宽为中心有效参数的相应比例。下表是同一 ROM 上各个**整个二维连续小盒**的上界，时间窗为 40×50 s。各盒计算用到全阶中心轨迹与参数灵敏度，不计相同的快照提取成本。

| Taylor 阶数 | 各轴半宽 / 中心值 | 全盒最坏瞬态相对上界 | 全盒最坏稳态相对上界 | 时间/盒 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2% | 1.9224e-2 | 1.1565e-3 | 2.7 s |
| 1 | 1% | 4.7564e-3 | 3.0326e-4 | 2.7 s |
| 1 | 0.5% | 1.1909e-3 | 9.4091e-5 | 2.6 s |
| 1 | 0.25% | **3.0569e-4** | 4.2218e-5 | 2.7 s |
| 2 | **2%** | **2.7238e-4** | **7.1558e-5** | 4.5 s |
| 2 | 5% | 4.1363e-3 | 7.6850e-4 | 4.2 s |
| 3 | **5%** | **2.4861e-4** | **6.4531e-5** | 6.5 s |
| 3 | 10% | 2.5314e-3 | 6.5120e-4 | 6.4 s |
| 4 | **10%** | **2.7099e-4** | **9.4873e-5** | 8.3 s |
| 4 | 20% | 7.0328e-3 | 2.4597e-3 | 8.3 s |

中心的实测最大瞬态相对误差为 `2.0321e-4`。**一阶证书**在此处只能覆盖约 ±0.25%，但将同一锚点的阶数提到 2、3、4 后，分别能证明 ±2%、±5%、±10% 的连续盒满足 `1e-3`，无须增加 HTC 锚点。阶数为 2、±2% 的最坏瞬态界比阶数为 1 的界缩小约 **71 倍**。一阶 ±0.25% 和四阶 ±10% 盒各自独立求解了四个角点；最坏实测瞬态相对误差分别为 `2.0354e-4` 和 `2.1571e-4`，各角点各步绝对误差均未越过对应的整盒界。这些有限角点实验只做交叉检查，**整个连续盒的理论上界来自上述余项证明**。这仍未证明能够高效覆盖 `[1,10000]^2`：不同中心的误差、谱半径、所需阶数都会变化，提升阶数也增加灵敏度求解成本。

为避免只挑选容易的中心，另取物理 HTC `(5000,2)`，ROM 和快照均完全相同。此处中心实测最大瞬态相对误差 `1.8637e-4`，但某些跨端口稳态传递分母很小：

| 各轴半宽 / 中心值 | 阶数 | 整盒瞬态相对上界 | 整盒稳态相对上界 | 时间/盒 |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 4 | 1.2090e-1 | 4.0169e-3 | 8.2 s |
| 2% | 4 | 2.0580e-4 | 9.0457e-5 | 8.6 s |
| 10% | 6 | 1.3266e-3 | 2.3022e-4 | 13.4 s |
| **10%** | **7** | **3.5485e-4** | **1.9808e-4** | **16.4 s** |

这里提升阶数恢复了较大的可证明盒，但每个锚点多出了大量全阶灵敏度求解。两个中心均**没有**给整个 `[1,10000]^2` 的运行时间、所需锚点数或误差上界；不能据此宣称总体提取优于随机基线。

## 下一步数学改进

下一步优先实现上述**向下封闭的各向异性稀疏 jet**，并以真实的算子范数/边界残差分配计算；不必给每个方向同样的 Taylor 阶数。另外可保留原始快照张成的较大空间 `W` 仅作证明见证者，交付基底仍为 `V`：`Y-Y_V=(Y-Y_W)+(Y_W-Y_V)`。后一项在任何给定参数和时间由两个小 ROM 精确相减；前一项可用 `W` 的残差及整盒变化界控制。在正实频率和稳态还能用 Galerkin 正交得到 `Y_W-Y_V` 的半正定能量恒等式；BDF1 每一步没有同样的半正定结论。两种方法能否进一步降低**全域**认证工作量，有待同指标实验，而不是预设为成功。

相关基础：[Extended FANTASTIC 原论文](https://ieeexplore.ieee.org/abstract/document/9507439) 的随机 HTC 与 SVD 步骤；[Bachmayr–Cohen–Migliorati，仿射椭圆问题的解析与稀疏逼近](https://arxiv.org/abs/1509.07045) 阐述参数解析性需要的结构；[DeVore–Petrova–Wojtaszczyk，greedy 与 Kolmogorov 宽度](https://arxiv.org/abs/1204.2290) 给出逼近空间的速率背景；[Yano，exact solution certificate](https://epubs.siam.org/doi/10.1137/16M1071341) 对真正可核查的后验误差证明提供参照。这些文献没有直接给出这里 (1)–(2) 的 BDF1/结温参数盒公式；这里的推导应视为针对当前有限维模型的工作推导。

复现：

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/test_taylor_box_certificate.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_taylor_box.py --mesh-mm 2.5 --steps 40 --fractions 0.02 0.01 0.005 0.0025 0.00125 --output /tmp/bci-taylor-box-2p5.json
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_taylor_box.py --mesh-mm 2.5 --steps 40 --order 2 --fractions 0.02 0.05 --audit-corners --output /tmp/bci-taylor-box-order2-2p5.json
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_taylor_box.py --mesh-mm 2.5 --steps 40 --order 4 --fractions 0.1 --audit-corners --output /tmp/bci-taylor-box-order4-2p5.json
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_taylor_box.py --mesh-mm 2.5 --steps 40 --order 7 --center-h 5000 2 --fractions 0.1 --output /tmp/bci-taylor-box-hard-order7-2p5.json
```

实现仅在 `taylor_box_certificate.py`；`probe_taylor_box.py` 是诊断脚本，生成的结果文件保存在仓库外。现有测试还检查了任意符号多端口、**非对易**的三个半正定参数项以及稀疏/稠密的一致性。

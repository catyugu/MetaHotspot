# weekly_report_0915

## 非共形接口耦合

参考之前看到的 *Connecting MOR-based boundary condition independent compact thermal models*，目前的处理方法是不要求两侧原始 face 一一对应，而是在两套界面网格上再构造一层**公共面片**。
对任一侧 $S$，边界 cell 到物理界面的半单元热导为

$$
g_{S,j}=\frac{k_{S,j}A_{S,j}}{d_{S,j}},
$$

非共形时，把两边所有 face 的切向边界线合在一起做细分。每一个同时落在左右 face 内的最小矩形就是一个公共面片 $\mathcal F_{if,i}$。记 $E_S$ 为公共面片到 $S$ 侧原始 face 的关联矩阵，

$$
\xi_{S,i}=\frac{A_{if,i}}{A_{S,j(i)}},
$$

则该侧分配到公共面片上的热导为

$$
h_{S,i}=\xi_{S,i}(E_Sg_S)_i.
$$

实际代进去就是

$$
h_{S,i}=\frac{k_{S,j(i)}A_{if,i}}{d_{S,j(i)}},
$$

每个公共面片增加一个独立、无质量的界面温度自由度 $T_{\Gamma,i}$。对于 FVM 一侧，公共面片上的边界温度就是 $E_ST_{face,S}$；对于 ROM 一侧则直接用基底在界面上的 trace：

$$
T_{S,if}=E_SV_{b}q.
$$

记

$$
R_S=E_SV_{b},\qquad H_S=\operatorname{diag}(h_S),
$$

则两侧和公共界面节点的耦合块直接装成

$$
\begin{bmatrix}
K_L+R_L^TH_LR_L & -R_L^TH_L & 0\\
-H_LR_L & H_L+H_R & -H_RR_R\\
0 & -R_R^TH_R & K_R+R_R^TH_RR_R
\end{bmatrix}.
$$

## Embeddable ROM 的接口模式

- 上周把 EROM 的连接形式基本弄清楚后，这周继续看一个更根本的问题：接口虽然已经可以自由外接，但是当前的降阶基底其实并没有显式学过“接口温度作为输入”时的响应。
- 目前 `extract_rom()` 中，接口面和普通 BCI 面一样，只是作为一组 Robin 仿射项加入系统矩阵，真正生成快照时右端项仍然只有内部热源 $G$。因此实际训练的是

$$
A(s,\mu)^{-1}G,
$$

而外接以后系统还会受到

$$
A(s,\mu)^{-1}B_\Gamma u_\Gamma
$$

这一类输入。前者能让内部热源响应适应不同边界条件，但是并不能保证外部接口温度本身激发的传播模式也在基底里。

- 这和之前 FloTHERM EROM 的实验现象比较吻合：结温通常还可以，但是一旦外接条件变化，全场误差明显增大。09-01 的压力测试中，最坏 steady global error 约 4.15%，transient global error 约 4.02%。

### 接口空间的低秩性

- 看了一些相关工作，比较直接的是 Smetana / Patera 的 optimal port space。基本思想是把外部边界到内部/共享端口的响应看成一个 transfer operator，再取其主奇异方向；Buhr / Smetana 后续又给了 randomized local MOR，可以避免把每个边界自由度都逐个训练。
- 这里不应当以“任意接口输入都保证 1e-3”为目标。实际使用时，外部热网络产生的接口温度通常远没有这么病态，真正有意义的是那些能够传播到组件内部、影响结温和主体温度场的低频方向。

### 按热传播可达性训练接口方向

- 对均匀热传导，如果把接口上的温度分布按二维 Laplacian 模态分解，空间频率越高，沿法向衰减越快。设

$$
L_\Gamma\phi_j=\lambda_jM_\Gamma\phi_j,
$$

则对应模式在 Laplace shift 为 $s$ 时的衰减尺度大致和

$$
\sqrt{\lambda_j+s/\alpha}
$$

成正比，其中 $\alpha=k/(\rho c)$。

- 先试了一个很朴素的做法：仍然计算当前 ROM 对每个接口候选方向的 residual，但是再乘一个根据扩散距离估计的可达权重：

$$
\eta_j(s)=w_j(s)\frac{\|B_\Gamma\phi_j-A(s)x_{r,j}\|}{\|B_\Gamma\phi_j\|},
$$

$$
w_j(s)\approx\operatorname{sech}\left(L_{eff}\sqrt{\lambda_j+s/\alpha}\right).
$$

- 每次只取 $\eta$ 最大的方向做一次完整求解，直到它小于原来的 `ROM_TOLERANCE`。没有再引入 `port_modes`、`port_tol` 之类的额外控制量。
- 这里的 $\operatorname{sech}$ 目前只是规则区域的传播近似，理论上肯定还可以做得更严谨，先用来判断“按传播可达性筛接口模式”这个想法到底值不值得继续。

### simple_erom_case1

- 重新独立组了一个 15×15×15 的 FVM reference，几个代表工况和仓库里的结果能对到 0.001 K 左右：

```text
baseline                 11.57851 K   repo 11.579 K
bottom_htc_strong         6.41848 K   repo  6.418 K
external_source_200w      34.05352 K   repo 34.054 K
all_stress                57.09233 K   repo 57.092 K
layered_extreme_source   243.78694 K   repo 243.787 K
```

- 再看这些 full-FVM 解在接口上的二维空间频谱。13 个 stress cases 去掉均匀分量以后，前大约 9 个 surface modes 就已经能解释 99.998% 以上的变化能量。也就是说，虽然数学上接口有 225 个自由度，实际这些工况产生的接口温度确实极其低频。
- 原来的 source/FANTASTIC 部分得到 8 个状态，再用上面的接口 residual greedy 自动补了 11 个，最后一共 19 阶。

```text
                               FloTHERM EROM       reachable ROM
order                             10~11                 19
worst steady global               4.151 %             0.0838 %
worst transient global            4.022 %             0.0838 %
worst steady junction             0.987 %             0.00087 %
```

原来几个误差比较大的 case：

```text
external_source_200w      4.151 %  ->  0.0322 %
bottom_htc_strong         2.765 %  ->  0.0782 %
all_stress                3.542 %  ->  0.0838 %
layered_extreme_source    3.231 %  ->  0.0471 %
```

- 这个结果说明，FloTHERM EROM 那几个百分点的 global error 不是简单把 tolerance 调小就一定能解决，主要还是缺了一部分 interface-driven manifold。
- 阶数从 10~11 增加到 19，代价并不算特别大，但全场误差下降接近两个数量级。我觉得这个结果比“为了任意边界像素输入把端口空间做到几十上百阶”有意义得多。

### PoP-like 复杂模型

- simple copper block 毕竟太简单，又搭了一个 Package-on-Package-like 模型。尺寸和材料数量级参考公开 PoP thermal model：15×15 mm package，bottom logic die + top memory，里面有 Si / mold / substrate / interconnect，多热源，并且上下都有可连接接口。
- 这里并不是想复刻某一个商业 FloTHERM PoP 文件，主要测试多材料、两个接口和多个热源存在时，上面的低频假设会不会很快失效。
- bottom reduced component 共 1125 cells。FANTASTIC/source 部分 6 阶，接口再补 16 阶，最后 22 阶，weighted residual 为 $2.2\times10^{-4}$。
- 有趣的是，16 次接口增广全部发生在 $s=0$。看起来 FANTASTIC 的 shifts 已经把各种时间尺度覆盖得比较好了，接口额外缺失的主要是稳态/低频下的空间传播模式。可以粗略理解为：
    - FANTASTIC 补各种“时间常数”；
    - interface enrichment 补各种“接口空间尺度”。

比较正常的几组外接工况：

```text
case                      global error     logic junction
nominal                      1.287 %          0.175 %
memory_hot                   1.837 %          0.199 %
weak_interconnect            1.663 %          0.200 %
solder_like_interconnect     0.543 %          0.085 %
weak_board_cooling           0.518 %          0.071 %
lowk_board                   1.604 %          0.340 %
highk_board                  0.961 %          0.066 %
```

- 这几组里 global error 最大约 1.84%，结温最大约 0.34%。以 22 阶来说，我觉得已经是可用的范围。
- 如果故意把条件推得比较远，比如很强的 board cooling、非常偏心的 memory heat source，global error 会升到 5~7%，更极端组合还会更高，但是结温多数还在 1% 左右。
- 暂时不打算为了这种输入继续把阶数往上堆。实际产品里如果大部分工况能维持结温 <1%、主体温度场约 1~2%，阶数在二三十以内，应该已经比追求任意接口输入一致精确更合理。

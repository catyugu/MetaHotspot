# 连续 HTC 动态全场认证：调研与原型判决

日期：2026-10-07。基础提交 `d7e229bca40b94d75d332827097d6ecd86fc2317`，分支 `agent/work`。

## 判决

得到一种有完整精确算术证明、能覆盖连续参数单元和无限时间轴的认证构造。
32 自由度导热链的完整二维 HTC 盒通过解析接受条件；Case1 矩阵重建模型的一个
很窄局部盒通过。**认证 RHS 成本不可接受，不采用为新提取方法。**

另一个保留 full/reduced 残差抵消的耦合原型也有上界证明，但实测比独立递推更差。
失败发生在约化误差补偿的标量算子范数上，而非“缺陷相减”本身。

所有运行 `floating_point_certified=False`。这里“通过”指精确算术接受定理的浮点实现
算得上界小于容差；线性解、Gram、特征值、插值和矩阵运算的舍入尚未区间包围。
样本用于实现核对，不被用来代替连续域证明。生产提取代码未修改。

## 调研所得与选择理由

已有的 TM/Laguerre 正交有理时间展开、ADI 的正因子与 residual-factor 思想适合
把 impulse 误差直接表达为有限系数缺陷加剩余能量。与此前失败路线相比，本构造：

- 不从有限 matching 点缺陷直接推断整个动态误差；精确计算系数递推并保留尾项。
- 不求完整 n 阶 Lyapunov/KYP SDP，也不对参数依赖的约化特征向量求导。
- 用 dual `K(h)^{-1}` 度量传播 RHS 缺陷，基本常数为 `1,2,1/sqrt(2)`，
  避免 `alpha^{-3}` 链条。没有宣称因此消除了病态性影响。
- 用多项式残差的 Bernstein 控制矩阵覆盖参数单元内部，而非取节点最大值。
- 用小输入空间的完整谱范数，保留所有输入组合。

完整推导、精确尾项恒等式、连续单元接受定理及有严格余量时的有限覆盖存在性见
[RATIONAL_DYNAMIC_PROOF.md](../RATIONAL_DYNAMIC_PROOF.md)。理论还给出公平枚举搜索的
有限终止条件；没有宣称当前固定 sweep 是高效有限终止提取器。

主要文献与阅读边界在证明文件第 8 节。正交有理时间基、ADI 与 rational Krylov
不是新贡献；连续 HTC、all-input relative full-state impulse、成本受控三者能否
形成新的可发表方法，仍需进一步工作。

## 复现条件

- Python 3.12，NumPy 2.3.5，SciPy 1.17.0，pyamg 5.3.0，pytest 9.1.1。
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。
- stock 源码不变：容差 `1e-3`、seed `20260805`、默认 3 个随机接受轮。
- stock 提取仍用 AMG-CG；原型 full trial/Riesz 用稀疏 LU。
- 全阶 RHS 和稀疏分解数分别计数，不能把当前 tiny FOM 墙钟外推到大型模型。
- 独立动态参考使用完整 FOM/ROM 广义特征分解与解析无限时间积分；这是外部参考成本，
  不算认证成本。小模型另有直接时间积分的独立单元测试。
- 最终 SVD 基与 raw 快照基分别绑定阶数与 SHA256，不继承彼此证书。

```bash
PYTHONPATH=python python -m pytest \
  playground/adaptive_bci_sampling/test_field_audit.py \
  playground/adaptive_bci_sampling/test_rational_dynamic.py -q -p no:cacheprovider

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
python playground/adaptive_bci_sampling/run_rational_dynamic.py \
  --model chain --basis-stage raw --terms 96 --degrees 3 --widths .02 \
  --cover-cells 4 --cover-degree 7 \
  --output playground/adaptive_bci_sampling/results/rational_chain_cover7.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
python playground/adaptive_bci_sampling/run_rational_dynamic.py \
  --model case1 --basis-stage raw --terms 128 --degrees 0 2 3 \
  --widths 1 .02 .002 .0002 0 \
  --output playground/adaptive_bci_sampling/results/rational_case1_raw_contraction.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
python playground/adaptive_bci_sampling/run_rational_dynamic.py \
  --model case1 --terms 128 --degrees 0 3 --widths 1 .0002 0 \
  --output playground/adaptive_bci_sampling/results/rational_case1_final_current.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
python playground/adaptive_bci_sampling/run_rational_dynamic.py \
  --model case1 --basis-stage raw --terms 128 --degrees 0 2 3 \
  --widths 1 .02 .002 .0002 0 --propagation coupled \
  --output playground/adaptive_bci_sampling/results/rational_case1_raw_coupled.json
```

`--terms/--degrees/--widths/--cover-*` 是判决实验设置；不是新提取器的用户调参接口，
不能将某个 sweep 的最佳项当作仅以目标容差为输入的完成算法。

## 实验 A：整个二维 HTC 盒的解析覆盖

32 单元保守导热链，两个功率输入，两个端点 Robin 组，参数盒 `[1,4]^2`。
stock 提取 26 次 RHS，最终 SVD 基 13 阶；同一批快照的 numerical-rank raw 基加
常温方向为 27 阶。以下完整覆盖使用 **27 阶 raw 基**，不代表 13 阶交付基通过。

| 设置 | 完整盒最大上界 | 独立中心样本最大误差 | 认证 RHS | 判决 |
|---|---:|---:|---:|---|
| 每轴 4 单元，p=6，N=96 | 1.35716e-3 | 约 6.95e-5 | 349232 | 未通过 1e-3 |
| 每轴 4 单元，p=7，N=96 | 4.37477307e-4 | 6.94847232e-5 | 448080 | 16 个连续单元全部通过 |

后者生成原始报告的整个运行约 8 秒，只说明 n=32 的诊断可运行，不证明生产成本低。
单独覆盖的端到端全阶 RHS 至少为 `26+448080=448106`，还未把独立谱参考折算成
同一种成本。这比 26 次 stock 提取高四个数量级。

raw 基 SHA256：`ae3c29c35ad1cfabc7d914170fb9a27a93d6f11c25cde607dbebe5512e01ca5f`。

## 实验 B：Case1 重建模型的局部收敛及失败

本环境缺少 `libmhs_c_api.so`；`--model native-case1` 因此失败。
为完成原型实验，根据 `Case1Config` 网格、层/芯片/空气材料与原生组装器的调和面
导热公式建立纯矩阵重建。10 mm 网格共 364 自由度、四个单位功率输入。
独立检查 K 行和、边界总面积、源列总功率及正质量；**未与原生矩阵数值逐项核对**。
所以以下结果不能替代此前更细网格/85 阶基的 Case1 记录。

物理 HTC 两轴端点均 `[1,1e4]`，实际仿射参数为有效系数：
`p1 in [0.9999923,9285.7143]`、`p2 in [0.9836066,59.6421471]`。
单元中心标记为 `c=sqrt(a*b)`；width fraction `f` 的盒为
`[c-f(c-a), c+f(b-c)]`，按**有效系数**计算，不是物理 HTC 的相对宽度。

stock 提取 94 次 RHS；最终 SVD 基 34 阶，raw numerical-rank 快照基加常温方向为
95 阶。固定 raw 基、N=128 的独立递推结果：

| width fraction | p | 连续单元上界 | 独立点最大误差 | 认证 RHS | 判决 |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 7.38146e3 | 1.02308e-3 | 2581 | 粗盒界太松；样本本身也超目标 |
| 1 | 3 | 2.20318e5 | 1.02308e-3 | 21097 | 增加次数不能解决粗盒问题 |
| .02 | 3 | 8.78963e-1 | 9.92119e-4 | 21097 | 余量主导 |
| .002 | 3 | 1.22545e-3 | 9.91897e-4 | 21097 | 未通过 |
| .0002 | 2 | 9.98690134e-4 | 9.91549902e-4 | 12869 | 极窄局部盒通过 |
| .0002 | 3 | 9.94772857e-4 | 9.91549902e-4 | 21097 | 更紧，但更贵 |
| 0（单点） | 0 | 9.91471286e-4 | 9.91471149e-4 | 2581 | 点态校验接近等式 |

p=3、f=.0002 的主项 `9.94750192e-4`，递推余量 `2.04576e-8`，两条尾项合计
`2.20763e-9`；余量已很小，剩下约 0.3% 的松弛主要来自单元统一分母/控制矩阵。
p=2 认证 12869 RHS，加提取至少 12963 RHS，比 94 RHS 提取贵约 138 倍。
没有构建整个 Case1 参数盒的通过树。

raw 基 SHA256：`a74ce9a2dba5cab8204dd4dcb21461189db7b43ec7a259eb290c24b0029cbec0`。
final 基 SHA256：`dcdd3a3e62a16e11de9b6b0ec2eefda4cebb9a24da7941898d51e21404524eae`。

## 实验 C：终止证书必须针对最终压缩基

同一组 94 次响应快照、同一参数中心，独立参考给出：

| 基 | 阶数 | all-input 动态误差 | 四列单独误差范围 |
|---|---:|---:|---:|
| raw + 常温 | 95 | 9.91471149e-4 | 6.6588e-4 至 8.9543e-4 |
| stock 最终 SVD | 34 | 1.75905853e-2 | 1.2475e-2 至 1.4826e-2 |

动态误差放大约 17.74 倍。这个点的 `cond(Q)=2.3725`，不是近秩亏分母导致的假象。
最终基单点的有限时间系数前缀约 `1.75905853e-2`，完整上界也约该值，说明本例
不只是连续盒证书太松：最终基本身不满足 `1e-3` 动态目标。

这支持检查“归一化 resolvent 快照的 Euclidean SVD cutoff 与当前动态指标失配”这一
假设。它不证明 stock 文献自己的误差指标有错，也不代表任意模型都会如此。
raw 基在全盒另有超目标样本，因此 raw 基也没有全域通过。

## 实验 D：耦合缺陷标量补偿的负结果

耦合变体用 `fE=fF-CV Cr^{-1}fr` 保留残差相减，但还必须包围约化系数误差通过
`R=KV-CV Cr^{-1}Kr` 注入全阶递推的贡献。不能省略该项。

| raw 基单元、p=3 | 独立递推上界 | 耦合递推上界 |
|---|---:|---:|
| 根盒 | 2.20318e5 | 1.01863e9 |
| width .002 | 1.22545e-3 | 4.23028e-1 |
| width .0002 | 9.94773e-4 | 1.04207e-3 |

根盒 `kappa_R≈9.724`、`kappa_L≈26.853`。慢 shift 上补偿含
`kappa_R/sqrt(2 sigma_k)`，约化误差被先变成标量后乘一个对全部方向取的最坏增益。
因此即使保留了 `fE` 的抵消，补偿仍支配界。该变体记录为**界太松**，不采用。
它不否定保留约化误差完整矩阵/频率方向的耦合方法。

## 测试与工程状态

已有场审计基线：`3 passed`。新测试先因原型模块缺失失败，再实现并核对。
最终相关测试：**15 passed**（3 个原场审计 + 12 个新测试）。
包括精确尾项/Parseval、独立无限时间 quadrature、非共址全部输入组合、状态坐标
变换不变性、matching 完美但动态不完美的反例、秩亏拒绝、合法谱区间及耦合变体。
这是研究原型，没有改动生产提取/求解接口，没有提交原始生成结果到版本库。

## 下一项值得做的判决实验

1. **先恢复原生 Case1 对照并重做 raw/final 动态审计。** 当前重建结果不能当作
   原生生产模型的确定结论。核对网格/空气背景/面导热/有效 HTC，并固定 basis hash。
2. **研究动态指标下的压缩验收。** 对最终基重新认证；可比较 C 加权时间系数的
   压缩候选，但 projection/POD 误差不自动等于 Galerkin 动态误差，不能继承证书。
3. **保留约化误差方向再做 compensation。** 不继续单纯增加多项式阶数或时间项数。
   先比较 `R delta y` 的实际 K-dual Gram 与 `kappa_R ||delta y||`，确认损失来自何处，
   再构造矩阵化的连续包围。将约化误差强行标量化这一版已经被本实验判掉。
4. **设端到端 RHS 判决门槛。** 本次连续覆盖比提取贵几个数量级，只能作为独立
   验证工具。共享分解/缓存可能缩短墙钟，不会改变当前数学构造的逆作用规模，
   不把这些工程优化当作主问题的解决。

当前没有得到满足全场动态目标且成本优于 Extended BCI FANTASTIC 的新方法。

# 输入可达空间、正性尾项及轮廓残差：补充文献核对

日期：2026-10-09。基于 `2f7a709` 的现有结构性文献记录；本次另行检索
positive reachability / Dyson--Phillips / Feynman--Kac / modal comparison /
rational Krylov residual / contour parametric MOR / time-uniform real poles。
结论是适用条件核对，不是“文献里没有此方法”的证明。

| 工作 | 本次证据 | 与本问题的关系和未闭合部分 |
|---|---|---|
| Gugercin & Manucci, *Contour integral methods and model order reduction for parametric linear control systems*, 2026, [arXiv:2608.05363](https://arxiv.org/abs/2608.05363) | 全文；Sec.3、Lemma 3.1、Algorithm 1、Remark 3.2 | 直接连接移位残差和有限时窗的参数输出误差；局部频率空间及 primal/dual 残差可用。指标是式(3.11a)对输入/初值归一化的 CIM 输出误差，并非每个 t>0 的相对全场阶跃误差。Algorithm 1 所需 max over P 仍要实现并认证，不能用训练网格冒充。全文里的部分行文称 Theorem 3.1，实际展示为 Lemma 3.1。 |
| Simunec, *Error bounds for the approximation of matrix functions with rational Krylov methods*, 2023 preprint, [arXiv:2311.02701](https://arxiv.org/abs/2311.02701) | 全文 Sec.2--3；式(1.4)、(2.10)、Theorem 3.1 | 通过移位线性系统残差表达矩阵函数误差；可保留最后真实输入残差及方向。移位残差共线来自固定 A 的 rational Arnoldi 关系；一般 HTC 算子族与九点快照空间不自动享有这一关系。 |
| Güttel & Shao, *Uniform-in-time rational approximation of the matrix exponential with real poles*, 2026, [arXiv:2607.18018](https://arxiv.org/abs/2607.18018) | 全文 HTML，Sec.1--4、Algorithm 1 | 共享实极点、全非负谱区间及有限正时间窗；明确处理线性求解误差、留数和求和误差。能够改善时间近似工具，但共享极点不等于共同 HTC 可达空间，且其函数绝对误差指标不等于本项目全时间相对阶跃指标。 |
| Beattie/Gugercin/Tomljanović, *Sampling-free model reduction of systems with low-rank parameterization*, 2020, [arXiv:1912.11382](https://arxiv.org/abs/1912.11382) | 本次获取作者全文，延续原记录 Sec.2 的核对 | 参数反馈整体保留已是先行工作；原热边界作用秩为 3335/6520，不能把二参数偷换成秩二。低秩前提、反馈保结构和范数指标要逐项满足。 |
| Duff/Grundel/Benner, *New Gramians for Linear Switched Systems: Reachability, Observability, and Model Reduction*, [arXiv:1806.00406](https://arxiv.org/abs/1806.00406) | 本次获取作者全文；原记录已核对定理条件 | 作用链、递归核和可达 Gramian 均不是新想法；共同包络要满足完整不等式，不能从有限轨迹协方差直接推出。 |
| Redmann, *Type II Balanced Truncation for Deterministic Bilinear Control Systems*, [arXiv:1709.05655](https://arxiv.org/abs/1709.05655) | 本次获取作者全文；沿用原记录条件核对 | 有界控制进入 Gramian，尾谱提供确定性 L2 指标。固定耗散 HTC 与物理功率缩放要分开，常输入无限时间和逐时刻相对误差仍需额外证明。 |
| Pitman & Yor, *Kac's Moment Formula and the Feynman--Kac Formula for Additive Functionals of a Markov Process*, [作者讲义](https://mathweb.ucsd.edu/~pfitz/downloads/kac/kac.html)、[Feynman--Kac 节](https://mathweb.ucsd.edu/~pfitz/downloads/kac/node11.html) | 读作者原始页面 | 状态依赖 killing 的正核及 Poisson 表述是经典工具；本次用它证明有限维对角 Robin 扰动的确定性 Poisson 余项，绝不把“Poisson”解释为概率置信保证。 |
| Li/Lam/Shu/Du, *H-infinity model reduction for positive systems*, 2010, [作者存档 PDF](https://hub.hku.hk/bitstream/10722/159019/1/Content.pdf)；2011 positivity-preserving H-infinity paper, [Brunel 存档](https://bura.brunel.ac.uk/handle/2438/6065) | 检索返回作者/机构摘要；后者全文未成功抓取 | 正性、稳定性、H-infinity 性能联合降阶已有工作。没有据摘要认定其满足连续 HTC、逐时刻相对全场的本项目指标。 |

## 本次数学尝试的位置

把真实输入的最后一个正 Dyson 核送入 K(cell_low)^-1，得到非交换扰动下的
连续盒、全时间绝对尾项；再组合 Poisson 核余项和四输入 spectral Jensen
下界，补齐该**参数尾项**的全时间相对指标。另用实际 Galerkin 模态中的
M-matrix 比较系统构造共同输入驱动锥，利用四输入终端残差响应界定全部
遗漏空间方向。完整独立推导在 `../POSITIVE_SOURCE_TAIL_PROOF.md`。

这些推导没有发现可直接套用的完整既有定理，但所用组件都是经典理论。
目前不存在可据此宣称的新低成本认证提取算法，也没有完成新颖性判定。
正文不把采样的误差 SVD 尾谱当作这两项确定性尾界。

## 对后续路线的实际影响

1. 轮廓残差路线应保留为候选：能够保持 signed residual，而不是先取绝对值。
   但需要真实的连续参数最大值证书、积分/无穷尾证书、近零和长时间组合。
2. 正性可为被舍弃的参数核提供不依赖整个能量球的安全尾项；它不能自动
   保证 signed Galerkin 残差的包络紧致。取绝对值会激活原来抵消的慢模。
3. 若继续构造可达 covariance，应保存输入与残差的交叉项，再用正终端
   响应控制剩余部分；不能把全部主项也塞进正 orthotope。
4. 更换极点或提高阶数只可作为工具/诊断，不足以构成学术创新。

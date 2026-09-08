# expr0908 方法调研笔记：BCI 提取算法的备选与近期进展

（供周报第三部分使用；结论均以仓库论文原文为准，标注出处。）

## 1. 当前实现所处的谱系（忠实性核对）

- 当前生产实现 = Extended FANTASTIC（Codecasa et al., “Galerkin's Projection
  Framework for BCI CTMs. Part I: Extended FANTASTIC Approach,” IEEE TCPMT
  11(11), 2021）Algorithm 1 的逐行忠实实现：
  - 每个热源 port 独立估计谱端点 (λmin, λmax)（FANTASTIC 2014 Step 1）；
  - 由容差 ε 经 Zolotarev 椭圆最优频点计数 m（4exp(−mπ²/log(4/q′))≤ε），
    dn 分布实位移（2014 Eq. 4-5；2021 版本指数对 m 线性，仓库已按 2021
    修正）；
  - 每 (port, shift) 随机抽 h∈H 探针，残差 ρ>ε 才做一次全阶求解
    (σM+K+Σh_kH_k)φ=g_k（2021 Eq. 26），增量投影扩展（2021 的增量式
    M̂, K̂, Ĥ 更新，论文显式注明可增量做），残差驱动停：连续 probe_rounds
    次随机 h 均通过才认证；
  - 收尾对快照矩阵做 SVD，按 ε·σmax 截断（2021 收尾；RomCore 反推显示其
    用 ε/10 截断，仓库按此对齐），并可加常数列以允许环境温度平移。
- 理论保证：MPMM 收敛指数级（2014 Sec. 2 的 zTH 能量不等式），对"有限
  仿射边界参数族 + 源的功率输入"族保证误差 ≤ε——这正是 BCI 语义。

## 2. 备选方向逐条评估

### 2.1 per-shift 矩迭代（moment continuation）——本周实验
- 来源：Codecasa 的 MPMM 原始谱系（“An Arnoldi based thermal network
  reduction method,” 2003；“Multipoint moment matching reduction from port
  responses of dynamic thermal networks,” 2005）本就是每频点多矩；FloTHERM
  RomCore 二进制反推（expr0907）显示其内部为 per-(source,shift) 的正交
  Krylov 矩迭代（~2x 更少全阶求解）。
- 与忠实实现的差异：失败探针处对同一算子 A(σ,h) 计算 k 个矩
  v_{j+1}=A⁻¹C v_j，共享一次 AMG 层级；认证语义、收尾 SVD 不变。
- 预期收益：AMG setup 占 enrich 57.6%（E1 实测），k 个矩共享 setup ⇒
  setup 次数 /≈k；代价是 CG 求解数 ×k（同算子 warm-start，迭代数应下降）。
- 风险：矩链饱和导致部分求解空转（正交化后无新方向）；SVD 收尾可能保留
  更多列（阶数略升）。本周用 k=1/2/3 A/B 实测（moment_enrich_ab）。

### 2.2 贪心/残差估计类（RB 方法）
- Reduced Basis（Patera-Rozza 谱系，RB 2016 综述）用 a posteriori 误差估计
  选点代替随机抽参。BCI 2015 论文明确说明"参数随机选取以避免贪心停滞"
  （vs. 文献 [10] 的 Greedy），并引用 Sommer 2015 说明精确残差计算开销。
- 结论：不改。当前随机 h + 残差驱动已达成贪心的自适应效果，且无
  误差估计器误差（污染）风险。RB 的"在线/离线"框架与 BCI 训练族语义
  一致，但误差估计器对无穷参数族（H 为连续区间）要做 min-max 化，实践中
  更贵。

### 2.3 多端口 RC 网路 MOR（TurboMOR / SMP-RCR / FlexRC，TCAD 线）
- 面向"端口极多"的寄生 RC 网络：TurboMOR 2015（矩匹配 + 端口低秩聚合）、
  SMP-RCR（稀疏多点矩匹配）、FlexRC 2023（非正交投影 + 稀疏带状降阶模型）。
- 与热 BCI 的差异：它们把"端口数"当主体降维对象（很多端口、每端口简单
  响应）；BCI 把"边界参数族 + 动态谱"当主体。热封装端口（4-18 个源 +
  若干边界）远少于 RC 寄生端口，直接借用的收益有限；但 FlexRC 的"保留
  稀疏结构"思想对多边界组 DCTM 的密度控制有启发（未纳入本周实验）。

### 2.4 可组合系统级方法（2024-2026，上海交大 Tang/Mao 组）
- DDR（Domain Decomposition and Reduction, 2024）：把系统分为核心区与非
  核心区，非核心区用界面等效耦合矩阵表示（DtN 型），10x 加速——这是
  "用 DtN 算子代替完整网格"的静态约简，与 BCI 动态 DCTM 分层不同，但
  与我们的"界面独立节点 + 单侧半导纳"耦合在数学上同源（都基于界面
  互阻抗/导纳等价）。
- DTDM（Domain Transformation and Decomposition, 2025）：Laguerre 时域
  展开 + 按模块分解、模块界面温度/热流映射的"可组合宏模型"。对瞬态
  **激励波形族**可组合是亮点，但模块间仍需界面变量一致；与我们 EROM
  的"提取一次、随处连接"目标一致，区别在于时域基底（Laguerre vs
  频域 MPMM）。
- HDI（Hierarchical Decomposition and Interconnection, 2026）：
  线性区 ZI/ZS 分离 + 非线性区局部迭代，处理温度相关参数。热物性随
  温度变化（非线性）是 BCI 之外的另一维，未纳入本周。
- 结论：这三个都是"系统仿真层"的组合策略，不回答"单个子域如何高效
  提取 BCI 基底"；作为耦合层的佐证（界面变量 + 导纳映射的普遍性）写入
  周报。

### 2.5 FloTHERM 商用对照（Simcenter Flotherm BCI-ROM Validation 2020.2）
- 商用 BCI-ROM 的提取语义即本文谱系；RomCore 以 per-(source,shift)
  矩迭代实现，等价于 2.1 的更激进版。仓库 reverse-engineering 结论
  （binary-rom-tracing / flotherm-romcore-re）与此一致。

## 3. 结论

- 算法级替代中，**per-shift 矩迭代**是唯一既有论文谱系支撑、又可直接
  差分验证的方向（本周 E2 实测）；其余方向（RB 贪心、多端口 RC、系统级
  组合）要么语义不匹配、要么层面不同，作为观察记录。

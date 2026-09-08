# BCI-ROM 提取方法调研笔记（expr0907, Q2）

日期：2026-09-08
范围：仓库 ref 库内论文 + 已确认的 FloTHERM RomCore 反推结论。
目标：为 "提取算法本身是否有更好的选择" 提供依据。

## 当前实现（已确认与论文/二进制一致的部分）

- 逐热源（per-source）谱估计 → 椭圆移位计数 m_i（Zolotarev 最优）
  → dn 分布移位 → 每个 (port, shift) 随机 h 探针 → 失败则单列全阶求解 enrich
  → 收尾 SVD（RomCore 语义: eps/10 · ||s||_2 前缀截断, +1 常数列）。
- 基线 122k 单元: 4 port × 14 shifts, 154 enrich solves / 360 probes,
  提取 88.5 s（AMG setup 54 s + CG 19 s），40 阶，相对响应误差 4.3e-7。
- 与 RomCore 二进制一致：srand(1)+2500 随机池、tol 派生迭代上限、
  tol/10 预处理、全局 eps·||s||_2 SVD 前缀截断 + 常数列。
- 差异/开放项（RE 笔记）：FloTHERM 每 shift 生成的是**正交 Krylov 矩基**
  （每次算子 matvec + 重正交），而非独立的频率响应全阶解；这就是它 ~2×
  少解的来源。我们的单列全阶解每候选一次 AMG setup。

## 论文对照（提取方法层面）

1. Extended FANTASTIC (TCPMT 2021 Part I) —— 已忠实实现。确认其停止准则
   就是随机 h 探针残差（无训练/测试集，全自动，容差即精度）。

2. FloTHERM RomCore 二进制反推（本仓库 skill 笔记）：
   - 内层是 per-(source, sigma) 的 **Krylov/Lanczos 矩迭代**（FUN_180157a80
     单步 = w -= A·x 稀疏 matvec + 2-范数 + beta 缩放）→ 比"每个移位做一次
     独立全阶频率响应"节省约一半全阶求解。
   - SVD 截断是全局 eps·||s||_2 前缀 + 常数列（已确认）。
   - 论文复现实验：纯 sigma 移位求逆 Krylov（m0=A^-1 g, m_i=A^-1 C m_{i-1}）
     在 case1 上精度不足（中心 h 灾难性，~3%）；随机 h p=1 达 57 阶 0.25%，
     p=2 达 89 阶 0.012%。当前落地的"独立频率解 + per-source top-m_k 压缩"
     是经验上唯一达 ~50 阶且精度≥FloTHERM 的构造。→ 忠实复刻其矩迭代
     仍是开放大任务（需要读 60KB 反编译内核），不作为本次实验目标。

3. Connecting MOR-based BCI CTMs (THERMINIC 2017) —— 我们非共形耦合机制的
   直接出处（Sec.5 公共面片 E/xi）。PoP 例子 2×25 万单元, BCI DCTM 各 30 阶
   提取 <5 min。**没有**对提取算法本身的改进。

4. DDR / HDI / DTDM (上海交大, 2024-2026, 2.5D chiplet)：
   - DDR：静态 N 端口等效热阻矩阵，把非核心区缩成一个端口网络。缺点：
     仅稳态、只对均匀接口热流假设好。与 BCI 不同——端口是"固定连接面"，不是
     边界条件无关。对我们的问题（BCI-CTM 提取优化）不是直接替代。
   - HDI：ZI/ZS 分解把非线性迭代限制在线性区。面向温度相关材料，非 BCI。
   - DTDM (ASP-DAC 2026)：Laguerre 时域变换（Galerkin 权函数正交消除时间），
     每模块界面热流-温度映射矩阵，485× 加速。有意思但面向系统级瞬态仿真的
     **可组合模块**，与我们"提取 BCI 动态紧模型"的目标不同层；其"时域基=
     加权 Laguerre 多项式"思想若用于替代椭圆移位-频域采样，需要大改
     认证语义，超出本次范围（周报里可作为"探索方向"提及）。

5. 多热源/大规模源 (2016-2018 Codecasa)：热源数量多时"逐源基底"的复杂度
   从二次降到线性。Case1 只有 4 源，不是瓶颈。

结论：**没有比"当前随机 h 探针 + 残差驱动 enrich"在精度语义上更优的
现成替代**——FloTHERM 同门更高效之处在于 Krylov 矩迭代的紧凑列构造
（同一 shift 内多列共享一次算子分解/矩阵向量模式），以及更省的 h 采样
（每 shift 响应数量更少）。这些是"如何用更少的全阶解得到同精度基底"的
方向，正好由我们的 E1（AMG 参数）、E2（求解语义/rtol）与潜在的
"multi-shift / block-Krylov 扩展"实验承接。

## 我们当前实验能落地的算法层改进（候选, 按证据强度排序）

A. AMG interpolation=direct（E1 已验证：setup 减半, 迭代不变）→ 改 utils.py
   默认求解器参数（若端到端验证通过）。
B. enrich rtol 从 1e-6 放宽（E2 验证中）：响应只进入快照池+重正交，
   认证在 1e-3；1e-6 可能过杀 → 减 CG 迭代。
C. warm-start 语义（E2）：跨 (shift,h) 复用 x0 是否真的省迭代，冷启动对照。
D. 同一 (shift) 的多 h 响应若共用 AMG hierarchy，可从 154 次 setup 降到
   ~56 次（每 shift 一次）——但这是 caching，用户已排除；不列入。
E. 每 (shift, h) 的多列/块 Krylov（如同时解 g 和 C g 方向）压缩 solve 次数；
   尚未实验，风险高（改变列构造会影响最终阶数/精度）。

## 与导师周报的衔接

- 提取性能对比（我们 vs FloTHERM）之前周报已报 12 万 66s vs 78s。
  本轮目标：用 E1/E2 的实测把提取成本再降（目标 ≥30%），并保持 40 阶
  与 ≤0.01% 结温误差。

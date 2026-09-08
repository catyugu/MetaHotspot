# BCI-ROM 提取优化工作计划（expreriment/expr0907）

日期：2026-09-08
基线（已验证）：
- 分支 expreriment/expr0907 @ a322c46，工作树干净
- python metahotspot 122400-cell case1（4 ports, 2 groups）：基础提取 40 阶，88.5 s
  （求解 79.2 s 其中 AMG setup 53.7 s，eigenbounds 5.5 s，probe 4.1 s，投影扩展 1.3 s）
- reproduce_case1: our ROM vs FloTHERM ROM vs full FVM 全部 < 0.01% steady 误差
- tests 93/93 通过

## 优化目标（三个问题）

Q1 提取性能： 现有 88.5s 中 AMG setup 占 ~54s（61%）。且 warm-start x0 语义可疑——
   `_enrich` 的 x0 跨不同 (shift, h) 复用，但不同 shift 的方程组差异大，warm start 收益存疑。
Q2 提取算法调研： 结合更近期论文（Codecasa 组 2019+、以及本仓库 ref 库中其它近期方法）
   寻找比当前"random-h 探针 + 单列 Krylov 补充"更优的替代/改进。
Q3 非共形网格 EROM： 理论与实验都做。仓库已有 `common_patches`（公共面片细分 +
   面积权重 E/ξ），需设计非共形案例验证精度/守恒。

## 已排除

- 不同 port 间共享基底：精度有问题（历史结论，见仓库 note）。
- 纯调参型工作（调 tolerance / max_order 之类无意义）。

## 实验计划（诊断先行，逐项验证）

E1 AMG 参数诊断（首要，成本低）：
   - Case1 122400-cell 模型上：用同一组 154 个 (shift,h) 候选，对比
     a) 当前 Ruge-Stueben 默认参数
     b) RS 强度阈值/插值参数微调
     c) 平滑聚合(SA) + CG
     d) 直接稀疏 LU（对比）
     e) CG 无预条件（对照）
   - 记录：setup 时间、每 solve 迭代数、总时间。看 setup 时间是否能降一个量级。
   - 判定：若 AMG 总成本 vs 直接 LU 无显著优势（<2x），考虑用带 fill-in 的
     Cholesky/ILU；若 setup 是主要成本，考虑 reuse（按 shift 分组复用同一个 AMG？）
     以及减少 solve 次数。

E2 求解调用次数与 warm-start 语义：
   - 数一下 probe 与 enrich 的调用分布：154 solves / 360 probes。
   - 实验：关闭跨 shift 的 x0 warm start（每个 shift 冷启动或只同 shift 内 warm），
     对比收敛性与总时间。若 x0 语义错误导致 CG 迭代更多，修正。
   - 若可能：单次候选内多右端（multi-RHS）一起解？ 目前 enrich 是单列，probe 是
     小稠密解。Multi-RHS 无直接收益（不同 h），但同一 (shift, h) 只解一次。
   - 核心疑问：154 次 enrich solves 中，有多少次其实 residual 已经小到
     不需要真正 enrich？（probe 失败才 enrich，但每次 enrich 后继续 probe 直到
     连续 probe_rounds 次通过——现有逻辑在 enrich 之后不会重复 probe 同一点，
     而是继续新的随机 h。也许可以在 enrich 后先在同一 (shift,h) 复查，若已过
     就不再额外 solve。这能减少 solves？注意 enrich 一次添加一列，probe 是相对
     当前基底+reduced solve 的残差 —— enrich 之后该点残差应显著下降。
     但算法设计上就是 1 enrich + (probe_rounds 随机点通过)。多次 enrich 在同一
     shift 不同 h 是需要的（不同 h 残差方向不同）。

E3 阶数与精度控制：
   - SVD 截断语义 (tol/10 · ||s||)，常数向量显式加入。基线 40 阶 vs FloTHERM 36 阶。
   - 对照实验：改变 relative epsilon 对阶数与精度的 Pareto。

E4（调研后决定）算法替代候选：
   - 论文库里有几篇 2017-2021 的近作，先精读再定实验：
     - Galerkin's Projection Framework ... Part I Extended FANTASTIC（已实现）
     - Domain Decomposition / Transformation ... 2.5D 异构集成 (DDM-ROM?)
     - Connecting MOR-based BCI CTMs
     - Ten Years of BCI CTM review
     - 待查：Codecasa 最近是否有多点矩匹配 + AMG 的大规模加速做法
   - 可能的方向：a) 每个 (port, shift) 用 multi-h 联合 enrich（同时解多个 h 的
     响应做 block Krylov），减少总 solve 次数；b) 探针策略用确定性低差异序列；
     c) tangential/block rational Krylov。

E5 EROM 非共形（重点）：
   - 案例：ROM 方体 (15×15×15 网格) 连接外部 FVM 域，但外部域的 XY 网格错开
     (非对齐)，或者外部网格更粗/更细（如 8×8 柱 vs 15×15）。使用 common_patches。
   - 验证：patch 面积守恒、E/ξ 归一化、对称性、逐 patch 与积分热流、identity
     耦合、ROM trace 误差、外部场误差。当前 simple_erom_case1 的
     nonconforming_showcase 已有雏形，先读它的结果再扩展。
   - 需要新案例：网格错开的 2.5D chiplet 结构（在 bci_rom_testcase1 风格上建
     一个更实际的非共形附着的案例）而非简单 cube-cube。

## 产出

1. playground/bci_rom_testcase1/experiments/optimization/ 下：
   - amg_diagnostic.py（E1）+ 结果 json/csv
   - order_accuracy_pareto.py（E3）+ 结果
2. playground 非共形案例（E5）：脚本 + results + 报告
3. docs/reports/ 周报（可交付给导师的完整内容）
4. 详实的实验数据（raw csv/json 先行）

## 里程碑（提交点）

- M1: 计划 + 基线
- M2: E1 AMG 诊断结果
- M3: E2 solver 语义实验结果
- M4: E4 算法调研笔记（记录到报告）
- M5: E5 非共形 EROM 案例实验
- M6: 周报合成

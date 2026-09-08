# expr0908：BCI-ROM 提取优化 + 非共形 EROM 深化实验计划

分支：experiment/expr0908。基线：03cef3e（工作树干净）。
沿用约定：h 范围、容差等为输入参数，不做扫参/调参；不做 AMG caching；
不共享跨 port 基底；不使用子代理。所有实验在 playground 隔离目录进行，
结果目录（results/0908）按既有 .gitignore 约定不入库，脚本与周报入库。

## 基线事实（承上周 expr0907）

- 生产提取 = Extended FANTASTIC 2021 Algorithm 1 的忠实实现（utils.py），
  已用 AMG direct 插值（E1b，-31%）。Case-1 122,400 单元：40 阶、
  ~61s、154 次全阶求解、relative_response_error 4.27e-7。
- 非共形 EROM：common_patches（E/ξ 面积权重）+ 独立界面节点装配已实现；
  单侧 ROM（ROM-FVM）结温误差 ~0.5%，但**双 ROM（EROM-EROM）结温误差
  ~10.5-12%**（rom_rom_diagnosis.json），界面 trace 重构误差达 3.6 K
  （36.7%）。判定为"可达到界面迹子空间"不足，本周做机理实验与修复验证。

## 实验计划（全部先测后改，脚本内保持 RED/GREEN 语义）

### E1 提取管线细分画像（新测量）
在 122,400 单元 Case-1 上给当前生产代码插桩：谱规划（eigsh/lobpcg+AMG）、
探针、enrich 全阶求解（AMG setup vs CG）、增量投影、收尾 SVD，
输出细分计时 + 求解/快照计数。为整个报告提供统一成本表。

### E2 算法级替代：per-shift 矩迭代（moment continuation）A/B
动机：FloTHERM RomCore 反推结论 + Codecasa MPMM 理论（2003/2005 的
per-frequency 多矩）。在失败探针触发全阶求解时，同一算子 A(σ,h) 上
做 k 个传递函数矩（v_{j+1} = A⁻¹(C v_j)），共享一次 AMG 层级。
对比 k=1（忠实基线）/2/3：AMG setup 次数、CG 迭代总数、快照数、
预 SVD 阶、收尾阶、墙钟、relative_response_error、以及同一 holdout
场景（h=50/1000，P=[0.1..0.4]）下的稳态/瞬态结温与全场误差。
另在 0.75 mm 网格（329,280 单元）验证规模收益。

### E5a 界面迹子空间机理实验（双 EROM 误差根因）
对上下子域：把耦合解的界面节点温度视为"需求迹" T*，计算
||T* − Π_{W_S}(T*)||（W_S = span(Vb,S) 为 ROM 侧可达到迹空间），
以及 ROM 侧迹重构误差 (Vb q − T*)。量化"迹子空间缺口"如何传导到
结温误差。顺带对比 ROM-FVM 与 ROM-ROM（单侧 ROM 时 FVM 侧给全
分辨率约束，误差小；双侧 ROM 时缺口叠加）。

### E5b 界面子分区富集（修复候选，机理上扩充可达到迹空间）
提取时把界面面按坐标带分为 k 个子群（k = 1/4/16），每个子群独立
随机 h（界面 h 范围仍 [1e0,1e4]，不调参）：训练载荷族从"均匀 h
载荷"扩充为"分片载荷"，可达到迹空间对任意外接的逼近能力增强。
扫 k 并测双 ROM 非共形耦合的结温/trace/场误差与提取阶数变化；
若修复有效，在网格比 4/5/8 mm 三档复测（对照 nonconforming_erom_matrix）。

### E5c 新案例（结合理论的压力测试）
1) 任意偏移非共形：下部 FR4 用**非均匀、与上部不对齐**的 XY 网格
（不规则边集），验证 common_patches 对完全错位网格的面积守恒/通量
平衡/拼接完整性。
2) 侧向（x 法向）非共形界面：Case-1 栈 x=0 竖直切分，左右子域用
不同 XY 网格，两侧各带自己的 crown/bottom 环境群，测详细-详细与
EROM-FVM 与 EROM-EROM。

### E6 回归
reproduce_case1（含 FloTHERM 三方对照）+ run_tests.py 全绿；
任何生产改动先加失败回归测试。

## 周报 0908 结构（中文，数学化、符号一致）
一、概述；二、提取性能优化（E1/E2 + 规模伸缩）；三、算法选型调研
（Extended FANTASTIC 2021、RomCore 矩迭代反推、RB/贪心、2024-2026
DDR/DTDM/HDI、TurboMOR/SMP-RCR/FlexRC 多端口 RC）；四、非共形 EROM
理论（Galerkin 投影框架、边界 DoF、独立界面节点、E/ξ 公共面片、
对称半正定、界面消去、可达到迹子空间误差界）+ 实验（E5a/b/c）；
五、压力测试表；六、成本与摊销；七、结论与后续；附录：提交与产物。

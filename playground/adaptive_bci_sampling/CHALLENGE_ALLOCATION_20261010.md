# 2026-10-10 续研：图逆下界与挑战中的任务分配

本阶段的全时间理论与新增算法见 [CHALLENGE_ALLOCATION_PROOF.md](CHALLENGE_ALLOCATION_PROOF.md)。
**图块版本在 47k 已通过全时间风险验收，但同保证随机对照的成本相同；尚未得到正式规模选点优势。**
不更改生产提取器，各输入独立响应空间，认证针对 SVD 前基底。

## 1. 诊断与严格构造

- 下端 ground-state 衰减率已接近真实最慢率：47k 五次逆迭代将
  4.20168e-5 提高到 4.28637e-5，仅约 2%，不能靠增加辅助逆迭代解决。
- 参数正见证给出全域有效的 max-min 仿射衰减下界。最多购买 12 RHS。
  scalar 全时间证书对衰减率严格单调，Galerkin 最小 Ritz 值只作乐观
  诊断；连它都不能解除拒绝时，合法衰减下界也不能单独修复该证书。
  **Ritz 值从不用作验收下界。** 正见证不进入任何输入的响应空间。
- 序贯混合超鞅允许少量参数证书拒绝，只有资本达到 1/delta_epoch
  才接受。任意停止时刻保持风险目标；不是按经验比例放松门槛。
- 保留 ground-state 图拉普拉斯的分块内边，构造全秩 B，证明
  D <= B <= A(h) 及 A(h)^(-1) <= B^(-1) <= D^(-1)。用小块 Cholesky
  白化残差替代纯对角双范数，严格控制所有状态方向，不使用经验尾项。
  因子化、每次白化、QR 及所有查询均计时；没有新增全局 FOM RHS。

所有新部件以显式开关启用，默认仍复现上一阶段。
图分块取整个竖向柱、水平每边 2 个单元；一般图分区具有相同证明。
矩阵版依然可复现，当前严格分流实验用 scalar 单调包围。

## 2. 冻结的小模型结果

Case1 重构 364 DOF，四输入、两有效 Robin 参数、种子 20261031。
稳态 epsilon=.001、全部时间阶跃 sqrt(epsilon)、连续 log-uniform HTC
联合超限质量目标 rho=.01、总 delta=1e-6。

| 方法 | 响应 RHS | 额外衰减见证 | 共同辅助 RHS | 总 RHS | 阶数 | 接受 |
|---|---:|---:|---:|---:|---:|---|
| 参数衰减分流、四提案 | 76 | 8 | 1 | 85 | 81 | 是 |
| 同证书同分流、单提案纯 log-uniform 对照 | 84 | 10 | 1 | 95 | 89 | 是 |
| 分流及序贯混合证据、四提案 | 76 | 7 | 1 | 84 | 81 | 是 |
| 图逆下界、四提案、零失败检验 | 57 | 0 | 1 | 58 | 62 | 是 |

生产随机 RHS 为 96，但它没有同一全时间联合风险验收，不能作
同保证胜出结论。上一阶段同种子 scalar 版为 83 RHS、87 阶。
图逆下界使本次小模型总 RHS 降至 58，减少约 30.1%。
图版小模型同保证随机对照为 65 RHS、69 阶，接受；本种子四提案 58 RHS
比它低约 10.8%。一个种子不证明普遍胜出。

图版最后一轮 1801 个新独立 HTC 全部通过，最大稳态界 0.000606749，
最大全时间阶跃界 0.0310512。独立完整 FOM 谱审计使用 4 角点和 16
新 HTC，每点 128 个时间加初始／稳态极限；最坏真实稳态 7.34372e-5、
阶跃 0.00130551，稳态、时间和衰减界违反数均为零。

参数衰减分流的独立谱审计最坏稳态 2.55772e-6、阶跃 0.000424303，
无界违反。序贯版本最后检查 3883 点，其中 6 个证书拒绝；资本对数
19.8627 达到阈值 -log(delta_20)。最坏证书界可超过逐点容差，
这是允许非零拒绝质量的风险验收，不能将其改写为整个样本全部通过。

图 lower 的独立代数核查在水平宽度 1/2/4 上检查 A_min-B 和 B-D，
最小特征值仅有 1e-15 量级负浮点裕量；三角白化与直接逆 Gram
相对差最大约 1.90e-13。严格性来自恒等式，不来自忽略这些浮点裕量。

## 3. 参数衰减分流的正式负面门槛

47,085 DOF，四提案 188 个响应 RHS、9 个新正见证及共同 1 RHS，
总 198 RHS，在 240 外循环预算内未接受。单提案同保证随机对照
185 响应 RHS、12 正见证及共同 1 RHS，也为 198 RHS、未接受。
生产参照为 141 RHS。很多拒绝处连乐观 Ritz 衰减率都无法让旧证书
通过，因此仅更新参数衰减界不足；必须改善残差双范数／状态传播。
122,400 DOF 的参数衰减配置运行被系统 OOM 终止，最后记录为 140
响应 RHS、145 阶；不记作算法失败或接受，需错峰补跑。并发运行内存
安排不当是执行问题，不能用来评判方法。

## 4. 复现

在仓库根目录，依赖 NumPy、SciPy、PyAMG，BLAS 线程固定为 1：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python playground/adaptive_bci_sampling/random_task_extraction.py \
  --seed 20261031 --certificate-mode scalar --graph-block-width 2 \
  --max-rounds 180 --audit 2 \
  --output playground/adaptive_bci_sampling/results/graph_small31.json
python playground/adaptive_bci_sampling/audit_random_task.py \
  --candidate playground/adaptive_bci_sampling/results/graph_small31.npz \
  --output playground/adaptive_bci_sampling/results/graph_small31_audit.json
python playground/adaptive_bci_sampling/random_task_extraction.py \
  --mesh-mm 1.5 --seed 20261031 --certificate-mode scalar --graph-block-width 2 \
  --max-rounds 240 --audit 2 \
  --output playground/adaptive_bci_sampling/results/graph_47k.json
python playground/adaptive_bci_sampling/audit_risk_sequence.py \
  --output playground/adaptive_bci_sampling/results/risk_sequence_audit.json
```

正式 122k 用 --mesh-mm 1；随机对照增加 --tournament 1 --corner-mixture 0。
参数衰减分流改用 --decay-budget 12、--graph-block-width 0；序贯证据
再增加 --risk-test mixture，默认最大检查数为同 epoch 零失败所需数的 2 倍。
不同开关对应不同冻结实验，不将后续配置收益追溯到旧结果。

普通浮点，没有外向舍入；native_validated=false。并发实验墙钟不作
隔离性能基准。不能把本地三角解当免费工作，也不能把实验性的图分块
默认替换到生产提取流程。Collatz–Wielandt、support graph 和 betting
均已有先例；新增组合的学术创新及正式效率仍待判决。


## 5. 图块版本的正式规模正面门槛

47,085 DOF，四提案和单提案纯 log-uniform 对照各为 176 响应 RHS
加共同 1 RHS，总 177 RHS、181 阶。两者均在第 41 次尝试完成
2117 个新独立 HTC 零拒绝的联合全时间风险验收。
四提案最大稳态／阶跃界为 0.000182520 / 0.0271096；对照为
0.000517469 / 0.0304500。正式规模首次从未接受转为接受，但
**两种发现方法 RHS 相同，不能宣称比纯随机更高效**。
生产随机参照 141 RHS，没有同一个全时间风险保证，仍需完整成本比较。

四提案独立六 HTC 稳态最大真实误差 1.89287e-6，对照 7.53035e-6。
图块共 814 块，最大 60 自由度，保留 90,161 边。四提案因子化
0.279 s，41 次加权帧白化共 15.57 s；并发提取加认证墙钟 1001.8 s，
对照 956.2 s。约 2000 次约化全时间检查和反复 QR 仍支配成本。
不能称为生产级整体加速，指定点阶跃独立 BE 核查另行记录。

## 6. 序贯检验独立概率核查

有限时域 Bernoulli 动态规划枚举全部计数路径，不用 Monte Carlo。
rho=.01、delta_epoch=1e-9、样本上限 4200：当真实拒绝质量 p=.01，
错误接受概率为 5.31064e-10，小于目标；p=.02 时为 7.96217e-20。
p=.001 时接受概率 0.89734，平均检查约 3090.7 点；p=0 时 2177 点。
这说明允许少量拒绝可改善检验功效，但查询代价增加，不能直接
宣布降低总时间。普通浮点枚举只核查实现，证明仍来自超鞅。

## 7. 后续严格下算子原型

正半定粗 Robin 下更新和二部图 modified LDL 的完整推导见证明
第 7、8 节。前者小模型仍为 58 RHS，没有观察到 RHS 进一步下降。
二部图下因子的小模型原型为 53 RHS、57 阶，已接受；其正式模型
仍在运行。后者所有完整状态三角 RHS 都计入辅助应用次数与耗时，
不混淆为廉价的标量操作或零成本完整辅助求解。

启用：`--inverse-lower-kind bipartite`；粗 Robin 更新增加
`--graph-robin-degree 1`（图块版本还须 --graph-block-width 2）。
这些都是显式实验选项，绝不将未完成的正式判决写成已验证的创新。

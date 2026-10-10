# 2026-10-10：全时间风险证书与同输入残差反馈判决

**得到了小模型的联合稳态／全部时间阶跃分布风险接受；尚未得到正式
规模上满足经济性要求的新提取算法。** 47,085 DOF 在冻结配置的 240
外循环轮预算内未接受，195 RHS 高于生产随机基线的 141 RHS。
122,400 DOF 同一第一版也未接受，195 RHS 高于生产的 165 RHS。

完整理论和算法见 [RANDOM_TASK_PROOF.md](RANDOM_TASK_PROOF.md)。
入口是 `random_task_extraction.py`、`diagonal_time_certificate.py`。
不替换生产 `utils.py`。各输入独立选点和独立残差校正，仅交付时拼接。
没有跨源共享响应空间，没有新增低秩逆像辅助响应库，没有单元测试。

## 1. 构造与局部陷阱

一次辅助 AMG-CG 求解产生正向量 $z$，利用实际 $A_-z$ 形成对角
$D\preceq A(h)$，并给出 $A(h)\succeq\alpha C$。由真实阶跃残差
$r(t)=R_0+Jf_\Lambda(t)B_r$，组合收缩范数界与能量微分界，保留
每一低阶残差项的抵消，解析包围初始段、中间区间和无穷尾段。
没有将有限时间抽样、冲激积分或 Hankel 范数当作全时间相对误差。

认证就在循环内：稳态失败直接解同输入稳态残差校正；时间失败在
同一 HTC、失败区间时间 $\tau$ 解 $(A+\tau^{-1}C)z_j=r_j(\tau)$。
只富集该输入自己的空间，每四个反馈 RHS 再检查。

发现提案保留全域随机成分，停止必须通过独立全时间联合风险挑战。
这防止固定池或局部优化停滞被误接受；它不是所有问题上快速终止的定理。
固定大小的新提案批次避免全域最大化，但约化谱、时间包围与每轮 QR
仍有明确成本，不声称参数维数或基阶数无关。

## 2. 第一版冻结的小模型复核

Case1 矩阵重构，364 DOF，两个 Robin 参数、四个输入。原始整个盒上
各维独立 log-uniform。稳态容差 $\epsilon=.001$，全时间阶跃容差
$\sqrt\epsilon$；联合 HTC 超限风险目标 $\rho=.01$，全部自适应轮次
合计 $\delta=10^{-6}$。每次尝试用 $\delta_k=\delta/[k(k+1)]$。

| 种子 | 生产提取 RHS | 新原型总 RHS | 阶数 | 最终新 HTC 检查数 | 最大稳态界 | 最大全时间阶跃界 |
|---|---:|---:|---:|---:|---:|---:|
| 20261030 | 98 | 89 | 93 | 1945 | 6.34719e-5 | 0.0228075 |
| 20261031 | 96 | 83 | 87 | 1920 | 4.01701e-4 | 0.0225465 |
| 20261032 | 98 | 93 | 97 | 1956 | 2.28232e-4 | 0.0214786 |

三次最终联合界零失败，接受的是连续参数**分布风险**，不是整盒处处
通过。时间和任意有符号输入组合由解析证书覆盖，不抽样时间。
新原型 RHS 包括一次共同对角界辅助求解；共同谱计划使用原生产算法，
单独计时，不把复用计划当免费研究收益。

生产三次随机残差接受与联合风险接受不同，83/96 等只是实际生产成本
参照，不是同保证最优性结论。同一证书与残差反馈下的单提案纯
log-uniform 参数随机对照（`--tournament 1 --corner-mixture 0`，
种子 20261031）为 97 RHS、101 阶、最终 1966 新 HTC 检查通过。
该种子新原型 83 RHS，下降约 14.4%；一个同保证控制种子不能推出普遍胜出。

新原型提取加认证墙钟约 86.6–93.2 s，同保证随机对照约 99.1 s，生产
随机提取器约 0.35–0.38 s。约 1900 次约化谱与解析时间检查支配成本。
部分实验并发执行，墙钟不是隔离性能基准；**不宣称整体更快**。

这些第一版数字对应 `--certificate-mode scalar --accept-corners --time-feedback failure`。
其角点额外准入把全盒式约束叠加到了分布风险目标上，导致正式模型
一直被角点拒绝、没有进入真正的随机风险检查。这是方法论上的过强
条件，默认实现已纠正为只用新独立连续分布样本接受，角点仍可用于发现。
同时提供能量卷积启发的最大强迫贡献时间选择；修正版正式判决见第 6 节。

## 3. 独立小 FOM 核查

`audit_random_task.py` 独立计算完整小 FOM 谱：4 个角点、16 个新
连续 HTC、每点 128 个对数时间，以及初始斜率和稳态极限。
这是对已经有全时间证明的实现作独立反证检查，不是接受来源。

| 种子 | 最坏稳态真实误差 | 最坏阶跃真实误差 | 上界违反数 |
|---|---:|---:|---:|
| 20261030 | 6.62193e-7 | 4.50431e-4 | 0 |
| 20261031 | 4.54297e-6 | 4.33343e-4 | 0 |
| 20261032 | 1.42927e-6 | 4.31959e-4 | 0 |

对角 ground-state 的最小特征值裕量单独报告，数值误差范围内非负。
证书比真实误差明显松；少量点核查不能证明它在所有参数上紧。

## 4. 第一版正式规模的负面门槛

47,085 DOF、种子 20261031、240 外循环轮预算：没有联合风险接受，
195 RHS、199 阶，生产参照 141 RHS。提取和认证约 207.7 s，生产
参照约 17.0 s（并发影响见上节）。最后的高 HTC 角点在约 1054.5 s
的解析时间区间仍未通过，首个失败阶跃包围 0.0318645。
这是**证书失败**，不是完整误差被证明超限。
最后一次验收尝试针对 197 阶空间；预算耗尽时空间为 199 阶，未再完成
一次接受尝试。首个失败区间的界也不是全时间最大界。

122,400 DOF 在同样 240 轮预算内没有接受，195 RHS、199 阶，生产
165 RHS；提取加认证约 556.5 s，生产参照约 50.6 s。
最后一次 197 阶检查在约 1092.9 s 被拒绝，首个失败区间界 0.0325510。
这同样不是实际误差超限证明。两个正式模型都没有新的概率接受结论。

独立 6 个 HTC 稳态参考最大真实相对误差为 1.10191e-4，满足稳态容差。
指定点阶跃 BE 参考另行核查，不替代全时间接受。

122k 独立 6 个 HTC 稳态参考最大误差为 1.57224e-4。
两种模型各自使用 3 个 HTC、t=.01 / 1 的独立 BE 64 步参考，计入解析
时间误差及实际 CG 缺陷；各 6 个点全部通过阶跃容差。最坏真实误差上界
分别为 0.0134561 / 0.0135362。每个模型的 BE 审计单独消耗 1536 RHS，
不计入提取成本、不用于接受整个连续域或全部时间。

共同对角衰减约 $4.2\times10^{-5}$。累计界不能识别真实误差强迫主要
进入哪些衰减方向，容易把早期强迫保留到远处时间。残差反馈修补一些
方向，但正式规模上仍需过多富集和重复 QR。这是结构性成本失败，
不是仅靠 AMG setup 或小型求解调优能解决的问题。

## 5. 复现与成本作用域

依赖 NumPy、SciPy、PyAMG，在仓库根目录执行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for task_seed in 20261030 20261031 20261032; do
  python playground/adaptive_bci_sampling/random_task_extraction.py \
    --seed "$task_seed" --max-rounds 180 --audit 2 --certificate-mode scalar --accept-corners --time-feedback failure \
    --output "playground/adaptive_bci_sampling/results/final_small${task_seed}.json"
  python playground/adaptive_bci_sampling/audit_random_task.py \
    --candidate "playground/adaptive_bci_sampling/results/final_small${task_seed}.npz" \
    --output "playground/adaptive_bci_sampling/results/final_small${task_seed}_audit.json"
done
python playground/adaptive_bci_sampling/random_task_extraction.py \
  --seed 20261031 --tournament 1 --corner-mixture 0 --max-rounds 200 --audit 2 --certificate-mode scalar --accept-corners --time-feedback failure \
  --output playground/adaptive_bci_sampling/results/final_control31.json
python playground/adaptive_bci_sampling/random_task_extraction.py \
  --mesh-mm 1.5 --seed 20261031 --max-rounds 240 --audit 2 --certificate-mode scalar --accept-corners --time-feedback failure \
  --output playground/adaptive_bci_sampling/results/final_47k.json
python playground/adaptive_bci_sampling/random_task_extraction.py \
  --mesh-mm 1 --seed 20261031 --max-rounds 240 --audit 2 --certificate-mode scalar --accept-corners --time-feedback failure \
  --output playground/adaptive_bci_sampling/results/final_122k.json
python playground/adaptive_bci_sampling/audit_output_step.py \
  --mesh-mm 1.5 --candidate playground/adaptive_bci_sampling/results/final_47k.npz \
  --times .01 1 --steps 64 \
  --output playground/adaptive_bci_sampling/results/final_47k_step.json
python -m py_compile playground/adaptive_bci_sampling/random_task_extraction.py \
  playground/adaptive_bci_sampling/diagonal_time_certificate.py \
  playground/adaptive_bci_sampling/audit_random_task.py
```

`--max-rounds` 包括证书尝试轮；`--max-rhs` 是兼容旧命令的别名，实际
RHS 以成本账本为准。独立 FOM 审计与提取成本分开。
原始矩阵、快照、检查流与完整 JSON 不提交；标量判决、证明、复现命令提交。
普通浮点没有外向舍入，Case1 native_validated=false。
最终 NPZ 保存 SVD 前空间及 Galerkin 算子，不将证书沿用到 SVD 后空间。

不能立即称为可发表创新。一次辅助求解的全时间相对证书和同输入时间
残差校正可以保留；下一项有价值的数学改进必须识别强迫的衰减方向，
避免共同最慢率支配累计界，再通过正式规模和同保证成本比较。
只优化 QR、setup 或验收代码不补齐这个问题。


## 6. 修正角点准入后的独立判决

去掉强制角点接受，保留角点随机发现。时间反馈改为最大累计强迫
贡献代理（启发式，不是严格的误差贡献最大化）。每次拒绝仍触发同输入
残差富集及新的独立连续风险流。以下是 scalar 证书的冻结复核：

| DOF | 生产 RHS | 新方法 RHS | 新方法接受 | 同证书纯随机对照 |
|---|---:|---:|---|---|
| 364 | 96 | 83 | 是 | 89 RHS，接受 |
| 47,085 | 141 | 195 | 否 | 195 RHS，未接受 |
| 122,400 | 165 | 195 | 否 | 未在此配置运行 |

新方法提取加认证为 78.4 / 203.3 / 551.1 s；47k 同保证随机对照
206.0 s。小模型另行完整谱审计最坏稳态 2.45978e-6、阶跃
4.18458e-4，无界违反。正式两模型独立六 HTC 稳态最大真实误差
1.34703e-6 / 2.34300e-6。正式失败不是仅由强制角点导致。
这些真实误差抽样不提供全部时间或连续域验收。

复现：不加 `--accept-corners`，使用 `--time-feedback dominant`
及 `--certificate-mode scalar`，其余第 5 节正式模型参数不变。
小模型最大轮数 180；对照增加 `--tournament 1 --corner-mixture 0`。

## 7. 输入 Gram 矩阵结构修正

保留四输入误差 Gram，以 Bernstein 控制矩阵、Loewner 正部上界及
解析 Taylor 余项覆盖每一个时间区间。全时间证明见证明文档末节。
这是输入方向结构修正，不能声称已有状态方向衰减证书。
入口默认现为 `--certificate-mode matrix`，旧结果必须显式用 scalar。

| DOF | 生产 RHS | 矩阵原型 RHS | 接受 | 阶数 | 提取加认证秒 |
|---|---:|---:|---|---:|---:|
| 364 | 96 | 85 | 是 | 89 | 240.6 |
| 47,085 | 141 | 195 | 否 | 199 | 264.5 |
| 122,400 | 165 | 195 | 否 | 199 | 562.1 |

47k 同保证纯随机对照也是 195 RHS、199 阶、未接受，257.2 s。
同证书控制仅改变发现阶段的 tournament/corner mixture；残差反馈、
独立验收和风险参数全部相同，因此它不是原论文的无反馈随机算法。
小模型独立 FOM 谱核查（20 HTC，128 时间加两极限）最坏稳态
1.91283e-6、阶跃 4.17271e-4，界违反数为零。
矩阵版小模型最终 1933 个新 HTC 风险检查完整通过，最大稳态界
0.000244309、最大全时间阶跃界 0.0278980；但比 scalar 版更贵，不能将其
作为总体性能改进。并发墙钟仅为本次成本记录，不是隔离性能基准。

47k 最后一次冻结的 197 阶风险尝试检查 56 个新 HTC 后拒绝，
拒绝参数约 [8359.98,349.740]，全时间阶跃界 0.0367408 超过
0.0316228。195 RHS 的最终 199 阶空间包含最后三次未经风险验收的
富集，不能继承前一轮接受结论。外循环预算 240，包含认证轮，
不是 240 RHS 预算；没有因通过有限样本而提前宣布成功。

复现矩阵版：第 5 节对应命令去掉 `--accept-corners`，改用
`--time-feedback dominant --certificate-mode matrix`，种子 20261031；
小模型 180 轮、两正式模型 240 轮，对照增加
`--tournament 1 --corner-mixture 0`。这些运行结果与第一版明确分开。

该研究目前没有满足正式规模比纯随机更高效的要求，也没有证明
局部陷阱完全消失或高参数维度的认证复杂度得到解决。保留下来的成果
是循环内全时间相对风险证书及独立核查的小模型原型，正式规模结果
作为负面边界；不能只按小模型 RHS 下降给出论文级效率结论。

矩阵版 47k 最终空间的独立 BE 审计同样在 3 HTC、t=.01 / 1、
64 步下核查六点，全部通过；最大真实误差上界 0.0134371。
参考时间离散误差和实际 CG 缺陷均计入，审计单独消耗 1536 RHS。
这没有提供全部时间或连续参数分布接受。

矩阵版 122k 最后一次 197 阶尝试检查 3 个新 HTC 后拒绝，
参数约 [664.358,527.904]，首个失败时间约 1601.47 s，
首个失败阶跃界 0.0316507。最终 199 阶仍未接受。
独立六 HTC 稳态最大真实误差 4.67115e-7，不能代替全时间验收。

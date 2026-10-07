# 连续 HTC 动态全场认证：第二轮优化

日期：2026-10-07。基于 agent/work，HEAD d7e229bca40b94d75d332827097d6ecd86fc2317。
本轮修改和前轮原型均在本地，未提交或推送；生产提取器没有改变。

本轮获得两个有精确算术证明的改进：联合创新能量界，以及已有多项式试验的精确参数包围
细分与子盒输入度量。32 节点链式模型的完整参数域认证 RHS 从 448,080 降到 126,608，
减少 71.74%。Case1 矩阵重建同一较窄局部单元的认证 RHS 从 12,869 降到 8,129，
减少 36.83%；另一通过单元的两轴宽度均扩大 10 倍。
**仍不是经济的提取加认证方案，也不是严格浮点证书。**

## 1. 改了什么，证明了什么

目标沿用前轮：固定 V，连续 HTC 盒中，所有输入组合的无限时间全场 impulse 相对误差
`epsilon_imp^2=lambda_max(J,Q)`，`J=integral(E^T C E)dt`，`Q=0.5 G^T K(h)^{-1}G`。
没有改成结温、输入逐列误差、有限时间或任意功率历程指标。

### 联合创新能量界

旧界在每个时间系数之前把历史残差变成标量源 tube，再累加系数余量和末端 tube。
新界把整个强迫递推的系数缺陷与末端源能量一起处理。第 j 个创新的精确输出列 Gram 为

\[
 A_{jj}(\lambda)=b_j,\qquad
 A_{jk}(\lambda)=-b_jb_k\prod_{\ell=j+1}^{k-1}t_\ell/2,
\quad b_j=2\sigma_j/(\lambda+\sigma_j),\quad
 t_j=(\lambda-\sigma_j)/(\lambda+\sigma_j).
\]

用合法谱区间包围该 Gram，得到全阶和约化联合缺陷余量，再加试验差的主项和试验尾项。
精确末端能量是推导的一部分，没有隐藏时间截断误差。实现取新旧合法上界的最小值。
这避免了部分重复的历史放大；当前绝对值谱包围仍丢失空间方向和部分符号信息。
因此尚未解决前轮耦合原型中的完整方向损失，也不宣称这个标准能量构件本身具有文献新颖性。

### 已有试验的参数包围细分

对 trial 前缀、残差和尾源的 Bernstein 控制张量作 de Casteljau 精确限制。
保留原盒 Riesz 度量和谱区间，局部输入分母使用子盒上角 `Q(b_sub)`。
控制向量与已有逆作用作相同线性组合，因逆作用线性，子盒 dual Gram 仍精确。
不增加全阶时间系数或 Riesz RHS；新增分母 RHS 全部计入成本。
这改变的是连续包围的紧度，不是采样验收或解算器缓存。

完整证明在 [RATIONAL_DYNAMIC_PROOF.md](../RATIONAL_DYNAMIC_PROOF.md) 第 10–11 节。
实现选项为 `--propagation energy --envelope-depth D`，均属判决实验设置；
未设计只有容差与 HTC 范围两个控制参数的自动提取器。

## 2. 32 节点链式模型：完整参数域

完全相同的 raw 固定基：27 阶，SHA256
`ae3c29c35ad1cfabc7d914170fb9a27a93d6f11c25cde607dbebe5512e01ca5f`。
外层仍为 `[1,4]^2` 的 4×4 覆盖。阈值始终 `1e-3`。

| 方法 | 时间项 N | 插值 p | 包围细分深度 | 全域最大界 | 认证 RHS | 覆盖墙钟秒 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| 前轮独立递推 | 96 | 7 | 0 | 4.374773e-4 | 448,080 | 7.74 | 通过 |
| 联合创新能量 | 96 | 6 | 0 | 3.048934e-4 | 349,232 | 6.15 | 通过 |
| 能量 + 参数限制 | 96 | 5 | 1 | 1.608255e-4 | 262,832 | 11.14 | 通过 |
| 能量 + 参数限制 | 96 | 4 | 2 | 3.734415e-4 | 189,072 | 14.17 | 通过 |
| 能量 + 参数限制 | 64 | 4 | 2 | 3.734421e-4 | 126,608 | 8.37 | 通过 |
| 能量 + 参数限制 | 64 | 3 | 3 | 2.32148e-3 | 86,832 | 见原始 JSON | 不通过 |

最后一个通过方案的端到端全阶 RHS 为 `126,608+26=126,634`，前轮为
`448,080+26=448,106`。外部独立稠密特征参考另计，没有用于接受判断。
全域接受基于连续子盒公式，不是有限参考点的最大值。

RHS 降低不等于运行时间加速：相对前轮 7.74 秒，最终覆盖 8.37 秒略慢。
参数限制增加控制向量操作、Gram 计算和分母分解，墙钟非严格隔离 benchmark。
相对仅 26 RHS 的 stock 提取，这个证书仍有约 4,870 倍的 RHS 成本。
另外，此通过基为 27/32 的 raw 基，不是 stock 最终 13 阶基。

## 3. Case1 重建：固定 raw 基上的局部认证

10 mm 网格矩阵重建：364 节点、4 输入，raw 95 阶基，SHA256
`a74ce9a2dba5cab8204dd4dcb21461189db7b43ec7a259eb290c24b0029cbec0`。
基提取为 94 RHS。没有修改网格、材料、端口、HTC 映射或基提取设置。

局部单元记为 `a=c-f(c-range_low), b=c+f(range_high-c)`，`c` 是范围几何中心。
这是有效 HTC 仿射坐标中的线性盒宽系数，不是物理 HTC 的相对半宽。

| f | 方法 | N / p / 细分深度 | 连续盒上界 | 独立样本最大误差 | 认证 RHS | 判决 |
|---:|---|---|---:|---:|---:|---|
| 0.0002 | 前轮独立递推 | 128 / 2 / 0 | 9.986901e-4 | 9.915499e-4 | 12,869 | 通过 |
| 0.0002 | 联合能量 | 128 / 2 / 0 | 9.953257e-4 | 9.915499e-4 | 12,869 | 通过 |
| 0.0002 | 能量 + 参数限制 | 80 / 2 / 2 | 9.92755e-4 | 9.915499e-4 | 8,129 | 通过 |
| 0.0002 | 能量 + 参数限制 | 64 / 2 / 2 | 1.00204e-3 | 9.915499e-4 | 6,529 | 不通过，时间尾项约 9.46e-6 |
| 0.002 | 前轮独立递推 | 128 / 3 / 0 | 1.225453e-3 | 9.918977e-4 | 21,097 | 不通过 |
| 0.002 | 联合能量 | 128 / 3 / 0 | 1.05544e-3 | 9.918977e-4 | 21,097 | 不通过 |
| 0.002 | 能量 + 参数限制 | 96 / 3 / 3 | 1.000789e-3 | 9.918977e-4 | 16,101 | 不通过 |
| 0.002 | 能量 + 参数限制 | 96 / 3 / 4 | 9.987875e-4 | 9.918977e-4 | 16,869 | 通过 |
| 0.002 | 能量 + 参数限制 | 80 / 3 / 4 | 9.989495e-04 | 9.918977e-4 | 14,245 | 通过 |

较宽通过盒为
`[96.17131644695168,114.74074503376474] × [7.645918026028506,7.763235107148344]`。
相对于此前 f=0.0002 通过盒，两轴宽度均为 10 倍，面积为 100 倍。
N=96 的合成界分量为主项 `9.941555426e-4`、联合缺陷 `4.617843391e-6`、
试验尾项合计 `1.408669741e-8`。分母追加 1,020 RHS，已含在 16,869 中。
该实验耗时约 67.77 秒，不能据此声称 wall-clock 改善。
时间项进一步减少到 N=80 后，较宽单元仍通过，认证成本为 14,245 RHS，
端到端为 14,339 RHS，墙钟约 53.07 秒。其尾项合计
`1.7612760446e-07`，没有为降低 N 而删除尾项。
较窄通过方案端到端成本为 `8,129+94=8,223`，仍远大于 94 RHS 的提取。

stock 最终 34 阶基在中心的真实参考动态误差仍约 `0.0175906`，已经超过阈值 17 倍。
优化证书不能把这个基变成满足 `1e-3` 的动态基。
raw 基在完整 Case1 参数范围也已有参考点约 `0.00102308` 超限，
本轮没有、也不能在该固定基上宣布完整范围通过。

## 4. 可复现和验证

从仓库根目录，设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python`。

```bash
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model chain --basis-stage raw --terms 64 --propagation energy --envelope-depth 2 --widths .02 --degrees 3 --cover-cells 4 --cover-degree 4 --output playground/adaptive_bci_sampling/results/rational_chain_energy64_cover4_envelope2.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 80 --propagation energy --envelope-depth 2 --widths .0002 --degrees 2 --output playground/adaptive_bci_sampling/results/rational_case1_energy80_envelope2.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 96 --propagation energy --envelope-depth 4 --widths .002 --degrees 3 --output playground/adaptive_bci_sampling/results/rational_case1_energy96_envelope4.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 80 --propagation energy --envelope-depth 4 --widths .002 --degrees 3 --output playground/adaptive_bci_sampling/results/rational_case1_energy80_envelope4.json
python -m pytest playground/adaptive_bci_sampling/test_field_audit.py playground/adaptive_bci_sampling/test_rational_dynamic.py -q -p no:cacheprovider
```

修改前相关基线 15 passed。先增加能量界测试并确认两个预期失败，再实现：17 passed。
再先增加参数限制测试并确认两个预期失败，再实现：19 passed。
补充显式完整强迫映射与签名 Gram 恒等式交叉检查后，最终 20 passed。
`git diff --check` 通过。
全部输入维度、无限时间 Gram、非节点参数检查、满空间、近相关输入、坐标变化、
单实移匹配反例等已有检查继续通过。
本轮不需要更改 native C++ 构建或生产算法。

本轮末尾添加的 `envelope_worst_cell` 元数据及试验尾项字段一致性修正不改变计算上界；
部分 JSON 生成于该元数据改动之前。压缩包清单保存最终代码与每份原始数据的 SHA256。

## 5. 仍然没有解决的问题

- 精确算术的合法界已经证明，但线性求解、谱界、输入 Cholesky、控制转换与 Gram
  尚未进行向外舍入；所有输出 `floating_point_certified=False`。
- native `libmhs_c_api.so` 未构建。Case1 矩阵重建未与 native 逐项比对，
  所有 Case1 结果都只能归于这个矩阵实验。
- 相比提取成本，认证仍昂贵；减少 RHS 不是端到端优于 stock 的证据。
- 当前能量 majorant 仍绝对值化谱交叉核，完整空间方向保留尚未完成。
- 生产最终 SVD 基的动态误差超限、只有 tau/HTC 范围控制参数的有效自动策略、
  基于完整目标的动态压缩与经济认证仍是下一阶段的实际障碍。

优先研究方向应是：以本目标控制最终空间压缩，并给出含全部余量的方向敏感能量 majorant。
不应把 raw 大基局部通过写成当前最终基已经实现全范围动态保证。

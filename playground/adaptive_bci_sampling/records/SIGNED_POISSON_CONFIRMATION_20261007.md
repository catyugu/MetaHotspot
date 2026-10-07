# 带符号 Poisson–Loewner 动态全场界：进一步实验与确认

日期：2026-10-07。基于 `d7e229bca40b94d75d332827097d6ecd86fc2317`，本地未提交研究原型。

本轮确认：原生 C++ 装配、多提取种子、更宽连续参数盒、1200 单元细网格，以及独立正时间积分均支持此前数学方案。也发现嵌套三角的固定分预算会产生可避免的拒绝；直接认证最终 SVD 基可在不改变容差的情况下通过。没有获得整个 HTC 参数域的认证，也没有获得浮点区间认证或经济性优势。

## 目标与不变条件

误差为全场、无限时间区间、所有固定输入组合的相对冲激响应能量误差：

\[
\epsilon(V,h)^2=\lambda_{\max}(J(V,h),Q(h)),\qquad
Q(h)=\tfrac12G^TK(h)^{-1}G.
\]

`J` 是完整温度场误差的 `C` 加权时间积分 Gram。此目标不等同于逐节点峰值误差、H2 trace 或任意时间输入的相对诱导范数。

所有提取均使用 stock `build_parametric_basis`，`tolerance=0.001`，最终 SVD 基接受阈值固定为 `0.002`。保持常数模态，使用固定快照的 SVD 方向；未更换生产提取器。认证默认使用 80 个 rational 时间项、二阶参数 trial、`envelope_depth=0`，因此本轮旧/新界比较不依靠参数包围细分。

参数盒采用有效 Robin 仿射坐标。设训练区间为 `[l,u]`，中心 `c=sqrt(l*u)`，盒端点是
`low=c-f*(c-l)`、`high=c+f*(u-c)`。`f` 不是物理 HTC 的相对扰动比例，也不是全域覆盖。不同网格对应的有效参数映射可能不同。

## 原生独立装配核对

成功构建 `mhs_c_api`，GNU C++ 13.3，Release，`USE_MKL=OFF`。在相同 Case1Config 下比较全部矩阵和有效参数区间，未进行事后参数拟合或排序。

| 单元数 | 网格上限 mm | K 相对 Frobenius 差 | G 相对 Frobenius 差 | C、H 差 |
|---:|---:|---:|---:|---:|
| 364 | 10 | 6.718e-17 | 1.250e-15 | 0 |
| 1200 | 8 | 8.260e-17 | 1.100e-15 | 0 |

有效区间最大绝对差为 `2.22e-16`。原生 364 单元模型重新提取并认证，仍得到最终 76 阶，直接上界 `0.0013635453332268533`，与此前重建实验吻合。此前“未经原生验证”的限制已在这两个配置上消除；没有据此宣称所有配置的重建均已验证。

数据：`results/native_matrix_comparison.json`、`results/guard_native_seed20260805.json`。

## 提取种子与最终 SVD

在 `f=0.0002` 的连续盒上：

| 提取种子 | raw 阶数 | stock 最终阶数 | 认证最终阶数 | 嵌套上界 | 最终基直接上界 |
|---:|---:|---:|---:|---:|---:|
| 20260805，原生 | 95 | 34 | 76 | 0.001943375928 | 0.001363545333 |
| 20260806 | 98 | 34 | 80 | 0.001656409368 | 0.001151527622 |
| 20261007 | 101 | 34 | 80 | 0.001876910707 | 0.001434856880 |

三个 stock 最终基的采样最大动态误差分别约 `0.01759, 0.01829, 0.01751`，均明显超过放宽后的 `0.002`。因此 `2*tolerance` 仍不能直接用作 stock 快照奇异值 cutoff 的动态保证。这里通过的是经动态证书验收的另一组最终基。阶数 76 并非种子无关，更不是全体子空间中的最小阶数结论。

## 同一设置下的数学改进复验

固定每个种子的最终基，在 `f=0.0022`、二阶 trial、80 项、无细分、相同 8069 个全阶 RHS 下：

| 种子 | 旧标量能量上界 | 新带符号矩阵上界 | 0.002 判决 |
|---:|---:|---:|---|
| 20260805，前轮同基 | 0.002126018243 | 0.001926756429 | 旧拒绝、新通过 |
| 20260806 | 0.001904045393 | 0.001704783498 | 两者通过 |
| 20261007 | 0.002198341095 | 0.001999079160 | 旧拒绝、新通过 |

新种子 20261007 仅有约 `9.21e-7` 的浮点数值余量；它是精确算术定理对应的数值通过，不能当作含舍入误差的严格机器认证。20260806 在 `f=0.0024` 也通过，上界 `0.00185669625`；20261007 在该宽度拒绝，上界 `0.00215206953`。两者在 `f=0.02,p=3` 均拒绝，上界约 `0.098`，即使采样误差依然很小。保留这些负结果，避免把采样平滑性误作全盒保证。

这些变化源于余量保留带符号时间/空间/输入交叉 Gram，并用连续谱 Loewner 支配包围它。此前证明的结构抵消例仍成立；本轮没有更改核心公式。有限样本压力实验只能检查实现，不能赋予新的全域证明或论文新颖性结论。

## 细网格：分预算失败与直接证书通过

1200 单元、种子 20260805、相同 `tolerance=0.001`，raw 为 113 阶，stock 最终为 36 阶。

嵌套路线先要求 raw 上界不超过 `tau`。实际 raw 上界为 `0.0016143322172876221`，因此拒绝并保存 `results/guard_mesh8_seed20260805.json`，不将其隐藏或覆盖。

但嵌套分预算

\[
B_{\rm raw}\le\tau,\quad B_{\rm compression}\le\tau
\quad\Longrightarrow\quad \epsilon(V,h)\le2\tau
\]

只是充分条件。用户要求最终基通过 `2*tau`，没有要求 raw 单独满足动态 `tau`。直接构造并认证最终基，无需满足上述两项分别成立。

试验预设保留 SVD 方向 `[100,104,108,110,112]`，逐项直接认证，首次通过即停止，没有假设动态误差随阶数单调。第一项 100 个 SVD 方向加常数模态即通过：

| 项目 | 数值 |
|---|---:|
| 最终阶数 | 101 |
| 连续盒 | f=0.0002 |
| 最终直接矩阵上界 | 0.0017623811264322203 |
| 旧能量上界 | 0.0017624634352307537 |
| low、center、high 参考最大值 | 0.0017571746473897404 |
| 最终接受阈值 | 0.002 |
| 单次全阶证书 RHS | 8069 |

此案例主要确认方法和最终验收目标可在细网格上成立；旧/新界差很小，不能据此宣称所有案例都有明显矩阵增益。没有证明 101 阶最优，也没有细网格全域覆盖。

## 独立时间积分与最坏输入

对三个 364 单元最终基，在 `f=0.0022` 盒上使用另外的种子 20261008 检查 35 个点，包括端点、中心和 32 个内部点。求完整广义特征值，而非仅单个端口或名义输入，保存最坏输入向量及 Rayleigh 商复核。

采样最大误差分别为 `0.001360593873, 0.001147382845, 0.001429816042`，均低于各自该盒的连续上界。每个基另在 low、center、high 三点进行 `quad_vec` 正积分：直接积累 `(T-Tv)^T C (T-Tv)` 与 `T^T C T`，不使用相减 Lyapunov Gram 作为积分参考。积分区间为 `[0,infinity)`，绝对/相对积分目标均为 `1e-9`。

这 9 项正积分与解析 eigenmode Gram 的相对能量误差数值之最大绝对差为 `6.99e-13`。细网格 101 阶另做三点正积分，差值不超过 `3.35e-13`，最大正积分误差 `0.0017571746471226888`。这排除了本批结果主要来自两个大 Gram 相减的数值假象，但不是区间积分认证；积分器估计误差也不能替代严格包围。

## 数学实现压力实验

100 个固定种子 20261007 的独立系统，使用稠密、互不对角化的 SPD `C,K`，`K>=Ka`，随机未排序的正 shifts，4–24 个时间项、3 个输入方向。独立强迫递推逐项积累全时间系数能量，加上**精确末端能量**，与联合控制矩阵界比较。

每个系统还检查连续谱区间内 101 个对数点和全部实际广义特征值的 Loewner 余量，并作非正交稠密坐标变换，检查对偶残差 Gram 的不变性。

结果：0 次观测违界，最小被检查 Loewner 余量 `4.80e-7`；上界/参考比在 `2.149–6.714`；最大坐标变换相对差 `6.71e-16`。这个随机余量测试并不普遍紧致，仍有明显保守性。连续谱覆盖的理由来自 Poisson 核比值证明，不来自这 101 个离散检查点。

数据：`results/signed_poisson_confirmation.json`、`results/guard_mesh8_positive_time.json`。

## 成本与剩余问题

单次完整模型证书包含 denominator 4、trial 2880、Riesz 5185 个 RHS，共 8069；提取、多个候选证书和独立参考均需额外计入。种子实验的小模型压缩证书仍产生 8068 个小模型 RHS；不能把这些计为免费。随机压力实验和稠密 eigenmode 参考属于外部验证成本。新矩阵界在同设置下不额外增加全阶 RHS，但本原型仍没有建立经济优势。

仍缺少：整个有效 HTC 区间的最终基认证、含线性求解与浮点舍入误差的外向包围，以及减少共同谱矩阵包围保守性的进一步数学方法。空间谱通道条件化仍是值得研究的方向，本轮没有用缓存或调参冒充该改进。

## 复现与代码验证

环境：Python 3.11、NumPy 2.3.5、SciPy 1.17.0、单线程 BLAS。运行前设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python`。

相关修改前基线：以下命令 **32 passed**；本轮加入 seed/mesh/native 配置与保存矩阵验证两个回归测试，分别观察预期 ImportError 后实现。相同完整命令最终 **34 passed**。

```bash
python -m pytest playground/adaptive_bci_sampling/test_field_audit.py playground/adaptive_bci_sampling/test_rational_dynamic.py playground/adaptive_bci_sampling/test_matrix_innovation.py -q -p no:cacheprovider
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --model native-case1 --output playground/adaptive_bci_sampling/results/guard_native_seed20260805.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --seed 20260806 --output playground/adaptive_bci_sampling/results/guard_seed20260806.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --seed 20261007 --output playground/adaptive_bci_sampling/results/guard_seed20261007.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --mesh-mm 8 --output playground/adaptive_bci_sampling/results/guard_mesh8_seed20260805.json
python playground/adaptive_bci_sampling/verify_guarded_svd_cells.py --basis playground/adaptive_bci_sampling/results/guard_seed20260806.npz --output playground/adaptive_bci_sampling/results/guard_seed20260806_wider.json
python playground/adaptive_bci_sampling/verify_guarded_svd_cells.py --basis playground/adaptive_bci_sampling/results/guard_seed20261007.npz --output playground/adaptive_bci_sampling/results/guard_seed20261007_wider.json
python playground/adaptive_bci_sampling/confirm_signed_poisson.py --output playground/adaptive_bci_sampling/results/signed_poisson_confirmation.json --basis playground/adaptive_bci_sampling/results/guard_native_seed20260805.npz playground/adaptive_bci_sampling/results/guard_seed20260806.npz playground/adaptive_bci_sampling/results/guard_seed20261007.npz
python playground/adaptive_bci_sampling/confirm_native_and_mesh.py --output-dir playground/adaptive_bci_sampling/results
```

原生构建先运行 CMake Release 配置（`USE_MKL=OFF`）和 `cmake --build build --target mhs_c_api -j 2`。最后一条实验脚本复现原生矩阵对比和细网格直接证书；本轮已实际运行同一计算路径。正时间积分通过 `verify_guarded_svd_cells.positive_time_reference` 对保存矩阵进行。

修正了独立验证脚本原先总是重新加载 10-mm Case1 且固定阈值 0.002 的限制：现在读取 NPZ 保存的矩阵、区间与 tolerance，防止换网格时误验证另一模型。第一轮旧 NPZ 缺少区间元数据，仍仅按其明确的 10-mm/0.001 历史配置兼容。

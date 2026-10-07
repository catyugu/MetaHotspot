# 无原始数据依赖的复现说明

本分支提交推导、算法、实验脚本、测试和整理后的 Markdown 结果。
不提交生成的矩阵、温度场、基底、NPZ、JSON、ZIP、日志或 native 编译产物。
这些脚本从仓库模型配置重新组装并生成输入；后续验证使用**本次运行生成**的基底文件。
所有输出均写入 Git 忽略的 `playground/adaptive_bci_sampling/results/`。

## 1. 环境

推荐 Python 3.12。最新稳态/阶跃批次实际使用 3.12.14、NumPy 2.3.5、SciPy 1.17.0、
pyamg 5.3.0、pytest 9.1.1。历史记录中的环境描述按当轮保留，不要求字节级基底哈希一致。
在仓库根目录使用隔离 Python 环境：

```bash
python -m venv /tmp/metahotspot-research-20261007
. /tmp/metahotspot-research-20261007/bin/activate
python -m pip install numpy==2.3.5 scipy==1.17.0 pyamg==5.3.0 pytest==9.1.1
export PYTHONPATH="$PWD/python"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p playground/adaptive_bci_sampling/results
```

PowerShell 可用对应 venv 激活命令，并设置 `$env:PYTHONPATH="$PWD/python"`、
`$env:OPENBLAS_NUM_THREADS='1'`、`$env:OMP_NUM_THREADS='1'`；以下 Python 参数不变。
除原生对比外，矩阵重建实验与测试不需要预先构建 C++ API。

## 2. 测试

```bash
python -m pytest playground/adaptive_bci_sampling/test_field_audit.py playground/adaptive_bci_sampling/test_rational_dynamic.py playground/adaptive_bci_sampling/test_matrix_innovation.py playground/adaptive_bci_sampling/test_steady_step_audit.py -q -p no:cacheprovider
```

已整理基线：37 passed。包含 Parseval 与精确尾、Poisson 频率积分、连续谱支配、
带符号抵消、全输入近共线反例、坐标不变性、SVD 常数模态与嵌套、steady 连续盒、
step 初始/中间/无限尾和缺失输入拒绝。

## 3. 最新稳态/阶跃目标与提取成本

```bash
python playground/adaptive_bci_sampling/steady_step_audit.py --seed 20260805 --output playground/adaptive_bci_sampling/results/steady_step_seed20260805.json
python playground/adaptive_bci_sampling/steady_step_audit.py --seed 20260806 --output playground/adaptive_bci_sampling/results/steady_step_seed20260806.json
python playground/adaptive_bci_sampling/steady_step_audit.py --seed 20261007 --output playground/adaptive_bci_sampling/results/steady_step_seed20261007.json
python playground/adaptive_bci_sampling/steady_step_audit.py --stock-sweep --output playground/adaptive_bci_sampling/results/steady_step_stock_tighter.json
```

默认 ε=0.001、网格上限 10mm、f=0.0002。预期最终阶数 48/48/53，steady 上界
0.001899/0.001846/0.001971，完整逆作用成本 134/137/140。
对照首次通过提取 RHS 150/176/153；失败 sweep 在输出中保留。
不把七点全时间通过标记为连续 HTC step 通过。
完整整理结果：[STEADY_STEP_COST_20261007.md](records/STEADY_STEP_COST_20261007.md)。

## 4. 历史 signed 冲激路线：先生成再验证

```bash
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --output playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_2tau.json
python playground/adaptive_bci_sampling/verify_guarded_svd_cells.py --basis playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_2tau.npz --output playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_wider_cells.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --seed 20260806 --output playground/adaptive_bci_sampling/results/guard_seed20260806.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --seed 20261007 --output playground/adaptive_bci_sampling/results/guard_seed20261007.json
python playground/adaptive_bci_sampling/verify_guarded_svd_cells.py --basis playground/adaptive_bci_sampling/results/guard_seed20260806.npz --output playground/adaptive_bci_sampling/results/guard_seed20260806_wider.json
python playground/adaptive_bci_sampling/verify_guarded_svd_cells.py --basis playground/adaptive_bci_sampling/results/guard_seed20261007.npz --output playground/adaptive_bci_sampling/results/guard_seed20261007_wider.json
python playground/adaptive_bci_sampling/confirm_signed_poisson.py --output playground/adaptive_bci_sampling/results/signed_poisson_confirmation.json --basis playground/adaptive_bci_sampling/results/rational_case1_guarded_svd_2tau.npz playground/adaptive_bci_sampling/results/guard_seed20260806.npz playground/adaptive_bci_sampling/results/guard_seed20261007.npz
```

预期冲激最终阶数 76/80/80。第一条默认使用矩阵重建；原生复验是下一节独立步骤。
验证器读取保存矩阵、区间和容差，不能用另一个网格重建替代保存对象。
100 个 seeded SPD 压力实验和每个基的 35 点最坏输入、三点正时间积分均从本次输入重新计算。
仅复现压力实验可以省略 `--basis`，无需任何已有 NPZ。

整理结果：[SIGNED_POISSON_EXPERIMENT_20261007.md](records/SIGNED_POISSON_EXPERIMENT_20261007.md)、
[SIGNED_POISSON_CONFIRMATION_20261007.md](records/SIGNED_POISSON_CONFIRMATION_20261007.md)。

## 5. 原生核对与 1200 单元细网格

需要 C++20 编译器、CMake、Ninja，以及首次获取 CMake/CPM 依赖的网络。
本轮成功构建环境为 GNU C++ 13.3、Release、USE_MKL=OFF。

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=OFF
cmake --build build --target mhs_c_api -j 2
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --model native-case1 --output playground/adaptive_bci_sampling/results/guard_native_seed20260805.json
python playground/adaptive_bci_sampling/svd_dynamic_guard.py --mesh-mm 8 --output playground/adaptive_bci_sampling/results/guard_mesh8_seed20260805.json
python playground/adaptive_bci_sampling/confirm_native_and_mesh.py --output-dir playground/adaptive_bci_sampling/results
```

预期分预算路线在细网格被拒绝，raw 界 0.001614>0.001；最后一条在相同容差下
直接认证最终 101 阶基，界 0.001762<0.002。第一项拒绝是实验结果，不是程序失败。
最后一条同时重新生成 364/1200 单元 native 矩阵比较。

细网格正时间积分可直接对本次生成矩阵调用已有函数：

```bash
python - <<'PY'
import sys, numpy as np
sys.path.insert(0, 'playground/adaptive_bci_sampling')
from verify_guarded_svd_cells import load_saved_system, positive_time_reference
p = np.load('playground/adaptive_bci_sampling/results/guard_mesh8_direct.npz')
s, ranges, threshold = load_saved_system(p)
for h in [p['low'], np.sqrt(ranges[:, 0]*ranges[:, 1]), p['high']]:
    print(h, positive_time_reference(s.operator(h), s.C, s.G, s.V))
PY
```

## 6. 早期正交有理与包围细分判决

以下保留当轮设置。raw 冲激阈值为 τ，final 为 2τ；均不是最新 step 的阈值。

```bash
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model chain --basis-stage raw --terms 96 --degrees 7 --widths .02 --cover-cells 4 --cover-degree 7 --output playground/adaptive_bci_sampling/results/rational_chain_cover7.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model chain --basis-stage raw --terms 64 --propagation energy --envelope-depth 2 --widths .02 --degrees 3 --cover-cells 4 --cover-degree 4 --output playground/adaptive_bci_sampling/results/rational_chain_energy64_cover4_envelope2.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model chain --basis-stage raw --terms 64 --propagation matrix --widths .02 --degrees 4 --cover-cells 4 --cover-degree 4 --output playground/adaptive_bci_sampling/results/rational_chain_signed_matrix64_cover4.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 80 --propagation matrix --widths .002 .0002 --degrees 1 2 3 --output playground/adaptive_bci_sampling/results/rational_case1_signed_matrix80.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 80 --propagation energy --envelope-depth 2 --widths .0002 --degrees 2 --output playground/adaptive_bci_sampling/results/rational_case1_energy80_envelope2.json
python playground/adaptive_bci_sampling/run_rational_dynamic.py --model case1 --basis-stage raw --terms 80 --propagation energy --envelope-depth 4 --widths .002 --degrees 3 --output playground/adaptive_bci_sampling/results/rational_case1_energy80_envelope4.json
```

完整早期对照设置及其他失败实验命令在
[RATIONAL_DYNAMIC_EXPERIMENT_20261007.md](records/RATIONAL_DYNAMIC_EXPERIMENT_20261007.md)、
[RATIONAL_DYNAMIC_OPTIMIZATION_20261007.md](records/RATIONAL_DYNAMIC_OPTIMIZATION_20261007.md) 中。
chain signed/no-subdivision 的完整覆盖应拒绝；不能把 energy+subdivision 的成功嫁接到它。

## 7. 结果核对与提交边界

只比较整理记录中的标量指标、阶数、范围、接受/拒绝和成本，不要求浮点基底逐字节相同。
本批没有外向舍入，靠近阈值的数值通过尤其不能宣称严格机器认证。
每条实验输出保留计算设置、候选历史与数值状态供本地检查；仓库只提交整理后的结果文档。
`git status --short --untracked-files=all` 不应将上述 results 文件列入提交。

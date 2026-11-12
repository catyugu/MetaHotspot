# 非对称 Case-1 提取实验（2026-11-13）

基于 `dev@32eb633` 的散热器堆叠模型，默认 0.5 mm 网格，146,704 个有效单元。
比较原版自适应 Extended FANTASTIC 和全参数域共享谱区间，并区分提取误差、
快照压缩误差、边界投影误差与 FloTHERM 模型差异。

阶段结论和数值见 [experiment_report.md](experiment_report.md)。
[experiment_design.md](experiment_design.md) 和 [data_contract.md](data_contract.md)
固定了比较协议。`python/` 下的正式算法保持原样；替换只在独立实验进程内生效。

从仓库根目录安装并运行：

```bash
uv venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python -r playground/bci_rom_testcase1/experiments/codex_1113/requirements.txt
# Editable 安装若不能自动发现本地动态库，指向构建产物：
export MHS_CAPI_PATH="$PWD/.cache/skbuild/src/api/libmhs_c_api.so"
.venv/bin/python playground/bci_rom_testcase1/experiments/codex_1113/run.py
```

编译器由 CMake 使用当前工具链；本次 Linux 环境使用 `CXX=g++ CC=gcc` 编译。
Windows 请在安装后将 `MHS_CAPI_PATH` 指向相应 `mhs_c_api.dll`。

`run.py` 顺序运行两种谱计划、三个种子，随后执行独立验证和首个种子的 SVD 消融。
约需 5–10 分钟 CPU 时间；各提取运行的实际耗时记录在输出中。完成的提取会自动跳过。
所有大型矩阵、快照、完整日志和 JSON 保存在被 git 忽略的 `results/`，需要约 2 GB。
不要复用不同代码或模型版本留下的 `results/`；`--out PATH` 可指定新的目录。

单独运行一次提取和验证：

```bash
.venv/bin/python playground/bci_rom_testcase1/experiments/codex_1113/train.py --method global --seed 20260805 --out /tmp/global_20260805
.venv/bin/python playground/bci_rom_testcase1/experiments/codex_1113/evaluate.py /tmp/global_20260805 --out /tmp/case1_reference --ablate
.venv/bin/python playground/bci_rom_testcase1/experiments/codex_1113/energy_pod.py /tmp/global_20260805 --out /tmp/case1_reference
```

`energy_pod.py` 固定最终阶数，复用已有响应，以能量度量替代欧氏快照压缩。
这是首种子上的探索性消融，不参与三种子主比较。
`make_figures.py` 从实际执行的 JSON 汇总统计并输出三张 PDF/SVG/PNG 图。

HTC 验证包括标称点、全部八个角点、八个独立随机内部点。
瞬态采用各模型一致的 BDF1：20 s 步长，1000 s 总时长。
结温是功率形状加权平均。FloTHERM 导出本身是 35 阶模型，不能当作全阶真值。
全部结果为实际数值求解输出，无人工填充数值；不包含实体实验测量。

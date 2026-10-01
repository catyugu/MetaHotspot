# 全场传递算子研究与审计

目标是 `X(h,s) = (K+sC+sum_i h_i H_i)^-1 G`：功率输入到**全部温度自由度**的传递算子。
固定实移验收使用所有输入组合下的相对 `A(h,s)`-能量误差，容差 `tau` 对应平方缺陷阈值
`tau^2`。动态目标是全场 impulse 的 `C` 加权时间积分误差；严格定义及尚缺的传递定理见
[THEORY.md](THEORY.md)。结温、共址输出 H2/Hankel、有限时刻 step 测量仅为辅助观测。

生产 `python/metahotspot/macromodel/utils.py` 保持 stock Extended BCI FANTASTIC，实验算法
留在 playground。新方法只能接受容差与 HTC 范围作为控制参数；网格、物理模型和审计分辨率
属于验证条件，不能成为改善候选成绩的调参手段。固定 seed 是复现基线所需的历史设置，不能
作为新方法输入。

当前目录保留一种基线审计和三项数学支撑；不再运行已排除的选点配方。

| 文件 | 用途 |
| --- | --- |
| `certify_extraction.py` | stock 最终基的 DC 连续盒上界与直接全场对照，输出原始 JSON |
| `certified_box.py` | 固定实移的 Riesz–Bernstein 单元上界；升阶与分支定界是研究工具 |
| `exact_error.py` | 对角 Robin 项的 Woodbury 误差映射，作为小模型独立核验工具 |
| `sparse_solve.py` | 大型 SPD 系统的 AMG 预条件 CG |
| `test_field_audit.py` | 全输入场误差、弱观测节点误差及完整空间的回归检查 |
| `records/FAILURE_ARCHIVE.md` | 唯一历史档案：负结果、测量更正、旧证书与预算证据 |

删除旧的确定性种子/网格贪心提取配方及其预算 sweep、仅端口 step 驱动；它们没有建立满足
当前全场动态目标且优于基线的方法。历史公式或负结果不等于所有上位数学方向被排除。
历史脚本在 Git 中保留，档案中的旧命令仅标识当时实验。

从根目录运行：

```powershell
$env:PYTHONPATH = "$PWD/python"
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& E:/env/miniconda3/envs/numerical/python.exe playground/adaptive_bci_sampling/certify_extraction.py 5 --tolerance 1e-3 --output "$env:TEMP/metahotspot-field-audit.json"
& E:/env/miniconda3/envs/numerical/python.exe -m pytest python/tests playground/adaptive_bci_sampling/test_field_audit.py -q -p no:cacheprovider
```

审计固定在 `s=0`、每轴 8 单元、三阶 trial、每轴 4 个 Gram 锚点块；每单元检查上下角及两个
独立内部点。JSON 同时记录这些条件、基线原始设置、直接场误差和所有 RHS 计数。该分辨率上的
上界可能很松，审计成功仅表示未发现违反上界，不表示达到给定容差。浮点 AMG-CG 与稠密代数
未作区间误差包围，`floating_point_certified=False`。

成本分别报告 `N_extract`、`N_certificate`、`N_reference` 与其总和。候选提取必须把为了选择、
停止或认证而执行的全阶逆作用纳入 `N_FOM`；独立外部参考成本单列。AMG setup、谱 matvec、CG
迭代、墙钟和内存另计，不能用快照数代替端到端成本。

Case 1 的 `playground/bci_rom_testcase1/reproduce_case1.py` 仍报告实际整场稳态与瞬态最终时刻
恢复误差，以及热源区观测；其名义功率组合不能代替所有输入方向或整个时域的保证。

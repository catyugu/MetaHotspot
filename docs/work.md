# 问题与修复

## 1. 边界条件的正则匹配（Regex）耦合在计算准备阶段

在 boundary_conditions.py 的 resolve_boundary_cells 中，目标单元的匹配使用了 Python 的 re 模块进行正则表达式匹配（target_regex）。

* 冲突点：如果把这个逻辑交给 C++，C++ 就必须引入正则表达式库来处理字符串匹配。

* 理想状态：计算内核不应该知道什么是“正则表达式”或“单元名称”。它只应该拿到一个数组：[边界ID, 关联的Cell ID数组, 边界参数值（键值对）]。

## 2. 时态（Transient）仿真与文件 IO 的耦合

在 thermal_solver.py 的 solve_transient 中，代码在一个 for i, step_power in enumerate(ptrace): 循环中，按步读取并映射字典。而且 _load_ptrace 是在 metahotspot_solver.py 里解析文本文件。

* 冲突点：绝对不应该在时间步长循环里去处理字符串字典（step_power.get(n, 0.0)）。

* 理想状态：应该提前将整个 ptrace 转换为一个连续的 2D 浮点数组（[num_steps, num_units] 的矩阵），并随着初始状态一次性传给 C++。
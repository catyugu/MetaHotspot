# 第二阶段交付：前处理案例准备

## 1. 本阶段实现内容

### 1.1 转换脚本能力增强

已完成 `scripts/adapter.py` 与 `metahotspot/converter.py` 的增强，支持：

- 单案例转换与批量四案例转换（example1~example4）
- 同时生成稳态/瞬态两套配置
  - `solver_config.toml`（默认稳态）
  - `solver_config_steady.toml`
  - `solver_config_transient.toml`
- `.lcf` 多层案例转换（example3、example4）
- 功率单元从层局部坐标到全局坐标映射
  - 采用“全局最大平面对齐 + 层平面居中偏移”策略
- 微流 CSV 编码转换（扩展支持，非 example1~4 必需）
  - 解析 `code` 网格（0/1/2/3）
  - 自动生成边界组编号并映射到 `boundary_conditions.selection`

### 1.2 运行方式

严格使用终端激活环境执行，不依赖 VS Code Python 包装层：

```bash
conda activate numerical
python scripts/adapter.py --batch-four --mode both
```

## 2. 四案例转换验证

### 2.1 产物检查

已成功生成：

- `examples/hotspot_converted/example1/mesh.msh`
- `examples/hotspot_converted/example1/solver_config_steady.toml`
- `examples/hotspot_converted/example1/solver_config_transient.toml`
- `examples/hotspot_converted/example2/mesh.msh`
- `examples/hotspot_converted/example2/solver_config_steady.toml`
- `examples/hotspot_converted/example2/solver_config_transient.toml`
- `examples/hotspot_converted/example3/mesh.msh`
- `examples/hotspot_converted/example3/solver_config_steady.toml`
- `examples/hotspot_converted/example3/solver_config_transient.toml`
- `examples/hotspot_converted/example4/mesh.msh`
- `examples/hotspot_converted/example4/solver_config_steady.toml`
- `examples/hotspot_converted/example4/solver_config_transient.toml`

### 2.2 配置一致性检查

检查项：

- 稳态配置 `simulation_type == "steady"`
- 瞬态配置 `simulation_type == "transient"`
- `mesh.msh` 文件存在

检查结果：通过。

## 3. 扩展验证（CSV 微流层）

为覆盖“CSV 编码 -> 边界组编号”需求，额外对 example5 做了烟测：

```bash
conda activate numerical
python scripts/adapter.py Hotspot/examples/example5 examples/hotspot_converted/example5 --mode both
```

结果：

- 生成成功
- 配置中包含 `microchannel_group_map`
- 配置中包含 `microchannel_cells`
- 生成了 `temperature`/`inlet_velocity`/`outlet_pressure` 三类边界条件

## 4. 与工作清单对应关系

已完成并在 `docs/work.md` 勾选：

- 第二步第 1 项：转换脚本
- 第二步第 2 项：四案例稳态/瞬态转换确认

未完成：

- 第二步第 3 项：提交到 git（等待你确认后统一提交）

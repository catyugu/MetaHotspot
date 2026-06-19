# Issue 06: CLI --fluid-overlay + 集成物理合理性验证

## Parent

.fluid-algorithm/PRD.md

## What to build

为命令行入口添加 `--fluid-overlay` 参数,完成从 CLI 到求解器的端到端流体-固体耦合传热集成。

### 修改点

**`bin/main.cpp`**:

```cpp
// 新增 --fluid-overlay 参数解析
std::optional<std::string> fluid_overlay_path;
if (args.contains("--fluid-overlay")) {
    fluid_overlay_path = args["--fluid-overlay"];
}

// 读取 overlay
auto io = mhs::io::read_xml(case_path);
auto fluid_overlay = fluid_overlay_path
    ? mhs::io::read_fluid_overlay_xml(*fluid_overlay_path)
    : std::nullopt;

// 传递给 Preprocessor
auto model = preprocessor.load(io, fluid_overlay);
```

### 验证

运行:

```bash
cmake --build build --parallel --config Release
bin/metahotspot --case cases/microfluid_cases/steady_case1.xml \
    --fluid-overlay cases/microfluid_cases/steady_case1_additional.xml \
    output.vtu output.xml
```

## Acceptance criteria

- [ ] `cmake --build build --parallel` 编译通过
- [ ] `python run_tests.py` 所有现有测试 + 流体测试 100% 通过
- [ ] 无 `--fluid-overlay` 时,行为与之前完全一致 (纯固体路径)
- [ ] 有 `--fluid-overlay` 时,steady_case1 输出:
    - [ ] 最高温度 < 343K (比纯固体解低,流体带走热量)
    - [ ] 流体入口附近 ≈ 298.15K
    - [ ] 整体 T 在 300K~340K 范围
    - [ ] 无负温度、无 NaN
- [ ] VTU 输出文件可正常被 ParaView 打开
- [ ] **HITL 验证**: 人工与 COMSOL 截图大致对比,确认温度分布趋势合理

## Blocked by

- Issue 05 (advection 上风组装 + 出口温度注入)

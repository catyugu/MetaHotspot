# 切片 1 — 拆 `assemble()`

> **Status**: needs-triage
> **依赖**: 切片 0
> **阻塞**: 切片 2（TimeScheme 需要 assemble_static / assemble_mass）

## 目标

把 `Assembler::assemble()` 拆成 `assemble_static()` + `assemble_mass()`。**删除**原 `assemble()`。瞬态项分支（`study_type==Transient && dt>0`）整段删除。

## 修改

### `src/assembler/assembler.hpp`

- 删除 `assemble()` 声明
- 新增 `assemble_static()`、`assemble_mass()` 声明
- 新增 `struct StaticOpsResult { K, f_static }` 与 `struct MassOpsResult { M_diag }`

### `src/assembler/assembler.cpp`

- 删除 `assemble()` 实现
- 删除瞬态项分支（末段 `mass_coeff = rho_c * vol / dt` 整段）
- 实现 `assemble_static()`：原 assemble 的非瞬态部分
- 实现 `assemble_mass()`：返回 `Eigen::VectorXd` 每单元 ρc·vol

### `src/nonlinear/nonlinear_solver.cpp`

- 临时在 `nonlinear_solve` 内部**用新接口**（`assemble_static` + `assemble_mass`）拼装 BDF1 LinearSystem
- 这是切片 1 的临时拼装；切片 2 引入 `Bdf1Scheme::build_system` 后删除

### `tests/test_assembler.cpp`

- **先删** `AssembleReturnsCorrectSize` 等旧测试
- 新增 `AssembleStaticReturnsKAndFStatic`
- 新增 `AssembleStaticHasNoTransient`：**关键** —— 验证 `K(c,c)` 不含 `ρc·vol/dt`
- 新增 `AssembleMassReturnsDiag`
- 新增 `AssembleStaticReadsTemperature`（温度非线性）

## 关于"是否暂留旧 assemble"

**最终决策**：直接删 `assemble()`。Feature branch 允许 build 短红到切片 3。`nonlinear_solve` 立即改用新接口 + 临时拼装。

## 验证

```bash
cmake --build build --parallel
python run_tests.py
```

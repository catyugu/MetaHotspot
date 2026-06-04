---
Status: ready-for-agent
---

# 05: 测试与 case

## 范围

### 单元测试

- IO 解析单测（`tests/test_io.cpp` 或对应文件）：
    - `<DaoreXishu>1,2,3</DaoreXishu>` → kx=1, ky=2, kz=3
    - `<DaoreXishu>  5  </DaoreXishu>` → kx=ky=kz=5
    - `<DaoreXishu>1, 2</DaoreXishu>` → panic
    - `<DaoreXishu>1,2,3,4</DaoreXishu>` → panic
    - `<DaoreXishu>1,,3</DaoreXishu>` → panic
- Preprocessor 单测：编译路径、非法表达式 panic
- 装配回归：5 个现有 case 结果差异 < 1e-9（或既有容差）

### 集成 case

- `cases/anisotropic_conductivity/case1.xml`：典型芯片级不等导热
    - kx ≠ ky ≠ kz，三方向明显差异
    - 在 `scripts/compare_steady_results.py` 框架内能跑出
    - 参考解：可手算 / 文献（小尺寸盒 + 均匀热源 + 稳态，热流各向异性）
    - 或：与一个解析解 / 简化二维情况做对照

### 失败用例

- `cases/anisotropic_conductivity/case_bad_2exprs.xml` / `case_bad_4exprs.xml`：
    - 期望 main 退出码非零、stderr 含 panic 信息

## 验收

- `python run_tests.py` 全部通过
- 失败 case 退出码非零
- PR 描述列出新增 / 修改文件

## 不做

- 实现代码（02-04）
- 文档 / ADR（属于 06）

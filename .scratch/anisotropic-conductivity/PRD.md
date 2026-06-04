# PRD: 各向异性热导率（kx / ky / kz）

## 目标

支持材料的**各向异性**热导率，前端通过逗号分隔的三个表达式分别指定 `kx` / `ky` / `kz`（W/(m·K)）。
维持后端内部模型为三轴分量的形式（`MaterialProps` 拆 `kx` / `ky` / `kz`），装配/边界条件中
按面的法向选用对应分量。

## XML / 前端格式

每个 `Material` 下的 `<DaoreXishu>` 元素接受**逗号分隔的表达式列表**（trim 空白）：

| 输入                                   | 行为                                                           |
| -------------------------------------- | -------------------------------------------------------------- |
| 三个表达式 `kx_expr, ky_expr, kz_expr` | 三个分量分别使用，方向化装配                                   |
| 一个表达式 `k_expr`                    | 退化为各向同性：`kx = ky = kz = k_expr`（向后兼容现有 case）   |
| 零个 / 两个 / 四个及以上表达式         | **错误**：进入 `panic`，状态码非零退出                         |
| 单个表达式为空、或解析失败             | **错误**：`panic`（继承现有 `expr::parse` 的常量回退要被绕开） |

> 错误要早：在 `io::read_xml` 阶段就校验清楚，不在装配阶段才爆。

## 后端数据流变更

```text
XML <DaoreXishu>"kx_expr,ky_expr,kz_expr"</DaoreXishu>
  → io::read_xml
      · 解析为 std::vector<std::string>{kx, ky, kz}
      · 校验 count ∈ {1, 3}，否则 panic("DaoreXishu must have 1 or 3 expressions ...")
  → IOStructure::Material
      · daore_xishu 字段替换为
          struct Conductivity {
              std::string kx, ky, kz;
          };
      · 旧单字符串字段移除
  → Preprocessor::load
      · expr::parse(kx), expr::parse(ky), expr::parse(kz)
      · 任意一个解析失败 → panic
  → InternalModel::MaterialProps
      · CompiledExpression kx, ky, kz   (旧 k 字段移除)
  → Assembler / Postprocessor
      · 内部面 / 虚拟邻居面：按法向取分量
          - FaceDir::XMinus / XPlus → kx (cell) / kx (neighbor)
          - FaceDir::YMinus / YPlus → ky
          - FaceDir::ZMinus / ZPlus → kz
      · 调和平均：单元分量 + 邻居分量
          cond = A / (d_half_cell/k_face + d_half_neighbor/k_face_neighbor)
      · Dirichlet 边界面：使用面法向对应分量
      · ThirdType 边界面：使用面法向对应分量（k·h·A/(k+h·d)）
      · Postprocessor 节点插值 / BC 面外推：仍按 cell 法向分量
```

> **关键不变量**：法向在装配层就是 `FaceDir`，现有代码已经按方向分支——这次只是把
> `materials[mat_id].k.eval(...)` 替换为 `materials[mat_id].<axis>.eval(...)`。

## 行为细节

1. **空格 / 截断**：逗号分隔后 `trim` 两侧空白；空字符串视作 0 表达式输入。
2. **方向选取**：用 `FaceDir` 决定 k 字段；不要在内层用 `(nx*ny+ny*nz+...)` 之类的奇技。
3. **错误信息**：必须打印原文 + 错误原因（`count=2`、第几段为空、parse 失败原因）。
4. **不修改 schema 文件**（`docs/xsd/...` 等如有）如果它们只是文档；只修代码 + 用例。
5. **数值与原各向同性 case 一致**：单表达式路径下结果与改前完全相同——这是回归基线。

## 验收

- 单表达式 case：`cases/simple_steady_tests/case1.xml` 等 5 个现有 case 结果与改前一致。
- 三表达式 case：新增 1 个用例，验证典型晶体不等导热（kx≠ky≠kz），结果与解析解 / 参考解一致。
- 错误用例：2 表达式、4 表达式、空字符串——`panic` 信息准确、退出码非零。
- 单元测试：
    - `io::read_xml` 对 `<DaoreXishu>` 解析的 1 / 3 / 2 / 4 路径
    - `preprocessor` 编译路径
    - 各向同性回退（1 表达式 → kx=ky=kz）

## 不在范围内

- 改单位体系
- 改后端 `MaterialProps` 之外的结构
- 修 FaceDir 枚举本身
- 重写装配 / 求解器算法
- 改 schema 文档（如果存在），仅代码 + case + 测试

## 关联

- 见 `docs/adr/` 中与导热相关的既有决策（待 PR 时在 ADR 增补一条：0006-各向异性）
- 涉及文件（实现时改动）：
    - `src/common/io_model.hpp` — `Material` / IOStructure
    - `src/common/internal_model.hpp` — `MaterialProps`
    - `src/io/io.cpp` — `read_xml` 解析 `<DaoreXishu>` 并校验
    - `src/preprocessor/preprocessor.cpp` — 编译 kx/ky/kz
    - `src/assembler/assembler.cpp` — 按 `FaceDir` 选分量
    - `src/postprocessor/postprocessor.cpp` — 同上
    - `cases/...` — 新增三表达式各向异性 case
    - `tests/...` — IO / preprocessor / 集成测试

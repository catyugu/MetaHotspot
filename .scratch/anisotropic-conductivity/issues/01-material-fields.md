---
Status: ready-for-agent
---

# 01: 改 Material 字段支持三轴 k

## 范围

- `src/common/io_model.hpp`
    - 新增 `struct Conductivity { std::string kx, ky, kz; }`（或类似）
    - `Material::daore_xishu` 替换为该结构
- `src/common/internal_model.hpp`
    - `MaterialProps`：`k` 替换为 `kx` / `ky` / `kz`（三个 `CompiledExpression`）
- 不改 IOStructure 其它部分；不改 FaceDir；不改 expr 模块 API

## 约束

- 命名遵循拼音：字段名仍用 `kx` / `ky` / `kz`（约定在 `docs/agents/domain.md` 中以英文为主）
- 旧 `daore_xishu` 字符串字段彻底删除，不要保留 shim
- 包含默认构造，使 `Material{}` 仍可用

## 验收

- 全部编译通过
- 现有包含 `Material` / `MaterialProps` 的引用同步更新（grep `daore_xishu` 与 `.k.eval(`)
  在 src/ 下应仅出现在 io / preprocessor / assembler / postprocessor 引用点）

## 不做

- XML 解析、装配、preprocessor 逻辑（属于后续 issue）
- 测试

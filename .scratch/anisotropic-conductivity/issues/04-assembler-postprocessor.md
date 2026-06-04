---
Status: ready-for-agent
---

# 04: 装配与后处理按 FaceDir 选 kx/ky/kz

## 范围

- `src/assembler/assembler.cpp`
    - 单元中心 k 评估处（约 123-125）：仍可一次取三个分量
    - 内部面调和平均（约 152-163）：按 `FaceDir` 取 kx/ky/kz 中的对应分量
    - FirstType BC（约 175）：按面法向取分量
    - SecondType BC（约 179-183）：原本 k 不参与 RHS，保持不变
    - ThirdType BC（约 194）：按面法向取分量
    - 虚拟邻居面：用法向取分量
- `src/postprocessor/postprocessor.cpp`
    - 节点插值的 k 权重（约 55）：仍按 cell 法向对应分量
    - BC 面外推的 k（约 95-106 范围）：按 cell 法向对应分量

## 关键不变量

- `FaceDir::XMinus / XPlus` → kx
- `FaceDir::YMinus / YPlus` → ky
- `FaceDir::ZMinus / ZPlus` → kz
- 不要硬编码 `(nx*ny+...)`，一律用 `FaceDir` 判别
- `MaterialProps` 现存 5 处 `.k.eval(...)` 调用全部替换；可用 `grep -n 'material_table.*\.k\.' src/` 自检

## 验收

- 单表达式 case（case1.xml 等）→ 结果与改前一致（基线回归）
- 新增三表达式 case → 方向化装配生效
- Postprocessor 输出与解析解一致

## 不做

- IO / preprocessor（02/03）
- 测试 / case（05）

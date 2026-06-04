---
Status: ready-for-agent
---

# 01: IO 解析 ObservePoints3D + 数据结构

## 范围

- `src/common/io_model.hpp`
    - 新增 `struct ObservationPoint3D { std::string name; double x, y, z; }`
    - `IOStructure` 加 `std::vector<ObservationPoint3D> observation_points`
- `src/common/internal_model.hpp`
    - `InternalModel` 加 `std::vector<ObservationPoint3D> observation_points`
- `src/io/io.cpp`
    - `read_xml` 解析 `<ObservePoints3D>` / `<ObservePoint3D>` 子元素
    - `<Name>` → `name`，`<X>` / `<Y>` / `<Z>` → `x` / `y` / `z`（`parse_double`）
    - `<ObservePoints2D>` 忽略（与 Dimension2D panic 语义一致）
    - 若 XML 无 `<ObservePoints3D>` → observation_points 为空向量（默认）
- `src/preprocessor/preprocessor.cpp`
    - `model->observation_points = ioStructure.observation_points`（纯搬运，不编译）

## 约束

- 不改 IOStructure 其它字段
- 不改 write_xml（属于 04）
- observation_points 默认空向量，稳态 case 不受影响

## 验收

- 编译通过
- case1.xml（瞬态，含 2 个观察点）能被 read_xml 正确解析出 `observation_points.size() == 2`
- 稳态 case 无 `<ObservePoints3D>` → observation_points.size() == 0

## 不做

- Postprocessor sample_point（02）
- Scheduler 回调（03）
- write_xml 回写（04）

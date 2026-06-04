---
Status: ready-for-agent
---

# 04: write_xml 回写 Result0DTransient

## 范围

- `src/io/io.hpp`
    - `write_xml` 签名变更：新增 `observation_traces` 参数

    ```cpp
    void write_xml(const std::string& input_path,
                   const std::string& output_path,
                   const InternalModel& model,
                   const std::vector<double>& node_temperature,
                   const std::vector<ProbeTrace>& observation_traces = {});
    ```

- `src/io/io.cpp`
    - `write_xml` 实现：
    1. 已有的 `Values/Data/a:double` 温度写入逻辑不动
    2. 在 `Results/a:anyType` 下对每个 `ProbeTrace`：
       - 按 `PointName` 搜索已存在的 `<a:anyType i:type="Result0DTransient">` 节点
       - 若存在 → 清空 `<Times>` / `<Values>` 内 `<a:double>` 子元素
       - 若不存在 → 创建新节点（含 `PhysicsName=温度`, `PointName`, `TimeUnit=S`, `UnitName=K`）
       - 按 trace.times / trace.values 顺序 append `<a:double>`
- `bin/main.cpp`
    - 调用 `write_xml(input_path, output_xml, *model, node_temperature, traces)`
    - 稳态 case → 传空 traces（默认参数生效）

## 约束

- `ProbeTrace` 定义在公共头文件（`src/common/types.hpp` 或单独头文件）

  ```cpp
  struct ProbeTrace {
      std::string name;      // 与 ObservationPoint3D::name 对应
      std::vector<double> times;
      std::vector<double> values;
  };
  ```

- 回写时 `TimeUnit` 取 `IOStructure.transient_time_unit`（从 XML 读取的 "S"/"MS" 等）
- `PhysicsName` 固定 "温度"，`UnitName` 固定 "K"

## 验收

- 瞬态 case1.xml → write_xml 输出包含 2 个 Result0DTransient，Times/Values 数量=101
  （t=0 到 t=100，101 个条目）
- 已有 Result0DTransient 节点 → 清空重写（不追加）
- 稳态 case → write_xml 输出不含 Result0DTransient（或空 traces 无操作）

## 不做

- IO 解析（01）
- sample_point（02）
- scheduler 回调（03）

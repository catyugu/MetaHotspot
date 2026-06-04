---
Status: ready-for-agent
---

# 03: Scheduler 时间步回调 + main 流程重构

## 范围

- `src/scheduler/scheduler.hpp`
    - 新增回调类型 / 结构：

    ```cpp
    struct StepCallback {
        std::function<void(double time, int step, const std::vector<double>& cell_T)> on_step_done;
    };
    ```

    - `Scheduler` 加 `StepCallback callback_` 字段 + `setCallback()` setter
- `src/scheduler/scheduler.cpp`
    - 瞬态循环中，每步 `nonlinear::solve` 完成后调用 `callback_.on_step_done(state_.current_time, state_.time_step, state_.T)`
    - 稳态分支：`run()` 结尾也调用一次（保持接口统一）
    - t=0（初始状态）也要在循环前回调一次，以包含初始温度
- `bin/main.cpp`
    - 定义 `ProbeTrace` 结构：`struct ProbeTrace { std::string name; std::vector<double> times; std::vector<double> values; }`
    - 在 scheduler.run() 前：
        - 若 observation_points 非空 && transient → 创建 traces 容器
        - 组装回调 lambda：每步做 `node_T = postprocessor.interpolate_cell_to_node(model, state.T)`，
      对每个点 `trace[i].append(time, postprocessor.sample_point(node_T, model, points[i]))`
    - 末步输出：
        - `write_vtu`（不变）
        - `write_xml` 传入 traces（需要改签名，04 负责）

## 约束

- Scheduler 不直接依赖 Postprocessor — 回调在 main.cpp 组装
- 回调默认为空（`std::function` 默认构造）→ 稳态 case 无开销
- t=0 必须回调：初始温度必须记录进 trace（与 reference XML 一致）

## 验收

- 稳态 case（无观察点）→ scheduler 行为不变，回调不执行
- 瞬态 case（有观察点）→ trace 包含 N+1 条目（t=0 + N 步）
- 回调中 `cell_T` 与 `state_.T` 一致

## 不做

- Postprocessor sample_point 实现（02）
- write_xml 回写（04）
- 回调频率控制（后续 issue）

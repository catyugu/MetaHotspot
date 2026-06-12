# 切片 0 — `TimeStepBuffer` 数据结构

> **Status**: needs-triage
> **依赖**: 无
> **阻塞**: 切片 1（拆 assemble 需要 history buffer）

## 目标

在 `mhs::core` 引入环形时间步历史缓冲。不引 Eigen、不引 sim。

## 新建文件

- `src/data/time_step_buffer.hpp`
- `src/data/time_step_buffer.cpp`（或 header-only）
- `tests/test_time_step_buffer.cpp`

## 接口

```cpp
namespace mhs::core {
class TimeStepBuffer {
public:
    explicit TimeStepBuffer(std::size_t cell_count, std::size_t capacity);
    void reset(const std::vector<double>& T_initial);
    void push(const std::vector<double>& T_new);
    const std::vector<double>& latest() const noexcept;
    const std::vector<double>& at(std::size_t i) const noexcept;
    double time_at(std::size_t i) const noexcept;
    double dt_to(std::size_t i) const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
private:
    std::vector<std::vector<double>> slots_;
    std::vector<double>              times_;
    std::size_t head_   = 0;
    std::size_t stored_ = 0;
    std::size_t cap_;
};
}
```

## 测试（红→绿）

- `PushThenLatest`：push 一次，`latest() == push 值`
- `AtRelative`：push 两次，`at(0)==T_2`、`at(1)==T_1`
- `WrapAround`：capacity=3，push 5 次，`at(0)` 第 5 次、`at(2)` 第 3 次
- `TimeAtAndDtTo`：push 配 time；`time_at(0)` 最新 time；`dt_to(1)` = `time_at(0) - time_at(1)`
- `Reset`：reset 后 `size()==1`、`latest()==reset 值`
- `EmptyBufferBehavior`：构造后 `size()==0`；`latest()` 返回空引用（UB 文档明示）

## 验证

```bash
conda activate cpp_env
cmake --build build --parallel
python run_tests.py
```

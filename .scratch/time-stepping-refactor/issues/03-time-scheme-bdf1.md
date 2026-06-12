# 切片 2 — `TimeScheme` 抽象 + `Bdf1Scheme`

> **Status**: needs-triage
> **依赖**: 切片 0、切片 1
> **阻塞**: 切片 3（Scheduler 需要 TimeScheme）

## 目标

建立算法抽象与第一个实现（BDF1）。

## 新建

- `src/time_scheme/time_scheme.hpp` — 抽象接口
- `src/time_scheme/time_scheme.cpp`
- `src/time_scheme/bdf1_scheme.hpp` / `.cpp`
- `tests/test_time_scheme.cpp`
- `tests/test_bdf1_scheme.cpp`

## 关键类型

```cpp
namespace mhs::sim::time_scheme {
enum class TimeSchemeKind { Bdf1, Bdf2, AdaptiveBdf };
struct TimeSchemeConfig { /* max_order, initial_dt, min_dt, max_dt, abs_tol, rel_tol, max_internal_steps, output_dt */ };
struct StepDecision { double dt; std::size_t order; };
enum class AcceptDecision { Accept, Reject };

class TimeScheme {
public:
    virtual ~TimeScheme() = default;
    virtual void initialize(TimeStepBuffer& history, GlobalState& state) const = 0;
    virtual StepDecision select_step(const TimeStepBuffer& history, double current_t) const = 0;
    virtual LinearSystem build_system(const StaticOpsResult& s, const MassOpsResult& m,
                                       const TimeStepBuffer& history, std::size_t order, double dt) const = 0;
    virtual AcceptDecision accept_or_reject(const TimeStepBuffer& history_before,
                                            const std::vector<double>& T_candidate,
                                            const std::vector<double>& error_estimate) const = 0;
};

class Bdf1Scheme : public TimeScheme { /* ... */ };
class StaticSchemeFactory { static std::unique_ptr<TimeScheme> create(const TimeSchemeConfig&); };
}
```

## BDF1 公式

```text
α0 = 1/Δt
A = α0·M + K
b = α0·M·T_n + f_static
```

## 修改

### `src/data/internal_model.hpp::GlobalState`

- 删除 `T_prev`（暂留此切片；切片 9 必删）
- 加 `TimeStepBuffer history;`
- 加 `int output_step = 0;`

### `src/scheduler/scheduler.hpp`

- 加 `std::unique_ptr<time_scheme::TimeScheme> scheme_;`
- 加 `time_scheme::TimeSchemeConfig scheme_cfg_;`
- 加 `void setTimeSchemeConfig(...)`
- 加 `std::unique_ptr<TimeScheme> schemeFactory(...)`

**注**：本切片不调 `scheme_`；仅暴露接口。切片 3 才接入主循环。

## 测试

### `test_time_scheme.cpp`

- `Bdf1SchemeSelectStepReturnsInitialDt`
- `Bdf1SchemeBuildSystemCoefficient`：验证 `A(c,c) == K(c,c) + M_diag(c)/dt`
- `Bdf1SchemeAcceptOrRejectAlwaysAccepts`

### `test_bdf1_scheme.cpp`

- `MatchesLegacyAt1ms`：与旧 fixed-step 行为一致（参考 `cases/simple_transient_tests/case1.xml` 历史数据或切片 1 临时 LinearSystem）
- `InitializeAfterPush`

## 验证

```bash
cmake --build build --parallel
python run_tests.py
```

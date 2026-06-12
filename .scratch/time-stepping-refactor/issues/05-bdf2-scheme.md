# 切片 4 — `Bdf2Scheme` + 起步降阶

> **Status**: needs-triage
> **依赖**: 切片 0、1、2、3
> **可与切片 5 并行**: 否（建议串行）

## 目标

固定步长 BDF2；起步时降阶到 BDF1（`history.size()==1` 时）。

## 新建

- `src/time_scheme/bdf2_scheme.hpp` / `.cpp`

## 公式

```text
h_n   = t_n - t_{n-1}
h_n-1 = t_{n-1} - t_{n-2}
δ     = h_n / h_{n-1}
α0    = (1 + 2δ) / (h_n · (1+δ))
α1    = -(1+δ) / (h_n · δ)
α2    = δ / (h_n · (1+δ))
A = α0·M + K
b = α0·M·T_n + α1·M·T_{n-1} + α2·M·T_{n-2} + f_static
```

固定步长退化（δ=1）：`α0=3/(2h)`, `α1=-2/h`, `α2=1/(2h)`，与教科书一致。

## 起步逻辑

- `select_step`：若 `history.size() < 2`，返回 `order=1`；否则 `order=2`
- `build_system`：根据传入 `order` 走 BDF1 或 BDF2

## 测试

### `tests/test_bdf2_scheme.cpp`

- `CoefficientsFixedStep`：固定 h 验证 A、b 公式
- `CoefficientsVariableStep`：手动设 δ=2，验证 α
- `StartsAsOrder1`：**关键** —— `history.size()==1` 时 `select_step` 返回 `order=1`；build_system 按 BDF1 拼
- `HandlesTwoStepHistory`：`history.size()==3` 时 BDF2 正常返回
- `Bdf2ConvergesToAnalytic`：1D 棒材、初值温阶跃、两端 T=0、无热源；解析解为 Fourier 级数；取 `t_end` 处中心点温度，比较 `dt=0.1, 0.05, 0.025` 的对数斜率 → 接近 2

## 验证

```bash
cmake --build build --parallel
python run_tests.py
```

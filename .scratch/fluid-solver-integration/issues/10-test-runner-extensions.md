# 10: run_cases.py / run_tests.py 扩展

Status: needs-triage · Type: tooling · Depends on: 08

## Context

新增 5 个流体单元测试 + 1 个集成 case 后，需要在 `run_tests.py` / `run_cases.py` 里登记，否则 CI 不会跑到。

## Goal

- `run_tests.py`：识别 `tests/test_fluid_*.cpp` 模式，自动加入构建目标
- `run_cases.py`：识别 `cases/microchannel_steady/` 子目录，添加 microchannel_steady 入口；case 跑通后与 `expected/temperatures.csv` 比对（误差 < 1%）

## Scope

- 修改 `run_tests.py` / `run_cases.py`
- 不重写工具脚本（保持现有风格）
- 不引入新依赖

## Acceptance

1. `python run_tests.py` 全绿，含 5 个流体新测试
2. `python run_cases.py` 全绿，含 microchannel_steady
3. microchannel_steady 跑通后自动与 fixture 比对；误差超阈报 `LOG_ERROR` 并退出非零
4. 既有测试 / case 不回归

## Notes

- 脚本风格保持既有约定（看现状后照抄）
- 不引入 pytest / unittest 等重框架
- microchannel_steady 跳过比对时（如 fixture 缺失）应 `LOG_WARN` 不报错（首次跑可允许）

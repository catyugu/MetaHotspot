# 08: 集成 case `microchannel_steady` + Python 参考 fixture

Status: needs-triage · Type: integration · Depends on: 06, 07

## Context

需要端到端验证：把 Poiseuille 解算 + 热装配串起来，与 Python `experiment-v1/examples/example4` 微通道参考解比对。

## Goal

新建 `cases/microchannel_steady/`：

```text
cases/microchannel_steady/
├── case.xml                    # 主 XML：5×1×3 通道 + 衬底
├── case_additional.xml         # sidecar：water_25C 流体材料 + 入口/出口 pressure BC
└── expected/
    ├── temperatures.csv        # 稳态单元中心温度
    └── config.json             # case 元信息（grid, materials, BCs）
```

主 XML 几何：

- 整体网格 5×1×3（X 通道方向 5 格、Y 窄 1 格、Z 高度 3 格）
- layer 1（衬底，硅）：所有单元为硅
- layer 2（微通道）：中央 1×1×5 单元是水，两侧单元是硅（通道壁）
- layer 3（顶盖）：所有单元为硅
- 入口 X=0 pressure=1.0e5 Pa，出口 X=5 pressure=0.0 Pa
- 底部 FirstType T=500 K（衬底被加热）；顶面 SecondType q=0；侧面 cauchy h=10 T_inf=300

稳态求解。

## Scope

- 主 XML 与 sidecar 编写（手工）
- 跑通 Poiseuille 解 + 热装配
- 输出稳态温度场（VTU）
- 用 Python `experiment-v1/examples/example4` 跑一遍同样的 case，导出参考 VTU / CSV 作为 fixture
- `run_cases.py` 新增 microchannel_steady 入口，比对 T 场误差 < 1%

## Acceptance

1. `cases/microchannel_steady/case.xml` + `case_additional.xml` 存在
2. `python run_cases.py` 全绿
3. 微通道 case 输出 T 场与 Python 参考误差 < 1%
4. 既有 case 不回归

## Notes

- Python 参考解生成方法：用 `experiment-v1/examples/example4/run.py` 跑同一几何、同一 BC，导出 VTU 后转为 CSV
- 误差容忍 1% 来自 Poiseuille 假设本身的精度边界；如要更严，可降到 0.1% 但需逐项排查
- 不在本 issue 写 `run_cases.py` 的扩展点；扩展本身单独 issue 09
- 参考 fixture 的生成脚本放 `scripts/gen_microchannel_reference.py`（新建），不可提交到 `cases/` 里

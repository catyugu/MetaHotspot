# 自适应 BCI 宏模型提取：研究工作台

本目录研究参数化热模型的 **功率输入 → 全温度场** 降阶；研究目标是比 stock Extended BCI FANTASTIC 获得更可靠、且计入认证成本后更廉价的宏模型提取。**截至 2026-10-17，尚无在异质模型与生产规模上同时稳定胜出、并对最终全场动态 ROM 给出完整保证的方法。** 小模型收益、固定频移残差风险和连续盒数学上界不能互相替代。

生产入口 `python/metahotspot/macromodel/utils.py` 仍是 stock 实现；这里的代码是审计与数学工具，不把历史对话中的实验原型误写成已合入的算法。

## 阅读顺序

| 文档 | 内容与证据边界 |
| --- | --- |
| [研究路线与实验台账](records/RESEARCH_SYNTHESIS_2026-10-17.md) | 2026-10-10 至 10-17 的跨对话路线、数值结果、证据等级、淘汰原因与开放问题 |
| [THEORY.md](THEORY.md) | 全场状态与冲激响应目标、固定实移恒等式、Riesz–Bernstein 盒界、连续分布风险定理及证明缺口 |
| [失败路线档案](records/FAILURE_ARCHIVE.md) | 原有逐条失败、反例、测量更正及其适用作用域；不据此宣布整个上位方向不可能 |

**当前共识。** 必须区分：（1）给定频移上的 SVD 前逐输入残差；（2）固定实移的全输入、全场能量误差；（3）最终 SVD 后空间的全场冲激/时域误差。对任一指标给出的连续参数分布风险保证，只对**该指标、该阶段、给定采样分布**成立。抽样认证不等于参数盒逐点上界，亦不自动覆盖任意时间输入或全部频率。用户可接受连续分布风险作为研究控制目标；更强的确定性全盒证书仍是独立研究方向。

### 现阶段最重要的实测判断

- [JFRGE 归档提交](https://github.com/catyugu/MetaHotspot/commit/56fbfea98a31b9a07c6553c41a23646f76faad54) 含可追溯的原生 Case1 47,085 / 122,400 DOF、24 条配对记录：对**同样补足 SVD 前残差风险认证的 stock**，引导式联合频移评价降低了平均 RHS 与总时间；但相对**未经风险补足的原生产停止规则**，总时间仍更高。不能省略这个基线差异。
- Project 历史中的 Riesz–Lanczos、单次 V-cycle 校正、动态感知选阶和 PAC 随机富集各有局部正面信号；也分别出现认证开销高、规模放大后反转、仅固定频移风险通过、或缺少异质模型重复验证的问题。具体条件与数值见研究台账；**对话报告不等于当前分支可复现的提交记录**。
- 已否决将共享跨输入响应空间、只看更小残差、简单目标导向 SVD 或未加证明的固定频移代理直接视为最终动态 ROM 的普适改进。保留各输入独立空间、生产既定频移、随机探索、Riesz 加权、AMG 迭代和闭环风险审计作为后续比较的约束，不能预设其收益。

## 目录代码

| 文件 | 维护用途 |
| --- | --- |
| `certify_extraction.py` | stock 最终基的 DC 连续盒上界与直接全场对照，输出审计 JSON |
| `certified_box.py` | 固定实移的 Riesz–Bernstein 单元上界；升阶与分支定界是数学研究工具 |
| `exact_error.py` | 对角 Robin 项的 Woodbury 误差映射，用于小模型独立核验 |
| `sparse_solve.py` | 大型实 SPD 系统的 AMG 预条件 CG |
| `test_field_audit.py` | 已存在的全输入场、弱观测节点及完整空间回归检查 |
| `records/` | 原失败档案与本次跨对话研究台账；不存放大型原始数据 |

已退役的确定性种子/网格贪心和旧预算 sweep 不在此恢复；历史命令只属于相应实验时的实现。JFRGE 的脚本和原始记录可以在上述**历史提交**读取，**不在当前 `agent/work` 工作树里**。

## 当前审计复现与口径

在仓库根目录、安装相应 Python 依赖并按需构建原生 API 后运行（以下为 POSIX shell；Windows 请按环境设置变量）：

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python playground/adaptive_bci_sampling/certify_extraction.py 5 \
    --tolerance 1e-3 --output /tmp/metahotspot-field-audit.json

PYTHONPATH=python python -m pytest \
  python/tests playground/adaptive_bci_sampling/test_field_audit.py \
  -q -p no:cacheprovider
```

该 DC 审计固定 `s=0`、每轴 8 单元、三阶 trial 和每轴 4 个 Gram 锚点块；每单元检查上下角及两个独立内部点。输出保存原始设置、直接场误差、分母条件与 RHS 计数。通过“未发现违反上界”不意味着达到 `tau=1e-3`。浮点 AMG-CG 与稠密代数未作区间包围，`floating_point_certified=False`。本次**仅整理文档**，没有重新运行大规模实验或新增测试。

审计成本必须分别记录快照/富集逆作用、选点逆作用、停止/认证逆作用与独立外部参考；报告 `N_FOM` 之外，还要报告 AMG setup、迭代数、谱运算、投影、SVD、墙钟及峰值内存。内部审核属于提取闭环成本，不能把独立 holdout 误计成免费或同一停止标准。不同终止判据、不同基底和不同空间阶段的实验不能直接按 RHS 或秒数排序。

Case 1 的 `playground/bci_rom_testcase1/reproduce_case1.py` 仍提供名义功率下的整场稳态、瞬态最终时刻和热源区诊断；它们不是所有输入方向、所有 HTC 参数或全部时域的认证。

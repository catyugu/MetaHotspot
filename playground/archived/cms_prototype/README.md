# 带空腔封装：部件 ROM + 完整 TIM/lid FOM

仅压缩 silicon core 到 substrate 的内部动态。上方 TIM 和铜 lid 保持三维 FOM，通过两块芯片顶面的全部物理温度自由度连接。保留两个逻辑芯粒、两个功率输入，不添加存储芯粒或额外热点。

## 运行

需要 NumPy、SciPy 和 Matplotlib。在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py --dx-mm 1.5 --dy-mm 1.5 --dz-mm 0.1 --materials isotropic weak_lateral --full-interface
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py --dx-mm 0.75 --dy-mm 0.75 --dz-mm 0.1
```

- `--dx-mm`、`--dy-mm`、`--dz-mm`：三个方向的**最大单元尺寸**，单位 mm，默认分别为 **0.75、0.75、0.1**。每个材料层按厚度和 z 上限向上取整划分；x/y 方向自动插入精确芯片边界线。z 方向也保留固定热源区边界，不再用单一 nx 或固定层单元数控制网格。
- 三个参数独立生效，例如 `--dx-mm 0.4 --dy-mm 0.7 --dz-mm 0.08`。几何边界附近允许更小单元，实际最大尺寸写入 JSON。
- `--patches`：每块芯片接口上每边的富集图案分段数，默认 2，每块形成 9 个双线性图案。全部接口节点仍然保留。
- `--shift-tolerance`：椭圆选点预算，默认 1e-3，**不是最终误差保证**。
- `--full-interface`：增加全部接口热流方向的实 Laplace 采样考核。
- `--time-step`：后向 Euler 最大步长，默认 0.005 s。
- `--output`：输出目录，默认本目录下 `results/`；图、VTK、JSON 和 CSV 均不提交 Git。

## 几何、空区域和离散

这是可复现的合成热学基准，物性与冷却条件是明确的建模假设，未经器件实测标定。封装占地 24 × 24 mm；两个芯粒各 7.5 × 12 mm，位置为 `(3,3)–(10.5,15)` 和 `(13.5,3)–(21,15)` mm。

| 所属区域 | 自下而上材料 | z 范围 mm | 默认 z 单元数 | 横向占据 | k_xy / k_z，W/(m K) | ρc，MJ/(m³ K) |
|---|---|---:|---:|---|---:|---:|
| ROM | 等效有机 substrate | 0–0.60 | 6 | 全占地 | 12 / 0.6 | 1.70 |
| ROM | 等效 bump/underfill | 0.60–0.65 | 1 | 仅两个芯片占地 | 5 / 3 | 1.80 |
| ROM | silicon core | 0.65–0.95 | 4 | 仅两个芯片占地 | 100 / 100 | 1.63 |
| 外界 FOM | TIM | 0.95–1.00 | 1 | 仅两个芯片占地 | 4 / 4 | 1.80 |
| 外界 FOM | 铜 lid | 1.00–1.80 | 8 | 全占地 | 390 / 390 | 3.45 |

**芯片之外、substrate 与 lid 之间的空间为空腔。** 不填充 mold 或低导热“空气材料”，不装配单元、导热边、热容或热源，不保留悬空节点，也不导出任何空腔六面体。空腔暴露表面按绝热处理，未引入腔内气体传热和辐射。lid 跨过空腔的部分仍是真实铜实体，保留横向扩散与热容。

单元采用节点集中热容、沿三轴单元边导热的守恒离散。先筛除不占据的单元，再压紧顶点编号；`points`、`cells`、温度和所有代数量使用同一套紧凑编号。`node_grid` 用 -1 表示不存在的节点，观察点落在空腔时明确拒绝采样。绘图直接切割占据单元，不再将紧凑温度数组强行 reshape 回实心长方体。

| 最大 dx / dy / dz，mm | ROM 区域原始自由度 | 完整 silicon/TIM 接口节点 | TIM/lid FOM 自由度 | 完整组装 FOM 自由度 |
|---|---:|---:|---:|---:|
| 1.5 / 1.5 / 0.1 | 2,563 | 108 | 2,709 | 5,164 |
| 0.75 / 0.75 / 0.1（默认） | 9,493 | 374 | 10,175 | 19,294 |
| 0.5 / 0.5 / 0.1 | 20,807 | 800 | 22,409 | 42,416 |
| 0.4 / 0.7 / 0.08 | 27,018 | 798 | 27,968 | 54,188 |

默认横向步长最大 0.75 mm，substrate 厚向 100 μm、连接层 50 μm、silicon 为底部有源区 37.5 μm 加其上 3 × 87.5 μm、TIM 50 μm、lid 100 μm。有源区厚度固定为 37.5 μm，若 dz 上限更小则在其中继续细分，改变网格不会改变物理发热区域。默认 ROM 区域有 7,744 个占据单元，外界有 8,512 个；仅 ROM 区域便排除了包围盒中的 3,520 个空单元。

`isotropic` 表示未额外削弱横向导热，基板自身仍各向异性。`weak_lateral` 将 silicon 和连接层的横向导热乘 0.02，`extreme` 对全部 ROM 层如此处理；这两者是人工降阶压力测试，不解释为真实硅物性。外界 TIM/lid 的物性保持不变。

## 完整模型装配与边界

部件接口从旧版的 substrate 底面改为**两个芯片的完整 silicon 顶面**，z=0.95 mm。静态提升和动态富集只对 ROM 区域进行，TIM/lid 不进入提取基。substrate 底面固定以 250 W/(m² K) 接到 25°C 参考环境；这是整个研究中的固定板侧边界，未另加 PCB 功能块。

TIM 底面与 silicon 顶面采用共形网格，所有接口节点逐一共享。令 P 将全系统坐标取回外界局部坐标，则

$$
K_{assembly}=\operatorname{diag}(K_{component},0)+P^T K_{external} P,
\qquad
C_{assembly}=\operatorname{diag}(C_{component},0)+P^T C_{external} P.
$$

ROM 装配只将上式中的部件矩阵替换为降阶矩阵，接口仍是物理温度。TIM 自身的有限厚度导热已经给出热阻，不再叠加人为接触系数。共享节点汇总两侧各自的热容与导热贡献，没有重复接口状态。完整 FOM 的自由度因此是两区域自由度之和减去接口节点数。

**没有一维串联终端。** 两芯片可经 substrate 与 lid 横向交换热量，外界的全部内部温度与部件同时求解。FOM+外界与 ROM+外界各自联立求解，接口轨迹不由另一套解给定。

lid 顶面左右半区名义 HTC 为 8000、12000 W/(m² K)，表示盖板至冷却介质的等效边界，不是静止空气对流。另将这两个 HTC 同时减半后复用同一个 ROM，完成第二组完整三维外接仿真；旧版固定终端与 PCB 示例不再运行。

## 简单功率工况

| 开始时间 s | A，W | B，W | 总功率 W |
|---:|---:|---:|---:|
| 0 | 10 | 10 | 20 |
| 0.5 | 65 | 55 | 120 |
| 3.0 | 0 | 0 | 0 |

6 s 结束，保持低载→高载→关断两次跳变。每个输入均匀作用于对应芯片底部固定厚度 37.5 μm 的 silicon 有源区（可含多个网格单元），以节点分配保证总功率精确为指定值，没有另设有源材料层。稳态另求持续 120 W。

后向 Euler 精确落在跳变与快照时刻，每段由 20 μs 步长增长到最大步长。观察点为 A/B 结平面中心、A 下方 substrate 内部，以及 A 上方 lid 顶面，全部使用实际单元的三线性采样。报告的 ROM 误差相对同一时间网格上的 FOM，不能代替时间步或空间网格收敛。

## 图表与 ParaView

- `package.png`：完整模型剖面及芯片布局，白色空腔不含单元。
- `transient_probes.png`：功率及 FOM/ROM 温度轨迹，共两个面板。
- `external_connections.png`：名义冷却、半 HTC 两种完整外接条件的轨迹对比。
- `transient_fields.png`、`steady_fields.png`：结平面和**完整封装纵剖面**的 FOM、ROM+外界 FOM 及误差，包含 TIM 和 lid，并保留空腔。
- `reduced_cooling/`：半 HTC 工况下相同的图表和 VTK。
- `vtk/assembly.pvd`：完整组件+TIM/lid 的瞬态序列；打开它即可查看两区域，不要只打开部件的 `transient.pvd`。
- `vtk/steady_assembly.vtm`：完整稳态装配，引用 `steady.vtu` 和 `cooler_steady.vtu`。
- CSV 记录功率与观察点轨迹；JSON 记录网格规模、误差、计时及守恒核验。

坐标为 m，温度为 °C，温差为 K。需要保留整个 VTK 文件目录与相对路径。VTU 仅导出占据单元及其真实顶点；`connectivity` 使用单分量 Int64，VTK_HEXAHEDRON 顶点遵循标准顺序。外界文件的 `layer_id=0/1` 分别为 TIM/lid，部件为上表的三层。`interface_node` 标识共享接口在各块中的局部节点。瞬态跳变时刻场连续，源密度记录刚结束区间的功率。

## 降阶方法

离散方程为

$$
C\dot x+Kx=Bu,
\qquad
x=\begin{bmatrix}x_\Gamma\\x_I\end{bmatrix}.
$$

这里 $C$ 为正对角矩阵。记 $A=K_{II}$、$F=-K_{I\Gamma}$，静态提升为

$$
\Psi=A^{-1}F,
\qquad
T=\begin{bmatrix}I&0\\\Psi&W\end{bmatrix},
\qquad
C_r=T^TCT,\quad K_r=T^TKT.
$$

接口坐标始终是原来的节点温度。静态 Schur 补由 $\Psi$ 保留，$W$ 只描述内部相对静态提升的动态变化。

包含接口热容的精确接口导纳为

$$
\begin{aligned}
Y(s)
&=K_{\Gamma\Gamma}+sC_{\Gamma\Gamma}-F^T(A+sC_{II})^{-1}F\\
&=S+s\bigl(C_{\Gamma\Gamma}+\Psi^TC_{II}\Psi\bigr)
-s^2\Psi^TC_{II}(A+sC_{II})^{-1}C_{II}\Psi,\\
S&=K_{\Gamma\Gamma}-F^T\Psi.
\end{aligned}
$$

因此以接口图案 $U$ 和内部功率分布 $B_I$ 形成右端

$$
R=\begin{bmatrix}C_{II}\Psi U&B_I\end{bmatrix},
$$

再把 $(A+\sigma C_{II})^{-1}R$ 加入内部空间。先加入 $\sigma=0$ 的块，随后使用正实的 MPMM 椭圆点。零频块同时包含 $A^{-1}B_I$，使指定功率源的稳态响应也得到保留。

选点覆盖估计的整个内部广义谱区间。谱端点用容量归一化后的对称算子估计；下端的移位求逆复用静态提升的分解。椭圆点数由

$$
4\exp\!\left[-\frac{m\pi^2}{\log(4\lambda_{\max}/\lambda_{\min})}\right]\le\varepsilon
$$

确定，再按椭圆 `dn` 分布生成正实频移。谱端点是数值估计，这个公式用于规划采样成本。

每个响应块先按容量缩放、逐列归一化，再对已有动态空间做两遍正交化，最后用带列主元的 QR 判断数值秩。**不要对静态提升列做内部容量正交化**：那会消掉接口与内部动态之间必要的容量耦合。

## 计算安排

- 静态提升按小批右端求解，避免同时保存稠密的接口耦合矩阵。其分解供谱下端估计和零频富集复用。
- 稀疏分解按对称图结构排序以减少填充；每个频移求解结束便释放分解，不让相邻两个大分解同时驻留。
- 动态基在预分配空间中增长，不反复拼接、复制已有方向。
- 投影直接计算接口块、交叉块和内部块，不构造完整的 $T$，也不复制一份完整的容量加权提升。
- 默认参考解直接求需要的载荷空间，而不是先求全部接口列。令 $D$ 为接口控制面积，则

  $$
  Q=\begin{bmatrix}DU&0\\0&B_I\end{bmatrix},
  \qquad H_Q(s)=Q^T(K+sC)^{-1}Q.
  $$

  这与先计算全部接口传递矩阵再压到相同图案空间等价。默认两块接口各有 9 个图案，每个采样点共 20 个右端。每个点只分解一次，并在全部嵌套降阶空间之间共用参考结果。
- 可选的全部接口考核使用 $\sqrt D$ 缩放的节点热流输入。参考结果逐点处理，求解右端分批展开，不保存全场响应集合。

这些安排仍然需要显式保存稠密的 $\Psi\in\mathbb R^{n_I\times n_\Gamma}$，以及完整接口对应的稠密降阶块；稀疏 LU 也仍有填充成本。这个原型没有解决任意大规模接口的提取问题。

## 如何解释结果

静态提升残差、零频矩、容量正交性、降阶正定性、稳态响应、嵌套空间收敛和能量平衡用于考核算法自洽性。实 Laplace 点误差按各参考 DC 通道块的 Frobenius 范数归一化；这些检查不是连续频域或 H∞ 认证。保留完整接口并不意味着任意接口图案具有相同精度。具体 TIM/lid 外接后的轨迹对比也不能证明所有外部系统上的统一精度。

`assembly.py` 装配仅含占据单元的部件，`cms.py` 提取并投影，`external.py` 独立装配和连接 TIM/lid，`transient.py` 联立时域求解，`visualization.py` 输出图表及 VTK，`run_study.py` 组织采样与验证。

## 核验记录（2026-10-27）

旧模型先执行 `--nx 4` 完成基线检查。改造后的正式命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py --dx-mm 1.5 --dy-mm 1.5 --dz-mm 0.1 --materials isotropic weak_lateral --full-interface --output playground/cms_prototype/results/final_coarse
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py --dx-mm 0.75 --dy-mm 0.75 --dz-mm 0.1 --output playground/cms_prototype/results/final_medium
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py --dx-mm 1.5 --dy-mm 1.1 --dz-mm 0.08 --output playground/cms_prototype/results/final_asymmetric
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/cms_prototype/run_study.py --dx-mm 1.5 --dy-mm 1.5 --dz-mm 0.1 --time-step 0.0025 --output playground/cms_prototype/results/final_halfstep
ruff check playground/cms_prototype
ruff format --check playground/cms_prototype
```

| 最大 dx/dy/dz，mm / 材料 | 外界冷却 | 瞬态快照峰值 °C | 最大观察点差 K | 最大部件快照场差 K |
|---|---|---:|---:|---:|
| 1.5/1.5/0.1 / 默认材料 | 名义 | 73.505 | 0.1253 | 0.00940 |
| 1.5/1.5/0.1 / 默认材料 | 半 HTC | 95.683 | 0.1283 | 0.01278 |
| 0.75/0.75/0.1 / 默认材料（默认网格） | 名义 | 73.856 | 0.1072 | 0.00810 |
| 0.75/0.75/0.1 / 默认材料（默认网格） | 半 HTC | 95.951 | 0.1099 | 0.01204 |
| 1.5/1.5/0.1 / weak_lateral | 名义 | 74.514 | 0.1890 | 0.01437 |
| 1.5/1.5/0.1 / weak_lateral | 半 HTC | 96.935 | 0.1936 | 0.02008 |
| 1.5/1.1/0.08 / 默认材料 | 名义 | 73.568 | 0.0919 | 0.00865 |

默认网格部件由 9,493 个原始自由度缩为 620 个（374 个原始接口温度 + 246 个内部动态方向）。加上未降阶的 TIM/lid，完整系统从 19,294 降至 10,421 阶。持续 120 W 的名义工况稳态峰值为 74.189°C。温度来自上述物性和冷却假设，不作为实测器件预测。

上述工况每步离散能量平衡相对残差均低于 5e-12，部件稳态场差低于 6e-12 K。默认网格相对 1.5 mm 网格的瞬态快照峰值变化约 0.35 K，但尚未完成系统的三方向网格收敛。粗网格最大步长 5→2.5 ms 后，FOM 观察点在原时间点的差约 0.163 K（名义）和 0.155 K（半 HTC），因此不能把 FOM/ROM 同网格误差解释为总预测误差。

独立把全部五层一次性装配，再与两区域共享接口装配比较：在三组尺寸上，导热矩阵相对差低于 4e-17，热容最大绝对差低于 6e-20 J/K，热源完全一致。另核对各轴实际单元尺寸不超过所给上限、全部自由度均被占据单元引用、热容为正、空腔观察点被拒绝。

使用 VTK 官方 XML 读取器核对 VTU/VTM、六面体体积为正、空腔内无单元、导出的单元尺寸符合三个上限。图和 VTK 包含完整 TIM/lid，而不是只有部件。粗网格全接口实 Laplace 采样的接口误差约 0.407%（默认材料）、0.656%（弱横向导热），不是连续频域或 H∞ 认证。

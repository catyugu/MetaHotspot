# MetaHotspot XML 配置文件格式

## 概述

MetaHotspot 使用 XML 格式作为主要配置文件，格式兼容 ThermalSim 软件导出的 XML 格式。

**命名空间**: `xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models"`

---

## 根元素: `<Structure>`

### 全局配置属性

| 元素                                            | 类型       | 说明                                            |
| ----------------------------------------------- | ---------- | ----------------------------------------------- |
| `AmbientTemperature`                            | double     | 环境温度 (K)                                    |
| `InitialTemperature`                            | double     | 初始温度 (K)                                    |
| `LengthUnit`                                    | string     | 长度单位，默认 "Mm" (毫米)                      |
| `Dimension`                                     | string     | 维度，"Dimension3D"                             |
| `StudyType`                                     | string     | 研究类型，"Steady" (稳态) 或 "Transient" (瞬态) |
| `TransientStudyDuration`                        | double     | 瞬态分析总时长                                  |
| `TransientStudyTimeStep`                        | double     | 瞬态分析时间步长                                |
| `TransientTimeUnit`                             | string     | 时间单位，默认 "S" (秒)                         |
| `EnabledPhysics`                                | string     | 启用的物理模型，"HeatTransfer"                  |
| `AlphaDegree`, `BetaDegree`, `GammaDegree`      | double     | 旋转角度参数                                    |
| `TopLayerIndex`                                 | int        | 顶层索引                                        |
| `DieLayerNum`, `DieLayerSizeX`, `DieLayerSizeY` | int/double | Die 层配置                                      |
| `DieMaterialName`, `TIMMaterialName`            | string     | Die/TIM 材料名称引用                            |
| `SoftwareMode`                                  | string     | 软件模式 ("PTA" 等)                             |
| `UseBlockBuildMode`                             | bool       | 使用块构建模式                                  |
| `MechanicalDampingFactor`                       | double     | 机械阻尼因子                                    |
| `DielectricLossFactor`                          | double     | 介电损耗因子                                    |
| `PreviewRowNum`                                 | int        | 预览行数                                        |
| `BasicGeometries`                               | array      | 基础几何体                                      |
| `TopThermalBoundary`                            | typed      | 顶部热边界 (可为 nil)                           |
| `BottomThermalBoundary`                         | typed      | 底部热边界 (可为 nil)                           |
| `OtherThermalBondary`                           | typed      | 其他热边界 (SecondTypeThermalBoundary)          |

### SAW (声表面波) 相关参数

| 元素                    | 类型   | 说明                         |
| ----------------------- | ------ | ---------------------------- |
| `SawFrequency`          | double | SAW 频率                     |
| `SawFrequencyStart`     | double | SAW 频率起始值               |
| `SawFrequencyStop`      | double | SAW 频率结束值               |
| `SawFrequencyStep`      | double | SAW 频率步长                 |
| `SawFrequencyUnit`      | string | SAW 频率单位 ("GHz")         |
| `SawFrequencySolveMode` | string | SAW 求解模式 ("SinglePoint") |

### 拓扑与几何相关

| 元素                                 | 类型  | 说明             |
| ------------------------------------ | ----- | ---------------- |
| `TopoGeometries`                     | array | 拓扑几何体       |
| `TopoGeometryMaxNum`                 | int   | 拓扑几何最大数量 |
| `TopoLayers`                         | array | 拓扑层           |
| `ObservePoints2D`, `ObservePoints3D` | array | 观测点           |
| `Functions`, `Variables`             | array | 函数与变量定义   |
| `ThermalMaskResults`                 | array | 热掩模结果       |
| `LayerPBResults`                     | array | 层 PB 结果       |

---

## `<Layers>` 层配置

每个 `<Layer>` 代表一个水平层，叠放形成 2.5D 模型。

### Layer 属性

| 元素                                     | 类型   | 说明                        |
| ---------------------------------------- | ------ | --------------------------- |
| `Name`                                   | string | 层名称，如 "层 1"、"层 2"   |
| `ThicknessExpression`                    | double | 层厚度                      |
| `MeshSizeXExpression`                    | double | X 方向网格尺寸 (0 表示自动) |
| `MeshSizeYExpression`                    | double | Y 方向网格尺寸 (0 表示自动) |
| `MeshSizeZExpression`                    | double | Z 方向网格尺寸 (0 表示自动) |
| `IsTopLayer`                             | bool   | 是否为顶层                  |
| `IsBottomPackaging`                      | bool   | 是否为底部封装层            |
| `IsDie`                                  | bool   | 是否为 Die 层               |
| `IsDoule`                                | bool   | 双层标志                    |
| `IsIDTSawModel`                          | bool   | IDT SAW 模型标志            |
| `IsNotIDTSawModel`                       | bool   | 非 IDT SAW 模型标志         |
| `IsSingle`                               | bool   | 单层标志                    |
| `IsTIM`                                  | bool   | 是否为 TIM (热界面材料) 层  |
| `IsSubstrate`                            | bool   | 是否为基板层                |
| `IsBasic`                                | bool   | 基础层标志                  |
| `CreatedByTopLayer`                      | bool   | 由顶层创建标志              |
| `SawBoundaryOption`                      | string | SAW 边界选项 ("None" 等)    |
| `PeriodWidth`                            | double | SAW 周期宽度                |
| `MetalConversionRate`                    | double | 金属转化率                  |
| `XOffsetExpression`, `YOffsetExpression` | double | X/Y 方向偏移                |

### `<Blocks>` 块几何

每个 `<Block>` 定义一个矩形块区域。

| 元素                                                          | 类型   | 说明                              |
| ------------------------------------------------------------- | ------ | --------------------------------- |
| `Name`                                                        | string | 块名称，如 "块 1"                 |
| `MaterialName`                                                | string | 材料名称 (如 "copper", "silicon") |
| `ThicknessExpression`                                         | double | 块厚度                            |
| `XOffsetExpression`, `YOffsetExpression`, `ZOffsetExpression` | double | 位置偏移                          |
| `MeshSizeXExpression`                                         | double | 网格尺寸                          |
| `Color`                                                       | RGBA   | 显示颜色                          |
| `IsVisible`                                                   | bool   | 是否可见                          |
| `IsChargeConservation`                                        | bool   | 电荷守恒标志                      |
| `NotChargeConservation`                                       | bool   | 非电荷守恒标志                    |
| `IsChargeConservationPiezoelectricity`                        | bool   | 压电电荷守恒标志                  |
| `TiReyuan`                                                    | double | 体热源强度 (W/m³)，0 表示无热源   |
| `IsPML`                                                       | bool   | PML (完美匹配层) 标志             |
| `IsPiezoelectricMaterial`                                     | bool   | 压电材料标志                      |
| `IsLinearElasticMaterial`                                     | bool   | 线弹性材料标志                    |
| `IsNormalMaterial`                                            | bool   | 正常材料标志                      |
| `IsElectrode`                                                 | bool   | 电极标志                          |
| `IsTerminal`                                                  | bool   | 端子标志                          |
| `NetDonorConcentration`                                       | double | 净施主浓度 (掺杂)                 |
| `TerminalVoltage`                                             | double | 端子电压                          |
| `PMLType`, `SelectedPMLType`                                  | string | PML 类型选择                      |
| `CanCreateArray`                                              | bool   | 可创建阵列标志                    |
| `HasBeenChanged`                                              | bool   | 已更改标志                        |
| `InitialName`                                                 | string | 初始块名称                        |
| `NotOccupiedByArray`                                          | bool   | 阵列未占用标志                    |

### `<AllRects>` 矩形操作列表

每个 `<Rect>` 定义一个矩形区域操作 (加或减)。

| 元素                                         | 类型   | 说明                                |
| -------------------------------------------- | ------ | ----------------------------------- |
| `Name`                                       | string | 操作名称，如 "加操作 1"、"减操作 1" |
| `Add_sub`                                    | bool   | true = 加操作，false = 减操作       |
| `XExpression`, `YExpression`                 | double | 矩形左下角坐标                      |
| `WidthExpression`, `HeightExpression`        | double | 矩形宽高                            |
| `XSizeExpression`, `YSizeExpression`         | double | 矩形总尺寸                          |
| `XIntervalExpression`, `YIntervalExpression` | double | 阵列间隔                            |
| `ArrayDisguise`, `CreatedByArray`            | bool   | 阵列相关标志                        |

---

## `<Boundaries>` 边界条件

每个 `<Boundary>` 定义一个边界条件。

| 元素               | 类型        | 说明                                                         |
| ------------------ | ----------- | ------------------------------------------------------------ |
| `Name`             | string      | 边界名称，如 "边界 1"                                        |
| `BoundaryCategory` | string      | 边界类别，"Electrical" 等                                    |
| `FaceKeys`         | string 数组 | 面标识符，格式: Face\|Direction\|LayerIndex\|X1,Y1,X2,Y2;... |

### FaceKeys 格式说明

格式: `Face|Direction|LayerIndex|X1_min,X1_max,Y1_min,Y1_max;X2_min,X2_max,Y2_min,Y2_max;...`

- `Face`: Z (顶/底), X, Y (所在坐标平面)
- `Direction`: E (负向), P (正向) (沿着该轴的方向)
- `LayerIndex`: 层索引 (0 = 顶层)
- 其后为该面上的矩形区域列表，每个区域用 4 个坐标值定义

**示例**: `Z|E|0|0,50,50,100;50,100,0,50;50,100,50,100`

含义: Z 面 (顶面)，负向，第 0 层，包含三个矩形区域:

- 区域1: X∈[0,50], Y∈[50,100]
- 区域2: X∈[50,100], Y∈[0,50]
- 区域3: X∈[50,100], Y∈[50,100]

### `<ThermalBoundary>` 热边界类型

#### 第一类边界 (恒温边界) - `FirstTypeThermalBoundary`

```xml
<ThermalBoundary i:type="a:FirstTypeThermalBoundary">
  <a:Temperature>500</a:Temperature>
</ThermalBoundary>
```

| 参数          | 说明         |
| ------------- | ------------ |
| `Temperature` | 边界温度 (K) |

#### 第二类边界 (热流边界) - `SecondTypeThermalBoundary`

```xml
<ThermalBoundary i:type="a:SecondTypeThermalBoundary">
  <a:HeatFlux>100</a:HeatFlux>
</ThermalBoundary>
```

| 参数       | 说明                        |
| ---------- | --------------------------- |
| `HeatFlux` | 热流密度 (W/m²)，0 表示绝热 |

#### 第三类边界 (对流边界) - `ThirdTypeThermalBoundary`

```xml
<ThermalBoundary i:type="a:ThirdTypeThermalBoundary">
  <a:HeatTransferCoefficient>10</a:HeatTransferCoefficient>
  <a:Temperature>300</a:Temperature>
</ThermalBoundary>
```

| 参数                      | 说明                    |
| ------------------------- | ----------------------- |
| `HeatTransferCoefficient` | 对流换热系数 (W/(m²·K)) |
| `Temperature`             | 环境温度 (K)            |

---

## `<Materials>` 材料库

格式为键值对集合:

```xml
<Materials xmlns:a="http://schemas.microsoft.com/2003/10/Serialization/Arrays">
  <a:KeyValueOfstringMaterialGyu7GfTz>
    <a:Key>copper</a:Key>
    <a:Value>
      <DaoreXishu>400</DaoreXishu>  <!-- 热导率 k (W/(m·K)) -->
      <Midu>8960</Midu>              <!-- 密度 (kg/m³) -->
      <BiRerong i:nil="true"/>       <!-- 比热容 (J/(kg·K)) -->
    </a:Value>
  </a:KeyValueOfstringMaterialGyu7GfTz>
</Materials>
```

### 标准材料属性

| 属性                   | 中文名      | 单位     | 说明         |
| ---------------------- | ----------- | -------- | ------------ |
| `DaoreXishu`           | 热导率 k    | W/(m·K)  | 导热系数     |
| `Midu`                 | 密度        | kg/m³    | 材料密度     |
| `BiRerong`             | 比热容      | J/(kg·K) | 恒压比热容   |
| `Eg0`                  | 能量Gap     | eV       | 电子相关     |
| `CouplingMatrixForSaw` | SAW耦合矩阵 | -        | 声表面波相关 |
| `Epsilon`              | 介电常数    | -        | 电学相关     |
| `MuN`, `MuP`           | 迁移率      | -        | 载流子相关   |

### 内置标准材料

| 材料名   | k (W/(m·K)) | cp (J/(kg·K)) | density (kg/m³) |
| -------- | ----------- | ------------- | --------------- |
| silicon  | 130         | 1.63e6        | 2330            |
| copper   | 400         | 3.44e6        | 8960            |
| aluminum | 237         | 2.42e6        | 2700            |
| tim      | 4.0         | 4.0e6         | 1000            |
| water    | 0.6069      | 4.17e6        | 1000            |

---

## `<Results>` 结果数据

`<Results>` 是一个数组，包含多个结果对象，每个为 `Result3D` 类型。

### Result3D 结构

| 元素     | 类型   | 说明                 |
| -------- | ------ | -------------------- |
| `Name`   | string | 结果名称 (如 "温度") |
| `Mesh`   | object | 网格坐标 (见下)      |
| `Values` | object | 场值数据 (见下)      |

### `<Mesh>` 网格坐标

```xml
<Mesh>
  <XArray>
    <a:double>0</a:double>
    <a:double>10</a:double>
    ...
  </XArray>
  <YArray>
    <a:double>0</a:double>
    ...
  </YArray>
  <ZArray>
    <a:double>0</a:double>
    <a:double>2</a:double>
    ...
  </ZArray>
</Mesh>
```

- `XArray`, `YArray`, `ZArray`: 网格点坐标数组，用于定义计算域的离散化
- 网格线索引用于标识面的位置

### `<Values>` 场值结果

```xml
<Values>
  <Data>
    <a:double>NaN</a:double>
    ...
  </Data>
  <SizeX>11</SizeX>
  <SizeY>11</SizeY>
  <SizeZ>10</SizeZ>
</Values>
```

- `Data`: 温度场或其他场量的值，按 X,Y,Z 顺序展平
- `SizeX/Y/Z`: 三个方向的网格点数

---

## 配置示例

### 稳态分析配置

```xml
<StudyType>Steady</StudyType>
<AmbientTemperature>300</AmbientTemperature>
<InitialTemperature>300</InitialTemperature>
```

### 瞬态分析配置

```xml
<StudyType>Transient</StudyType>
<TransientStudyDuration>100</TransientStudyDuration>
<TransientStudyTimeStep>1</TransientStudyTimeStep>
<TransientTimeUnit>S</TransientTimeUnit>
```

### 带热源的块

```xml
<Block>
  <Name>块 1</Name>
  <MaterialName>copper</MaterialName>
  <TiReyuan>1e8</TiReyuan>  <!-- 体热源 W/m³ -->
  ...
</Block>
```

### 恒温边界

```xml
<Boundary>
  <Name>边界 1</Name>
  <FaceKeys>...</FaceKeys>
  <ThermalBoundary i:type="a:FirstTypeThermalBoundary">
    <a:Temperature>500</a:Temperature>
  </ThermalBoundary>
</Boundary>
```

### 对流边界

```xml
<Boundary>
  <Name>对流边界</Name>
  <ThermalBoundary i:type="a:ThirdTypeThermalBoundary">
    <a:HeatTransferCoefficient>10</a:HeatTransferCoefficient>
    <a:Temperature>300</a:Temperature>
  </ThermalBoundary>
</Boundary>
```

---

## 解析要点

1. **命名空间处理**: XML 使用 DataContract 命名空间，解析时需注意 `i:type` 属性
2. **坐标系统**: 解析时长度单位需统一转换为米用于准确计算
3. **层叠顺序**: XML 中层的顺序即 Z 轴堆叠顺序 (从上到下)
4. **FaceKeys 解析**: 需解析面标识符字符串以确定边界所在面
5. **材料属性**: 优先使用层/块定义的材料，否则继承层级材料的默认值

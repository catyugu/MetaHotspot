# MetaHotspot开发计划

## 一、环境准备

## 参考项目：Hotspot

* 提供了CTM模型的基本处理手法
* 提供了几个经典的集成电路封装算例`example1`~`example4`
  * example1：瞬态+稳态固体传热
  * example2：瞬态+稳态固体传热（3D异构）
  * example3：瞬态+稳态固体传热（3D异构）
  * example4：流固耦合传热（3D异构）

## 流体解算/流固耦合传热的相关算法参考：3D-ICE

* 提供了对微流道散热的详细处理手法。
* 可参考其中部分算法和经验公式等。

## 运行环境

```bash
conda activate numerical
```

这个conda环境中已经安装好了基础的编译器和常用库，如果有其他需要，可在里面安装。

## 一、 技术栈选型与链路设计

### 1. 兼容性前处理模块 (Adapter & Mesher)
**目标**：将 HotSpot 的私有格式`.config`/`.materials`/`flp`转化为标准几何网格与统一配置文件。
* **开发语言**：`Python 3.10+`（非常适合文本解析和调用外部网格库）。
* **网格生成工具**：**`PyVista` (Python API)**或者**gmsh**。
* **网格形式**：纯笛卡尔六面体网格（切割允许不均匀），须带有每个域和边界的id信息。

### 2. 核心求解器模块 (The Core FVM Solver)
**目标**：读取标准网格和配置以及`ptrace`文件，执行纯粹的矩阵组装与求解。
* **MVP 阶段**：`Python` + `SciPy` (`scipy.sparse`)。利用 Python 快速验证 FVM 矩阵（热导矩阵 G 和热容矩阵 C）的组装逻辑和边界条件，直接进行求解。
* **长期工业/学术阶段**：`C++17/20`。
  * **矩阵运算库**：**`Eigen`**（处理稀疏矩阵的构建和基础求解）或 **`PETSc`**（如果未来要上多核/集群并行求解和超大规模网格）。
  * **网格读取/写入库**：**`VTK C++ 源码库`** 或轻量级的 **`pugixml`** (如果采用基于 XML 的 VTU 格式)。

### 3. 后处理与展示模块 (Post-processor & GUI)
**目标**：读取求解器输出的带温度场数据的标准网格文件并进行渲染。
* **快速验证/学术作图**：**`PyVista`** (Python)。
* **长期工业前端开发**：**`Vue 3` / `React`** + **`vtk.js`**。在浏览器中直接渲染三维温度场，配合 **`Tauri`** 打包为轻量级跨平台桌面应用。

## 二、 接口与文件标准定义

**文件一：格式化配置文件 `solver_config.toml`**
* 放弃所有散乱的参数，向求解器传递纯粹的物理信息和求解控制信息。
* 示例：[example.toml](example.toml)

**文件二：标准 FVM 网格文件 `domain.vtu` (VTK Unstructured Grid)**
* 这是一个二进制/XML混合文件，由前处理模块生成。
* **核心内容**：包含所有网格节点 (Points) 的坐标，单元 (Cells) 的连接拓扑。
* **数据域 (Cell Data)**：必须包含每个 Cell 的 `Material_ID`（映射到 toml 中的材料），以及 `Power_Density_W_m3`（体积热源）。求解器直接读取这个数据作为方程的右端项（RHS）。

**文件三：格式化结果文件 `result_temp.vtu`**
* 求解器运行结束后，直接在输入的 `domain.vtu` 基础上，增加一个新的 Cell Data 字段 `Temperature_K`，并另存为该文件。前端拿到这个文件即可无脑渲染。

# MetaHotspot 进展报告（2026-07-06 ~ 2026-07-13）

## 一、功能开发和修订

- 完成了降阶宏模型的训练脚本和集成路径，具体理论陈述附后，此处作一概要。
    - 根本做法是学习边界处 DtN (Dirichlet to Neumann) 映射，得到边界面片局部的方程。
    - 线性模型：直接求解！（类似于做 Schur 补）
    - 非线性模型：使用感知机（通常单层 $\mathbf{\beta} = \sigma(\mathbf{A\alpha} + \mathbf{b})$ 就够，或者其他具有更好先验结构的神经网络？）通过给若干组经过设计的一些边界输入，学习参数。
    - 原始的参数太大，怎么办？
    - 低秩性很重要，对 $\mathbf{A}$ 可以进行 POD 正交分解，一般选取能量最高的 10~20 模态就很准确了，对 $\mathbf{b}$ 也同样做模态截断。
    - 或者考虑稀疏性，训练时做稀疏性约束！但是这个通常有点难度，因为表面的 DtN 可以想见是广泛联系的，一头的热效应必然会影响到另一头，尤其当导热系数比较高时，追求稀疏性很可能会损伤准确性。
    - 加载时，直接前向推理，通过下面陈述的方法，可以将边界或者其他 Cell 附加到端口上。
- 当前只做了线性稳态的降阶 Block，非线性（还需要巩固外部训练脚本，以及探索训练方法）、瞬态（需要额外加入状态量，可能需要更细致的处理）的待之后做。
- 暂未支持非共形网格投影。

## 二、稳态问题处理的详细方法论

### 外部输入的统一表达

无论宏模型外部连接的是普通的 FVM 单元，还是哪类边界，外部对端口的约束都可以表达为一个等效的戴维南/诺顿热学定理公式：

$$Q_{in, i} = C_i (T_{ref, i} - T_{face, i}) + Q_{ext, i}$$

其中：$T_{face, i}$ 是第 $i$ 个端口面的温度。$Q_{in, i}$ 是从外部流入该端口面的总热流（W）。$C_i, T_{ref, i}, Q_{ext, i}$ 是外部环境提供的三个统一参数：

- 外部是普通 FVM 单元：$C_i = \frac{k_c A_i}{d_c}$, $T_{ref, i} = T_c$ , $Q_{ext, i} = 0$。
- 外部是对流边界条件：$C_i = h A_i$, $T_{ref, i} = T_{\infty}$, $Q_{ext, i} = 0$。
- 外部是热流边界：$C_i = 0$, $Q_{ext, i} = q_{ext} A_i$。
- 外部是恒温边界：使用罚函数法：令 $C_i \to \infty$, $T_{ref, i} = T_{fixed}$, $Q_{ext, i} = 0$。

### 模态分解 + 稳态线性/非线性问题实装

#### 获取宏模型：线性版本

1. 生成精确的 Schur 补矩阵：在离线阶段，提取子域的内部矩阵 $\mathbf{K}_{ii}$、交界矩阵 $\mathbf{K}_{i\Gamma}$ 和边界矩阵 $\mathbf{K}_{\Gamma\Gamma}$。计算稠密的 Schur 补矩阵：$$\mathbf{K}_{schur} = \mathbf{K}_{\Gamma\Gamma} - \mathbf{K}_{\Gamma i} \mathbf{K}_{ii}^{-1} \mathbf{K}_{i\Gamma}$$
2. 求解特征值问题：对这个对称半正定矩阵求解特征值和特征向量：$$\mathbf{K}_{schur} \mathbf{\phi}_i = \lambda_i \mathbf{\phi}_i$$
3. 截断与构造 $\mathbf{\Phi}$：将特征值从小到大排序，提取前 $r$ 个最小特征值对应的特征向量，将它们按列排布，就构成了正交基底 $\mathbf{\Phi} = [\mathbf{\phi}_1, \mathbf{\phi}_2, \dots, \mathbf{\phi}_r]$。
4. 此时，我们可以定义模态等效刚度矩阵 $\mathbf{K}_{r}$ ，它是一个纯对角矩阵，对角线元素就是这 $r$ 个特征值：$$\mathbf{K}_{r} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_r)$$

#### 获取宏模型：非线性版本

1. 设计边界激励信号：使用低阶的 2D 空间傅里叶级数或切比雪夫多项式等叠加作为输入温度场。随机生成 $M$ 组温度分布样本 $\mathbf{T}_{face}^{(1)}, \mathbf{T}_{face}^{(2)}, \dots, \mathbf{T}_{face}^{(M)}$。
2. 运行全量仿真：将这 $M$ 组边界温度作为第一类边界条件，给全量求解器进行非线性迭代求解。待每组仿真收敛后，提取界面上对应的热流响应 $\mathbf{Q}_{face}^{(m)}$。将所有温度和热流向量分别按列拼接，形成快照矩阵 $\mathbf{X}_T$ 和 $\mathbf{X}_Q$（尺寸均为 $N_{face} \times M$）。
3. SVD 提取空间模态对温度快照矩阵 $\mathbf{X}_T$ 执行奇异值分解（SVD）：$\mathbf{X}_T = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$。根据奇异值 $\mathbf{\Sigma}$ ，选取 $r$ 个主导奇异值（保证累计能量占比足够高）。截取左奇异矩阵的这 $r$ 列，得到正交基底 $\mathbf{\Phi} = \mathbf{U}_{[:, 0:r]}$。
4. 投影到模态空间：将物理空间的训练数据转换为训练 AI 的模态数据：

- 输入特征（模态温度）： $\mathbf{\alpha}_{train} = \mathbf{\Phi}^T \mathbf{X}_T$ （尺寸 $r \times M$）
- 输出标签（模态热流）： $\mathbf{\beta}_{train} = \mathbf{\Phi}^T \mathbf{X}_Q$ （尺寸 $r \times M$）

5. 训练单层感知机：现在有低维的数据对 $(\mathbf{\alpha}, \mathbf{\beta})$。尝试训练一个模型预测 $\mathbf{\beta}(\mathbf{\alpha})$。

#### 非线性模型（DtN 映射）的局部线性化修正

我们这里只讨论和邻接的、和宏模型有耦合的单元。其他和宏块不相邻的单元不受影响。

符号约定：

- $\mathbf{\Phi}_f$：面上的纯几何模态基矩阵，尺寸为 $nfaces \times r$，满足正交性 $\mathbf{\Phi}_f^T \mathbf{\Phi}_f = \mathbf{I}_{r \times r}$。
- $\mathbf{C}$：界面面片的半热导对角矩阵，尺寸为 $nfaces \times nfaces$。（本质向量，计算很便宜）
- $\mathbf{\Phi}_c = \mathbf{C} \mathbf{\Phi}_f$：热导加权投影矩阵（整体方程中实际装配的东西），尺寸为 $nadj \times r$
- $\mathbf{C}_\alpha = \mathbf{\Phi}_f^T \mathbf{C} \mathbf{\Phi}_f$：界面半热阻在模态空间的等效投影，尺寸为 $r \times r$ 的对称正定矩阵。

代入界面的热平衡投影方程 $\mathbf{\Phi}_f^T \mathbf{Q}_{in}^{k+1} = \mathbf{\beta}(\mathbf{\alpha^k})$ 中：

$$\mathbf{\Phi}_c^T \mathbf{T}_{adj}^{k+1} - \mathbf{C}_\alpha \mathbf{\alpha}^{k+1} = \mathbf{\beta}(\mathbf{\alpha}^k)$$

当然，这里的 $\mathbf{C_{\alpha}}$ 是根据第 k 步来装配的，因为我们还没有第 k+1 步的温度值。

而另一方面，对于邻接单元有守恒律：

$$\mathbf{K}_{adj} \mathbf{T}^{k+1} -\mathbf{\Phi_c}\mathbf{\alpha}^{k+1}=\mathbf{f}_{adj}^{k+1}$$

将含有 $k+1$ 的未知量移到左边，已知量留在右边，即可得到非线性 DtN 映射下最终的增广系统矩阵结构：

$$\begin{bmatrix} \mathbf{K}_{adj} & -\mathbf{\Phi}_c \\ -\mathbf{\Phi}_c^T & \mathbf{C}_\alpha \end{bmatrix} \begin{bmatrix} \mathbf{T}_{adj}^{k+1} \\ \mathbf{\alpha}^{k+1} \end{bmatrix} = \begin{bmatrix} \mathbf{f}_{adj}^{k+1} \\  - \mathbf{\beta}(\mathbf{\alpha}^k) \end{bmatrix}$$

考虑宏模型自己的边界条件：

$$\begin{bmatrix} \mathbf{K}_{phys} & -\mathbf{\Phi}_c \\ -\mathbf{\Phi}_c^T & \mathbf{C}_\alpha + (\mathbf{\Phi}_f^T \mathbf{C}_{env} \mathbf{\Phi}_f) \end{bmatrix} \begin{bmatrix} \mathbf{T}_{phys}^{k+1} \\ \mathbf{\alpha}^{k+1} \end{bmatrix} = \begin{bmatrix} \mathbf{f}_{phys} \\ \mathbf{\Phi}_f^T (\mathbf{C}_{env} \mathbf{T}_{ref} + \mathbf{Q}_{ext}) - \mathbf{\beta}(\mathbf{\alpha}^k) \end{bmatrix}$$

特别地，宏模型内部为线性的情况下，有显式的模态刚度矩阵 $\mathbf{K}_r \mathbf{\alpha} = \mathbf{\beta}$，因而：

$$\begin{bmatrix} \mathbf{K}_{adj} & -\mathbf{\Phi}_c \\ -\mathbf{\Phi}_c^T & \mathbf{C}_\alpha + \mathbf{K}_r \end{bmatrix} \begin{bmatrix} \mathbf{T}_{adj}^{k+1} \\ \mathbf{\alpha}^{k+1} \end{bmatrix} = \begin{bmatrix} \mathbf{f}_{adj}^{k+1} \\ \mathbf{0} \end{bmatrix}$$

### 瞬态问题的处理思路

瞬态问题需要额外的状态量，外部训练的 DtN 映射变为：

$$[\mathbf{\beta}^n, \mathbf{h}_{trial}^n] = f_{predict}(\mathbf{\alpha}^n, \mathbf{h}^{n-1}, \Delta t)$$

非线性迭代时的系统变为：

$$\begin{bmatrix} \mathbf{K}_{adj} + \frac{1}{\Delta t}\mathbf{M}_{adj} & -\mathbf{\Phi}_c \\ -\mathbf{\Phi}_c^T & \mathbf{C}_\alpha \end{bmatrix} \begin{bmatrix} \mathbf{T}_{adj}^{k+1} \\ \mathbf{\alpha}^{k+1} \end{bmatrix} = \begin{bmatrix} \mathbf{f}_{adj} \\- \mathbf{\beta}(\mathbf{\alpha}^k, \mathbf{h}^{n-1}, \Delta t) \end{bmatrix}$$

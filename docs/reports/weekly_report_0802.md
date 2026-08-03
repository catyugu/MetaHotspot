# weekly_report_0802

## 方法论

### 模型陈述

为了做瞬态问题，需转换观点：不再试图直接构建稠密的稳态端口响应矩阵，而是试图降阶宏域内部自由度。端口 + 宏域内的方程可写成

$$\begin{bmatrix}  
C_{pp}&C_{pi}\\  
C_{ip}&C_{ii}  
\end{bmatrix}  
\begin{bmatrix}  
\dot T_p\\  
\dot T_i  
\end{bmatrix}  
+  
\begin{bmatrix}  
K_{pp}&K_{pi}\\  
K_{ip}&K_{ii}  
\end{bmatrix}  
\begin{bmatrix}  
T_p\\  
T_i  
\end{bmatrix}  
=  
\begin{bmatrix}  
f_p\\  
f_i  
\end{bmatrix}.$$

端口温度始终是全量自由度，只有 $T_i$ 被压缩。我们想要的关系是：

$$\begin{bmatrix}  
T_p\\  
T_i  
\end{bmatrix}  
\approx  
\underbrace{  
\begin{bmatrix}  
I&0\\  
0&W  
\end{bmatrix}}_{\mathcal T}  
\begin{bmatrix}  
T_p\\  
q  
\end{bmatrix},  
\qquad T_i\approx Wq.$$
而且出于性能考虑，$W$ 应该尽可能稀疏。由于我们的宏模型接口不是少量端口，而是大量面片，照搬 MPMM 或者 Stationary DDM-ROM 的处理手法会导致预处理时间过长，以及导致运行时稠密的子块。

### 局部化基底

直接全域降阶难度比较大，我们尝试进行局部化的降阶构造。对宏域，按垂直网格柱构造局部基底。对于第 $j$ 个物理端口，取其正上方同一 $x$-$y$ 位置的宏域单元集合 $\mathcal I_j$。从全局矩阵中截取：

$$k_0^{(j)}  
=  
K_{ii}^{(0)}[\mathcal I_j,\mathcal I_j],$$
$$c_0^{(j)}  
=  
C_{ii}[\mathcal I_j,\mathcal I_j],$$

以及该端口到该柱内部的耦合：

$$b_0^{(j)}  
=  
K_{ip}^{(0)}[\mathcal I_j,j].$$

每个柱分别生成若干候选模态，最后把各柱局部基底组装为稀疏全局矩阵 $W$。由于每个局部模态只在本柱内部非零，$W$ 很稀疏。虽然**基底构造是局部的，但最终降阶投影使用完整全局矩阵**。因此横向导热在

$$W^\mathsf TK W$$

中被重新引入。

### 5. 四类局部候选模态

#### 5.1 静态约束模态

每个柱首先求解

$$k_0^{(j)}\psi_s^{(j)}  
=  
-b_0^{(j)}.$$

也就是

$$\psi_s^{(j)}  
=  
-\left(k_0^{(j)}\right)^{-1}b_0^{(j)}.$$

它表示：当第 $j$ 个端口施加单位温度扰动时，该柱内部的准静态温度形状。

#### 5.2 对流参数灵敏度模态

宏域算子写成

$$K(\theta)=K_0+\theta K_1,  
\qquad  
K_{ip}(\theta)=B_0+\theta B_1,$$

其中

$$\theta=\frac{h}{h_a}.$$

设参数相关的静态约束形状满足

$$K_{ii}(\theta)\psi(\theta)  
+  
K_{ip}(\theta)e_j=0.$$

在 $\theta=0$ 处求导：

$$K_{ii}^{(0)}  
\frac{\partial\psi}{\partial\theta}  
+  
K_{ii}^{(1)}\psi_s  
+  
K_{ip}^{(1)}e_j  
=0.$$

因此一阶灵敏度为

$$\psi_h^{(j)}  
=  
-\left(k_0^{(j)}\right)^{-1}  
\left(  
k_1^{(j)}\psi_s^{(j)}  
+b_1^{(j)}  
\right).$$

它使基底不仅能表示 $h=0$ 时的静态形状，还能表示静态形状随 $h$ 变化的一阶方向。因而对于 $h$ 跨越 500–8000 W/m²K 的情况，不必为每个 $h$ 单独生成基底。

#### 5.3 常数模态

每个柱显式加入

$$\psi_c^{(j)}=\mathbf 1.$$

其主要意义是：

* 精确表示均匀环境温度；
* 保证零温度梯度状态容易被表示；
* 改善初始条件投影；
* 避免静态模态和动态模态不能精确覆盖常数场。

#### 5.4 固定界面动态模态

每个柱还求解广义特征值问题

$$k_0^{(j)}\phi_\ell^{(j)}  
=  
\lambda_\ell^{(j)}  
c_0^{(j)}\phi_\ell^{(j)}.$$

对于热传导系统，$\lambda_\ell$ 的单位是 $1/\mathrm{s}$，对应热衰减速率：

$$T_\ell(t)\sim e^{-\lambda_\ell t}.$$

实践中我们保留：

$$\lambda_\ell\leq \frac{\pi}{\Delta t},$$

且最多保留 `dynamic_modes_per_column` 个，目前默认为 2。低 $\lambda$ 模态是慢热模态，主导当前时间范围内的记忆效应；高 $\lambda$ 模态衰减很快，通常在一个或几个时间步内消失，因此被截断。

### 局部正交化与秩压缩

每个柱的候选矩阵为

$$V_j=  
\begin{bmatrix}  
\psi_s^{(j)}&  
\psi_h^{(j)}&  
\mathbf 1&  
\phi_1^{(j)}&  
\phi_2^{(j)}&
\dots
\end{bmatrix}.$$

代码对它执行带列主元的 QR 分解：

$$V_jP_j=Q_jR_j.$$

然后根据 $R_j$ 的对角元删除数值线性相关的列。最终：

$$W=\operatorname{blockdiag}(Q_1,Q_2,\ldots,Q_{n_p})$$

### 对流边界的处理

Robin 对流边界为

$$-q_n=h(T-T_\infty).$$

离散后，其贡献形式是

$$K_h=hK_{\Gamma},  
\qquad  
f_h=hK_{\Gamma}T_\infty.$$

因此对于固定几何、固定 $T_\infty$，算子关于 $h$ 严格线性。

代码只装配两个全阶孤立宏域：

1. $h=0$： $$A_0=(K_0,C,f_0)$$
2. $h=h_a$，例如设 $h_a=2500$： $$A_a=(K_a,C,f_a)$$

定义

$$\Delta A=A_a-A_0.$$

任意 $h$ 下：

$$A(h)  
=  
A_0+\frac{h}{h_a}\Delta A.$$

具体为

$$K(h)=K_0+\theta\Delta K,$$ $$f(h)=f_0+\theta\Delta f,  
\qquad  
\theta=\frac{h}{h_a}.$$
### 离线投影

全局投影矩阵是

$$\mathcal T=  
\begin{bmatrix}  
I_{n_p}&0\\  
0&W  
\end{bmatrix}.$$

分别投影基准算子和参数增量：

$$K_{r,0}=\mathcal T^\mathsf TK_0\mathcal T,  
\qquad  
\Delta K_r=\mathcal T^\mathsf T\Delta K\mathcal T,$$ $$C_{r,0}=\mathcal T^\mathsf TC_0\mathcal T,  
\qquad  
\Delta C_r=\mathcal T^\mathsf T\Delta C\mathcal T,$$ $$f_{r,0}=\mathcal T^\mathsf Tf_0,  
\qquad  
\Delta f_r=\mathcal T^\mathsf T\Delta f.$$
## 实验结果

```bash
# 由于时间步数较多，瞬态求解较慢，暂不用很大规模的网格，已能看到显著加速
# 较小网格
Grid 12x12x20; exact ports=144; macro internal 1,584->580
Basis column order min/mean/max=4/4.03/5, density=6.944e-03; residual base/anchor=4.479e-03/4.591e-03
h=  500.0: error steady/transient=0.26544/0.04690 K; transient full/ROM=0.222/0.110s, speedup=2.02x
h= 2500.0: error steady/transient=0.05889/0.04691 K; transient full/ROM=0.195/0.107s, speedup=1.83x
h= 8000.0: error steady/transient=0.10512/0.04693 K; transient full/ROM=0.192/0.105s, speedup=1.82x
# 较大网格
Grid 28x28x28; exact ports=784; macro internal 12,544->2,356
Basis column order min/mean/max=3/3.01/4, density=1.276e-03; residual base/anchor=2.569e-02/2.608e-02
h=  500.0: error steady/transient=0.27411/0.09697 K; transient full/ROM=44.774/8.997s, speedup=4.98x PASS
h= 2500.0: error steady/transient=0.22726/0.09935 K; transient full/ROM=44.253/9.119s, speedup=4.85x PASS
h= 8000.0: error steady/transient=0.26220/0.10401 K; transient full/ROM=44.585/9.182s, speedup=4.86x PASS
```

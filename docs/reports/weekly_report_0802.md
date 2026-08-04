# weekly\_report\_0802

## 切向有理 Krylov 降阶

### 概述

主要方法论来自：

> P. Li et al., “Modeling Interconnect Variability Using Efficient Parametric Model Order Reduction,” DATE 2005, pp. 958–963.

离散后的宏模型满足

$$C\dot T+K(h)T=f(h),$$

“有理”的意义在于，它不像经典的 Krylov 方法一样考虑矩阵级数，而是考虑把方程在 Laplace 域写成：

$$A(s,h)T=(K(h)+sC(h))T=f.$$

把矩阵按照端口和内部状态分块：

$$K+sC=  
\begin{bmatrix}  
A_{pp}&A_{pi}\\  
A_{ip}&A_{ii}  
\end{bmatrix}.$$

若将接口温度视为给定输入，则内部温度响应满足

$$A_{ii}(s,h)T_i+  
A_{ip}(s,h)T_p=0.$$

因此

$$T_i  
=  
-A_{ii}(s,h)^{-1}A_{ip}(s,h)T_p.$$

我们试图用一个低阶的算子来逼近：

$$X(s,h)  
=  
-\left(K_{ii}(h)+sC_{ii}(h)\right)^{-1}  
\left(K_{ip}(h)+sC_{ip}(h)\right).$$

### 参数—时间尺度训练域

在两个维度上建立训练点。

#### 对流参数点

在实验边界的最小值和最大值之间按几何级数采样：

$$h_j\in[h_{\min},h_{\max}],$$

同时并加入仿射中心锚点 $h_a$。

#### 热时间尺度点

脚本使用实数正移位，训练点在：

$$s_0 = 0, s_{min}=\frac{1}{duration}, s_{max}=\frac{2}{\Delta t},$$
中间按几何尺度插采样点。

### 每个训练点计算完整端口响应

对于每个 $(s,h)$，构造

$$A=K_{ii}(h)+sC_{ii}(h),$$
$$B=K_{ip}(h)+sC_{ip}(h).$$

做一次稀疏 LU 分解，针对全部接口端口求解：

$$X(s, h)=-A^{-1}B.$$

它的每一列表示一个接口端口施加单位温度时，宏模型内部的响应。这一步与单端口逐次求解相比更高效，因为同一训练点上的所有右端项共享一次稀疏分解。

*注意：如果忽略 h 只看 s，这里实际上的操作高度类似于 FANTASTIC 里用的零阶 MPMM（可匹配零阶和一阶导数）。需要更高阶频域精度的手段也呼之欲出：再考虑高阶展开$AX_0=-B, AX_1  =  -\left(C_{ii}X_0+C_{ip}\right), A X_k = -C_{ii}X_{k-1}, k\ge2.$，则是一种高阶 MPMM 的变体实现了。但是工程上，通常增加零阶点的收益高于添加高阶匹配。*

### 单点的训练

设当前宏模型内部已有的降阶基为 $V$

在某个训练点上，降阶响应坐标满足

$$V^\mathsf TAVZ  
=  
-V^\mathsf TB.$$

因此

$$Z  
=  
-\left(V^\mathsf TAV\right)^{-1}V^\mathsf TB,$$

恢复到全阶内部空间的近似响应为：

$$X_r=VZ.$$

由这个训练点实际的 $X(s, h)$ 可以推算响应误差向量：

$$E=X-X_r.$$

构造误差 Gram 矩阵：

$$G_e=E^\mathsf TAE.$$对于任意单位端口方向 $r$：

$$r^\mathsf TG_er  
=  
(Er)^\mathsf TA(Er)  
=  
\|Er\|_A^2.$$

因此 $G_e$ 的最大特征值对应的主特征向量 $r_1$ 就是当前关于 $A$ 误差最大的接口温度空间方向。所谓“切向”的意义就在于，看到一个新的训练点 $(s, h)$ 时，我们只把 $v_1​=Er_1​$（或者较大的若干特征值对应的方向）加入投影基底矩阵 $V$ 中，而不是全部往里并，从而降低了降阶子空间的维度。

停止指标考虑设定为：

$$\eta(s,h) = \sqrt{\frac{\lambda_{\max}(E^\mathsf TAE)}{\lambda_{\max}(X^\mathsf TAX)}}.$$
这个量在所有训练点上足够小时（例如，小于 0.005），则说明降阶误差基本合格。

### 训练循环

1. 预先建立整个训练集；
2. 在当前基底下评估所有训练点；
3. 贪婪：选择相对误差最大的训练点；
4. 在该点选择切向方向；
5. 增广基底；
6. 重新扫描全部训练点，回到 2，直到所有训练点上降阶误差合格。

### 实验结果

```bash
# 小网格
Grid 22x22x9; exact ports=144; macro states 1,904->345 (5.52x); Krylov residual=4.956e-03
h=500 W/(m^2 K): reference range steady=340.919..412.429 K, transient final=304.144..393.627 K; rise error steady=0.25851 K/0.230%, transient final=0.20039 K/0.214%; full/ROM=0.150/0.200s, speedup=0.75x PASS
h=2500 W/(m^2 K): reference range steady=304.614..371.659 K, transient final=301.238..362.706 K; rise error steady=0.00000 K/0.000%, transient final=0.00142 K/0.002%; full/ROM=0.145/0.195s, speedup=0.74x PASS
h=8000 W/(m^2 K): reference range steady=300.478..362.848 K, transient final=300.389..355.578 K; rise error steady=0.07023 K/0.112%, transient final=0.05992 K/0.108%; full/ROM=0.141/0.197s, speedup=0.72x PASS
# 大网格
Grid 42x42x17; exact ports=576; macro states 15,156->1,208 (12.55x); Krylov residual=4.983e-03
h=500 W/(m^2 K): reference range steady=341.052..401.625 K, transient final=304.484..383.862 K; rise error steady=0.13628 K/0.134%, transient final=0.11002 K/0.131%; full/ROM=14.281/6.041s, speedup=2.36x PASS
h=2500 W/(m^2 K): reference range steady=304.672..360.910 K, transient final=301.328..352.996 K; rise error steady=0.00000 K/0.000%, transient final=0.00100 K/0.002%; full/ROM=13.007/5.692s, speedup=2.28x PASS
h=8000 W/(m^2 K): reference range steady=300.490..352.165 K, transient final=300.398..345.921 K; rise error steady=0.02833 K/0.054%, transient final=0.02438 K/0.053%; full/ROM=14.208/5.804s, speedup=2.45x PASS
```

### 优缺点

优点：

- 一定范围内误差可控且稳定
- 压缩比相当可观，压缩后的矩阵尺寸很小
- 可通过调节训练点、投影模式数来平衡精度/误差
- 有比较完善的理论基础

缺点：

- 预处理时间相当长。往往显著超过通常情况下完整瞬态求解的时间，只有外接多工况反复跑才能见到收益。
- 压缩后的矩阵稠密，虽然看起来尺寸小，实际上加速比也没有那么好看。

## 局部化降阶

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
\begin{bmatrix}  
I&0\\  
0&W  
\end{bmatrix}
\begin{bmatrix}  
T_p\\  
q  
\end{bmatrix},  
\qquad T_i\approx Wq.$$

而且出于性能考虑，$W$ 应该尽可能稀疏。由于我们的宏模型接口不是少量端口，而是大量面片，照搬 MPMM 或者 Stationary DDM-ROM 的处理手法会导致预处理时间过长，以及导致运行时稠密的子块。

### 局部化基底

直接全域降阶的代价通常比较大，而且降阶后可导致很稠密的矩阵，因此我们试进行局部化的降阶构造。对宏域，按垂直网格柱构造局部基底。对于第 $j$ 个物理端口，取其正上方同一 $x$-$y$ 位置的宏域单元集合 $\mathcal I_j$。从全局矩阵中截取：

$$k_0^{(j)}  
=  
K_{ii}^{(0)}[\mathcal I_j,\mathcal I_j],$$
$$c_0^{(j)}  
=  
C_{ii}[\mathcal I_j,\mathcal I_j],$$

及该端口到该柱内部的耦合：

$$b_0^{(j)}  
=  
K_{ip}^{(0)}[\mathcal I_j,j].$$

每个柱分别生成若干候选模态，最后把各柱局部基底组装为稀疏全局矩阵 $W$。由于每个局部模态只在本柱内部非零，$W$ 很稀疏。另外，虽然**基底构造是局部的，但最终降阶投影使用完整全局矩阵**。横向导热被重新引入，所以这不导致精度的丧失。

### 5. 柱局部的候选模态

#### 5.1 静态约束模态

也就是

$$\psi_s^{(j)}  
=  
-\left(k_0^{(j)}\right)^{-1}b_0^{(j)}.$$

它表示：当第 $j$ 个端口施加单位温度扰动时，该柱内部的准静态温度形状。

#### 5.2 对流参数灵敏度模态

记：
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
其中：$K_{ii}^{(1)}=K_{ii}^{\theta=1}-K_{ii}^{\theta=0}$，$K_{ip}^{(1)}$ 同理。
因此一阶灵敏度为

$$\psi_h^{(j)}  
=  
-\left(k_0^{(j)}\right)^{-1}  
\left(  
k_1^{(j)}\psi_s^{(j)}  
+b_1^{(j)}  
\right).$$

它使基底不仅能表示 $h=0$ 时的静态形状，还能表示静态形状随 $h$ 变化的一阶方向。因而不必为每个 $h$ 单独生成基底。

#### 5.3 常数模态

每个柱显式加入

$$\psi_c^{(j)}=\mathbf 1.$$

其主要意义是：

* 表示均匀环境温度；
* 保证零温度梯度状态容易被表示；
* 避免静态模态和动态模态不能精确覆盖常数场。

#### 5.4 固定界面动态模态

服务于瞬态，每个柱求解广义特征值问题

$$k_0^{(j)}\phi_\ell^{(j)}  
=  
\lambda_\ell^{(j)}  
c_0^{(j)}\phi_\ell^{(j)}.$$

实践中关注到时间分辨率 $\Delta t$，可只保留：

$$\lambda_\ell\leq \frac{\pi}{\Delta t},$$

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

对它执行带列主元的 QR 分解：

$$V_jP_j=Q_jR_j.$$

然后根据 $R_j$ 的对角元删除数值线性相关的列。最终：

$$W=\operatorname{blockdiag}(Q_1,Q_2,\ldots,Q_{n_p})$$
然后：
$$K_{r,0}=W ^\mathsf TK_0 W,  
\qquad  
\Delta K_r=W^\mathsf T\Delta KW,$$ $$C_{r,0}=W^\mathsf TC_0W,  
\qquad  
\Delta C_r=W^\mathsf T\Delta CW,$$ $$f_{r,0}=W^\mathsf Tf_0,  
\qquad  
\Delta f_r=W^\mathsf T\Delta f.$$
### 实验结果

```bash
# 预处理极快，几乎不用什么时间。
# 较小网格
Grid 22x22x9; exact ports=144; macro states 1,904->1,204 (1.58x)
h=500 W/(m^2 K): reference range steady=340.919..412.429 K, transient final=304.144..393.627 K; rise error steady=0.32329 K/0.288%, transient final=0.24827 K/0.265%; full/ROM=0.148/0.109s, speedup=1.36x PASS
h=2500 W/(m^2 K): reference range steady=304.614..371.659 K, transient final=301.238..362.706 K; rise error steady=0.18850 K/0.263%, transient final=0.15324 K/0.244%; full/ROM=0.134/0.108s, speedup=1.25x PASS
h=8000 W/(m^2 K): reference range steady=300.478..362.848 K, transient final=300.389..355.578 K; rise error steady=0.21182 K/0.337%, transient final=0.17715 K/0.319%; full/ROM=0.135/0.106s, speedup=1.28x PASS
# 较大网格
Grid 42x42x17; exact ports=576; macro states 15,156->4,644 (3.26x)
h=500 W/(m^2 K): reference range steady=341.052..401.625 K, transient final=304.484..383.862 K; rise error steady=0.28095 K/0.276%, transient final=0.22301 K/0.266%; full/ROM=12.741/2.162s, speedup=5.89x PASS
h=2500 W/(m^2 K): reference range steady=304.672..360.910 K, transient final=301.328..352.996 K; rise error steady=0.26106 K/0.429%, transient final=0.21236 K/0.401%; full/ROM=12.933/2.277s, speedup=5.68x PASS
h=8000 W/(m^2 K): reference range steady=300.490..352.165 K, transient final=300.398..345.921 K; rise error steady=0.23312 K/0.447%, transient final=0.19483 K/0.424%; full/ROM=13.023/2.275s, speedup=5.72x PASS
```

### 优劣

优点：

- 预处理极快，对瞬态问题，单轮运行就有性能收益。
- 对 Z 轴上传播特征压缩效果良好。
- 得到的投影矩阵是分块对角的稀疏矩阵，投影后可以保证稀疏性，利于求解。
- 可以通过添加每柱的模态数量来调节精度/效率平衡。

缺点：

- 无法高效压缩 XY 平面的传输特征，因此压缩率有限。
- 缺乏理论上的误差界控制论证，准确性保障有限。

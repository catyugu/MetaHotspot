# 建模与 PTA 适配边界

内核输入为 `ModelDefinition`：按创建顺序持有 Component，每个 Component 包含笛卡尔或降阶的数值表示、物理端口、热边界与观测。模型共享材料、热源、表达式函数与连接定义。所有坐标统一为 SI 米。

## 有限体积建模

笛卡尔 Component 由严格递增的三个坐标轴定义。物理体由按顺序应用的轴对齐 Add/Subtract box 构造；后声明的体覆盖先前的材料归属。体热源根据几何交叠体积分配。

`SurfaceRegion` 描述轴对齐面上的矩形，切向轴顺序是 X→(y,z)、Y→(x,z)、Z→(x,y)。热边界后声明的区域覆盖先前条件；未覆盖的外表面采用 Component 的默认热边界。流体的实际入流或压力出口逆流必须有显式入口温度。

跨 Component 的物理接口通过同名空间语义的 Port 和 Connection 连接；降阶 Component 的端口须提供物理温度迹、半热阻和必要的切向梯度映射。公开 C/Python 调用契约见 [API 使用说明](../api.md)，此处不重复函数签名。

## PTA 适配

PTA 的 XML、`LengthUnit`、`Layer`、`Block`、`Rect`、`FaceKey` 和表达式重写仅位于 `apps/metahotspot`。适配器展开三维绝对 SI box，进行材料映射、单位换算、边界转换与探针定位，然后生成普通笛卡尔 Component。

PTA `Results/Mesh` 和 `GeneratedMesh` 中的坐标轴由应用层解析并传给内核的笛卡尔网格规格。内核不接收外部 `DiscreteMesh`。PTA 体热源使用 W/m³；引用裸函数名时适配器补全 `(t)`。周期分段常数函数的 Interval 表示每段时长。

`OutputMesh`、PTA XML 回写和 VTU 输出也属于应用层。公开 SDK 不依赖 PTA XML 解析器。

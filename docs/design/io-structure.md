# ModelDefinition / Library 与 PTA 适配

LibraryDefinition 保存 Material、Source、Function、ThermalBoundary 与 FluidBoundary 五类共享物理定义，每类资源有独立顺序 ID。ModelDefinition 保存网格、体几何、模型局部变量、探针、默认热边界 ID 和热/流体表面赋值。已注册的物理条件与其施加的表面区域相互独立。

所有坐标采用 SI 米；材料表达式 T 表示单元温度，热边界表达式 T 表示表面温度迹。热通量正号表示流入。Dirichlet 不允许依赖温度迹。热边界后赋值的区域覆盖先前区域；未显式指定默认热边界时采用内建 Neumann(0)，流体边界只使用显式赋值。投影仍通过现有的笛卡尔网格表面矩形交叠算法，编译结果仍使用按单元排序的 BoundaryFace 及紧凑条件表。

共享 Library 使用独立句柄；创建 Model 时保留底层 Library 的共享所有权。Compiled 在编译时取得自己的物理参数快照，不依赖后续资源注册。Library 资源 ID 不支持删除或重排。

PTA 的 XML、LengthUnit、Layer、Block、Rect、FaceKey 和表达式形式参数规则仍位于 apps/metahotspot；导入时创建当前项目自己的 Library，对其表达式进行 x→T/t 规范化，然后进行几何单位换算。CLI 的 XML 网格读取不进入内核。

## Python 示例

```python
from metahotspot import Library, Model, Dirichlet, SurfaceRegion
from metahotspot.enums import Axis

library = Library()
material = library.add_material("solid", "1", "1", "1", "1000", "500")
cold = library.add_thermal_boundary(Dirichlet("300"))

with Model(library) as model:
    model.set_cartesian_mesh([0, 0.001], [0, 0.001], [0, 0.001])
    volume = model.add_volume(material)
    model.add_box(volume, (0, 0, 0), (0.001, 0.001, 0.001))
    model.add_thermal_boundary([SurfaceRegion(Axis.Z, 0, 0, 0.001, 0, 0.001)], cold)
    compiled = model.compile()
    solution = compiled.solve()
```

函数通过 Library 注册，变量仍由 Model 注册。Probe 仍是模型局部几何观测。PTA 导出、节点插值与 VTU 均保持应用层职责。

# weekly\_report\_0901

## 连接模型

- 看到一篇 17 年的相关工作，

> Connecting MOR-based boundary condition independent compact thermal models。

- 方法论和我们之前提到的方法论一样，把边界节点作为独立节点处理，不参与降阶。为了可以处理非共形，引入面积权重，让不同网格的边界节点对齐到一个公共接口上。处理可以直接在模型定义层面完成，无需知道原始网格细节。
- 但是并未提供训练方法的细节（也就是说，仍然没有交代如何才能将接口边界的传播模式加入降阶模型中）
- 如果用朴素方法施加各种边界模式激励，则阶数将显著增加。
- 就物理直观看，以下几个东西是不可能同时实现的：
    - 低阶数
    - 低误差
    - 能自由外接任意热网络（对任意边界输入模态的良好响应）
- 根据对相关 RomCore.dll 的逆向工程结果看，FloTHERM 在提取时，对于端口面也和 BCI 面一样，用一致的一个对流换热系数进行提取。显然，这种做法会有一定的误差。

## 压力测试

- 拿一个简单模型做测试，中间热源，上面提取成 BCI，下面端接，使用的是 Flotherm 生成的 Embeddable BCI ROM 基底：
- 首先注意到的是，FloTHERM 的 EROM 相比非嵌入 ROM 模型，在单独求解时的固有误差有所增加（相同的容差下，非嵌入 vs 全量：0.00018 % 的相对误差，嵌入 vs 全量：0.73 % 的相对误差）可能是由于增加了接口面，导致模态投影有所偏差。
- 随便测一些外接的工况：

```text
[baseline_copper         ] order= 11  dTmax=11.579 K  junction_rel=0.139%  global_rel=0.742%  transient_junction_rel=0.608%  transient_global_rel=1.777%
[lowk_insulator          ] order= 11  dTmax=11.582 K  junction_rel=0.147%  global_rel=0.950%  transient_junction_rel=0.071%  transient_global_rel=0.731%
[highk_aluminum          ] order= 11  dTmax=11.580 K  junction_rel=0.141%  global_rel=0.775%  transient_junction_rel=0.351%  transient_global_rel=1.332%
[bottom_htc_weak         ] order= 10  dTmax=10.434 K  junction_rel=0.496%  global_rel=1.692%  transient_junction_rel=0.620%  transient_global_rel=2.284%
[bottom_htc_strong       ] order= 10  dTmax=6.418 K  junction_rel=0.944%  global_rel=2.765%  transient_junction_rel=1.147%  transient_global_rel=4.022%
[external_source_50w     ] order= 11  dTmax=16.988 K  junction_rel=0.280%  global_rel=2.354%  transient_junction_rel=0.252%  transient_global_rel=1.243%
[external_source_200w    ] order= 11  dTmax=34.054 K  junction_rel=0.443%  global_rel=4.151%  transient_junction_rel=0.427%  transient_global_rel=2.426%
[lowk_source             ] order= 10  dTmax=86.330 K  junction_rel=0.987%  global_rel=3.709%  transient_junction_rel=0.795%  transient_global_rel=2.098%
[all_stress              ] order= 10  dTmax=57.092 K  junction_rel=0.927%  global_rel=3.542%  transient_junction_rel=0.825%  transient_global_rel=2.276%
[layered_soft            ] order= 11  dTmax=11.580 K  junction_rel=0.142%  global_rel=0.809%  transient_junction_rel=0.083%  transient_global_rel=0.772%
[layered_soft_source     ] order= 10  dTmax=41.534 K  junction_rel=0.970%  global_rel=3.631%  transient_junction_rel=0.735%  transient_global_rel=2.187%
[layered_extreme         ] order= 11  dTmax=11.579 K  junction_rel=0.139%  global_rel=0.741%  transient_junction_rel=0.164%  transient_global_rel=1.025%
[layered_extreme_source  ] order= 10  dTmax=243.787 K  junction_rel=0.956%  global_rel=3.231%  transient_junction_rel=0.738%  transient_global_rel=2.089% 
```

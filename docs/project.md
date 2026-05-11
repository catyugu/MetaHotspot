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
# Transformer
参考：https://blog.csdn.net/jokerxsy/article/details/116299343


## 结构

![](Images/1.png)

左图是一个序列对齐的方式。Transformer改变了序列对齐，以self-attention的方式处理。

## Self-attention

![](Images/2.png)

* V：输入
* 括号内为对齐函数

![](Images/4.png)


![](Images/3.png)

左图是右图中间部分的核心结构。

MatMul： 将Q和K矩阵相乘
Scale: 将得到的权重加权到每个通道特征上。


完整结构：

![](Images/6.png)

* Multi-headed 的意思是有多重学习手段，Q、K、V
* Masked Multi-head attention: 只关注前序信息

####简化后的结构：

![](Images/7.png)

####High-level:

可以被抽象为下图

![](Images/8.png)

#### A specific level look

![](Images/9.png)

![](Images/10.png)

#### multi headed 训练效果demo

不同的颜色代表不同head

![](Images/11.png)


## 为什么Transfromer更重要

![](Images/12.png)



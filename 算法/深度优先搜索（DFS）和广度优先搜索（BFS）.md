# 深度优先搜索（DFS）和广度优先搜索（BFS）

参考：https://blog.csdn.net/weixin_40953222/article/details/80544928

## 图的基本概念

### 无向图

顶点对(u，v)是无序的，即（u，v）和（v，u）是同一条边。常用一对圆括号表示。

![](../Images/1.png)


### 有向图

顶点对<u,v>是有序的，它是指从顶点u到顶点 v的一条有向边。其中u是有向边的始点，v是有向边的终点。常用一对尖括号表示。

![](../Images/2.png)


## 权和网

图的每条边上可能存在具有某种含义的数值，称该数值为该边上的权。而这种带权的图被称为网。

## 连通图与非连通图

连通图：在无向图G中，从顶点v到顶点v'有路径，则称v和v'是联通的。若图中任意两顶点v、v'∈V，v和v'之间均联通，则称G是连通图。上述两图均为连通图。

非连通图：若无向图G中，存在v和v'之间不连通，则称G是非连通图。

![](../Images/3.png)


## 深度优先与广度优先的区别

bfs是按一层一层来访问的，所以适合有目标求最短路的步数，你想想层层搜索每次层就代表了一步。bfs优先访问的是兄弟节点，只有这一层全部访问完才能访问下一层，也就是说bfs第几层就代表当前可以走到的位置(结点).而dfs是按递归来实现的，它优先搜索深度，再回溯，优先访问的是没有访问过的子节点

DFS多用于连通性问题因为其运行思想与人脑的思维很相似，故解决连通性问题更自然。BFS多用于解决最短路问题，其运行过程中需要储存每一层的信息，所以其运行时需要储存的信息量较大，如果人脑也可储存大量信息的话，理论上人脑也可运行BFS。


## 广度优先搜索


### 算法的基本思路

广度优先搜索类似于树的层次遍历过程。它需要借助一个队列来实现。如图2-1-1所示，要想遍历从v0到v6的每一个顶点，我们可以设v0为第一层，v1、v2、v3为第二层，v4、v5为第三层，v6为第四层，再逐个遍历每一层的每个顶点。

具体过程如下：  

1.准备工作：创建一个visited数组，用来记录已被访问过的顶点；创建一个队列，用来存放每一层的顶点；初始化图G。  

2.从图中的v0开始访问，将的visited[v0]数组的值设置为true，同时将v0入队。

3.只要队列不空，则重复如下操作：   

 (1)队头顶点u出队。   

 (2)依次检查u的所有邻接顶点w，若visited[w]的值为false，则访问w，并将visited[w]置为true，同时将w入队。


### 算法的实现过程

白色表示未被访问，灰色表示即将访问，黑色表示已访问。

visited数组：0表示未访问，1表示以访问。

队列：队头出元素，队尾进元素。

1.初始时全部顶点均未被访问，visited数组初始化为0，队列中没有元素。

![](../Images/4.png)


2.即将访问顶点v0。

![](../Images/5.png)

3.访问顶点v0，并置visited[0]的值为1，同时将v0入队。

![](../Images/6.png)

4.将v0出队，访问v0的邻接点v2。判断visited[2]，因为visited[2]的值为0，访问v2。

![](../Images/7.png)

5.将visited[2]置为1，并将v2入队

![](../Images/8.png)

6.访问v0邻接点v1。判断visited[1],因为visited[1]的值为0，访问v1。

![](../Images/9.png)

7.将visited[1]置为0，并将v1入队

![](../Images/10.png)

8.判断visited[3],因为它的值为0，访问v3。将visited[3]置为0，并将v3入队。

![](../Images/11.png)

9.v0的全部邻接点均已被访问完毕。将队头元素v2出队，开始访问v2的所有邻接点。

开始访问v2邻接点v0，判断visited[0]，因为其值为1，不进行访问。

继续访问v2邻接点v4，判断visited[4]，因为其值为0，访问v4，如下图：

![](../Images/12.png)

10.将visited[4]置为1，并将v4入队。

![](../Images/13.png)

11.v2的全部邻接点均已被访问完毕。将队头元素v1出队，开始访问v1的所有邻接点。开始访问v1邻接点v0，因为visited[0]值为1，不进行访问。继续访问v1邻接点v4，因为visited[4]的值为1，不进行访问。继续访问v1邻接点v5，因为visited[5]值为0，访问v5，如下图：

![](../Images/14.png)


12.将visited[5]置为1，并将v5入队。

![](../Images/15.png)

13.v1的全部邻接点均已被访问完毕，将队头元素v3出队，开始访问v3的所有邻接点。

开始访问v3邻接点v0，因为visited[0]值为1，不进行访问。

继续访问v3邻接点v5，因为visited[5]值为1，不进行访问。

![](../Images/16.png)

14.v3的全部邻接点均已被访问完毕，将队头元素v4出队，开始访问v4的所有邻接点。

开始访问v4的邻接点v2，因为visited[2]的值为1，不进行访问。

继续访问v4的邻接点v6，因为visited[6]的值为0，访问v6，如下图：

![](../Images/17.png)

15.将visited[6]值为1，并将v6入队。


![](../Images/18.png)


16.v4的全部邻接点均已被访问完毕，将队头元素v5出队，开始访问v5的所有邻接点。

开始访问v5邻接点v3，因为visited[3]的值为1，不进行访问。

继续访问v5邻接点v6，因为visited[6]的值为1，不进行访问

![](../Images/19.png)


17.v5的全部邻接点均已被访问完毕，将队头元素v6出队，开始访问v6的所有邻接点。

开始访问v6邻接点v4，因为visited[4]的值为1，不进行访问。

继续访问v6邻接点v5，因为visited[5]的值文1，不进行访问。

![](../Images/20.png)

18.队列为空，退出循环，全部顶点均访问完毕

![](../Images/21.png)


## 深度优先搜索

深度优先搜索类似于树的先序遍历，具体过程如下：

准备工作：创建一个visited数组，用于记录所有被访问过的顶点。

1.从图中v0出发，访问v0。

2.找出v0的第一个未被访问的邻接点，访问该顶点。以该顶点为新顶点，重复此步骤，直至刚访问过的顶点没有未被访问的邻接点为止。

3.返回前一个访问过的仍有未被访问邻接点的顶点，继续访问该顶点的下一个未被访问领接点。4.重复2,3步骤，直至所有顶点均被访问，搜索结束。

## 算法的实现过程

1.初始时所有顶点均未被访问，visited数组为空。

![](../Images/22.png)

2.即将访问v0。

![](../Images/23.png)

3.访问v0，并将visited[0]的值置为1。

![](../Images/24.png)


4.访问v0的邻接点v2，判断visited[2]，因其值为0，访问v2。

![](../Images/25.png)

5.将visited[2]置为1。

![](../Images/26.png)

6.访问v2的邻接点v0，判断visited[0]，其值为1，不访问。

继续访问v2的邻接点v4，判断visited[4]，其值为0，访问v4。

![](../Images/27.png)

7.将visited[4]置为1

![](../Images/28.png)

8.访问v4的邻接点v1，判断visited[1]，其值为0，访问v1。

![](../Images/29.png)

9.将visited[1]置为1

![](../Images/30.png)

10.访问v1的邻接点v0，判断visited[0]，其值为1，不访问。

继续访问v1的邻接点v4，判断visited[4]，其值为1，不访问。

继续访问v1的邻接点v5，判读visited[5]，其值为0，访问v5。

![](../Images/31.png)

11.将visited[5]置为1

![](../Images/32.png)

12.访问v5的邻接点v1，判断visited[1]，其值为1，不访问。

继续访问v5的邻接点v3，判断visited[3]，其值为0，访问v3。

![](../Images/33.png)

13.将visited[1]置为1。

![](../Images/34.png)

14.访问v3的邻接点v0，判断visited[0]，其值为1，不访问。

继续访问v3的邻接点v5，判断visited[5]，其值为1，不访问。

v3所有邻接点均已被访问，回溯到其上一个顶点v5，遍历v5所有邻接点。

访问v5的邻接点v6，判断visited[6]，其值为0，访问v6。

![](../Images/35.png)

15.将visited[6]置为1

![](../Images/36.png)

16.访问v6的邻接点v4，判断visited[4]，其值为1，不访问。

访问v6的邻接点v5，判断visited[5]，其值为1，不访问。

v6所有邻接点均已被访问，回溯到其上一个顶点v5，遍历v5剩余邻接点。

![](../Images/37.png)


17.v5所有邻接点均已被访问，回溯到其上一个顶点v1。v1所有邻接点均已被访问，回溯到其上一个顶点v4，遍历v4剩余邻接点v6。v4所有邻接点均已被访问，回溯到其上一个顶点v2。v2所有邻接点均已被访问，回溯到其上一个顶点v1，遍历v1剩余邻接点v3。v1所有邻接点均已被访问，搜索结束。

![](../Images/38.png)
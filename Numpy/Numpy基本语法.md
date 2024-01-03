# Numpy基本语法



## numpy.arange

### 从数值范围创建数组

```
numpy.arange(start, stop, step, dtype)
```

例子1：

```
x = np.arange(11)
x
```

输出：

```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```



例子2：

```
# 设置了 dtype 

x = np.arange(5, dtype =  float)   

print (x)


```

输出：

```
[0.  1.  2.  3.  4.]
```



例子3：

```
#设置了起始值、终止值及步长：

x = np.arange(10,20,2)  

 print (x)
```

输出：

```
[10  12  14  16  18]
```





## numpy.linspace

numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的，格式如下：

```
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

例子1:

```
import numpy as np 

a = np.linspace(1,1,10) 

print(a)
```

输出：

```
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```

例子2:



```
import numpy as np 

a =np.linspace(1,10,10,retstep= True)  

print(a) 
# 拓展例子 
b =np.linspace(1,10,10).reshape([10,1]) 
print(b)
```

输出：

```
(array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]), 1.0)
[[ 1.]
 [ 2.]
 [ 3.]
 [ 4.]
 [ 5.]
 [ 6.]
 [ 7.]
 [ 8.]
 [ 9.]
 [10.]]
```





## numpy.logspace

numpy.logspace 函数用于创建一个于等比数列。格式如下：

```
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
```

例子：

```
import numpy as np 
# 默认底数是 10 
a = np.logspace(1.0,  2.0, num =  10)
print (a)
```

输出：

```
[ 10.           12.91549665     16.68100537      21.5443469  27.82559402      
  35.93813664   46.41588834     59.94842503      77.42636827    100.    ]
```
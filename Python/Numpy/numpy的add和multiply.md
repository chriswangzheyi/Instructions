# numpy的add和multiply

参考资料：https://blog.csdn.net/shu15121856/article/details/76206891

## 加法运算

### np.add.accumulate()

适用于python序列(串不行)和numpy数组，每一个位置的元素和前面的所有元素加起来求和，得到的始终是numpy数组。

	import numpy as np
	
	print(np.add.accumulate([1,2,3]) )

输出：

	[1 3 6]


### np.add.reduce()

所有元素加在一起求和

	import numpy as np
	
	print(np.add.reduce([1,2,3,4,5]) )

输出：

	15

###　np.add.at()

入的数组中制定下标位置的元素加上指定的值

	import numpy as np
	
	x=np.array([1,2,3,4])
	#下标为0,2的元素加3
	np.add.at(x,[0,2],3)
	print(x)

输出：

	[4 2 6 4]


###　np.add.outer()

将第一个列表或数组中的每个元素依次加到第二个列表或数组中的每个元素，得到每一行。

	import numpy as np
	
	print(np.add.outer([1,2,3],[4,5,6,7]))

输出：

	[[ 5  6  7  8]
	 [ 6  7  8  9]
	 [ 7  8  9 10]]


### np.add.reduceat()

对于传入的数组，根据传入的list(第二个参数)作指定的变化，传入的list中的数字是成对出现的。

	import numpy as np
	
	x=np.arange(8)
	ans = np.add.reduceat(x,[0,4,1,5,2,6,3,7]) #在各切片上作reduce运算
	print(ans)

输出：

	[ 6  4 10  5 14  6 18  7]

例子中是将x中0,4部分切片作np.add.reduce()运算(也就是连加)，放在第一个位置，然后第二个位置就是下标4在x中的值，也就是4，以此类推。

## 乘法

乘法跟加法类似

### np.multiply.at()

表示将某个数组中的制定下标元素乘以指定值

	import numpy as np
	
	x=np.arange(8)
	print(x)
	np.multiply.at(x,[0,1,2],5)
	print(x)

输出：

	[0 1 2 3 4 5 6 7]
	[ 0  5 10  3  4  5  6  7]


### np.multiply.accumulate()

表示累乘

### np.multiply.outer()

表示将第一个列表或数组中的每个元素依次乘到第二个列表或数组中的每个元素，得到每一行。

### np.multiply.reduce

表示连乘，所有元素相乘

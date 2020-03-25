# 经典Demo

## 自定义函数

格式：

def function_name(parameter):
	

	import numpy as np
	
	def basic_sigmoid(x):
	    s = 1/(1+np.exp(-x))
	    return s
	
	my_array = np.arange(-11,11)
	print(my_array)
	print(basic_sigmoid(my_array))


## 勾股定理
	
	import numpy as np
	
	number_of_triangles = 9
	
	# +1的原因是arange是从0开始的
	base = np.arange(number_of_triangles) +1
	height = np.arange(number_of_triangles)+ 1
	print(base)
	print(height)
	
	
	# np.add.outer()将第一个列表或数组中的每个元素依次加到第二个列表或数组中的每个元素，得到每一行
	hypotenuse_squared = np.add.outer(base **2, height**2)
	
	# 斜边
	hypotenuse = np.sqrt(hypotenuse_squared)
	
	print(hypotenuse)

# Machine Learning Demo

## 经典库

## 统计Demo

	import os;
	import pandas as pd;
	import requests;
	
	#训练数据存放地址
	Path = 'E:\projects'
	
	#从远端下载训练集
	r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
	
	#将训练集存入本地
	with open(Path+'\iris.data', 'w') as f:
	    f.write(r.text)
	
	#改变当前工作目录到指定的路径
	os.chdir(Path)
	
	#读文档
	df = pd.read_csv(Path+'\iris.data', names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','种类'])
	
	#使得数据对齐
	pd.set_option('display.unicode.ambiguous_as_wide', True)
	pd.set_option('display.unicode.east_asian_width', True)
	
	#输出前几项
	print(df.head())
	
	print("-------多条件查询------")
	
	#多条件查询
	print( df[ (df['花萼宽度']>3.1) & (df['种类']=='Iris-setosa')])
	
	print("-------各项统计------")
	
	#各项统计数据
	print(df.describe())
	
	print("-------自定义各项统计------")
	
	#增加了20%百分比， 40%百分比， 80%百分比
	print(df.describe(percentiles=[.20, .40, .80]))
	
	
	print("-------相关系数矩阵------")
	# 相关系数矩阵，即给出了任意两个变量之间的相关系数
	print(df.corr())


![](../Images/1.png)


## 直方图Demo

	import matplotlib.pyplot as plt
	import pandas as pd;
	from pylab import mpl
	
	#显示中文
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	
	#数据库样式
	plt.style.use('ggplot')
	
	# 训练数据存放地址
	Path = 'E:\projects'
	
	#读取文件
	df = pd.read_csv(Path+'\iris.data', names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','种类'])
	
	#形成数据库
	fig, ax = plt.subplots(figsize=(6,4))
	
	# 根据'花萼宽度'这个属性值，画图，图像为黑色。 hist是散点图
	ax.hist(df['花萼宽度'], color='black')
	
	#y轴属性
	ax.set_ylabel('数量',fontsize=12)
	
	#x轴属性
	ax.set_xlabel('花萼宽度',fontsize=12)
	
	#统计图抬头
	plt.title('Iris 花瓣宽度',fontsize=14,y=1.01)
	
	#显示数据
	plt.show()

![](../Images/2.png)


## 散点图Demo

	import matplotlib.pyplot as plt
	import pandas as pd;
	from pylab import mpl
	
	#显示中文
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	
	#数据库样式
	plt.style.use('ggplot')
	
	# 训练数据存放地址
	Path = 'E:\projects'
	
	#读取文件
	df = pd.read_csv(Path+'\iris.data', names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','种类'])
	
	#形成数据库
	fig, ax = plt.subplots(figsize=(6,6))
	
	# 根据'花萼宽度'这个属性值，画散点图，图像为蓝色。 Scatter函数第一个参数为x轴函数，第二个函数为y轴函数
	ax.scatter(df['花萼宽度'], df['花萼长度'],color='blue')
	
	#y轴属性
	ax.set_ylabel('花瓣长度')
	
	#x轴属性
	ax.set_xlabel('花萼宽度')
	
	#统计图抬头
	plt.title('Iris 花瓣散点图')
	
	#显示数据库
	plt.show()

主要的区别为：使用了 **scatter** 函数

![](../Images/4.png)

## 线图Demo

	import matplotlib.pyplot as plt
	import pandas as pd;
	from pylab import mpl
	
	#显示中文
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	
	#数据库样式
	plt.style.use('ggplot')
	
	# 训练数据存放地址
	Path = 'E:\projects'
	
	#读取文件
	df = pd.read_csv(Path+'\iris.data', names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','种类'])
	
	#形成数据库
	fig, ax = plt.subplots(figsize=(6,6))
	
	# 根据'花萼宽度'这个属性值，画线图，图像为蓝色
	ax.plot(df['花萼宽度'], color='blue')
	
	#y轴属性
	ax.set_ylabel('花瓣长度')
	
	#x轴属性
	ax.set_xlabel('花萼宽度')
	
	#统计图抬头
	plt.title('Iris 花瓣散点图')
	
	#显示数据库
	plt.show()


主要的区别为：使用了 **plot** 函数

![](../Images/3.png)


## 条形图Demo

	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy as np
	
	from pylab import mpl
	
	#显示中文
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	
	#数据库样式
	plt.style.use('ggplot')
	
	# 训练数据存放地址
	Path = 'E:\projects'
	
	#读取文件
	df = pd.read_csv(Path+'\iris.data', names=['花萼长度','花萼宽度','花瓣长度','花瓣宽度','种类'])
	
	#形成数据库
	fig, ax = plt.subplots(figsize=(6, 6))
	
	# 柱状图宽度
	bar_width = .8
	
	# 统计所有'花萼长度'和'花萼宽度' 参数
	labels = [x for x in df.columns if '花萼长度' in x or '花萼宽度' in x]
	
	vir_y= [df[df['种类']=='Iris-virginica'][x].mean() for x in labels]
	ver_y= [df[df['种类']=='Iris-versicolor'][x].mean() for x in labels]
	set_y= [df[df['种类']=='Iris-setosa'][x].mean() for x in labels]
	
	# 生成数组
	x = np.arange(len(labels))
	
	
	# bar() 函数来生成条形图
	ax.bar(x,vir_y,bar_width, bottom=set_y,color='darkgrey')
	ax.bar(x,set_y,bar_width, bottom=ver_y,color='white')
	ax.bar(x,ver_y,bar_width,color='black')
	
	ax.set_xticks(x + (bar_width/2))
	ax.set_xticklabels(labels, rotation=-70, fontsize=12)
	
	#统计图抬头
	ax.set_title('每个类别中特征的平均测量值', y=1.01)
	
	ax.legend(['Virginica','Setosa','Versicolor'])
	
	#显示数据库
	plt.show()


![](../Images/5.png)
# Matplotlib



## 引入

import matplotlib.pyplot as plt

## 例子

	import numpy as np 
	from matplotlib import pyplot as plt 
	 
	x = np.arange(1,11) 
	y =  2  * x +  5 
	plt.title("Matplotlib demo") 
	plt.xlabel("x axis caption") 
	plt.ylabel("y axis caption") 
	plt.plot(x,y) 
	plt.show()

![](../Images/1.png)


## 中文显示

	import numpy as np
	import matplotlib.pyplot as plt
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
	mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
	
	
	x = np.arange(1, 11)
	y = 2 * x + 5
	plt.title("菜鸟教程 - 测试")
	
	# fontproperties 设置中文显示，fontsize 设置字体大小
	plt.xlabel("x 轴")
	plt.ylabel("y 轴")
	plt.plot(x, y)
	plt.show()

![](../Images/2.png)


## 柱状图

使用plt.hist(x, num_bins, density=1, facecolor='blue', alpha=0.5)  

	x : (n,) array or sequence of (n,) arrays
	
	这个参数是指定每个bin(箱子)分布的数据,对应x轴
	
	num_bins : integer or array_like, optional
	
	这个参数指定bin(箱子)的个数,也就是总共有几条条状图

	density : boolean, optional
	
	If True, the first element of the return tuple will be the counts normalized to form a probability density, i.e.,n/(len(x)`dbin)
	
	这个参数指定密度,也就是每个条状图的占比例比,默认为1
	
	color : color or array_like of colors or None, optional
	
	这个指定条状图的颜色

	alpha设置透明度，0为完全透明
	

### 计算可能性demo

	import numpy as np
	import matplotlib.pyplot as plt
	
	mu, sigma = 100, 15
	data_set = mu+ sigma * np.random.randn(10000)
	
	plt.hist(data_set,50,density=1,facecolor='b', alpha = 0.75 )
	plt.xlabel('Smarts')
	plt.ylabel("Probability")
	plt.title("Histogram of IQ")
	plt.axis([40,160,0,0.03])
	plt.grid(True)
	plt.show()

![](../Images/3.png)


#### randn 和 rand 区别

numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。 
numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。
	

## figure 和 sublpot

figure定义了一个整体，subplot是其中的子图

	import numpy as np
	import matplotlib.pyplot as plt
	
	my_first_figure = plt.figure('my first figure')
	
	subplot1 = my_first_figure.add_subplot(2,3,1)  # row = 2, col = 3 , index = 1
	plt.plot(np.random.rand(50).cumsum(),'k--')  # cumsum 用于计算累加值 
	
	subplot2 = my_first_figure.add_subplot(2,3,6)  # row = 2, col = 3 , index = 1
	plt.plot(np.random.rand(50),'go')
	
	plt.show()

## 曲线图

### 使用plt.plot(x1,y1,x2,y2)的格式  曲线图

	import numpy as np
	import matplotlib.pyplot as plt
	
	data_set_size =15
	low_mu, low_sigma =50, 4.3
	low_data_set = low_mu + low_sigma* np.random.rand((data_set_size))
	
	high_mu, high_sigma= 57, 5.2
	high_data_set = high_mu + high_sigma* np.random.rand(data_set_size)
	
	days = list(range(1,data_set_size+1))
	
	plt.plot(days, low_data_set,
	         days,high_data_set)
	plt.show()


![](../Images/4.png)

### 使用plt.plot(x1,y1,'格式',x2,y2,'格式)的格式  点状分布图

	import numpy as np
	import matplotlib.pyplot as plt
	
	data_set_size =15
	low_mu, low_sigma =50, 4.3
	low_data_set = low_mu + low_sigma* np.random.rand((data_set_size))
	
	high_mu, high_sigma= 57, 5.2
	high_data_set = high_mu + high_sigma* np.random.rand(data_set_size)
	
	days = list(range(1,data_set_size+1))
	
	plt.plot(days, low_data_set, "vm",
	         days,high_data_set,"^k")
	plt.show()

![](../Images/5.png)


### 合二为一

	import numpy as np
	import matplotlib.pyplot as plt
	
	data_set_size =15
	low_mu, low_sigma =50, 4.3
	low_data_set = low_mu + low_sigma* np.random.rand((data_set_size))
	
	high_mu, high_sigma= 57, 5.2
	high_data_set = high_mu + high_sigma* np.random.rand(data_set_size)
	
	days = list(range(1,data_set_size+1))
	
	plt.plot(days, low_data_set,
	         days, low_data_set, "vm",
	         days,high_data_set,
	         days,high_data_set,"^k")
	plt.show()


![](../Images/6.png)

## 多曲线综合示例

	import numpy as np
	import matplotlib.pyplot as plt
	
	t1 = np.arange(0.0, 2.0, 0.1)
	t2 = np.arange(0.0, 2.0, 0.01)
	
	l1, = plt.plot(t2, np.exp(-t2)) #numpy.exp()：返回e的幂次方，e是一个常数为2.71828
	l2, l3 = plt.plot(t2, np.sin(2 * np.pi * t2), '--o', t1, np.log(1 + t1), '.')
	l4, = plt.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 's-.')
	
	plt.legend((l1, l4), ('oscillatory', 'damped'), loc='upper right', shadow=True)  #plt.legend加上图例，参数l1,l4指的是对应的两条线
	plt.xlabel('time')
	plt.ylabel('volts')
	plt.title('Damped oscillation')
	plt.show()

![](../Images/7.png)


## Tick图和Grid图


Tick 可以理解为数据的代表

	import numpy as np
	import matplotlib.pyplot as plt
	
	# 画曲线图
	number_data_points = 1000
	
	my_figure = plt.figure()
	subplot_1 = my_figure.add_subplot(1,1,1)
	subplot_1.plot(np.random.rand(number_data_points).cumsum())
	
	# 画ticks
	number_of_ticks =5
	ticks= np.arange(0,number_data_points,number_data_points//number_of_ticks)  # 从0开始到1000，步长是1000/5
	subplot_1.set_xticks(ticks)
	
	labels = subplot_1.set_xticklabels(['one','two','three','four','five'],rotation=45, fontsize='small')
	
	subplot_1.set_title("my first ticked plot")
	subplot_1.set_xlabel("Groups")
	
	## 画网格
	subplot_1.grid(True)
	gridlines = subplot_1.get_xgridlines()+subplot_1.get_ygridlines()
	
	for line in gridlines:
	    line.set_linestyle(":") # 用点状画网格
	
	plt.show()

![](../Images/8.png)


## Annotation 标记

	import numpy as np
	import matplotlib.pyplot as plt
	
	number_of_data_points =10
	data = np.random.rand(10)
	
	my_figure = plt.figure()
	subplot_1=my_figure.add_subplot(1,1,1)
	subplot_1.plot(data.cumsum())
	
	subplot_1.text(1,0.5, r'an equation: $E=mc^2',fontsize=18,color='red')
	subplot_1.text(1,1.5,"hello, moutain climbing!", fontsize=14,color='green')
	
	subplot_1.text(0.5,0.5, "we are centered, now", transform=subplot_1.transAxes)
	# transform表示以某一个标准为参考物
	
	subplot_1.annotate('shoot arrow', xy=(2,1), xytext=(3,4),
	                arrowprops=dict(facecolor='red',shrink=0.05))
	# xy是被注释的坐标点，二维元组形如(x,y)，xytext：注释文本的坐标点，也是二维元组，默认与xy相同
	# arrowprops定义了箭头的样式，shrink是指粗细
	
	plt.show()


![](../Images/9.png)

## 复杂曲线Demo

	import numpy as np
	import matplotlib.pyplot as plt
	
	# 画曲线图
	x = np.arange(0, 10, 0.005)
	y = np.exp(-x/2.) * np.sin(2*np.pi*x)
	
	fig = plt.figure()
	ax = fig.add_subplot(111) #等价于fig.add_subplot(1，1，1)
	ax.plot(x, y)
	ax.set_xlim(0, 10)
	ax.set_ylim(-1, 1)
	
	
	xdata, ydata = 5, 0
	xdisplay, ydisplay = ax.transData.transform_point((xdata, ydata)) #坐标转换
	bbox = dict(boxstyle="round", fc="0.8") #  圆角round， 程度rc
	
	arrowprops = dict(
	    arrowstyle = "->",
	    connectionstyle = "angle,angleA=0,angleB=90,rad=10")
	
	offset = 72
	
	#上面的annotation图
	ax.annotate('data = (%.1f, %.1f)'%(xdata, ydata), #显示内容
	            xy=(xdata, ydata), xytext=(-2*offset, offset), textcoords='offset points', #xytext制定了box的偏移方向
	            bbox=bbox, arrowprops=arrowprops)
	# textcoords ：注释文本的坐标系属性   offset points':相对于被注释点xy的偏移量（单位是点）. 'offset pixels'	相对于被注释点xy的偏移量（单位是像素）
	
	#下面的annotation图
	disp = ax.annotate('display = (%.1f, %.1f)'%(xdisplay, ydisplay),
	            (xdisplay, ydisplay), xytext=(0.5*offset, -offset),
	            xycoords='figure pixels',
	            textcoords='offset points',
	            bbox=bbox, arrowprops=arrowprops)
	
	plt.show( )

![](../Images/10.png)

## 饼状图 pie charts 直方图 Bar chart

### 饼状图 

	import matplotlib.pyplot as plt
	
	labels = 'China','Japan','Turkey', 'Germany'
	sizes=[15,30,45,10]
	colors =['yellowgreen','gold','lightskyblue','lightcoral']
	explode= (0,0.1,0,0) # only explore the 2nd slice
	
	plt.pie(x=sizes, explode=explode, labels=labels,
	        colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)
	
	# set aspect raton to be qual so that pie is drawn as a cirle
	plt.axis('equal')
	
	plt.show()


![](../Images/11.png)

### 直方图

	import numpy as np
	import matplotlib.pyplot as plt
	
	#基本设置
	N=5
	ind = np.arange(N)
	width = 0.35 # 柱状图的宽度
	
	#柱状图数据
	memMeeans = (20,35,30,35,27) #男性平均数
	menStd = (2,3,4,1,2) #男性方差
	womenMens = (25,32,34,20,25) # 女性平均数
	womenStd= (3,5,2,3,3) #女性方差
	
	#参数： bar(x轴坐标，y轴坐标，柱状图宽度，颜色，yerr：y方向error bar)
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind,memMeeans,width,color='r',yerr=menStd)
	rects2 = ax.bar(ind+width ,womenMens,width,color='y',yerr=womenStd)
	
	ax.set_label('Scores')
	ax.set_title('Scores by group and gender')
	
	# 设置tick标签
	ax.set_xticks(ind+width)
	ax.set_xticklabels(('G1','G2','G3','G4','G5'))
	
	#设置
	ax.legend((rects1[0],rects2[0]),('Men','Women'))
	
	plt.show()

![](../Images/12.png)


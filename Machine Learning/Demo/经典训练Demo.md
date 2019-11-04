# Machine Learning Demo

## 经典训练Demo

	import requests;
	import os;
	import pandas as pd;
	
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
	
	print(df.head())


![](../Images/1.png)
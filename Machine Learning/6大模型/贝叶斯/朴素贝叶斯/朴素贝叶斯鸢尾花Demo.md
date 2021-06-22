# 朴素贝叶斯鸢尾花Demo

## 代码

	# coding=utf-8
	#朴素贝叶斯sklearn代码实现
	import numpy as np
	from sklearn import naive_bayes as nb
	from sklearn.metrics import accuracy_score
	from sklearn.datasets import load_iris
	
	iris = load_iris()
	X = iris.data
	y = iris.target
	
	model = nb.GaussianNB() #连续数据
	model.fit(X,y)
	
	y_hat = model.predict(X)
	print("精确度=",accuracy_score(y,y_hat))

运行结果：

	精确度= 0.96
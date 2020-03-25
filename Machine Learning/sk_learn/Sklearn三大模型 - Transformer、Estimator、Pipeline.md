# Sklearn三大模型 - Transformer、Estimator、Pipeline

参考：https://www.jianshu.com/p/16169fbe68ba

---

在Sklearn当中有三大模型：Transformer 转换器、Estimator 估计器、Pipeline 管道




## Transformer 转换器

Transformer 转换器 (StandardScaler，MinMaxScaler)

	## 数据标准化
	## StandardScaler 画图纸
	ss = StandardScaler() 
	## fit_transform训练并转换 
	## fit在计算，transform完成输出
	X_train = ss.fit_transform(X_train) 
	X_train

Transformer有输入有输出，同时输出可以放入Transformer或者Estimator 当中作为输入。


## Estimator 估计器

Estimator 估计器（LinearRegression、LogisticRegression、LASSO、Ridge， 所有的机器学习算法模型，都被称为估计器。

	## 模型训练
	lr = LinearRegression()
	## LinearRegression 是一个有监督的算法，所以要把特征值和目标值一起放入
	lr.fit(X_train,Y_train) #训练模型
	
	## 模型校验
	y_predict = lr.predict(X_test) #预测结果

y_predict 是估计器的输出模型，估计器输出无法再放入Transformer 或 Estimator当中再获取另一个输出了。

## Pipeline 管道

将Transformer、Estimator 组合起来成为一个大模型。

管道： 输入→□→□→□→■→ 输出

□：Transformer ； ■：Estimator ；

Transformer放在管道前几个模型中，而Estimator 只能放到管道的最后一个模型中。


	## 一般用到sklearn的子库
	import sklearn
	from sklearn.model_selection import train_test_split #训练集测试集划分，最新版本中该库直接归到了sklearn的子库
	from sklearn.linear_model import LinearRegression # 线性模型
	from sklearn.preprocessing import StandardScaler # 预处理的库
	from sklearn.preprocessing import MinMaxScaler
	
	## 管道相关的包
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.model_selection import GridSearchCV

ipeline 的参数是一个列表，列表中存放着每一个模型的信息。

第0个模型名字： ss，告诉系统我要做数据标准化。

第1个模型名字： Poly，告诉系统我要做一个多项式扩展。
PolynomialFeatures即进行了ss= StandardScaler()的操作,并做了3阶的扩展

第2个模型名字： Linear，告诉系统进行模型训练。
fit_intercept=False 表示截距为0
截距：y=ax+b, b是截距。一般推荐使用fit_intercept=True。

如果输入特征包含x1,x2，将特征放入多项式扩展的图纸后，我们会得到一个针对x1,x2扩展的特征集，并把数据输出出来。因此在多项式扩展的算法中，存储的特征集合将是扩展后的结果

	## 可以设置多个管道，放进models里
	models = [
	    Pipeline([
	            ('ss',StandardScaler()),
	            ('Poly',PolynomialFeatures(degree=3)),#给定多项式扩展操作-3阶扩展
	            ('Linear',LinearRegression(fit_intercept=False))
	        ])，
	    Pipeline([
	            ('ss',StandardScaler()),
	            ('Poly',PolynomialFeatures(degree=5)),#给定多项式扩展操作-5阶扩展
	            ('Linear',LinearRegression(fit_intercept=False))
	        ])
	]
	model_0 = models[0] # 获取第一个管道
	model_1 = models[1] # 获取第二个管道


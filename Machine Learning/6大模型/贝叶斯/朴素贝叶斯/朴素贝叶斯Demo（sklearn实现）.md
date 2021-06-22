# 朴素贝叶斯Demo（sklearn实现）

## 代码

	# coding=utf-8
	#朴素贝叶斯sklearn代码实现
	import numpy as np
	from sklearn import naive_bayes as nb
	from sklearn.preprocessing import LabelEncoder
	
	def loaddata():
	    X = np.array([[1,'S'],[1,'M'],[1,'M'],[1,'S'],
	         [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
	         [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
	         [3, 'M'], [3, 'L'], [3, 'L']])
	    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
	    return X, y
	
	X,y = loaddata()
	X[:,1] = LabelEncoder().fit_transform(X[:,1])  #将字母映射成数字  ['2' '1' '1' '2' '2' '2' '1' '1' '0' '0' '0' '1' '1' '0' '0']
	X= X.astype(int) #将上面的字符转成int类型
	model = nb.MultinomialNB() #适用于离散数据的朴素贝叶斯
	
	model.fit(X,y)
	#[2,'S']
	result = model.predict([[2,2]])
	print(result)

运行结果

	[-1]
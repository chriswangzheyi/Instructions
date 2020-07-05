# 线性回归Demo代码

## 正规方程


	import  numpy as np
	from sklearn.model_selection import train_test_split
	from numpy.linalg import  inv
	
	#读取数据
	data = np.loadtxt("aqi1.csv",delimiter=",",skiprows=1,dtype=np.float)
	
	# 增加theta0的feature
	index = np.ones((data.shape[0],1))
	
	# 合并列
	data = np.hstack((data,index))
	
	# 区分x,y
	X=data[:,1:]
	Y=data[:,0]
	
	# 将数据分割为训练集和测试集,训练集80%
	train_x,test_x,train_y,test_y=train_test_split(X,Y,train_size=0.8)
	
	#求theta
	theta =np.dot(  np.dot( inv( np.dot(train_x.T,train_x) ),train_x.T), train_y )
	
	# 预测值
	predict=np.dot(test_x,theta)
	print(predict)



## SKlearn 实现正规方程

	import numpy as np
	from sklearn import  linear_model
	from sklearn.model_selection import train_test_split
	import matplotlib.pyplot as plt
	import pickle
	
	data =np.loadtxt("aqi1.csv",delimiter=",",skiprows=1, dtype=np.float)
	x = data[:,1]
	y = data[:,0]
	
	# 将数据分割为训练集和测试集,训练集80%
	train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.8)
	
	#线性回归模型
	model = linear_model.LinearRegression()
	model.fit(train_x.reshape(-1,1),train_y)
	
	print(model.coef_) # 系数
	print(model.intercept_) # 截距
	
	with open("model.pickel","wb") as f:
	    pickle.dump(model,f)
	
	y_predict = model.predict(test_x.reshape(-1,1))
	
	plt.title('AQI predict')
	plt.xlabel("PM2.5")
	plt.ylabel("AQI")
	plt.scatter(test_x,test_y,c="r")
	plt.plot(test_x,y_predict,c="blue")
	plt.show()

## 调用模型

	import  pickle
	
	with open("model.pickel","rb") as f:
	    model =pickle.load(f)
	
	test = [ [56]]
	
	y_predict = model.predict(test)
	print(y_predict)

## 批量梯度下降算法

	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
	
	#归一化处理
	def featureNomlize(X):
	    mu = X.mean(0) # mean
	    sigma = X.std(0) # 标准差
	    X = (X - mu)/sigma
	    index = np.ones((len(X),1))
	    X = np.hstack((X,index))
	    return X
	
	#读取数据
	def getDataFromFile():
	    data =np.loadtxt("aqi1.csv",delimiter=",",skiprows=1, dtype=np.float)
	    X = data[:,1:] # 特征集
	    Y = data[:,0] # 目标集
	    Y = Y.reshape(-1,1)
	
	    # 数据归一化
	    X= featureNomlize(X)
	    return  X,Y, len(X)
	
	#代价函数
	def computerCost(X,Y,theta):
	    m = len(Y) # 样本集的数量
	    J = np.sum( (np.dot(X,theta)-Y)**2 )/(2*m) # 代价函数
	    return J
	
	#梯度下降
	def gradientDescent(X,Y,theta,alpha,num_iters):
	    J_history = []
	    m = len(X)
	    for i in range(num_iters):
	
	        theta = theta - alpha*np.dot(X.T,(np.dot(X,theta)-Y))/m
	        J = computerCost(X,Y,theta)
	        J_history.append(J)
	        print("第%d次的损失值为：%f"%(i+1,J))  #观察损失值是否越来越小
	
	    return theta,J_history
	
	if __name__ == "__main__":
	    #读取文件
	    X,Y,amount=getDataFromFile()
	    #初始化theta
	    theta = np.ones([X.shape[1],1])
	    #设置初始值
	    alpha =0.01 #学习率
	    num_iters =1000 #迭代次数
	    theta,J_history=gradientDescent(X,Y,theta,alpha,num_iters)
	    print("-------theta-------")
	    print(theta)
	    #预测
	    predict_y= np.dot(X,theta)
	    #评价
	    print("-------判断模型好坏-----------------")
	    print("mae:",mean_absolute_error(Y,predict_y))
	    print("mse:",mean_squared_error(Y,predict_y))
	    print("median-ae:",median_absolute_error(Y,predict_y))
	    print("r2:",r2_score(Y,predict_y))
	    #5.可视化形式展示
	    plt.plot(np.arange(num_iters),J_history,"r")
	    plt.xlabel("iterations")
	    plt.ylabel("cost")
	    plt.show()

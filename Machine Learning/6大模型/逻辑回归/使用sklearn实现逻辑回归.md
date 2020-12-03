# 使用sklearn实现逻辑回归

## 代码

	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import roc_curve
	from sklearn.metrics import precision_score,recall_score
	
	def process_data(path):
	    data = np.loadtxt(path,delimiter=",",skiprows=1,dtype=np.float)
	    X = data[:,:-1]
	    y = data[:,-1]
	
	    mu = X.mean(0)
	    std = X.std(0)
	    X = (X-mu)/std
	    print(X.shape)
	
	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	    return X_train, X_test, y_train, y_test
	
	if __name__ == "__main__":
	    X_train, X_test, y_train, y_test = process_data("pima-indians-diabetes.data.csv")
	
	    logist = LogisticRegression()
	    logist.fit(X_train,y_train)
	    y_predict = logist.predict(X_test)
	    print(y_predict)
	    print("准确率：",np.sum((y_predict == y_test))/len(y_test))
	
	    # 6.评价指标
	    print("精确率评价指标：", precision_score(y_test, y_predict))
	    print("召回率评价指标：", recall_score(y_test, y_predict))
	
	    # 图表展示
	    fpr,tpr,thresholds = roc_curve(y_test,y_predict)
	    plt.rcParams["font.sans-serif"]=["SimHei"]
	    plt.rcParams["axes.unicode_minus"]=False
	    plt.xlim(0,1)#设定x轴的范围
	    plt.ylim(0.0,1.1)#设定y轴的范围
	    plt.title("ROC曲线")
	    plt.xlabel("假正率FPR")
	    plt.ylabel("真阳率TPR")
	    plt.plot(fpr,tpr,linewidth=2,linestyle="-",color="red")
	    plt.show()

## 输出

	(768, 8)
	[1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.
	 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0. 0. 0.
	 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 1. 1. 0. 0. 0. 0. 0.
	 0. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0.
	 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 1. 0. 1. 0. 1.
	 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0.
	 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 1. 1. 1. 0. 0. 1. 0. 0. 0.
	 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0.
	 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
	 0. 0. 1. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0.]
	准确率： 0.7792207792207793
	精确率评价指标： 0.7090909090909091
	召回率评价指标： 0.527027027027027
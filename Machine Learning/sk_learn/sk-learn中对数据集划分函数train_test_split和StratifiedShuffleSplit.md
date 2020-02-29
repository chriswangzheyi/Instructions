# sk-learn中对数据集划分函数train_test_split和StratifiedShuffleSplit

参考资料：https://blog.csdn.net/qq_30815237/article/details/87904205

## 随机划分训练集和测试集train_test_split

rain_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train_data和test_data，形式为：

	from sklearn.model_selection import train_test_split
	#展示不同的调用方式
	 
	train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
	 
	#cross_validation代表交叉验证
	X_train,X_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)

参数解释：

	train_data：所要划分的样本特征集
	train_target：所要划分的样本结果
	test_size：样本占比，如果是整数的话就是样本的数量
	random_state：是随机数的种子。


**随机数种子：**其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。

随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

## StratifiedShuffleSplit

	from sklearn.model_selection import StratifiedShuffleSplit
	StratifiedShuffleSplit(n_splits=10,test_size=None,train_size=None, random_state=None)

参数说明

	参数 n_splits是将训练数据分成train/test对的组数，可根据需要进行设置，默认为10，其实就是讲数据集划分了5次，得到5组(train，test)

**注意每组(train，test)都是包含了所有的数据集。**

	参数test_size和train_size是用来设置(train，test)中train和test所占的比例。例如： 
	
	1.提供10个数据num进行训练和测试集划分 
	2.设置train_size=0.8 test_size=0.2 
	3.train_num=num*train_size=8 test_num=num*test_size=2 
	4.即10个数据，进行划分以后8个是训练数据，2个是测试数据
	
	参数 random_state控制是将样本随机打乱


首先将样本随机打乱，然后根据设置参数划分出train/test对。 其创建的每一组划分将保证每组中类比比例相同。即第一组训练数据的类别比例为2:1，则后面每组类别都满足这个比例。
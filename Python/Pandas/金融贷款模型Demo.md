# 金融贷款模型Demo

## 数据分析

	import pandas as pd
	import re
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.naive_bayes import GaussianNB
	from sklearn.model_selection import KFold
	from sklearn import metrics
	
	# 读取数据
	load_data =pd.read_csv('E:\data\load\LoanStats3a.csv',skiprows=1,low_memory=False)
	
	# 将一列都是空的数据drop掉
	load_data = load_data.dropna(axis=1,how='all') # axis=1 表示一列都是空，How表示把所有的数据都drop掉
	
	# 数据特征选取
	load_data_clean = load_data[['loan_amnt','funded_amnt','term','int_rate',
	                             'installment','emp_length','dti',
	                             'annual_inc','total_pymnt','total_pymnt_inv',
	                             'total_rec_int','loan_status']]
	
	# 对一列只提取数字
	def extract_number(string):
	    num = re.findall('\\d+',str(string))  # re.findall  是一个正则表达式：findall(pattern, string, flags=0) \d+ means one or more digit [0-9] (depending on LOCALE)
	    if len(num)>0:
	       return int(num[0])
	    else:
	        return 0
	
	# 避免浅拷贝营销load_data_clean，将原始数据删除
	del load_data
	
	#将对应的列提取
	load_data_clean.emp_length = load_data_clean.emp_length.apply(extract_number)
	load_data_clean.term = load_data_clean.term.apply(extract_number)
	
	#将String去掉百分号转为float
	load_data_clean.int_rate = load_data_clean.int_rate.apply(lambda x: float( str(x).replace('%','') ) )
	
	
	###########################
	
	### 数据分析方法一 直接plot所有要素：
	
	# 显示需取消代码注销
	#load_data_clean.plot(figsize=(15,6)) # 可以看到某些数据有异常
	
	
	### 数据处理方法二：使用箱线图
	
	#箱形图（Box-plot）又称为盒须图、盒式图或箱线图，是一种用作显示一组数据分散情况资料的统计图。
	# 因形状如箱子而得名。在各种领域也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，
	# 还可以进行多组数据分布特征的比 较。
	# 显示需取消代码注销
	#sns.boxenplot(x=load_data_clean.loan_amnt)  # 由箱线图可以得知，异常数据占比很少，可以不处理
	
	### 数据处理方法三 使用散点图
	
	# 显示需取消代码注销
	# plt.figure(figsize=(15,6))
	# plt.scatter(load_data_clean.loan_amnt,load_data_clean.installment) # 绘制散点图,可以看到是正相关
	# plt.xlabel('loan_amnt')
	# plt.ylabel('installment')
	
	### 数据处理方法四：使用合并要素
	
	 ## 可以看到loan_status元素有多个要素占比很少
	
	def correct_label(string):
	    if 'not' in str(string) or not str(string):
	        return 'reject'
	    else:
	        return  str(string).replace(' ','_')
	
	load_data_clean.loan_status = load_data_clean.loan_status.apply(correct_label)
	load_data_clean.loan_status.value_counts().plot(kind='pie')
	
	plt.show()


## 数据预测及模型评估

	import pandas as pd
	import re
	from sklearn.naive_bayes import GaussianNB
	from sklearn.model_selection import train_test_split, StratifiedKFold
	from sklearn import metrics
	
	# 读取数据
	load_data =pd.read_csv('E:\data\load\LoanStats3a.csv',skiprows=1,low_memory=False)
	
	# 将一列都是空的数据drop掉
	load_data = load_data.dropna(axis=0,how='all') # axis=0 表示一列都是空，How表示把所有的数据都drop掉
	load_data = load_data.dropna(axis=1,how='all') # axis=1 表示一列都是空，How表示把所有的数据都drop掉
	
	# 数据特征选取
	load_data_clean = load_data[['loan_amnt','funded_amnt','term','int_rate',
	                             'installment','emp_length','dti',
	                             'annual_inc','total_pymnt','total_pymnt_inv',
	                             'total_rec_int','loan_status']]
	
	# 对一列只提取数字
	def extract_number(string):
	    num = re.findall('\\d+',str(string))  # re.findall  是一个正则表达式：findall(pattern, string, flags=0) \d+ means one or more digit [0-9] (depending on LOCALE)
	    if len(num)>0:
	       return int(num[0])
	    else:
	        return 0
	
	# 避免浅拷贝营销load_data_clean，将原始数据删除
	del load_data
	
	#将对应的列提取
	load_data_clean.emp_length = load_data_clean.emp_length.apply(extract_number)
	load_data_clean.term = load_data_clean.term.apply(extract_number)
	
	#将String去掉百分号转为float
	load_data_clean.int_rate = load_data_clean.int_rate.apply(lambda x: float( str(x).replace('%','') ) )
	
	#去掉占比很低的要素
	def correct_label(string):
	    if 'not' in str(string) or not str(string):
	        return 'reject'
	    else:
	        return  str(string).replace(' ','_')
	
	load_data_clean.loan_status = load_data_clean.loan_status.apply(correct_label)
	
	## 删除特殊行
	load_data_clean = load_data_clean.drop(39788)
	load_data_clean = load_data_clean.drop(42453)
	load_data_clean = load_data_clean.drop(42536)
	load_data_clean = load_data_clean.drop(42483)
	load_data_clean = load_data_clean.drop(42540)
	load_data_clean = load_data_clean.drop(42541)
	load_data_clean = load_data_clean.drop(42452)
	
	#### 建模
	
	Y = load_data_clean.loan_status   #是否还款
	X = load_data_clean.drop('loan_status', axis=1)
	
	#print(X.isnull().any())
	#test =   X[X.annual_inc.isnull()]
	#test.to_csv('E:/1.csv')
	
	# test_size是测试集占比，random_state：随机种子数
	X_train, X_test, Y_train,Y_test =train_test_split(X,Y,test_size=0.3,random_state=0)
	
	#产生模型
	gnb_model = GaussianNB()
	gnb_model.fit(X_train, Y_train)
	
	#验证
	predict = gnb_model.predict(X_test)  #根据x_test，推测是否会还款 Y
	ans = metrics.classification_report(Y_test, predict)
	print(ans)



输出：
	
	              precision    recall  f1-score   support
	
	 Charged_Off       0.69      0.26      0.38      1782
	  Fully_Paid       0.87      0.86      0.86     10177
	      reject       0.17      0.43      0.25       801
	
	    accuracy                           0.75     12760
	   macro avg       0.58      0.52      0.50     12760
	weighted avg       0.80      0.75      0.76     12760


precision: 表示准确率
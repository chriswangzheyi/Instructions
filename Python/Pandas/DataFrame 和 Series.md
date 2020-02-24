# DataFrame 和 Series

参考资料：https://geektutu.com/post/pandas-dataframe-series.html

## DataFrame

DataFrame可以理解为Excel中的一张表

### 创建DataFrame


#### 从字典中创建

	import pandas as pd
	
	# 不指定 index
	df = pd.DataFrame({'单价': [100, 200, 30], '数量': [3, 3, 10]})
	print("不指定索引")
	print(df)
	
	# 指定 index
	df = pd.DataFrame({'单价': [100, 200, 30], '数量': [3, 3, 10]}, index=['T001', 'T002', 'T003'])
	print("指定索引")
	print(df)

输出：

	不指定索引
	    单价  数量
	0  100   3
	1  200   3
	2   30  10
	指定索引
	       单价  数量
	T001  100   3
	T002  200   3
	T003   30  10


#### 通过Series创建

	import pandas as pd
	
	price_series = pd.Series([100, 200, 30], index=['T001', 'T002', 'T005'])
	quantity_series = pd.Series([3, 3, 10, 2], index=['T001', 'T002', 'T003', 'T004'])
	df = pd.DataFrame({'单价': price_series, '数量': quantity_series})
	
	print(df)

#### 从Excel文件中读取

	df = pd.read_excel("path/demo.xlsx", sheetname=0)
	# 指定 sheetname
	df = pd.read_excel("path/demo.xlsx", sheetname='销售记录')
	# 带分隔符
	df = pd.read_csv('demo.dat', delimiter='|') # csv默认是逗号分隔的，如果不是，需要指定delimiter

输出：

	         单价    数量
	T001  100.0   3.0
	T002  200.0   3.0
	T003    NaN  10.0
	T004    NaN   2.0
	T005   30.0   NaN

### 获取列与行

	df['标签'] # 按行
	df[ ['标签','标签'] ]  #按行与列
	df.iloc[0] # 按行号获取


### 修改

	df.要素 = 赋值


### 删除

	del df['日期']



## Series

Series可以理解为一维数组，它和一维数组的区别，在于Series具有索引。

### 创建Series

#### 默认索引

	import pandas as pd
	
	money_series = pd.Series([200, 300, 10, 5], name="money") # 未设置索引的情况下，自动从0开始生成
	print("原始数据：")
	print(money_series)
	print("按索引取数：")
	print(money_series[0])  # 根据索引获取具体的值
	money_series = money_series.sort_values() # 根据值进行排序，排序后索引与值对应关系不变
	print("按值排序：")
	print(money_series)


输出：

	原始数据：
	0    200
	1    300
	2     10
	3      5
	Name: money, dtype: int64
	按索引取数：
	200
	按值排序：
	3      5
	2     10
	0    200
	1    300
	Name: money, dtype: int64

#### 自定义索引

	import pandas as pd
	
	money_series = pd.Series([200, 300, 10, 5], index=['d', 'c', 'b', 'a'], name='money')
	print("原始数据:")
	print(money_series)
	money_series['a'] # 根据索引获取具体的值
	print("根据索引取值:")
	print(money_series['a'])
	money_series = money_series.sort_index() # 根据索引排序
	print("根据索引排序:")
	print(money_series)

输出：

	原始数据:
	d    200
	c    300
	b     10
	a      5
	Name: money, dtype: int64
	根据索引取值:
	5
	根据索引排序:
	a      5
	b     10
	c    300
	d    200
	Name: money, dtype: int64

### 切片与取值


#### 根据索引

	import pandas as pd
	
	money_series = pd.Series({'d': 200, 'c': 300, 'b': 10, 'a': 5}, name='money')
	print("原始数据：")
	print(money_series)
	
	print("根据loc取值:")
	print(money_series.loc['a'])  # 等价于 money_series['a']
	
	print("倒序输出:")
	print(money_series.loc['c':'a':-1]) # 从c取到 a，倒序
	
	money_series.loc[['d', 'a']] # d, a的值，等价于 money_series[['d', 'a']]
	print("根据loc取多个值：")
	print(money_series)

输出：

	原始数据：
	d    200
	c    300
	b     10
	a      5
	Name: money, dtype: int64
	根据loc取值:
	5
	倒序输出:
	Series([], Name: money, dtype: int64)
	根据loc取多个值：
	d    200
	c    300
	b     10
	a      5
	Name: money, dtype: int64

#### 根据序号

	import pandas as pd
	
	money_series = pd.Series({'d': 200, 'c': 300, 'b': 10, 'a': 5}, name='money')
	print("根据标签取值：")
	print( money_series.iloc[0] ) # 根据标签取值
	
	print("根据标签取值(范围)：")
	print( money_series.iloc[1:3] ) # 根据序号取值，不包含结束，等价于 money_series[1:3]
	
	print("根据标签取值(按位数)：")
	print( money_series.iloc[[3, 0]] ) # 取第三个值和第一个值

输出：

	根据标签取值：
	200
	根据标签取值(范围)：
	c    300
	b     10
	Name: money, dtype: int64
	根据标签取值(按位数)：
	a      5
	d    200
	Name: money, dtype: int64


#### 根据条件

	import pandas as pd
	
	money_series = pd.Series({'d': 200, 'c': 300, 'b': 10, 'a': 5}, name='money')
	
	print(money_series[money_series > 50]) # 选取大于50的值
	print(money_series[lambda x: x ** 2 > 50]) # 选取值平方大于50的值

输出：

	根据标签取值：
	200
	根据标签取值(范围)：
	c    300
	b     10
	Name: money, dtype: int64
	根据标签取值(按位数)：
	a      5
	d    200
	Name: money, dtype: int64
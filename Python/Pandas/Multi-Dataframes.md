# Multi-Dataframes

## Apply

批量执行某个操作

例子：

	import pandas as pd
	
	data = pd.read_csv('E://olympics.csv',skiprows=4)
	print ( data.Athlete.apply(str.lower).head() )
	#把所有对象变为小写


## Concat

pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

参数含义

 

	objs：Series，DataFrame或Panel对象的序列或映射。如果传递了dict，则排序的键将用作键参数，除非它被传递，在这种情况下，将选择值（见下文）。任何无对象将被静默删除，除非它们都是无，在这种情况下将引发一个ValueError。
	axis：{0,1，...}，默认为0。沿着连接的轴。
	join：{'inner'，'outer'}，默认为“outer”。如何处理其他轴上的索引。outer为联合和inner为交集。
	ignore_index：boolean，default False。如果为True，请不要使用并置轴上的索引值。结果轴将被标记为0，...，n-1。如果要连接其中并置轴没有有意义的索引信息的对象，这将非常有用。注意，其他轴上的索引值在连接中仍然受到尊重。
	join_axes：Index对象列表。用于其他n-1轴的特定索引，而不是执行内部/外部设置逻辑。
	keys：序列，默认值无。使用传递的键作为最外层构建层次索引。如果为多索引，应该使用元组。
	levels：序列列表，默认值无。用于构建MultiIndex的特定级别（唯一值）。否则，它们将从键推断。
	names：list，default无。结果层次索引中的级别的名称。
	verify_integrity：boolean，default False。检查新连接的轴是否包含重复项。这相对于实际的数据串联可能是非常昂贵的。
	copy：boolean，default True。如果为False，请勿不必要地复制数据


例如：

	result = pd.concat([df1, df4], axis=1)


## Merge

	import pandas as pd
	
	data = pd.read_csv('E://olympics.csv', skiprows=4)
	country = pd.read_csv('E://countrys.csv')
	
	ans = pd.merge(data,country, left_on='NOC', right_on='Int Olympic Committee code',how='right')
	print(ans.tail())


merge 中可以定义连接方式：

1. left
2. right
3. outer
4. inner


## column 重命名

	ans = ans.rename(index=str, columns={'Int Olympic Committee code':'NOC'})
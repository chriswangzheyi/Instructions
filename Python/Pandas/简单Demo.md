# 简单Demo

## 数据源

olympics.csv

## 统计类题目

### Q1

In which events did Jesse Owens win a medal?


### Q2

Which country has won men's gold medals in singles badminton over the years? Sort the results alphabetically by the player's names.



### Q3

Which three countries have won the most medals in recent year (1984 to 2008)?


### Q4

Display the male gold medal winners for the 100m sprint event over the years. List results starting with the most recent. Show the Olympic City, Edition, Athlete and the country they reresent.


### Q1-Q4解答

	import pandas as pd
	
	data = pd.read_csv('E://olympics.csv',skiprows=4)
	
	q1= data['Event'] [(data.Athlete=='OWENS, Jesse')].value_counts()
	
	q2 = data[ ['NOC','Athlete'] ][ (data.Event_gender=='M' ) & (data.Sport=='Badminton') & (data.Event=='singles') & (data.Medal=='Gold')].sort_values(by='Athlete', ascending= True)
	
	q3 = data[ 'NOC'][ (data.Edition>=1984) & (data.Edition<=2008) & (data.Medal=='Gold') ].value_counts().sort_values(ascending=False).head(3)
	
	q4 = data [ ['City','Edition','Athlete','NOC'] ][ (data.Event=='100m') & (data.Medal=='Gold') ].sort_values(by='Edition', ascending= False)
	
	print(q4)


### Q5

Which countries did not win a medal in the 2008 Olympic Games? 

### Q5解答

	import pandas as pd
	
	# 导入数据
	data = pd.read_csv('E://olympics.csv',skiprows=4)
	country = pd.read_csv('E://countrys.csv')
	
	# data表2008年数据
	bj =data[ data.Edition==2008 ]
	
	# 各国
	madel_2008 = bj.NOC.value_counts()
	
	
	# country表将国家简写作为索引
	country.set_index('Int Olympic Committee code',inplace=True)
	
	# 在country dataframe新增madel2008栏
	country['madel2008']= madel_2008
	
	country_no_madel= country[ country.madel2008.isnull() ]
	print(country_no_madel)



## 画图类题目

### Q1：

Plot the number of medals achieved by the Chinese team (men and women) in Beijing 2008？ Using matplotlib and Seaborn


### Q1解答

	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	
	data = pd.read_csv('E:/olympics.csv',skiprows=4)
	
	# matplotlib
	#data.Gender[ (data.Edition==2008) & (data.NOC=='CHN')].value_counts().plot(kind='bar')
	
	# seaborn
	sns.countplot(x='Gender', data=data, palette='bwr')
	
	plt.show()


### Q2:

Plot the number of Gold, Silver and Bronze medals for each gender.


### Q2解答

	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	
	data = pd.read_csv('E:/olympics.csv',skiprows=4)
	
	# seaborn
	sns.countplot(x='Medal', data=data, hue='Gender', palette='bwr')
	
	plt.show()



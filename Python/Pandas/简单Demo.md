# 简单Demo

## 数据源

olympics.csv

## 题目

### Q1

In which events did Jesse Owens win a medal?


### Q2

Which country has won men's gold medals in singles badminton over the years? Sort the results alphabetically by the player's names.



### Q3

Which three countries have won the most medals in recent year (1984 to 2008)?


### Q4

Display the male gold medal winners for the 100m sprint event over the years. List results starting with the most recent. Show the Olympic City, Edition, Athlete and the country they reresent.


## 解答

	import pandas as pd
	
	data = pd.read_csv('E://olympics.csv',skiprows=4)
	
	q1= data['Event'] [(data.Athlete=='OWENS, Jesse')].value_counts()
	
	q2 = data[ ['NOC','Athlete'] ][ (data.Event_gender=='M' ) & (data.Sport=='Badminton') & (data.Event=='singles') & (data.Medal=='Gold')].sort_values(by='Athlete', ascending= True)
	
	q3 = data[ 'NOC'][ (data.Edition>=1984) & (data.Edition<=2008) & (data.Medal=='Gold') ].value_counts().sort_values(ascending=False).head(3)
	
	q4 = data [ ['City','Edition','Athlete','NOC'] ][ (data.Event=='100m') & (data.Medal=='Gold') ].sort_values(by='Edition', ascending= False)
	
	print(q4)


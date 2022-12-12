# ParitionBy Vs Repartition

参考：https://www.youtube.com/watch?v=tIEg3TlJaQk

## 前提条件
上传文件 personal_transactions.csv到databricks上

![](Images/19.png)

## 代码

### 导入数据

	val df = spark.read.option("header", "true").format("csv").load("/FileStore/tables/personal_transactions-3.csv")
	df.show()

结果

	+-----------+-------------+---------+-------------------+----------------+-------+
	|Customer_No|    Card_type|     Date|           Category|Transaction Type| Amount|
	+-----------+-------------+---------+-------------------+----------------+-------+
	|    1000501|Platinum Card| 1/1/2018|           Shopping|           debit|  11.11|
	|    1000501|     Checking| 1/2/2018|    Mortgage & Rent|           debit|1247.44|
	|    1000501|  Silver Card| 1/2/2018|        Restaurants|           debit|  24.22|
	|    1000501|Platinum Card| 1/3/2018|Credit Card Payment|          credit|2298.09|
	|    1000501|Platinum Card| 1/4/2018|      Movies & DVDs|           debit|  11.76|
	|    1000501|  Silver Card| 1/5/2018|        Restaurants|           debit|  25.85|
	|    1000501|  Silver Card| 1/6/2018|   Home Improvement|           debit|  18.45|
	|    1000501|     Checking| 1/8/2018|          Utilities|           debit|     45|
	|    1000501|  Silver Card| 1/8/2018|   Home Improvement|           debit|  15.38|
	|    1000501|Platinum Card| 1/9/2018|              Music|           debit|  10.69|
	|    1000501|     Checking|1/10/2018|       Mobile Phone|           debit|  89.46|
	|    1000501|Platinum Card|1/11/2018|         Gas & Fuel|           debit|  34.87|
	|    1000501|Platinum Card|1/11/2018|          Groceries|           debit|  43.54|
	|    1000501|     Checking|1/12/2018|           Paycheck|          credit|   2000|
	|    1000531|Platinum Card|1/13/2018|          Fast Food|           debit|  32.91|
	|    1000531|Platinum Card|1/13/2018|           Shopping|           debit|  39.05|
	|    1000531|  Silver Card|1/15/2018|          Groceries|           debit|  44.19|
	|    1000531|  Silver Card|1/15/2018|        Restaurants|           debit|  64.11|
	|    1000531|     Checking|1/16/2018|          Utilities|           debit|     35|
	|    1000531|     Checking|1/16/2018|          Utilities|           debit|     60|
	+-----------+-------------+---------+-------------------+----------------+-------+

查看现在的partition

	df.rdd.partitions.size
	res7: Int = 1
	
	
	
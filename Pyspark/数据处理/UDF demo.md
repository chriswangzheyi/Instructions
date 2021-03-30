# UDF demo

## 定义

UDF 是用户自定义函数。被广泛用于数据处理，以便对DataFrame应用某些转换。Pyspark提供两种类型的UDF：Conventional UDF 和 Pandas UDF。


##  UDF

### 传统的python函数

	from pyspark.sql import SparkSession
	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType
	
	
	def price_range(brand):
	    if brand in ["Samsung", "Apple"]:
	        return 'High Price'
	    elif brand == 'MI':
	        return "Middle Price"
	    else:
	        return 'Low Price'
	
	if __name__ == '__main__':
	    spark = SparkSession.builder.appName("data_processing").getOrCreate()
	    df = spark.read.csv(
	        "/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_2_Data_Processing/sample_data.csv",
	        inferSchema=True, header=True)
	
	    print(df.show())
	
	    brand_udf = udf(price_range, StringType())
	
	    ans = df.withColumn('price_range', brand_udf(df['mobile'])).show(10, False)

    print(ans)

#### 显示

	+-------+---+----------+------+-------+
	|ratings|age|experience|family| mobile|
	+-------+---+----------+------+-------+
	|      3| 32|       9.0|     3|   Vivo|
	|      3| 27|      13.0|     3|  Apple|
	|      4| 22|       2.5|     0|Samsung|
	|      4| 37|      16.5|     4|  Apple|
	|      5| 27|       9.0|     1|     MI|
	|      4| 27|       9.0|     0|   Oppo|
	|      5| 37|      23.0|     5|   Vivo|
	|      5| 37|      23.0|     5|Samsung|
	|      3| 22|       2.5|     0|  Apple|
	|      3| 27|       6.0|     0|     MI|
	|      2| 27|       6.0|     2|   Oppo|
	|      5| 27|       6.0|     2|Samsung|
	|      3| 37|      16.5|     5|  Apple|
	|      5| 27|       6.0|     0|     MI|
	|      4| 22|       6.0|     1|   Oppo|
	|      4| 37|       9.0|     2|Samsung|
	|      4| 27|       6.0|     1|  Apple|
	|      1| 37|      23.0|     5|     MI|
	|      2| 42|      23.0|     2|   Oppo|
	|      4| 37|       6.0|     0|   Vivo|
	+-------+---+----------+------+-------+
	
	
    +-------+---+----------+------+-------+------------+
	|ratings|age|experience|family|mobile |price_range |
	+-------+---+----------+------+-------+------------+
	|3      |32 |9.0       |3     |Vivo   |Low Price   |
	|3      |27 |13.0      |3     |Apple  |High Price  |
	|4      |22 |2.5       |0     |Samsung|High Price  |
	|4      |37 |16.5      |4     |Apple  |High Price  |
	|5      |27 |9.0       |1     |MI     |Middle Price|
	|4      |27 |9.0       |0     |Oppo   |Low Price   |
	|5      |37 |23.0      |5     |Vivo   |Low Price   |
	|5      |37 |23.0      |5     |Samsung|High Price  |
	|3      |22 |2.5       |0     |Apple  |High Price  |
	|3      |27 |6.0       |0     |MI     |Middle Price|
	+-------+---+----------+------+-------+------------+


### lambda 函数

	from pyspark.sql import SparkSession
	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType
	
	if __name__ == '__main__':
	    spark = SparkSession.builder.appName("data_processing").getOrCreate()
	    df = spark.read.csv(
	        "/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_2_Data_Processing/sample_data.csv",
	        inferSchema=True, header=True)
	
	    print(df.show())
	
	    age_udf = udf (lambda age: "young" if age <=30 else "senior", StringType())
	
	    ans = df.withColumn("age_group",age_udf(df.age)).show(10,False)
	
	    print(ans)
	    
	    
#### 显示

	+-------+---+----------+------+-------+
	|ratings|age|experience|family| mobile|
	+-------+---+----------+------+-------+
	|      3| 32|       9.0|     3|   Vivo|
	|      3| 27|      13.0|     3|  Apple|
	|      4| 22|       2.5|     0|Samsung|
	|      4| 37|      16.5|     4|  Apple|
	|      5| 27|       9.0|     1|     MI|
	|      4| 27|       9.0|     0|   Oppo|
	|      5| 37|      23.0|     5|   Vivo|
	|      5| 37|      23.0|     5|Samsung|
	|      3| 22|       2.5|     0|  Apple|
	|      3| 27|       6.0|     0|     MI|
	|      2| 27|       6.0|     2|   Oppo|
	|      5| 27|       6.0|     2|Samsung|
	|      3| 37|      16.5|     5|  Apple|
	|      5| 27|       6.0|     0|     MI|
	|      4| 22|       6.0|     1|   Oppo|
	|      4| 37|       9.0|     2|Samsung|
	|      4| 27|       6.0|     1|  Apple|
	|      1| 37|      23.0|     5|     MI|
	|      2| 42|      23.0|     2|   Oppo|
	|      4| 37|       6.0|     0|   Vivo|
	+-------+---+----------+------+-------+



	+-------+---+----------+------+-------+---------+
	|ratings|age|experience|family|mobile |age_group|
	+-------+---+----------+------+-------+---------+
	|3      |32 |9.0       |3     |Vivo   |senior   |
	|3      |27 |13.0      |3     |Apple  |young    |
	|4      |22 |2.5       |0     |Samsung|young    |
	|4      |37 |16.5      |4     |Apple  |senior   |
	|5      |27 |9.0       |1     |MI     |young    |
	|4      |27 |9.0       |0     |Oppo   |young    |
	|5      |37 |23.0      |5     |Vivo   |senior   |
	|5      |37 |23.0      |5     |Samsung|senior   |
	|3      |22 |2.5       |0     |Apple  |young    |
	|3      |27 |6.0       |0     |MI     |young    |
	+-------+---+----------+------+-------+---------+


## pandas udf

### 向量化UDF

	from pyspark.sql import SparkSession
	from pyspark.sql.functions  import pandas_udf
	from pyspark.sql.types import IntegerType
	
	
	def remaining_yrs(age):
	    yrs_left = (100-age)
	    return yrs_left
	
	
	if __name__ == '__main__':
	    spark = SparkSession.builder.appName("data_processing").getOrCreate()
	    df = spark.read.csv(
	        "/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_2_Data_Processing/sample_data.csv",
	        inferSchema=True, header=True)
	
	    print(df.show())
	
	    length_udf = pandas_udf(remaining_yrs, IntegerType())
	
	    ans = df.withColumn("yrs_left", length_udf(df['age'])).show(10,False)
	
	    print(ans)


#### 显示

	+-------+---+----------+------+-------+
	|ratings|age|experience|family| mobile|
	+-------+---+----------+------+-------+
	|      3| 32|       9.0|     3|   Vivo|
	|      3| 27|      13.0|     3|  Apple|
	|      4| 22|       2.5|     0|Samsung|
	|      4| 37|      16.5|     4|  Apple|
	|      5| 27|       9.0|     1|     MI|
	|      4| 27|       9.0|     0|   Oppo|
	|      5| 37|      23.0|     5|   Vivo|
	|      5| 37|      23.0|     5|Samsung|
	|      3| 22|       2.5|     0|  Apple|
	|      3| 27|       6.0|     0|     MI|
	|      2| 27|       6.0|     2|   Oppo|
	|      5| 27|       6.0|     2|Samsung|
	|      3| 37|      16.5|     5|  Apple|
	|      5| 27|       6.0|     0|     MI|
	|      4| 22|       6.0|     1|   Oppo|
	|      4| 37|       9.0|     2|Samsung|
	|      4| 27|       6.0|     1|  Apple|
	|      1| 37|      23.0|     5|     MI|
	|      2| 42|      23.0|     2|   Oppo|
	|      4| 37|       6.0|     0|   Vivo|
	+-------+---+----------+------+-------+


	+-------+---+----------+------+-------+--------+
	|ratings|age|experience|family|mobile |yrs_left|
	+-------+---+----------+------+-------+--------+
	|3      |32 |9.0       |3     |Vivo   |68      |
	|3      |27 |13.0      |3     |Apple  |73      |
	|4      |22 |2.5       |0     |Samsung|78      |
	|4      |37 |16.5      |4     |Apple  |63      |
	|5      |27 |9.0       |1     |MI     |73      |
	|4      |27 |9.0       |0     |Oppo   |73      |
	|5      |37 |23.0      |5     |Vivo   |63      |
	|5      |37 |23.0      |5     |Samsung|63      |
	|3      |22 |2.5       |0     |Apple  |78      |
	|3      |27 |6.0       |0     |MI     |73      |
	+-------+---+----------+------+-------+--------+



### pandas udf 多列

	from pyspark.sql import SparkSession
	from pyspark.sql.functions  import pandas_udf
	from pyspark.sql.types import IntegerType, DoubleType
	
	
	def prod(rating, exp):
	    x= rating*exp
	    return  x
	
	
	if __name__ == '__main__':
	    spark = SparkSession.builder.appName("data_processing").getOrCreate()
	    df = spark.read.csv(
	        "/Users/zheyiwang/Downloads/machine-learning-with-pyspark-master/chapter_2_Data_Processing/sample_data.csv",
	        inferSchema=True, header=True)
	
	    print(df.show())
	
	    prod_udf = pandas_udf(prod,DoubleType())
	
	    ans = df.withColumn("product", prod_udf(df['ratings'], df['experience'])).show(10,False)
	
	    print(ans)

#### 显示

	+-------+---+----------+------+-------+
	|ratings|age|experience|family| mobile|
	+-------+---+----------+------+-------+
	|      3| 32|       9.0|     3|   Vivo|
	|      3| 27|      13.0|     3|  Apple|
	|      4| 22|       2.5|     0|Samsung|
	|      4| 37|      16.5|     4|  Apple|
	|      5| 27|       9.0|     1|     MI|
	|      4| 27|       9.0|     0|   Oppo|
	|      5| 37|      23.0|     5|   Vivo|
	|      5| 37|      23.0|     5|Samsung|
	|      3| 22|       2.5|     0|  Apple|
	|      3| 27|       6.0|     0|     MI|
	|      2| 27|       6.0|     2|   Oppo|
	|      5| 27|       6.0|     2|Samsung|
	|      3| 37|      16.5|     5|  Apple|
	|      5| 27|       6.0|     0|     MI|
	|      4| 22|       6.0|     1|   Oppo|
	|      4| 37|       9.0|     2|Samsung|
	|      4| 27|       6.0|     1|  Apple|
	|      1| 37|      23.0|     5|     MI|
	|      2| 42|      23.0|     2|   Oppo|
	|      4| 37|       6.0|     0|   Vivo|
	+-------+---+----------+------+-------+
	
	
	+-------+---+----------+------+-------+-------+
	|ratings|age|experience|family|mobile |product|
	+-------+---+----------+------+-------+-------+
	|3      |32 |9.0       |3     |Vivo   |27.0   |
	|3      |27 |13.0      |3     |Apple  |39.0   |
	|4      |22 |2.5       |0     |Samsung|10.0   |
	|4      |37 |16.5      |4     |Apple  |66.0   |
	|5      |27 |9.0       |1     |MI     |45.0   |
	|4      |27 |9.0       |0     |Oppo   |36.0   |
	|5      |37 |23.0      |5     |Vivo   |115.0  |
	|5      |37 |23.0      |5     |Samsung|115.0  |
	|3      |22 |2.5       |0     |Apple  |7.5    |
	|3      |27 |6.0       |0     |MI     |18.0   |
	+-------+---+----------+------+-------+-------+
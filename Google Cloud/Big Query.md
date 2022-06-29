# Big Query

## 特性

###Column Storage，
背后对应的格式是Capacitor。类似于Hadoop/Spark生态圈中的Parquet

###Pay as you go，
这个是一个非常大的优势，你不再需要自己维护和组建集群，也不同于BigTable或者很多云服务商的Hadoop/Hive方案，需要自己运维，并且起步就是至少3台机器。BigQuery完全按照数据的存储量，以及Query扫描的数据量来计费。这个对于大量技术强，但是团队小的公司非常有利，一方面，你不需要早期先搞个5-6台机器就能有一个基本可以无限水平扩展的数据方案。

###高吞吐量
100K/s级别的写入TPS，并且可以做到在数据写入后几秒到几分钟级别的情况下，就能Query出来。我已经有一段时间没有关注过Hive了，但是在2年前来说，传统的Hive之类的方案这样做并不方便。遵循SQL Standard，还支持Nested的Struct以及Array，这个其实是Dremel Paper中的基本实现，Parquet之类也都支持。

## 产品角度

### 权限管理
通过View和Permission的管理，容易做细粒度的数据权限管理。

### 分享机制
通过Save Query + Share，方便数据分析团队做基本的分享和培训。

### 数据分析
配合Dataflow/Beam，Pub/Sub，可以快速搭建起一个Lambda架构的数据分析框架。

### 生态
能够直接和Google Analytics 360以及一系列的Google的相关产品Ads/Youtube无缝集成。这意味着，你买了Google Analytics 360的Premium服务，你可以直接自动把所有通过GA追踪的Session/Event的数据同步到BigQuery，自己做自定义的分析，不局限于Google Analytics已经给了你的功能。


## 局限性

### 不支持OLTP
BigQuery并不处理任何OLTP类型的数据处理，所以不要直接拿来替代MySQL。

### 跨区传输数据
BigQuery并没有做全球性质的数据同步处理，创建BigQuery的Dataset的时候，还是要选择US/EU两个Region之一，现在也还不能创建Asia区域的Dataset。



## Demo

	SELECT COUNT(*) AS cnt, country_code
	FROM (
	  SELECT ANY_VALUE(country_code) AS country_code
	  FROM `patents-public-data.patents.publications` AS pubs
	  GROUP BY application_number
	)
	GROUP BY country_code
	ORDER BY cnt DESC

### 如何导出分析结果

在分析表格的右上角有四个按钮，分别是Download as CSV,Download as JSON,Save As Table,Save As Google Sheet一般来讲，普通的分析使用Download as CSV即可。CSV文件，即逗号分隔文件，可以用EXCEL直接打开。如果想使用编程的方式对结果进行分析或者在Web端对数据进行呈现，也可以方便的使用Download as JSON。


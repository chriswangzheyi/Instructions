**采用技术**

Spring AOP来切换数据源

**技术原理**
继承AbstractRoutingDataSource重写determineCurrentLoookupKey()方法，来决定使用哪个数据库。在开启事务之前，通过改变lookupKey来达到切换数据源的目的。

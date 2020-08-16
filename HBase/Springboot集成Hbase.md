# Springboot集成Hbase


## 在windows运行的时候

需要增加ip 映射

	47.112.142.231 wangzheyi 


## 项目结构

![](../Images/4.png)

## 代码

### HbaseConfig

	package com.wzy.springboot_hbase.config;
	
	import org.apache.hadoop.hbase.HBaseConfiguration;
	import org.springframework.boot.context.properties.EnableConfigurationProperties;
	import org.springframework.context.annotation.Bean;
	import org.springframework.context.annotation.Configuration;
	
	import java.util.Map;
	import java.util.Set;
	
	@Configuration
	@EnableConfigurationProperties(HbaseProperties.class)
	public class HbaseConfig {
	
	    private final HbaseProperties properties;
	
	    public HbaseConfig(HbaseProperties properties) {
	        this.properties = properties;
	    }
	
	    @Bean
	    public org.apache.hadoop.conf.Configuration configuration() {
	
	        org.apache.hadoop.conf.Configuration configuration = HBaseConfiguration.create();
	
	        Map<String, String> config = properties.getConfig();
	
	        Set<String> keySet = config.keySet();
	        for (String key : keySet) {
	            configuration.set(key, config.get(key));
	        }
	        return configuration;
	    }
	
	}

### HBaseClient


	package com.wzy.springboot_hbase.utils;
	
	import com.wzy.springboot_hbase.config.HbaseConfig;
	import lombok.extern.slf4j.Slf4j;
	import org.apache.commons.lang3.StringUtils;
	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.hbase.*;
	import org.apache.hadoop.hbase.client.*;
	import org.apache.hadoop.hbase.filter.FilterList;
	import org.apache.hadoop.hbase.filter.PageFilter;
	import org.apache.hadoop.hbase.filter.SingleColumnValueFilter;
	import org.apache.hadoop.hbase.util.Bytes;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.context.annotation.DependsOn;
	import org.springframework.stereotype.Component;
	
	import javax.annotation.PostConstruct;
	import java.io.IOException;
	import java.util.*;
	
	@Slf4j
	@Component
	@DependsOn("hbaseConfig")
	public class HBaseClient {
	
	    @Autowired
	    private HbaseConfig config;
	
	    private static Admin admin = null;
	
	    public static Configuration conf=null;
	
	    private static Connection connection = null;
	
	    private ThreadLocal<List<Put>> threadLocal = new ThreadLocal<List<Put>>();
	
	    private static final int CACHE_LIST_SIZE = 1000;
	
	
	
	    @PostConstruct
	    private void init() {
	        if (connection != null) {
	            return;
	        }
	
	        try {
	            connection = ConnectionFactory.createConnection(config.configuration());
	            admin = connection.getAdmin();
	        } catch (IOException e) {
	            log.error("HBase create connection failed: {}", e);
	        }
	    }
	
	    /**
	     * 创建表
	     * experssion : create 'tableName','[Column Family 1]','[Column Family 2]'
	     * @param tableName 	  表名
	     * @param columnFamilies 列族名
	     * @throws IOException
	     */
	    public void createTable(String tableName, String... columnFamilies) throws IOException {
	        TableName name = TableName.valueOf(tableName);
	        boolean isExists = this.tableExists(tableName);
	
	        if (isExists) {
	            throw new TableExistsException(tableName + "is exists!");
	        }
	
	        TableDescriptorBuilder descriptorBuilder = TableDescriptorBuilder.newBuilder(name);
	        List<ColumnFamilyDescriptor> columnFamilyList = new ArrayList<>();
	
	        for (String columnFamily : columnFamilies) {
	            ColumnFamilyDescriptor columnFamilyDescriptor = ColumnFamilyDescriptorBuilder
	                    .newBuilder(columnFamily.getBytes()).build();
	            columnFamilyList.add(columnFamilyDescriptor);
	        }
	
	        descriptorBuilder.setColumnFamilies(columnFamilyList);
	        TableDescriptor tableDescriptor = descriptorBuilder.build();
	        admin.createTable(tableDescriptor);
	    }
	
	    /**
	     * 插入或更新
	     *  experssion : put <tableName>,<rowKey>,<family:column>,<value>,<timestamp>
	     * @param tableName 	表名
	     * @param rowKey		行id
	     * @param columnFamily  列族名
	     * @param column 		列
	     * @param value 		值
	     * @throws IOException
	     */
	    public void insertOrUpdate(String tableName, String rowKey, String columnFamily, String column, String value)
	            throws IOException {
	        this.insertOrUpdate(tableName, rowKey, columnFamily, new String[]{column}, new String[]{value});
	    }
	
	    /**
	     *   插入或更新多个字段
	     * experssion : put <tableName>,<rowKey>,<family:column>,<value>,<timestamp>
	     * @param tableName 	表名
	     * @param rowKey        行id
	     * @param columnFamily  列族名
	     * @param columns		列
	     * @param values		值
	     * @throws IOException
	     */
	    public void insertOrUpdate(String tableName, String rowKey, String columnFamily, String[] columns, String[] values)
	            throws IOException {
	        Table table = connection.getTable(TableName.valueOf(tableName));
	
	        Put put = new Put(Bytes.toBytes(rowKey));
	
	        for (int i = 0; i < columns.length; i++) {
	            put.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes(columns[i]), Bytes.toBytes(values[i]));
	            table.put(put);
	        }
	    }
	
	    /**
	     * 删除行
	     * @param tableName		表名
	     * @param rowKey		行id
	     * @throws IOException
	     */
	    public void deleteRow(String tableName, String rowKey) throws IOException {
	        Table table = connection.getTable(TableName.valueOf(tableName));
	
	        Delete delete = new Delete(rowKey.getBytes());
	
	        table.delete(delete);
	    }
	
	    /**
	     * 删除列族
	     * @param tableName		表名
	     * @param rowKey		行id
	     * @param columnFamily	列族名
	     * @throws IOException
	     */
	    public void deleteColumnFamily(String tableName, String rowKey, String columnFamily) throws IOException {
	        Table table = connection.getTable(TableName.valueOf(tableName));
	
	        Delete delete = new Delete(rowKey.getBytes());
	        delete.addFamily(Bytes.toBytes(columnFamily));
	
	        table.delete(delete);
	    }
	
	    /**
	     * 删除列
	     * experssion : delete 'tableName','rowKey','columnFamily:column'
	     * @param tableName		表名
	     * @param rowKey		行id
	     * @param columnFamily	列族名
	     * @param column		列名
	     * @throws IOException
	     */
	    public void deleteColumn(String tableName, String rowKey, String columnFamily, String column) throws IOException {
	        Table table = connection.getTable(TableName.valueOf(tableName));
	
	        Delete delete = new Delete(rowKey.getBytes());
	        delete.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes(column));
	
	        table.delete(delete);
	    }
	
	    /**
	     * 删除表
	     * experssion : disable 'tableName' 之后 drop 'tableName'
	     * @param tableName 	表名
	     * @throws IOException
	     */
	    public void deleteTable(String tableName) throws IOException {
	        boolean isExists = this.tableExists(tableName);
	
	        if (!isExists) {
	            return;
	        }
	
	        TableName name = TableName.valueOf(tableName);
	        admin.disableTable(name);
	        admin.deleteTable(name);
	    }
	
	    /**
	     * 获取值
	     * experssion : get 'tableName','rowkey','family:column'
	     * @param tableName		表名
	     * @param rowkey		行id
	     * @param family		列族名
	     * @param column		列名
	     * @return
	     */
	    public String getValue(String tableName, String rowkey, String family, String column) {
	        Table table = null;
	        String value = "";
	
	        if (StringUtils.isBlank(tableName) || StringUtils.isBlank(family) || StringUtils.isBlank(rowkey) || StringUtils
	                .isBlank(column)) {
	            return null;
	        }
	
	        try {
	            table = connection.getTable(TableName.valueOf(tableName));
	            Get g = new Get(rowkey.getBytes());
	            g.addColumn(family.getBytes(), column.getBytes());
	            Result result = table.get(g);
	            List<Cell> ceList = result.listCells();
	            if (ceList != null && ceList.size() > 0) {
	                for (Cell cell : ceList) {
	                    value = Bytes.toString(cell.getValueArray(), cell.getValueOffset(), cell.getValueLength());
	                }
	            }
	        } catch (IOException e) {
	            e.printStackTrace();
	        } finally {
	            try {
	                table.close();
	                connection.close();
	            } catch (IOException e) {
	                e.printStackTrace();
	            }
	        }
	        return value;
	    }
	
	    /**
	     * 查询指定行
	     * experssion : get 'tableName','rowKey'
	     * @param tableName		表名
	     * @param rowKey		行id
	     * @return
	     * @throws IOException
	     */
	    public String selectOneRow(String tableName, String rowKey) throws IOException {
	        Table table = connection.getTable(TableName.valueOf(tableName));
	        Get get = new Get(rowKey.getBytes());
	        Result result = table.get(get);
	        NavigableMap<byte[], NavigableMap<byte[], NavigableMap<Long, byte[]>>> map = result.getMap();
	
	        for (Cell cell : result.rawCells()) {
	            String row = Bytes.toString(cell.getRowArray());
	            String columnFamily = Bytes.toString(cell.getFamilyArray());
	            String column = Bytes.toString(cell.getQualifierArray());
	            String value = Bytes.toString(cell.getValueArray());
	            // 可以通过反射封装成对象(列名和Java属性保持一致)
	            System.out.println(row);
	            System.out.println(columnFamily);
	            System.out.println(column);
	            System.out.println(value);
	        }
	        return null;
	    }
	
	
	    /**
	     *   根据条件取出点位指定时间内的所有记录
	     * @param tableName		表名("OPC_TEST")
	     * @param family 列簇名("OPC_COLUMNS")
	     * @param column 列名("site")
	     * @param value 值(采集点标识)
	     * @param startMillis 开始时间毫秒值(建议传递当前时间前一小时的毫秒值，在保证查询效率的前提下获取到点位最新的记录)
	     * @param endMillis 结束时间毫秒值(当前时间)
	     * @return
	     * @throws IOException
	     */
	    @SuppressWarnings("finally")
	    public Map<String,String> scanBatchOfTable(String tableName, String family, String [] column, String [] value, Long startMillis, Long endMillis) throws IOException {
	
	        if(Objects.isNull(column) || Objects.isNull(column) || column.length != value.length) {
	            return null;
	        }
	
	        FilterList filterList = new FilterList();
	
	        for (int i = 0; i < column.length; i++) {
	            SingleColumnValueFilter filter =  new SingleColumnValueFilter(Bytes.toBytes(family), Bytes.toBytes(column[i]), CompareOperator.EQUAL, Bytes.toBytes(value[i]));
	            filterList.addFilter(filter);
	        }
	
	        Table table = connection.getTable(TableName.valueOf(tableName));
	
	        Scan scan = new Scan();
	        scan.setFilter(filterList);
	
	        if(startMillis != null && endMillis != null) {
	            scan.setTimeRange(startMillis,endMillis);
	        }
	
	        ResultScanner scanner = table.getScanner(scan);
	        Map<String,String> resultMap = new HashMap<>();
	
	        try {
	            for(Result result:scanner){
	                for(Cell cell:result.rawCells()){
	                    String values=Bytes.toString(CellUtil.cloneValue(cell));
	                    String qualifier=Bytes.toString(CellUtil.cloneQualifier(cell));
	
	                    resultMap.put(qualifier, values);
	                }
	            }
	        } finally {
	            if (scanner != null) {
	                scanner.close();
	            }
	            return resultMap;
	        }
	    }
	
	    /**
	     *   根据条件取出点位最近时间的一条记录
	     * experssion : scan 't1',{FILTER=>"PrefixFilter('2015')"}
	     * @param tableName		表名("OPC_TEST")
	     * @param family 列簇名("OPC_COLUMNS")
	     * @param column 列名("site")
	     * @param value 值(采集点标识)
	     * @param startMillis 开始时间毫秒值(建议传递当前时间前一小时的毫秒值，在保证查询效率的前提下获取到点位最新的记录)
	     * @param endMillis 结束时间毫秒值(当前时间)
	     * @return
	     * @throws IOException
	     */
	    @SuppressWarnings("finally")
	    public Map<String,String> scanOneOfTable(String tableName,String family,String column,String value,Long startMillis,Long endMillis) throws IOException {
	        Table table = connection.getTable(TableName.valueOf(tableName));
	
	        Scan scan = new Scan();
	        scan.setReversed(true);
	
	        PageFilter pageFilter = new PageFilter(1); //
	        scan.setFilter(pageFilter);
	
	        if(startMillis != null && endMillis != null) {
	            scan.setTimeRange(startMillis,endMillis);
	        }
	
	        if (StringUtils.isNotBlank(column)) {
	            SingleColumnValueFilter filter =  new SingleColumnValueFilter(Bytes.toBytes(family), Bytes.toBytes(column), CompareOperator.EQUAL, Bytes.toBytes(value));
	            scan.setFilter(filter);
	        }
	
	        ResultScanner scanner = table.getScanner(scan);
	        Map<String,String> resultMap = new HashMap<>();
	
	        try {
	            for(Result result:scanner){
	                for(Cell cell:result.rawCells()){
	                    String values=Bytes.toString(CellUtil.cloneValue(cell));
	                    String qualifier=Bytes.toString(CellUtil.cloneQualifier(cell));
	
	                    resultMap.put(qualifier, values);
	                }
	            }
	        } finally {
	            if (scanner != null) {
	                scanner.close();
	            }
	            return resultMap;
	        }
	    }
	
	
	    /**
	     * 判断表是否已经存在，这里使用间接的方式来实现
	     * @param tableName 表名
	     * @return
	     * @throws IOException
	     */
	    public boolean tableExists(String tableName) throws IOException {
	        TableName[] tableNames = admin.listTableNames();
	        if (tableNames != null && tableNames.length > 0) {
	            for (int i = 0; i < tableNames.length; i++) {
	                if (tableName.equals(tableNames[i].getNameAsString())) {
	                    return true;
	                }
	            }
	        }
	
	        return false;
	    }
	
	
	    /**
	     *  批量添加
	     *
	     * @param tableName HBase表名
	     * @param rowkey    HBase表的rowkey
	     * @param columnFamily        HBase表的columnFamily
	     * @param columns    HBase表的列key
	     * @param values     写入HBase表的值value
	     * @param flag      提交标识符号。需要立即提交时，传递，值为 “end”
	     */
	    public void bulkput(String tableName, String rowkey, String columnFamily, String [] columns, String [] values,String flag) {
	        try {
	            List<Put> list = threadLocal.get();
	            if (list == null) {
	                list = new ArrayList<Put>();
	            }
	
	            Put put = new Put(Bytes.toBytes(rowkey));
	
	            for (int i = 0; i < columns.length; i++) {
	                put.addColumn(Bytes.toBytes(columnFamily), Bytes.toBytes(columns[i]), Bytes.toBytes(values[i]));
	                list.add(put);
	            }
	
	            if (list.size() >= HBaseClient.CACHE_LIST_SIZE || flag.equals("end")) {
	
	                Table table = connection.getTable(TableName.valueOf(tableName));
	                table.put(list);
	
	                list.clear();
	            } else {
	                threadLocal.set(list);
	            }
	
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	    }
	}



### HbaseProperties

	package com.wzy.springboot_hbase.config;
	
	import lombok.Data;
	import org.springframework.boot.context.properties.ConfigurationProperties;
	
	import java.util.Map;
	
	@Data
	@ConfigurationProperties(prefix = "hbase")
	public class HbaseProperties {
	
	    private Map<String, String> config;
	} 

### SpringbootHbaseApplication

	package com.wzy.springboot_hbase;
	
	import org.springframework.boot.SpringApplication;
	import org.springframework.boot.autoconfigure.SpringBootApplication;
	
	@SpringBootApplication
	public class SpringbootHbaseApplication {
	
	    public static void main(String[] args) {
	        SpringApplication.run(SpringbootHbaseApplication.class, args);
	    }
	
	}


### application.yml

	## HBase 配置
	hbase:
	  config:
	    hbase.zookeeper.quorum: 47.112.142.231
	    hbase.zookeeper.port: 2181
	    hbase.zookeeper.znode: /hbase
	    hbase.client.keyvalue.maxsize: 1572864000

### SpringbootHbaseApplicationTests

	package com.wzy.springboot_hbase;
	
	import com.wzy.springboot_hbase.utils.HBaseClient;
	import org.junit.jupiter.api.Test;
	import org.springframework.beans.factory.annotation.Autowired;
	import org.springframework.boot.test.context.SpringBootTest;
	
	import java.io.IOException;
	
	@SpringBootTest
	class SpringbootHbaseApplicationTests {
	
	    @Autowired
	    HBaseClient client;
	
	    @Test
	    void contextLoads() throws IOException {
	
	        //删除namespace
	        client.deleteTable("ns1");
	
	        //创建namespace
	        client.createTable("ns1","test","cf");
	
	        //判断是否存在某个namespace
	        Boolean flag = client.tableExists("ns1");
	        System.out.println(flag);
	
	        //插入数据
	        client.insertOrUpdate("ns1","test","cf","a","111");
	
	        //获得数据
	        String val= client.getValue("ns1","test","cf","a");
	        System.out.println(val);
	
	    }
	
	}

### pom.xml

	<?xml version="1.0" encoding="UTF-8"?>
	<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
	    <modelVersion>4.0.0</modelVersion>
	    <parent>
	        <groupId>org.springframework.boot</groupId>
	        <artifactId>spring-boot-starter-parent</artifactId>
	        <version>2.3.2.RELEASE</version>
	        <relativePath/> <!-- lookup parent from repository -->
	    </parent>
	    <groupId>com.wzy</groupId>
	    <artifactId>springboot_hbase</artifactId>
	    <version>0.0.1-SNAPSHOT</version>
	    <name>springboot_hbase</name>
	    <description>Demo project for Spring Boot</description>
	
	    <properties>
	        <java.version>1.8</java.version>
	    </properties>
	
	    <dependencies>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter</artifactId>
	        </dependency>
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-configuration-processor</artifactId>
	            <optional>true</optional>
	        </dependency>
	
	
	        <dependency>
	            <groupId>org.springframework.boot</groupId>
	            <artifactId>spring-boot-starter-test</artifactId>
	            <scope>test</scope>
	            <exclusions>
	                <exclusion>
	                    <groupId>org.junit.vintage</groupId>
	                    <artifactId>junit-vintage-engine</artifactId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	        <dependency>
	            <groupId>org.apache.hbase</groupId>
	            <artifactId>hbase-client</artifactId>
	            <version>2.1.1</version>
	            <exclusions>
	                <exclusion>
	                    <groupId>javax.servlet</groupId>
	                    <artifactId>servlet-api</artifactId>
	                </exclusion>
	            </exclusions>
	        </dependency>
	        <dependency>
	            <groupId>org.projectlombok</groupId>
	            <artifactId>lombok</artifactId>
	        </dependency>
	    </dependencies>
	
	    <build>
	        <plugins>
	            <plugin>
	                <groupId>org.springframework.boot</groupId>
	                <artifactId>spring-boot-maven-plugin</artifactId>
	            </plugin>
	        </plugins>
	    </build>
	
	</project>



注意：

**注意jar包版本和服务器的对应关系
这里服务器版本是2.1.3，jar包是2.1.1
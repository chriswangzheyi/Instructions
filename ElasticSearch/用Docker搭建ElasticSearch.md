**用Docker构建ElasticSearch**

lucene_version" : "6.6.1

**1.启动服务**

    docker run -e ES_JAVA_OPTS="-Xms256m -Xmx256m" -d -p 9200:9200 -p 9300:9300  -p 9100:9100  --name myes  docker.io/elasticsearch


**2.验证**

访问{ip}:9200 

    http://47.112.142.231:9200/

得到:

![](../Images/1.png)






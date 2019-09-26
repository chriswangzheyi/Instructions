**用Docker搭建Kibana**


**1.启动服务**

    docker run -it -d -e ELASTICSEARCH_URL=http://47.112.142.231:9200 -p 5601:5601 --name kibana kibana

注意：需要将ELASTICSEARCH_URL 替换为ElasticSearch的URL


**2.验证**

访问{ip}:5601

    http://47.112.142.231:5601/

得到:

![](../Images/2.png)





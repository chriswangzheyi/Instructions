# Docker 安装redis



## 安装 



```
docker run --restart=always -p 6379:6379 --name myredis -d redis:6.2.1 
```





## 验证



```
# 进入容器
docker exec -it  [容器名 | 容器ID ] bash

#测试
redis-cli
set k1 v1
get k1
```


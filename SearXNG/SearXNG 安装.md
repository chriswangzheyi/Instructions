# SearXNG 安装



```
cd /usr/local
git clone https://github.com/searxng/searxng-docker.git
cd searxng-docker
```



### Generate the secret key

 

```
sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|g" searxng/settings.yml
```





## 修改setting

修改 earxng/settings.yml 的端口

```
- "127.0.0.1:31111:8080
```



启动：

```
docker compose up -d
```


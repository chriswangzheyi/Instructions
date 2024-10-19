# Docker 解决被墙的方案



在镜像前加上：

```
docker.fxxk.dedyn.io
```





比如：

```
docker run -it -d --init --name metaso-free-api -p 8000:8000 -e TZ=Asia/Shanghai docker.fxxk.dedyn.io/vinlic/metaso-free-api:latest
```





修改tag



```
docker tag docker.fxxk.dedyn.io/XX XX
```
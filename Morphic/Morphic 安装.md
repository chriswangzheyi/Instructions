# Morphic 安装



```
git clone https://github.com/miurla/morphic.git

# 安装bun
curl -fsSL https://bun.sh/install | bash
source /root/.bashrc

# 检查是否安装好
bun --version

# 安装Morphic
cd morphic
bun install
```





## 填写key

```
cp .env.local.example .env.local
```



填写：

```
# OpenAI API key retrieved here: https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Tavily API Key retrieved here: https://app.tavily.com/home
TAVILY_API_KEY=

# Upstash Redis URL and Token retrieved here: https://console.upstash.com/redis
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
```



## 启动



```
docker compose up -d
```





访问：

[http://localhost:3000](http://localhost:3000/)



## 解决问题



### bun: /lib64/libc.so.6: version `GLIBC_2.18' not found

```

curl -O http://ftp.gnu.org/gnu/glibc/glibc-2.18.tar.gz
tar zxf glibc-2.18.tar.gz 
cd glibc-2.18/
mkdir build
cd build/
../configure --prefix=/usr
make -j2
make install
```



### bun: /lib64/libc.so.6: version `GLIBC_2.24' not found 

````
### 注意：回到上一步的文件夹外


curl -O http://ftp.gnu.org/gnu/glibc/glibc-2.24.tar.gz
tar zxf glibc-2.24.tar.gz 
cd glibc-2.24/
mkdir build
cd build/
../configure --prefix=/usr
make -j2
make install
```
````





### bun: /lib64/libc.so.6: version `GLIBC_2.25' not found (required by bun)

````
### 注意：回到上一步的文件夹外


curl -O http://ftp.gnu.org/gnu/glibc/glibc-2.25.tar.gz
tar zxf glibc-2.25.tar.gz 
cd glibc-2.25/
mkdir build
cd build/
../configure --prefix=/usr
make -j2
make install
```
````


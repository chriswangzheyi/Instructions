# morphic 安装



## 安装步骤



```
sudo yum update
sudo yum install git
sudo yum install -y nodejs

#安装npm
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs

#验证
node -v
npm -v


#npm 源切换为淘宝的镜像源，这样 bun install 时会使用更快的镜像地址：
npm config set registry https://registry.npmmirror.com
```


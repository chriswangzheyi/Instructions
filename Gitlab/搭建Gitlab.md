**搭建Gitlab**




**1.搭建步骤**

拉取镜像

     docker pull gitlab/gitlab-ce


创建目录

    mkdir -p /root/gitlab_test/{config,data,logs}


启动容器

    docker run -d --name gitlab --hostname gitlab --restart always -p 4443:443 -p 8888:80 -p 2222:22 -v /root/gitlab_test/config:/etc/gitlab  -v /root/gitlab_test/data:/var/opt/gitlab -v /root/gitlab_test/logs:/var/log/gitlab gitlab/gitlab-ce:latest


修改配置文件


    sudo docker exec -it gitlab /bin/bash
    vim /etc/gitlab/gitlab.rb


增加内容：
    
    external_url 'http://{宿主机ip}'
    gitlab_rails['gitlab_shell_ssh_port'] = 2222


注意：

--需要替换宿主机ip参数。

--gitlab_rails的端口跟创建容器时候22的映射端口一致


![](../Images/6.png)


重启容器

    docker restart gitlab



**2.登录**

访问：

{ip}:8888

例如：

    http://47.112.142.231:8888



**3.设置ssh**

打开本地git bash,使用如下命令生成ssh公钥和私钥对


    ssh-keygen -t rsa -C 'root'


![](../Images/1.png)


然后一路回车(-C 参数是你的邮箱地址)


然后输入命令：

    # ~表示用户目录，比如我的windows就是C:\Users\Administrator，并复制其中的内容
    $ cat ~/.ssh/id_rsa.pub


打开文件：

    ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDbszLy9d0U8Vk5g/y0a1yGtXoksJDVSyEw5dG87hDgonBbmOskqd4VDrMSGGKIiTzzMlMJQa2vLpVX3Mn/E0UWw58y1osQBOYFyODsCOpYLjd5lE/8uyhCEyTcq14lhTIv2W0AUqFpS5S7lgoDkm8jrlUj4tFO/Cu0sri4WBeuPknePu8uqqs3PilYE0VDwihwGuImkDVH9tvcVlhT7TUJj1nikYBLYBPlI1XXIVkcXhEM46ecGLNBU8RRyHmEMCuIIcP9nPf5FwQ4IFCRkBZx5WoEedxpTmGGAmTSsrJaRNfQ3J3dcIB6Y2iXSfz+LsNVxrSABHG0ctdRfoX+J/PQrdnMFmF6Dm5FvWMli4kasjptCoQ+LZZ4LmVmjD+1IVwexk5f0n0t81JLO0rML7zfEI5t5Qg3dFoBsarVMSFFvFzypNAE6uMe8fpwGZxUrZ3cux1L7U/SVMoHnPgCIYSrZP8Ovt1Y97zr/LW8Z8nO2AWdlzolUdyYDecc8+N8XL0= root



打开gitlab,找到Profile Settings-->SSH Keys--->Add SSH Key,并把上一步中复制的内容粘贴到Key所对应的文本框


![](../Images/2.png)



**3. 克隆项目**

查看项目克隆地址：

![](../Images/4.png)



在本地需要克隆项目的地方点击右键，然后选择Git Bash Here

![](../Images/3.png)


使用命令： 

git clone {项目克隆地址]

例如

    git clone ssh://git@47.112.142.231:2222/root/test.git


可以看到项目开始下载到本地：

![](../Images/5.png)
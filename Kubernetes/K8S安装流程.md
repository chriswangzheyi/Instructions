# K8S安装流程

参考： https://mp.weixin.qq.com/s/KezmW60LF43jFNETpVH2Ag

## 集群组成

通过VMare Workstation搭建。处理器需要是2核及以上。

### 3个Master

master1: 192.168.195.140

master2: 192.168.195.141

master3: 192.168.195.142

###1个Node

node1: 192.168.195.143

## 设置静态ip地址

### master1:

vi /etc/sysconfig/network-scripts/ifcfg-ens33


将其中的

ONBOOT=no改成ONBOOT=yes
	
BOOTPROTO = static

插入：

	IPADDR=192.168.195.140
	NETMASK=255.255.255.0
	GATEWAY=192.168.195.2

其中GATEWAY的IP地址为

![](../Images/1.png)


### master2:

vi /etc/sysconfig/network-scripts/ifcfg-ens33

ONBOOT=no改成ONBOOT=yes
	
BOOTPROTO = static

插入：

	IPADDR=192.168.195.141
	NETMASK=255.255.255.0
	GATEWAY=192.168.195.2


### master3:

vi /etc/sysconfig/network-scripts/ifcfg-ens33

ONBOOT=no改成ONBOOT=yes
	
BOOTPROTO = static

插入：

	IPADDR=192.168.195.142
	NETMASK=255.255.255.0
	GATEWAY=192.168.195.2

### node1:

vi /etc/sysconfig/network-scripts/ifcfg-ens33


vi /etc/sysconfig/network-scripts/ifcfg-ens33

ONBOOT=no改成ONBOOT=yes
	
BOOTPROTO = static

插入：

	IPADDR=192.168.195.142
	NETMASK=255.255.255.0
	GATEWAY=192.168.195.2

### 对于所有服务器：

设置resolv.conf

	vi /etc/resolv.conf

插入：

    nameserver 192.168.195.2


重启服务

	sudo service network restart


## 安装基础软件包

在各个节点执行：

	yum -y install wget net-tools nfs-utilslrzsz gcc gcc-c++ make cmake libxml2-devel openssl-devel curl curl-devel unzipsudo ntp libaio-devel wget vim ncurses-devel autoconf automake zlib-devel  python-devel epel-release openssh-server socat  ipvsadm conntrack ntpdate



## 配置防火墙

在各个节点执行：

	systemctl stop firewalld  &&systemctl  disable  firewalld

## 时间同步

### 时间同步

	ntpdate cn.pool.ntp.org

### 编辑计划任务，每小时做一次同步

	crontab -e
	* */1 * * * /usr/sbin/ntpdate   cn.pool.ntp.org


## 关闭selinux

在各个节点执行：

关闭selinux，设置永久关闭，这样重启机器selinux也处于关闭状态
 
### 修改/etc/sysconfig/selinux文件

vi /etc/sysconfig/selinux

替换为：

	SELINUX=disabled


### 修改/etc/selinux/config

vi /etc/selinux/config

替换为：

	SELINUX=disabled


修改完毕后

	reboot -f


## 关闭selinux

在各个节点执行：

	swapoff -a
	# 永久禁用，打开/etc/fstab注释掉swap那一行。
	sed -i 's/.*swap.*/#&/' /etc/fstab


## 修改内核参数

	cat <<EOF >  /etc/sysctl.d/k8s.conf
	net.bridge.bridge-nf-call-ip6tables = 1
	net.bridge.bridge-nf-call-iptables = 1
	EOF
	sysctl --system


## 修改主机名

192.168.195.140：

	hostnamectl set-hostname master1

192.168.195.141：

	hostnamectl set-hostname master2

192.168.195.142：

	hostnamectl set-hostname master3

192.168.195.142：

	hostnamectl set-hostname node1


## 配置hosts文件，各个节点操作

在各个节点执行：

vi /etc/hosts

	192.168.195.140 master1
	192.168.195.141 master2
	192.168.195.142 master3
	192.168.195.143 node1

## 配置master1到node无密码登陆，配置master1到master2、master3无密码登陆

###　在master1上操作

	ssh-keygen -t rsa

一直回车就可以

	#需要输入密码，输入master2物理机密码即可
	ssh-copy-id master2

	#需要输入密码，输入master3物理机密码即可
	ssh-copy-id master3

	#需要输入密码，输入node1物理机密码即可
	ssh-copy-id node1


## 安装准备工作


### 修改yum源

在各个节点操作：

#### （1）备份原来的yum源

	mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup

#### （2）下载阿里的yum源 

	wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo

####（3）生成新的yum缓存

	yum makecache fast

#### （4）配置安装k8s需要的yum源

	cat <<EOF > /etc/yum.repos.d/kubernetes.repo
	[kubernetes]
	name=Kubernetes
	baseurl=https://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64
	enabled=1
	gpgcheck=0
	EOF

#### （5）清理yum缓存 

	yum clean all

#### （6）生成新的yum缓存 
	
	yum makecache fast

#### （7）更新yum源 

	yum -y update

####（8）安装软件包

	yum -y install yum-utilsdevice-mapper-persistent-data lvm2 yum-utils

####（9）添加新的软件源 
	
	yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo


## 安装docker19.03，各个节点操作

### （1）下载19.03.7版本

	yum install -y docker-ce-19*
	systemctl enable docker &&systemctl start docker

查看docker状态，如果状态是active（running），说明docker是正常运行状态
	
	systemctl status docker



### (2)修改docker配置文件，配置镜像加速器

	 cat > /etc/docker/daemon.json <<EOF
	{
	 "exec-opts": ["native.cgroupdriver=systemd"],
	 "log-driver": "json-file",
	 "log-opts": {
	   "max-size": "100m"
	  },
	 "storage-driver": "overlay2",
	 "storage-opts": [
	   "overlay2.override_kernel_check=true"
	  ]
	}
	EOF


### (3)重启docker使配置生效 

	systemctl daemon-reload && systemctl restart docker

### (4) 设置网桥包经IPTables，core文件生成路径，配置永久生效

	echo 1 > /proc/sys/net/bridge/bridge-nf-call-iptables
	echo 1 >/proc/sys/net/bridge/bridge-nf-call-ip6tables
	 
	echo """
	vm.swappiness = 0
	net.bridge.bridge-nf-call-iptables = 1
	net.ipv4.ip_forward = 1
	net.bridge.bridge-nf-call-ip6tables = 1
	""" > /etc/sysctl.conf
	sysctl -p

### （5）开启ipvs，不开启ipvs将会使用iptables，但是效率低，所以官网推荐需要开通ipvs内核

	cat >/etc/sysconfig/modules/ipvs.modules <<EOF
	#!/bin/bash
	ipvs_modules="ip_vs ip_vs_lc ip_vs_wlcip_vs_rr ip_vs_wrr ip_vs_lblc ip_vs_lblcr ip_vs_dh ip_vs_sh ip_vs_fo ip_vs_nqip_vs_sed ip_vs_ftp nf_conntrack"
	for kernel_module in \${ipvs_modules}; do
	 /sbin/modinfo -F filename \${kernel_module}> /dev/null 2>&1
	 if [$? -eq 0 ]; then
	 /sbin/modprobe \${kernel_module}
	 fi
	done
	EOF
	chmod 755 /etc/sysconfig/modules/ipvs.modules && bash /etc/sysconfig/modules/ipvs.modules && modprobe ip_vs && lsmod| grep ip_vs


## 安装kubernetes1.17.3

### 在所有节点安装kubeadm和kubelet

	yum install kubeadm-1.17.3 kubelet-1.17.3 -y
	systemctl enable kubelet


### 上传镜像

上传镜像到所有节点


#### 镜像提取方式

	链接：https://pan.baidu.com/s/1UCldCLnIrDpE5NIoMqvXIg
	提取码：xk3y

#### 加载镜像

	docker load -i   kube-apiserver.tar.gz
	docker load -i   kube-scheduler.tar.gz
	docker load -i   kube-controller-manager.tar.gz
	docker load -i   pause.tar.gz
	docker load -i   cordns.tar.gz
	docker load -i   etcd.tar.gz
	docker load -i   kube-proxy.tar.gz
	 
	docker load -i   cni.tar.gz
	docker load -i   calico-node.tar.gz
	 
	docker load -i  kubernetes-dashboard_1_10.tar.gz
	docker load -i  metrics-server-amd64_0_3_1.tar.gz
	docker load -i  addon.tar.gz

说明

> pause版本是3.1，用到的镜像是k8s.gcr.io/pause:3.1
> etcd版本是3.4.3，用到的镜像是k8s.gcr.io/etcd:3.4.3-0        
> cordns版本是1.6.5，用到的镜像是k8s.gcr.io/coredns:1.6.5
> cni版本是3.5.3，用到的镜像是quay.io/
> /cni:v3.5.3
> calico版本是3.5.3，用到的镜像是quay.io/calico/node:v3.5.3          
> apiserver、scheduler、controller-manager、kube-proxy版本是1.17.3，用到的镜像分别是
> k8s.gcr.io/kube-apiserver:v1.17.3
> k8s.gcr.io/kube-controller-manager:v1.17.3
> k8s.gcr.io/kube-scheduler:v1.17.3
> k8s.gcr.io/kube-proxy:v1.17.3
>  
> kubernetes dashboard版本1.10.1，用到的镜像是k8s.gcr.io/kubernetes-dashboard-amd64:v1.10.1
> metrics-server版本0.3.1，用到的镜像是k8s.gcr.io/metrics-server-amd64:v0.3.1       
> addon-resizer版本是1.8.4，用到的镜像是k8s.gcr.io/addon-resizer:1.8.4

## 部署keepalive+lvs实现master节点高可用-对apiserver做高可用

### （1）部署keepalived+lvs，在各master节点操作

	yum install -y socat keepalived ipvsadm conntrack

###　（２）修改master1的keepalived.conf文件

修改/etc/keepalived/keepalived.conf


	vi /etc/keepalived/keepalived.conf


注意：修改的vip必须跟各节点的网段一致。



master1节点修改之后的keepalived.conf如下所示:


	global_defs {
	   router_id LVS_DEVEL
	}
	
	vrrp_instance VI_1 {
	    state BACKUP
	    nopreempt
	    interface ens33
	    virtual_router_id 80
	    priority 100
	    advert_int 1
	    authentication {
	        auth_type PASS
	        auth_pass just0kk
	    }
	    virtual_ipaddress {
	        192.168.195.199
	    }
	}
	
	virtual_server 192.168.195.199 6443 {
	    delay_loop 6
	    lb_algo loadbalance
	    lb_kind DR
	    net_mask 255.255.255.0
	    persistence_timeout 0
	    protocol TCP
	
	
	    real_server 192.168.195.140 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	    real_server 192.168.195.141 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	    real_server 192.168.195.142 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	}

###　（3）修改master2的keepalived.conf文件


	global_defs {
	   router_id LVS_DEVEL
	}
	
	vrrp_instance VI_1 {
	    state BACKUP
	    nopreempt
	    interface ens33
	    virtual_router_id 80
	    priority 50
	    advert_int 1
	    authentication {
	        auth_type PASS
	        auth_pass just0kk
	    }
	    virtual_ipaddress {
	        192.168.195.199
	    }
	}
	
	virtual_server 192.168.195.199 6443 {
	    delay_loop 6
	    lb_algo loadbalance
	    lb_kind DR
	    net_mask 255.255.255.0
	    persistence_timeout 0
	    protocol TCP
	
	
	    real_server 192.168.195.140 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	    real_server 192.168.195.141 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	    real_server 192.168.195.142 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	}

###　（4）修改master3的keepalived.conf文件


	global_defs {
	   router_id LVS_DEVEL
	}
	
	vrrp_instance VI_1 {
	    state BACKUP
	    nopreempt
	    interface ens33
	    virtual_router_id 80
	    priority 30
	    advert_int 1
	    authentication {
	        auth_type PASS
	        auth_pass just0kk
	    }
	    virtual_ipaddress {
	        192.168.195.199
	    }
	}
	
	virtual_server 192.168.195.199 6443 {
	    delay_loop 6
	    lb_algo loadbalance
	    lb_kind DR
	    net_mask 255.255.255.0
	    persistence_timeout 0
	    protocol TCP
	
	
	    real_server 192.168.195.140 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	    real_server 192.168.195.141 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	    real_server 192.168.195.142 6443 {
	        weight 1
	        SSL_GET {
	            url {
	              path /healthz
	              status_code 200
	            }
	            connect_timeout 3
	            nb_get_retry 3
	            delay_before_retry 3
	        }
	    }
	
	}


### 重要知识点，必看，否则生产会遇到巨大的坑

keepalive需要配置BACKUP，而且是非抢占模式nopreempt，假设master1宕机，
启动之后vip不会自动漂移到master1，这样可以保证k8s集群始终处于正常状态，
因为假设master1启动，apiserver等组件不会立刻运行，如果vip漂移到master1，
那么整个集群就会挂掉，这就是为什么我们需要配置成非抢占模式了

启动顺序master1->master2->master3，在master1、master2、master3依次执行如下命令

	
	systemctl enable keepalived  && systemctl start keepalived  && systemctl status keepalived


keepalived启动成功之后，在master1上通过ip addr可以看到vip已经绑定到ens33这个网卡上了

	2: ens33:<BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UPgroup default qlen 1000
	   link/ether 00:0c:29:9d:7b:09 brd ff:ff:ff:ff:ff:ff
	   inet 192.168.124.16/24 brd 192.168.124.255 scope global noprefixrouteens33
	      valid_lft forever preferred_lft forever
	   inet 192.168.124.199/32 scopeglobal ens33
	      valid_lft forever preferred_lft forever
	   inet6 fe80::e2f9:94cd:c994:34d9/64 scope link tentative noprefixroutedadfailed
	      valid_lft forever preferred_lft forever
	   inet6 fe80::ed0d:a5fc:e36a:a563/64 scope link noprefixroute
	      valid_lft forever preferred_lft forever


**注意：只有这里显示正确，后边的步骤才能进行**

## 在master1节点初始化k8s集群

在master1上操作如下


vi kubeadm-config.yaml

	apiVersion: kubeadm.k8s.io/v1beta2
	kind: ClusterConfiguration
	kubernetesVersion:  v1.17.3
	apiServer:
	  certSANs:    #填写所有kube-apiserver节点的hostname、IP、VIP
	  - master1
	  - master2
	  - master3
	  - node1
	  - 192.168.195.199
	controlPlaneEndpoint: "192.168.195.199:6443"
	networking:
	  podSubnet: "10.244.0.0/16"


执行：

	kubeadm init --config kubeadm-config.yaml

成功以后会显示：

![](../Images/3.png)


需要将红色部分记录下来。

####　如果初始化失败可以执行

	kubeadm reset
	rm -rf $HOME/.kube


### 在master1节点执行如下，这样才能有权限操作k8s资源

	mkdir -p $HOME/.kube
	sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
	sudo chown $(id -u):$(id -g) $HOME/.kube/config

###  查看状态

在master1节点执行 

	kubectl get nodes

显示如下：

	NAME        STATUS     ROLES    AGE    VERSION
	master1  NotReady   master   2m13s  v1.17.3

在master1节点执行 

	kubectl get pods -n kube-system

可看到cordns也是处于pending状态

显示如下：

	coredns-6955765f44-bzssk          0/1     Pending   0          113s
	coredns-6955765f44-wmt6x          0/1     Pending   0          113s
	etcd-master1                      1/1     Running   0          98s
	kube-apiserver-master1            1/1     Running   0          98s
	kube-controller-manager-master1   1/1     Running   1          98s
	kube-proxy-tr4q4                  1/1     Running   0          113s
	kube-scheduler-master1            1/1     Running   1          98s



是因为没有安装网络插件，需要安装calico或者flannel，接下来我们安装calico

## 安装calico网络插件


将calico.yaml 上传到Master1中。


执行

	kubectl apply -f calico.yaml

因为网络的原因可能要多执行几次。

caLico安装成功后：

执行 

	kubectl get nodes	
	kubectl get pods -n kube-system

![](../Images/2.png)



## 把master1节点的证书拷贝到master2和master3上

### （1）在master2和master3上创建证书存放目录

cd /root && mkdir -p /etc/kubernetes/pki/etcd &&mkdir -p ~/.kube/

### 在master1节点把证书拷贝到master2和master3上，在master1上操作

	scp /etc/kubernetes/pki/ca.crt master2:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/ca.key master2:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/sa.key master2:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/sa.pub master2:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/front-proxy-ca.crt master2:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/front-proxy-ca.key master2:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/etcd/ca.crt master2:/etc/kubernetes/pki/etcd/
	scp /etc/kubernetes/pki/etcd/ca.key master2:/etc/kubernetes/pki/etcd/
	scp /etc/kubernetes/pki/ca.crt master3:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/ca.key master3:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/sa.key master3:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/sa.pub master3:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/front-proxy-ca.crt master3:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/front-proxy-ca.key master3:/etc/kubernetes/pki/
	scp /etc/kubernetes/pki/etcd/ca.crt master3:/etc/kubernetes/pki/etcd/
	scp /etc/kubernetes/pki/etcd/ca.key master3:/etc/kubernetes/pki/etcd/


### 证书拷贝之后在master2和master3上执行如下，形成集群

将Master1 执行k8s初始化的命令，分别在master2 和 master3中执行：

	kubeadm join 192.168.195.199:6443 --token rrgym5.iqki02sjp67uzfoz \
	  --discovery-token-ca-cert-hash sha256:7b52de1524dd1c1d90b1b72f92c6f301e8a64ae63785204ac006163b8da19690 \
	  --control-plane 



### 在master2和master3上操作：

	mkdir -p $HOME/.kube
	sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
	sudo chown $(id -u):$(id -g) $HOME/.kube/config


### 验证

在master1上：

	kubectl get nodes 


显示：

	[root@master1 etcd]# kubectl get nodes 
	NAME      STATUS   ROLES    AGE     VERSION
	master1   Ready    master   33m     v1.17.3
	master2   Ready    master   2m20s   v1.17.3
	master3   Ready    master   47s     v1.17.3



## 把node1节点加入到k8s集群，在node节点操作

	kubeadm join 192.168.195.199:6443 --token t804ua.g39ih5r9b0w3ltp5 \
	--discovery-token-ca-cert-hash sha256:b53d6136c6deb3669527885fa34b269b980b350e05df69d21d1af14ea0a6f65a 

这里的token和discovery-token-ca-cert-hash 都是master1初始化的时候生成的。

### 在master1节点查看集群节点状态

	kubectl get nodes

结果：

	NAME      STATUS   ROLES    AGE    VERSION
	master1   Ready    master   102m   v1.17.3
	master2   Ready    master   99m    v1.17.3
	master3   Ready    master   99m    v1.17.3
	node1     Ready    <none>   94m    v1.17.3


至此集群搭建完毕。


## 安装dashboard


### 下载镜像

dashboard_2_0_0.tar.gz

metrics-scrapter-1-0-1.tar.gz

链接：https://pan.baidu.com/s/1k1heJy8lLnDk2JEFyRyJdA
提取码：udkj


### 加载镜像（在每一个节点）

上传镜像到每一个节点后：

	docker load -i dashboard_2_0_0.tar.gz
	docker load -i metrics-scrapter-1-0-1.tar.gz


### 安装dashboard

上传 kubernetes-dashboard.yaml 文件到master1中

执行：

	kubectl apply -f kubernetes-dashboard.yaml


看到下面内容表示成功：

	NAME                                        READY   STATUS    RESTARTS  AGE 
	dashboard-metrics-scraper-694557449d-8xmtf   1/1    Running   0          60s  
	kubernetes-dashboard-5f98bdb684-ph9wg        1/1    Running   2          60s


查看：

	kubectl get svc -n kubernetes-dashboard


显示：

	NAME                       TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)   AGE  
	dashboard-metrics-scraper ClusterIP  10.100.23.9       <none>        8000/TCP   50s  
	kubernetes-dashboard       ClusterIP   10.105.253.155   <none>        443/TCP    50s


### 修改service type类型变成NodePort：

	kubectl edit svc kubernetes-dashboard -n kubernetes-dashboard

把type: ClusterIP变成 type: NodePort，保存退出即可

验证：

	kubectl get svc -n kubernetes-dashboard

![](../Images/4.png)

访问ip:port 就可以看到dashboard. port是上图红色框所示。 例如： https://192.168.195.140:30804


### 创建管理员token，可查看任何空间权限

	kubectl create clusterrolebinding dashboard-cluster-admin --clusterrole=cluster-admin --serviceaccount=kubernetes-dashboard:kubernetes-dashboard


1）查看kubernetes-dashboard名称空间下的secret

	kubectl get secret -n kubernetes-dashboard

显示

	NAME                               TYPE                                  DATA   AGE
	default-token-spgpq                kubernetes.io/service-account-token   3      18m
	kubernetes-dashboard-certs         Opaque                                0      18m
	kubernetes-dashboard-csrf          Opaque                                1      18m
	kubernetes-dashboard-key-holder    Opaque                                2      18m
	kubernetes-dashboard-token-6fvhs   kubernetes.io/service-account-token   3      18m

2） 找到对应的带有token的kubernetes-dashboard-token-6fvhs

	kubectl  describe  secret kubernetes-dashboard-token-6fvhs  -n   kubernetes-dashboard

得到token。



3）进入dashboard,填入token。

![](../Images/5.png)

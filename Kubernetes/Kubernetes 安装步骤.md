# Kubernetes 安装步骤

---

## 测试服务器构成

需要6台主机：

- K8S-master01：Master节点   192.168.195.140
- K8S-master02：Master节点   192.168.195.141
- K8S-master03：Master节点   192.168.195.142
- K8S-master-lb：Keepalived节点    192.168.195.143
- K8S-node01:worker节点    192.168.195.144
- K8s-node02 :worker节点   192.168.195.145



## 安装步骤

### 从VMare Workstation中安装Centos7.5

首先安装单台即可

### 配置网络地址

vi /etc/sysconfig/network-scripts/ifcfg-ens33


将其中的

ONBOOT=no改成ONBOOT=yes
	
BOOTPROTO = static

插入：

	IPADDR=192.168.195.140
	NETMASK=255.255.255.0
	GATEWAY=192.168.195.2

IPPADDR的前三个网段需要和Gateway一样


![](../Images/1.png)


设置resolv.conf

	vi /etc/resolv.conf

插入：

    nameserver 192.168.195.2


重启系统

reboot

### 设置虚拟机核数为2

![](../Images/5.png)

### 配置Host

vi /etc/hosts

插入

	192.168.195.140 k8s-master01
	192.168.195.141 k8s-master02
	192.168.195.142 k8s-master03
	192.168.195.143 k8s-master-lb
	192.168.195.144 k8s-node01
	192.168.195.145 k8s-node02

### 关闭防火墙、Selinux、Dnsmasq、swap

	# 关闭防火墙并清空防火墙规则
	systemctl disable --now firewalld
	iptables -F && iptables -X && iptables -F -t nat && iptables -X -t nat

	#关闭Network Manager
	systemctl disable --now NetworkManager

	# 关闭selinux  --->selinux=disabled 需重启生效!
	setenforce 0 && sed -i 's/^SELINUX=.*/SELINUX=disabled/' /etc/selinux/config
	
	# 关闭swap --->注释掉swap那一行, 需重启生效!
	swapoff -a && sed -i '/ swap / s/^\(.*\)$/# \1/g' /etc/fstab

	# 将SELINUX 改为Disabled
	grep -vE "#|^$" /etc/sysconfig/selinux

	# 注释swap 挂载选项
	grep "swap" /etc/fstab


### 时间同步及其他设置

所有节点同步时间，并且需要加到开机自动启和任务计划中。如果时间不同步，会造成Etcd储Kubernets信息的键值（Key-value）数据库同步数据不正常。

	ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime 

	echo 'Asia/Shanghai' >/etc/timezone 

	ntpdate

	ntpdate time2.aliyun.com
	
	#加到crontab
	 *  /5  *  *  * ntpdate time2.aliyun.com
	
	#加入开机自动同步
	ntpdate time2.aliyun.com

	# 加入crontab
	crontab  -e

	插入：
	* 5 * * * ntpdate time2.aliyun.com

	#加入开机启动
	vi /etc/rc.local
	ntpdate time2.aliyun.com

	#设置连接数最大值
	ulimit -SHn 65535

	# 升级系统并重启
	yum install wget jq psmisc vim net-toools -y
	yum update -y --exclude=kernel* && reboot

### 升级内核

	cd /root
	
	#在对应地方填写版本号
	wget http://mirror.rc.usf.edu/compute_lock/elrepo/kernel/el7/x86_64/RPMS/kernel-ml{,-devel}-${Kernel_Version}.el7.elrepo.x86_64.rpm
	#例如以4.18为例：
	wget http://mirror.rc.usf.edu/compute_lock/elrepo/kernel/el7/x86_64/RPMS/kernel-ml-4.18.9-1.el7.elrepo.x86_64.rpm
	
	 
	#在对应地方填写版本号
	wget http://mirror.rc.usf.edu/compute_lock/elrepo/kernel/el7/x86_64/RPMS/kernel-ml{,-devel}-${4.20.13-1}.el7.elrepo.x86_64.rpm
	#例如以4.18为例：
	wget http://mirror.rc.usf.edu/compute_lock/elrepo/kernel/el7/x86_64/RPMS/kernel-ml-devel-4.18.9-1.el7.elrepo.x86_64.rpm
	
	yum localinstall -y kernel-ml*

	# 设置开机从新内核启动
	grub2-set-default 0 && grub2-mkconfig -o /etc/grub2.cfg
	grubby --args="user_namespace.enable=1" --update-kernel="$(grubby --default-kernel)"

	#重启
	reboot

	#确认内核版本
	uname -r

	#采用ipvs模块，需要安装ipvsadm﻿
	yum install ipvsadm ipset sysstat conntrack libseccomp -y

	#配置ipvs模块
	modprobe -- ip_vs​
	modprobe -- ip_vs_rr
	modprobe -- ip_vs_wrr
	modprobe -- ip_vs_sh
	modprobe -- nf_conntrack_ipv4
	modprobe -- ip_tables
	modprobe -- ip_set
	modprobe -- xt_set
	modprobe -- ipt_set
	modprobe -- ipt_rpfilter
	modprobe -- ipt_REJECT
	modprobe -- ipip

	#加入开机启动
	vi /etc/sysconfig/modules/k8s.modules
	(将上面modprobe命令都写入)

	#查看是否加载
	lsmod | grep -e ip_vs -e nf_conntrack_ipv4

![](../Images/2.png)

	#开启k8s 集群的内核参数
	cat <<EOF >  /etc/sysctl.d/k8s.conf
	net.ipv4.ip_forward = 1
	net.bridge.bridge-nf-call-iptables = 1
	fs.may_detach_mounts = 1
	vm.overcommit_memory = 1
	vm.panic_on_oom = 0
	fs.inotify.max_user_watches = 89100	
	fs.file-max = 52706963
	fs.nr_open = 52706963
	net.netfilter.nf_conntrack_max = 2310720
	
	net.ipv4.tcp_keepalive_time = 600
	net.ipv4.tcp_keepalive_probes = 3
	net.ipv4.tcp_keepalive_intvl = 15
	net.ipv4.tcp_max_tw_buckets = 36000
	net.ipv4.tcp_tw_reuse = 1
	net.ipv4.tcp_max_orphans = 327680
	net.ipv4.tcp_orphan_retries = 3
	net.ipv4.tcp_syncookies = 1
	net.ipv4.tcp_max_syn_backlog = 16384
	net.ipv4.ip_conntrack_max = 65536
	net.ipv4.tcp_timestamps = 0
	net.core.somaxconn = 16384
	EOF

### 基本插件安装

#### 安装docker

	#更新yum包
	yum update

	#安装需要的软件包
	yum install -y yum-utils device-mapper-persistent-data lvm2
	
	#设置yum源
	yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

	#查看所有仓库中所有docker版本，并选择特定版本安装
	yum list docker-ce --showduplicates | sort -r

	#安装Docker，命令：yum install docker-ce-版本号,这里我选择了18.06.3.ce-3.el7
	yum install docker-ce-18.06.3.ce-3.el7

	#启动Docker，命令：systemctl start docker，然后加入开机启动
	systemctl start docker
	systemctl enable docker

	#验证是否安装成功
	docker version

	#使用Docker 中国加速器
	vi  /etc/docker/daemon.json
	#添加后：
	{
	    "registry-mirrors": ["https://registry.docker-cn.com"],
	    "live-restore": true
	}
	
或者选择DaoCloud镜像站点安装 https://www.daocloud.io/mirror#accelerator-doc

	curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://f1361db2.m.daocloud.io

	#重启docker服务
	systemctl restart docker


### 安装kubeadm组件
	
	yum install -y ebtables socat

	# 配置源
	cat <<EOF > /etc/yum.repos.d/kubernetes.repo
	[kubernetes]
	name=Kubernetes
	baseurl=https://mirrors.aliyun.com/kubernetes/yum/repos/kubernetes-el7-x86_64
	enabled=1
	gpgcheck=1
	repo_gpgcheck=1
	gpgkey=https://mirrors.aliyun.com/kubernetes/yum/doc/yum-key.gpg https://mirrors.aliyun.com/kubernetes/yum/doc/rpm-package-key.gpg
	EOF
	
	# 安装 
	yum install -y kubelet kubeadm kubectl 

	# 设置开机启动
	systemctl daemon-reload
	systemctl enable --now kubelet

### 安装keepalive

	yum install keepalived haproxy -y


## 复制虚拟机

克隆master01主机

![](../Images/3.png)


### 进入Master02

vi /etc/sysconfig/network-scripts/ifcfg-ens33

修改：

	IPADDR=192.168.195.141

重启

	reboot


### 进入Master03

vi /etc/sysconfig/network-scripts/ifcfg-ens33

修改：

	IPADDR=192.168.195.142

重启

	reboot

### 进入Master-lb

vi /etc/sysconfig/network-scripts/ifcfg-ens33

修改：

	IPADDR=192.168.195.143

重启

	reboot


### 进入Node-01

vi /etc/sysconfig/network-scripts/ifcfg-ens33

修改：

	IPADDR=192.168.195.144

重启

	reboot


### 进入Node-02

vi /etc/sysconfig/network-scripts/ifcfg-ens33

修改：

	IPADDR=192.168.195.145 

重启

	reboot


## 集群初始化


### Master01 Maser02 Maser03

#### 配置HAProxy：

	vi /etc/haproxy/haproxy.cfg

修改内容：

	global
	  maxconn  2000
	  ulimit-n  16384
	  log  127.0.0.1 local0 err
	  stats timeout 30s
	
	defaults
	  log global
	  mode  http
	  option  httplog
	  timeout connect 5000
	  timeout client  50000
	  timeout server  50000
	  timeout http-request 15s
	  timeout http-keep-alive 15s
	
	frontend monitor-in
	  bind *:33305
	  mode http
	  option httplog
	  monitor-uri /monitor
	
	listen stats
	  bind    *:8006
	  mode    http
	  stats   enable
	  stats   hide-version
	  stats   uri       /stats
	  stats   refresh   30s
	  stats   realm     Haproxy\ Statistics
	  stats   auth      admin:admin
	
	frontend k8s-master
	  bind 0.0.0.0:16443
	  bind 127.0.0.1:16443
	  mode tcp
	  option tcplog
	  tcp-request inspect-delay 5s
	  default_backend k8s-master
	
	backend k8s-master
	  mode tcp
	  option tcplog
	  option tcp-check
	  balance roundrobin
	  default-server inter 10s downinter 5s rise 2 fall 2 slowstart 60s maxconn 250 maxqueue 256 weight 100
	  server k8s-master01	192.168.195.140:6443  check
	  server k8s-master02	192.168.195.141:6443  check
	  server k8s-master03	192.168.195.142:6443  check

#### 配置keepalived

	vi /etc/keepalived/keepalived.conf


Master01，Master02，Master03的interface(服务器网卡)，proority（优先级，不同即可），mcast_src_ip(本机Ip)需要注意修改

enterfance的查看方式为： ip addr


#### Master01

	! Configuration File for keepalived
	global_defs {
	    router_id LVS_DEVEL
	}
	vrrp_script chk_apiserver {
	    script "/etc/keepalived/check_apiserver.sh"
	    interval 2
	    weight -5
	    fall 3  
	    rise 2
	}
	vrrp_instance VI_1 {
	    state BACKUP
	    interface ens160
	    mcast_src_ip 192.168.195.140
	    virtual_router_id 51
	    priority 100
	    advert_int 2
	    authentication {
	        auth_type PASS
	        auth_pass FMVm6NFFccY8WjhK
	    }
	    virtual_ipaddress {
	        192.168.20.10
	    }
	#    track_script {
	#       chk_apiserver
	#    }
	}


注意：track_script是注释掉的，等到集群简历完成后再开启

#### Master02

	! Configuration File for keepalived
	global_defs {
	    router_id LVS_DEVEL
	}
	vrrp_script chk_apiserver {
	    script "/etc/keepalived/check_apiserver.sh"
	    interval 2
	    weight -5
	    fall 3  
	    rise 2
	}
	vrrp_instance VI_1 {
	    state BACKUP
	    interface ens160
	    mcast_src_ip 192.168.195.141
	    virtual_router_id 51
	    priority 101
	    advert_int 2
	    authentication {
	        auth_type PASS
	        auth_pass FMVm6NFFccY8WjhK
	    }
	    virtual_ipaddress {
	        192.168.20.10
	    }
	#    track_script {
	#       chk_apiserver
	#    }
	}

#### Master03

	! Configuration File for keepalived
	global_defs {
	    router_id LVS_DEVEL
	}
	vrrp_script chk_apiserver {
	    script "/etc/keepalived/check_apiserver.sh"
	    interval 2
	    weight -5
	    fall 3  
	    rise 2
	}
	vrrp_instance VI_1 {
	    state BACKUP
	    interface ens160
	    mcast_src_ip 192.168.195.142
	    virtual_router_id 51
	    priority 102
	    advert_int 2
	    authentication {
	        auth_type PASS
	        auth_pass FMVm6NFFccY8WjhK
	    }
	    virtual_ipaddress {
	        192.168.20.10
	    }
	#    track_script {
	#       chk_apiserver
	#    }
	}


### 健康检查文件(在Master001, Master002, Master003中修改)

vi /etc/keepalived/check_apiserver.sh

	#!/bin/bash
	
	function check_apiserver() {
	  for ((i=0;i<5;i++));do
	    apiserver_job_id=$(pgrep kube-apiserver)
	    if [[ ! -z $apiserver_job_id ]];then
	       return
	    else
	       sleep 2
	    fi
	    apiserver_job_id=0
	  done
	}
	
	# 1: running 0: stopped
	check_apiserver
	if [[ $apiserver_job_id -eq 0 ]]; then
	    /usr/bin/systemctl stop keepalived
	    exit 1
	else
	    exit 0
	fi



然后 /etc/keepalived/keepalived.conf 文件中的注释取消掉


### 启动haporxy 和 keepalived

	systemctl enable --now haproxy
	systemctl enable --now keepalived


### 查看kubernetes版本

kubeadm version

![](../Images/7.png)

该版本会用到kubeadm-config.yaml中


### 将kubeadm-config.yaml上传到对应的服务器中

在目录中可以找到：

![](../Images/4.png)

source /root/.bashrc


### 下载镜像

	kubeadm config images pull --config /root/kubeadm-config.yaml

![](../Images/6.png)


### 初始化


	kubeadm init --config /root/kubeadm-config.yaml --v=6


![](../Images/8.png)

初始化后会生成Token值，需记录下来：

例如：

	#Master01
	kubeadm join 192.168.195.140:6443 --token v92ta8.znrzwsdbczedewa4 \
    --discovery-token-ca-cert-hash sha256:cceaa92d649d174dfe383e6513400b5a4e9e27aa5e70e145bf35e9bce070bc59 

	#Master02
	kubeadm join 192.168.195.141:6443 --token 4rnkbo.6nzarp9061t0eiq2 \
    --discovery-token-ca-cert-hash sha256:542cc38e4e4efb0566dfe63e6530f5b49a3acaed351602fdcc3816f0baebde9d 


	#Master03:
	kubeadm join 192.168.195.142:6443 --token oyijmw.kbynv8cvai97twem \
    --discovery-token-ca-cert-hash sha256:d1f4c50961256d675b626ef1186557bfb135f610fcebc567197aa74d3d4d5309 




![](../Images/9.png)

### 如果初始化失败需要重置
	
	kubeadm reset


### 全部初始化后

vi /etc/keepalived/keepalived.conf

将被注释的代码还原


### 所有Master配置环境变量

	cat <<EOF >> /root/.bashrc
	export KUBECONFIG=/etc/kubernetes/admin.conf
	EOF
	
	source /root/.bashrc


### 查看节点状态

	kubectl get nodes


![](../Images/10.png)

采用初始化安装方式，所有的系统组件均已容器的方式运行并且在kube-system命名空间内，此时可以查看Pod状态

	kubectl get pods -n kube-system -o wide

![](../Images/11.png)
# Docker 配置

## Installation

### 基础环境

   ubuntu 16.04,  cuda8.0(cudnn6.0),  1080Ti

## 安装docker

[docker][https://blog.csdn.net/wd2014610/article/details/80340991]

[docker_official][https://docs.docker.com/install/linux/docker-ce/ubuntu/#prerequisites]

### 安装准备

```
$ apt-get update
$ apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
阿里云可能报错，添加主机名
$ vi /etc/hosts
之后重复以上两步
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
查看本机配置
$ lsb_release -cs
$ dpkg --print-architecture
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

### 正式安装

  

```
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

此处会报错[dpkg error][https://www.xinrui520.cn/index.php/2019/02/05/%E5%AE%89%E8%A3%85docker%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%EF%BC%8Cdpkg%E5%A4%84%E7%90%86%E8%BD%AF%E4%BB%B6%E5%8C%85-docker-configure%E6%97%B6%E5%87%BA%E9%94%99/] 解决办法如下

```
$ sudo mv /var/lib/dpkg/info /var/lib/dpkg/info_old
$ sudo mkdir /var/lib/dpkg/info
$ sudo apt-get update
$ sudo apt-get -f install
$ sudo mv /var/lib/dpkg/info/* /var/lib/dpkg/info_old 
$ sudo rm -rf /var/lib/dpkg/info
$ sudo mv /var/lib/dpkg/info_old /var/lib/dpkg/info 
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

### Nvidia-docker

参考[nvidia][https://github.com/NVIDIA/nvidia-docker]

```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list $ sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docke
```

### Docker 指令

```
$ sudo docker version
$ sudo docker info
下载docker
$ sudo docker pull image_name
$ sudo docker images
$ sudo docker rmi image_name -f
启动容器
$ sudo docker run -it image_name /bin/bash 
安装程序
$ apt-get install -y app_name
保存修改
$ sudo docker commit ID new_image_name
保存加载镜像
$ sudo docker save image_name -o file_path
$ sudo docker load -i file_path
机器a
$ sudo docker save image_name > /home/save.tar
使用scp将save.tar拷到机器b上, 然后
$ sudo docker load < /home/save.tar
挂载
$ sudo docker run -it --gpus all --name cxy -v /home/cxy/smartCityCode2/:/home/cxy/smartCityCode2 -e NVIDIA_VISIBLE_DEVICE=0 smartcity_cpcn
上传:
(1): docker tag <existing-image> <hub-user>/<repo-name>[:<tag>]
(2): docker push <hub-user>/<repo-name>:<tag>
```



```
.
├── dataset
│   ├── fishEye
│   ├── gt
│   ├── mvpic_ofFish.py
│   ├── mvpic_ofPano.py
│   ├── officialData
│   ├── pano
│   ├── panoSplit
│   └── panoSplit8
├── scripts
│   ├── colmap
│   ├── contextDesc
│   └── pipeline.pdf
├── scripts.sh
├── software
│   ├── colmap
│   └── openMVG
├── upload
│   ├── cnn
│   ├── dsp
│   ├── final
│   ├── fishEye
│   ├── org
│   ├── org8
│   └── selectBest.py
├── upload.txt
└── workspace
    ├── cnn
    ├── dsp
    ├── fishEye
    ├── org
    └── org8
```
## Getting Started

运行之前，需要修改两处路径为绝对路径：

```
./scripts/colmap/matchForCnn.data　　　　　　　　　　　　 第 1 行
./scripts/colmap/scriptsOfCnn/matching_pipeline.m     第 10-13 行
```

将测试数据置于以下目录(数据组织方式与预赛相同)：

```
./dataset/officialData/
```

在根目录下运行：

```
sh scripts.sh
```

若运行有问题，请及时联系！

## Contact Us

如果有问题请及时联系．
Tel: 156 5075 8779 (常同学),  182 9288 5460 (蒋同学)
Email: cxy19960919@163.com, 1143958845@qq.com
星期一, 08. 七月 2019 05:14下午 
	
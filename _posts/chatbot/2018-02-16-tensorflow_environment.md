---
layout: post
comments: true
title:  우분투에서 Tensorflow, Virtualenv, PyCharm, Jupyter 설치 (2019.08.15 업데이트)
categories: ChatBot

tags:
- ChatBot(챗봇)
- Tensorflow
---

**<span style='color:DarkRed'>Tensorflow 환경 구축</span>**
> 우분투에서 tensorflow, Virtualenv, PyCharm, jupyter 설치 방법을 다룬 글입니다. 
> 파일명이나 버전이 조금씩 다들 수 있으니 개인 파일명에 맞도록 수정만 해주시면 됩니다. 전반적인 흐름은 일반화 되어 있습니다.

- ```ubuntu 16.04의 경우```: jupyter의 패스가 자동으로 잡히지 않음
- ```ubuntu 18.04의 경우```: jupyter의 패스가 자동으로 잡힘(따라서 jupyter에 대한 아래의 내용은 생략)

<br>


**<span style='color:DarkRed'>Nvidia graphic driver</span>**
-  ```ubuntu 16.04의 경우```: 우분투를 설치하고 나면 그래픽 드라이버를 잡고, ```CUDA-> cuDNN-> tensorflow ```순서로 잘 설치 했는데도, 그래픽 kernel문제로 tensorflow-gpu가 안잡히는 경우가 있다. 이 글에서는 ```시스템설정 >> 소프트웨어 & 업데이트 >> 추가 드라이버```에서 ```사용 NVIDA binary driver - version 430.40 출처 nvidia-430 (독점)```을 설치 하여 그래픽 카드를 잡고 CUDA 10.0설치 때 deb 대신 runfile를 사용해 CUDA 10.0과 nvidia-430.40가 충돌이 나지 않게 하였다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/zL1niQ7.png"></p>

- ```ubuntu 18.04의 경우```: terminal로 설치합니다.

	```shell
	# package loading
	sudo apt-add-repository ppa:graphics-drivers/ppa -y

	# 해당 그래픽 카드과 호환이되는 드라이버 탐색
	sudo apt-cache search ^nvidia-driver

	# 드라이버 설치(RTX 2080 ti blower의 경우, 430 version으로 선택)
	sudo apt install nvidia-driver-430 -y
	```



<br>

**<span style='color:DarkRed'>CUDA 10.0 설치</span>**

- https://developer.nvidia.com/cuda-downloads 에서 CUDA 10.0을 다운받아 설치
- 여기서 주의 할점은 runfile의 설치과정에서 **cuda-10.0만 설치**하고, graphic driver은 설치하지 **않는** 옵션을 줘야 runfile이 제대로 작동됨

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/vjpPBHE.png"></p>

```bash
~$ cd Downloads/
~/Downloads/$ sudo sh cuda_10.0.176_384.81_linux.run
```
<br>

**<span style='color:DarkRed'>CUDA 10.0 환경변수</span>**

- bashrc파일을 열어 CUDA 10.0의 환경변수를 설정

```bash
~$ sudo gedit ~/.bashrc
```

- bashrc에 붙여지는 내용

```bash
export CUDA_HOME=/usr/local/cuda-10.0

export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- 변경사항 적용

```bash
~$ source ~/.bashrc
```

<br>

**<span style='color:DarkRed'>Cudnn 설치</span>**

> https://developer.nvidia.com/cudnn 에서 해당 운영체제에 맞는 버전을 받으면 된다. 하지만 Tensorflow v1.8이상부터 cuDNN 7.0과 호환이 되므로 본 저자는 cuDNN 7.0을 설치하였다.
- Tensorflow v1.3 미만 : CUDA 8, cuDNN 5.1 버전
- Tensorflow v1.3 부터 : CUDA 8, cuDNN 6 버전 (2017.08월 기준)
- Tensorflow v1.8 부터 : CUDA 9.0, cuDNN 7 버전 (2018.06월 기준)
- Tensorflow v1.13 : ```CUDA 10.1, cuDNN 7.4``` (2019.8.15기준 테스트) 
	- cuDNN 7* 의 낮은 버전을 받을 경우, GPU session 제대로 켜질지라도 CNN 파라마터가 안 올라 가는 경우가 있으니 **7.4이상 버전**을 권장 

<p align="center"><img width="700" height="auto" src="https://i.imgur.com/0lVV0Ws.png"></p>

<br>
```bash
~$ cd Downloads
~/Downloads/$ sudo tar -xzvf cudnn-10.0-linux-x64-v7.4.tgz
```

- cuDNN 파일 이동 및 환경변수 추가

```bash
~/Downloads/$ cd cuda
~/Downloads/cuda/$ sudo cp include/cudnn.h /usr/local/cuda-10.0/include
~/Downloads/cuda/$ sudo cp lib64/libcudnn* /usr/local/cuda-10.0/lib64
~/Downloads/cuda/$ sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
```

<br>

**<span style='color:DarkRed'>virtualenv 생성 </span>**

- tensorflow로 이름을 가지는 virtualenv가 ```~/tensorflow```에 생성됨

```bash
~$ sudo apt-get install python3-pip python3.6-dev python-virtualenv
~$ virtualenv --system-site-packages -p python3.6 tensorflow
```

- virtualenv 실행

```bash
~$ source tensorflow/bin/activate
```

<br>

**<span style='color:DarkRed'>Tensorflow 설치 </span>**
- virtualenv 실행하여 tensorflow가 activate된 후에 설치
- https://pypi.python.org/pypi/tensorflow-gpu/1.13.1 앞서 정의한 대로 호환이 되는 버전을 받아야 함

```bash
(tensorflow) ~$ easy_install -U pip
(tensorflow) ~$ cd Downloads/
(tensorflow) ~/Downloads/$ sudo pip install tensorflow_gpu-1.13.1-cp36-cp36m-manylinux1_x86_64.whl 
```
- 설치 확인
<p align="center"><img width="500" height="auto" src="https://i.imgur.com/btiDiKH.png"></p>

<br>

**<span style='color:DarkRed'>PyCharm 설치 및 연동 </span>**

- https://www.jetbrains.com/pycharm/download/#section=linux 에서  pycharm-community 다운로드

```bash
~$ cd Downloads/
~/Downloads$ sudo tar xf pycharm-community-*.tar.gz -C /opt/
~/Downloads$ cd /opt/pycharm-community-2017.3.3/bin

/opt/pycharm-community-2017.3.3/bin$ sudo apt-get install default-jre
/opt/pycharm-community-2017.3.3/bin$ sudo chmod +x pycharm.sh
/opt/pycharm-community-2017.3.3/bin$ ./pycharm.sh
```
- 기존에 설정한 Virtualenv에 있는 python(```~/tensorflow/bin/python```)의 경로를 Project Interpreter에 추가

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/JUdlyhz.png"></p>

<br>

- ```~/tensorflow/bin/python``` 연결확인

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/rFUXNoF.png"></p>

<br>

- 단축아이콘을 사용할때 CUDA path가 안잡히는 경우

```bash
# command in Pycharm 
(tensorflow) ~$ sudo ldconfig /usr/local/cuda-10.0/lib64
```

<br>

**<span style='color:DarkRed'>Jupyter notebook </span>**
- jupyter notebook를 설치하고 위에 설치된 tensorflow 커널을 연결

```bash
(tensorflow) ~$ sudo pip3 install jupyter
(tensorflow) ~$ ipython kernelspec install-self

# env이름 바꾸기
(tensorflow) ~$ cd /usr/local/share/jupyter/kernels/
(tensorflow) ~/usr/local/share/jupyter/kernels/$ sudo mv python3/ tensorflow/
(tensorflow) ~/usr/local/share/jupyter/kernels/$ cd tensorflow
(tensorflow) ~/usr/local/share/jupyter/kernels/tensorflow/$ sudo gedit kernel.json
```

- Virtualenv python 경로를 입력

```bash
{
 "argv": [
  "/home/donghwa/tensorflow/bin/python",
  "-m",
  "ipykernel",
  "-f",
  "{connection_file}"
 ],
 "display_name": "tensorflow",
 "language": "python"
}
```
<br>

<span style='color:blue'>**library path(cuDNN) not work in jupyter notebook 문제 해결방법**</span>

```bash
(tensorflow) ~$ sudo jupyter notebook --generate-config
(tensorflow) ~$ sudo gedit ~/.jupyter/jupyter_notebook_config.py
```

- jupyter --config-dir(```/home/donghwa/.jupyter```) 에 생성된 ```jupyter_notebook_config.py```을 열어 ```아래 코드```를 맨 앞 줄에 입력

```bash
import os
c = get_config()
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.0/lib64:usr/local/cuda-10.0/lib64/libcudart.so.10.0'
c.Spawner.env.update('LD_LIBRARY_PATH')
```

- e.g
<p align="center"><img width="500" height="auto" src="https://i.imgur.com/sYpa4nQ.png"></p>

<br>

- jupyter notebook 실행

```bash
(tensorflow) ~$ sudo jupyter notebook --allow-root
```

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/sME284g.png1"></p>

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/8XqhTNB.png1"></p>
---
layout: post
comments: true
title:  우분투에서 Tensorflow, Virtualenv, PyCharm, Jupyter 설치
categories: ChatBot

tags:
- ChatBot(챗봇)
- Tensorflow
---

**<span style='color:DarkRed'>Tensorflow 환경 구축</span>**
> ChatBot을 tensorflow로 구현하고자 이러한 글을 올린다. 또한 우분투에서 tensorflow, Virtualenv, PyCharm, jupyter 설치 4단계가 잘 구성된 블로그가 없어 직접 작성하게 되었다.

<br>


**<span style='color:DarkRed'>Nvidia graphic driver</span>**
>우분투를 설치하고 나면 그래픽 드라이버를 잡고, ```CUDA-> cuDNN-> tensorflow ```순서로 잘 설치 했는데도, 그래픽 kernel문제로 tensorflow-gpu가 안잡히는 경우가 있다. 이 글에서는 ```시스템설정 >> 소프트웨어 & 업데이트 >> 추가 드라이버```에서 ```사용 NVIDA binary driver - version 384.111 출처 nvidia-384 (독점)```을 설치 하여 그래픽 카드를 잡고 CUDA 8.0설치 때 deb 대신 runfile를 사용해 CUDA 8과 nvidia-384가 충돌이 나지 않게 하였다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/zL1niQ7.png"></p>

<br>

**<span style='color:DarkRed'>CUDA 8.0 설치</span>**

- https://developer.nvidia.com/cuda-downloads 에서 CUDA 8.0을 다운받아 설치
<p align="center"><img width="500" height="auto" src="https://i.imgur.com/vjpPBHE.png"></p>

```bash
~$ cd Downloads/
~/Downloads/$ sudo sh cuda_8.0.61_375.26_linux.run
```
<br>

**<span style='color:DarkRed'>CUDA 8.0 환경변수</span>**

- bashrc파일을 열어 CUDA 8.0의 환경변수를 설정

```bash
~$ sudo gedit ~/.bashrc
```

- bashrc에 붙여지는 내용

```bash
export CUDA_HOME=/usr/local/cuda-8.0

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- 변경사항 적용

```bash
~$ source ~/.bashrc
```

<br>

**<span style='color:DarkRed'>Cudnn 설치</span>**

> https://developer.nvidia.com/cudnn 에서 해당 운영체제에 맞는 버전을 받으면 된다. 하지만 Tensorflow v1.3이상부터 cuDNN 6.0과 호환이 되므로 본 저자는 cuDNN 6.0을 설치하였다.
- Tensorflow v1.3 미만 : CUDA 8, cuDNN 5.1 버전
- Tensorflow v1.3 부터 : CUDA 8, cuDNN 6 버전 (2017.08년 기준)

```bash
~$ cd Downloads
~/Downloads/$ sudo tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
```
- cuDNN 파일 이동 및 환경변수 추가

```bash
~/Downloads/$ cd cuda
~/Downloads/cuda/$ sudo cp include/cudnn.h /usr/local/cuda/include
~/Downloads/cuda/$ sudo cp lib64/libcudnn* /usr/local/cuda/lib64
~/Downloads/cuda/$ sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

<br>

**<span style='color:DarkRed'>virtualenv 생성 </span>**

- tensorflow로 이름을 가지는 virtualenv가 ```~/tensorflow```에 생성됨

```bash
~$ sudo apt-get install python3-pip python3.5-dev python-virtualenv
~$ virtualenv --system-site-packages -p python3.5 tensorflow
```

- virtualenv 실행

```bash
~$ source tensorflow/bin/activate
```

<br>

**<span style='color:DarkRed'>Tensorflow 설치 </span>**
- virtualenv 실행하여 tensorflow가 activate된 후에 설치
- https://pypi.python.org/pypi/tensorflow-gpu/1.4.1 앞서 정의한 대로 호환이 되는 버전을 받아야 함

```bash
(tensorflow) ~$ easy_install -U pip
(tensorflow) ~$ cd Downloads/
(tensorflow) ~/Downloads/$ sudo pip install tensorflow_gpu-1.4.1-cp35-cp35m-manylinux1_x86_64.whl 
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
(tensorflow) ~$ sudo ldconfig /usr/local/cuda-8.0/lib64
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
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64:usr/local/cuda-8.0/lib64/libcudart.so.8.0'
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
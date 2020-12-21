---
layout: post
comments: true
title:  Docker push and pull
categories: docker

tags:
- Docker & Git
---

**<span style='color:DarkRed'>Docker push and pull</span>**

- Container를 이미지로 변환후 hub에 올리는 내용입니다.

<br>

**1) Check the container**

```
$ docker ps --all
```

<br>

**2) Commit** 
  - `docker commit [container name] [image name to save]`

```
$ docker commit seg_rec fashion:TTA 
```

<br>


**3) Login**

```
$ docker login
username
pwd
```

<br>


**4) Push**

```
$ export DOCKER_ID_USER="donghwa89"
$ docker tag fashion:TTA $DOCKER_ID_USER/fashion
$ docker push $DOCKER_ID_USER/fashion
```
<p align="center"><img width="600" height="auto" src="../assets/figure/docker_hub.png"></p>



<br>


5) Pull

```
$ docker pull donghwa89/fashion
```
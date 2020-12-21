---
layout: post
comments: true
title:  Docker 사용법
categories: docker

tags:
- Docker & Git
---

**<span style='color:DarkRed'>Docker 사용법</span>**

- 주로 사용되는 Docker 명령어입니다.

<br>

**1) Image list 확인**

```
docker images -a
```
```
(base) donghwa@dh-server:~$ docker images
REPOSITORY          TAG                            IMAGE ID            CREATED             SIZE
fashion             latest                         23f3f15cb75e        7 weeks ago         7.4GB
```
<br>

**2) container 확인**

```
docker ps -a
```

```
(base) donghwa@dh-server:~$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                                            NAMES
c27fb22dc3ba        fashion:latest      "/bin/bash"         7 weeks ago         Up 5 weeks          0.0.0.0:6006->6006/tcp, 0.0.0.0:8282->8282/tcp   fashion
```

<br>

**3) Container build**
- `-it`: runing docker
- `--shm-size`: shared memory
- `-p`: port {host}:{docker}
- `-v`: shared folder {host}:{docker}
- `--name`: container name
- `fashion:latest`: used image

```
docker run -it -p 8282:8282 --shm-size=8gb --env="DISPLAY" -v /home/korea/fashion-recommendation/dataset/:/home/appuser/fashion_repo/dataset -v /home/korea/fashion-recommendation/model:/home/appuser/fashion_repo/model -v /home/korea/fashion-recommendation/src:/home/appuser/fashion_repo/src -v /home/korea/fashion-recommendation/script:/home/appuser/fashion_repo/script --name=seg_rec fashion:latest
```



<br>


**4) Image 제거**

```
docker rmi {image id}
```

<br>

**5) Container 제거**

```
docker rm ${container id}
```

<br>

**6) Docker stop**

```
sudo container kill {id}
```

<br>

**7) Docker start**

```
sudo docker start {id}
```

<br>

**8) Docker runing**

```
sudo docker exec -it {c_id} /bin/bash
```

<br>

**9) File copy from/to Docker**

```
docker cp foo.txt mycontainer:/foo.txt
docker cp mycontainer:/foo.txt foo.txt
```

- An example

```
 docker cp kfashion_wok_history_json_1012.zip 063873750cf0:/home/appuser/detectron2_repo/data
```


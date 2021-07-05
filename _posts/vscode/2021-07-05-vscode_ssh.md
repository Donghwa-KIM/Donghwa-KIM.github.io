---
layout: post
comments: true
title: Docker container를 vscode로 원격제어하기
categories: vscode

tags:
- vscode
- Docker & Git
---

**<span style='color:DarkRed'>Docker container 원격접속</span>**

데이터분석 프로젝트를 Docker 환경에서 작업을 할 때가 많습니다. 이때 원격으로 container에 접속해 작업을 할때가 있는데요. 이 부분을 ssh keygen을 이용해서 접속하는 방법과 옵션설정을 기록하려고 합니다. 먼저 아래의 같이 `host`를 docker container가 설치되어 있는 환경이라하고, `local`을 접속하려는 노트북이라고 합시다. 

**프레임 워크**

<p align="center"><img width="600" height="auto" src="../assets/figure/docker_ssh/docker_ssh.png"></p>

<br>

**1) Local pc에서 ssh-keygen 생성 인증키 등록**

- 먼저, host가 local pc를 받아들일수 있도록 `local pc`에서 인증 key를 생성합니다.
- 생성된 파일은 `~/.ssh/id_rsa.pub` 경로에 생성될 것 입니다.

```bash
# local pc
ssh-keygen -t rsa
```

<br>


**2) Host pc에 ssh-keygen 인증키 등록**

- 발급된 key를 `server pc`에 `~/.ssh/authorized_keys` 경로에 복사 붙여넣기를 해줍니다. (=새로 파일을 생성)
- local pc를 여러개 등록하고 싶다면 개행으로 이어서 붙여주면 됩니다.

<p align="center"><img width="600" height="auto" src="../assets/figure/docker_ssh/keygen.png"></p>

<br>


**3) vscode를 이용한 원격제어**

- `ctrl`+`shift`+`p`를 통해 extension을 실행시킨뒤 `ssh`를 입력해서 `config`파일을 열어준 뒤, `HostName`, `IdentityFile`를 입력해줍니다.
    - `HostName`은 Host PC의 ip주소
    - `IdentityFile` Host PC에 존재하는 local pc 인증키 path

<p align="center"><img width="350" height="auto" src="../assets/figure/docker_ssh/ssh2.png"></p>

<br>



**4) Docker container 연결설정**

- 사전에 Local pc에 docker가 설치되어 있어야 합니다.
- vscode의 ssh remote extension 설치되어 있어야 합니다.
- Preference setting (json)에 `docker.host`(원격하고자하는 PC의 ip주소)를 기입합니다.

<p align="center"><img width="600" height="auto" src="../assets/figure/docker_ssh/option.png"></p>

<br>


**5) Docker container 연결실행**

- `ctrl`+`shift`+`p`를 통해 extension을 실행시킨뒤, `attach to running container` 실행

<p align="center"><img width="300" height="auto" src="../assets/figure/docker_ssh/docker.png"></p>

<br>
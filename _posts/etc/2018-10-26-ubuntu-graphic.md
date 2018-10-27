---
layout: post
comments: true
title:  우분투에서 로그오프 화면이 무한하게 반복되는 현상 해결
categories: Ubuntu

tags:
- etc.
---

- 그래픽카드를 교체하면, 로그인을 반복되는 현상이 생기기 마련이다. 따라서 아래의 글은 무한로그오프 해결방법과 새로운 그래픽카드에 대한 재 설치를 다루고 있다.

<br>

**<span style='color:DarkRed'>무한 로그오프 해결방법</span>**


- Getting terminal screen</span>
    - 로그오프화면에서 ```Ctrl```+```Alt```+```F1```를 눌러 terminal screen으로 들어간다.


<br>


- Nvidia reinstall

    
    ```bash
    sudo apt-get purge nvidia-*
    sudo apt-get install nvidia-current
    ```

<br>

- Get permission denied about infinity log

    - e.g. ```sudo chown donghwa.donghwa .Xauthority```


    ```python
    sudo chown <username>.<username> .Xauthority
    ```
<br>

- Rebooting


    ```python
    sudo reboot
    ```
<br>


**<span style='color:DarkRed'>Graphic Driver Reinstall</span>**

- 로그인 후에 ```시스템설정 >> 소프트웨어 & 업데이트 >> 추가 드라이버```에서 ```사용 NVIDA binary driver - version 384.111 출처 nvidia-384 (독점)```을 설치 하면, 기존에 설치된 tensorflow들이 잘 작동한다. 
<p align="center"><img width="500" height="auto" src="https://i.imgur.com/zL1niQ7.png"></p>


- 위 ```version```은 그래픽드라이버마다 다른데 본 저자는 NVIDIA GeForce RTX 2080 TI(11,264MB)를 사용하여 ```사용 NVIDA binary driver - version 410.57```를 선택하였다.



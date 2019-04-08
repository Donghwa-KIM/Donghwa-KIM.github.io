---
layout: post
comments: true
title:  우분투에서 GPU 메모리 체크
categories: Ubuntu

tags:
- etc.
---

**<span style='color:DarkRed'>GPU 메모리</span>**

- tensorflow를 돌리면 이 run이 제대로 되고 있는지, GPU 메모리를 얼마나 차지하는지 알고 싶을 때가 있다. 아래의 명령어로 쉽게 확인할 수 있다.

```bash
:~$ sudo watch nvidia-smi
```

<p align="center"><img width="500" height="auto" src='https://imgur.com/wkkhEmV.png'></p>

<br>

- CPU, 메모리 사용량도 시각적인 툴을 써보자.


```bash
:~$ sudo apt-get install htop
:~$ htop
```
<p align="center"><img width="500" height="auto" src='https://imgur.com/o1Oj5zo.png'></p>


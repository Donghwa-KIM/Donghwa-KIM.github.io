---
layout: post
comments: true
title: 보안을 위한 ssh 비번 틀린횟수 제한 걸기 및 Reset
categories: ssh
tags:
- etc.
---

**<span style='color:DarkRed'>SSH 비번 틀린횟수 제한</span>**

- 먼저 아래의 코드와 같이 ```/etc/pam.d/common-auth``` 텍스트를 실행시킨다.


```python
userName@userName-main:~$ sudo vim /etc/pam.d/common-auth
```

<br>

- 회색으로 줄쳐진 위치에 적혀있는 내용(```auth    required        pam_tally2.so onerr=fail even_deny_root deny=5 unlock_time=600```)을 추가해준다.
- ```deny=5```: 최대 5번까지 허용
- ```unlock_time=600```: 잠금시간(초)

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/wDdH4So.png"></p>

<br>

- 비번이 한번 틀렸다고 가정하면, 아래와 같이 Failures에 1번 카운팅된다.

```python
userName@userName-main:~$ sudo pam_tally2 -u `userName`
[sudo] password for userName:
Login           Failures Latest failure     From
userName             1    04/30/19 20:13:20  /dev/pts/1
```

<br>

- failure를 reset하고 싶으면, 위 코드에 ```-r```을 붙여준 후 실행한 뒤 한번 더 실행해보면 reset된 것을 확인할 수 있다.


```python
userName@userName-main:~$ sudo pam_tally2 -r -u `userName`
userName@userName-main:~$ sudo pam_tally2 -u `userName`
Login           Failures Latest failure     From
userName             0
```

<br>

**<span style='color:DarkRed'>참고사항</span>**

- 우분투 터미널 모드
	- in: ```Ctrl```+```Alt```+```F1```~```F6```
	- out: ```Ctrl```+```Alt```+```F7```
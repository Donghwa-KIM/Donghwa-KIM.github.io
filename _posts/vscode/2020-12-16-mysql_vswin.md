---
layout: post
comments: true
title: mysql 윈도우 원격 연결방법
categories: MySQL

tags:
- MySQL
---

**<span style='color:DarkRed'> mysql 윈도우 원격 연결방법</span>**

<br>

**<span style='color:blue'> 1) Inbound 규칙생성</span>**

- 윈도우에 설치된 mysql을 원격으로 연결하려면, 아래와 같은 규칙을 설정해줘야 한다.

- `window defense` → `inbound rules` → `port (3306 입력)` → `규칙 생성 완료`

<p align="center"><img width="600" height="auto" src="../assets/figure/window_bound.png"></p>

<br>

**<span style='color:blue'> 2) 유저 등록 및 권한 체크</span>**

- root 계정에서, 접속할 유저아이디 및 비밀번호를 설정한다.

<p align="center"><img width="600" height="auto" src="../assets/figure/mysql_user.png"></p>

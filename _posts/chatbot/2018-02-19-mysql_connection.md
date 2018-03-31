---
layout: post
comments: true
title:  MySQL 외부 접근 허용하기
categories: ChatBot

tags:
- ChatBot(챗봇)
- MySQL
---


**<span style='color:DarkRed'>MySQL 외부 접근 허용하는 방법</span>**

> 외부연결을 허용시키기 위해서는 3가지만 수행해 주면 된다.
> - bind-address 주석하기
> - MySQL에서 허용할 IP 등록하기
> - MySQL 재시작

<br>

**<span style='color:DarkRed'>bind-address 주석하기</span>**

```bash
sudo gedit /etc/mysql/mysql.conf.d/mysqld.cnf
```

<p align="center"><img width="600" height="auto" src="https://i.imgur.com/lTm15mm.png"></p>

<br>

**<span style='color:DarkRed'>MySQL에서 허용할 IP 등록하기 (코드안에 `패스워드`에 비밀번호 대입)</span>**

> 원하는 기호에 맞춰 설정해주면 된다.

Case01)
- 모든 IP 허용하는 방법
- 오픈소스로 허용하고 싶을 때


```sql
INSERT INTO mysql.user (host,user,authentication_string,ssl_cipher, x509_issuer, x509_subject) VALUES ('%','root',password('패스워드'),'','','');때
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%';
FLUSH PRIVILEGES;
```

<br>

Case02)
- IP 대역 허용 ( 예: 111.222.xxx.xxx )하는 방법
- 학교나 기관별로 공유하고 싶을 때

```sql
INSERT INTO mysql.user (host,user,authentication_string,ssl_cipher, x509_issuer, x509_subject) VALUES ('111.222.%','root',password('패스워드'),'','','');
GRANT ALL PRIVILEGES ON *.* TO 'root'@'111.222.%';
FLUSH PRIVILEGES;
```

<br>
Case03)
- 특정 IP 1개 허용 ( 예: 111.222.33.44 )하는 방법때
- 특정한 사람에게 공유하고 싶을 때

```sql
INSERT INTO mysql.user (host,user,authentication_string,ssl_cipher, x509_issuer, x509_subject) VALUES ('111.222.33.44','root',password('패스워드'),'','','');
GRANT ALL PRIVILEGES ON *.* TO 'root'@'111.222.33.44';
FLUSH PRIVILEGES;
```

<br>

- 원래 상태로 복구시키는 방법

```sql
DELETE FROM mysql.user WHERE Host='%' AND User='root';
FLUSH PRIVILEGES;
```

<br>

**<span style='color:DarkRed'>MySQL 재시작</span>**

```bash
~$ sudo /etc/init.d/mysqld restart
```

<br>

##### Reference
https://zetawiki.com/wiki/MySQL_%EC%9B%90%EA%B2%A9_%EC%A0%91%EC%86%8D_%ED%97%88%EC%9A%A9 
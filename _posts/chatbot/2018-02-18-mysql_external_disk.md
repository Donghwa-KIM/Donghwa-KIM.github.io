---
layout: post
comments: true
title:  MySQL 외장하드에 연결하기 
categories: ChatBot

tags:
- ChatBot(챗봇)
- MySQL
---

**<span style='color:DarkRed'>MySQL with an external disk</span>**
>  챗봇모델에 학습되는 데이터는 2006~2017 질의응답 comment 데이터는 대략 1.5 TB가 된다. 이것을 데이터로 저장하고 DB에 업로드하기 위해서 외장하드에 MySQL을 연결할 필요성이 있는데 그 방법을 설명하고자 한다. 

<br>

**<span style='color:DarkRed'> 상태 확인</span>**
- 기존에 존재하고 있는 ```/var/lib/mysql/``` 의 경로를 externel disk의 경로로 바꾸려고 함

```bash
~$ mysql -u root -p
```

```sql
mysql> select @@datadir;
+-----------------+
| @@datadir       |
+-----------------+
| /var/lib/mysql/ |
+-----------------+
1 row in set (0.00 sec)
```
<br>

**<span style='color:DarkRed'> MySQL 서비스중지</span>**
 

```bash
~$ sudo service mysql stop
```

<br>

**<span style='color:DarkRed'> data dir 설정</span>**

- 현재 사용 중인 MySQL의 data dir를 새로운 데이터 경로로 복사

```bash
~$ sudo cp -R /var/lib/mysql /저장되는PATH
~$ sudo chown -R mysql:mysql /저장되는PATH
```

- 예를들어 외장디스크의 경로가 ```/mnt/00B267E9B267E1A0/MYSQL``` 일 경우 아래와 같정이 입력


```bash
# e.g
~$ sudo cp -R /var/lib/mysql /mnt/00B267E9B267E1A0/MYSQL
~$ sudo chown -R mysql:mysql /mnt/00B267E9B267E1A0/MYSQL
```

- ```mysqld.cnf``` 파일을 실행

```bash
~$ sudo gedit /etc/mysql/mysql.conf.d/mysqld.cnf
```

- 이미지 지정되어 있는 datadir을 새로운 데이터 경로로 수정

```bash
...
- datadir = /var/lib/mysql/ -> /저장되는PATH/mysql
- innodb_data_home_dir = /var/lib/mysql/ -> /저장되는PATH/mysql
- innodb_log_group_home_dir = /var/lib/mysql/ -> /저장되는PATH/mysql
...
```

- 예를 들어 경로가 ```/mnt/00B267E9B267E1A0/MYSQL``` 일 경우 아래와 같이 입력

```bash
[mysqld]
#
# * Basic Settings
#
user		= mysql
pid-file	= /var/run/mysqld/mysqld.pid
socket		= /var/run/mysqld/mysqld.sock
port		= 3306
basedir		= /usr
datadir		= /mnt/00B267E9B267E1A0/MYSQL/mysql
innodb_data_home_dir = /mnt/00B267E9B267E1A0/MYSQL/mysql
innodb_log_group_home_dir = /mnt/00B267E9B267E1A0/MYSQL/mysql
tmpdir		= /tmp
lc-messages-dir	= /usr/share/mysql
skip-external-locking
```

<br>

**<span style='color:DarkRed'> data dir access 설정</span>**

- ```usr.sbin.mysqld``` 파일을 실행

```bash
~$ sudo gedit /etc/apparmor.d/usr.sbin.mysqld 
```

- 기존에 존재하는 PATH는 주석으로 처리하고 저장될 경로를 추가
- 예를 들어 경로가 ```/mnt/00B267E9B267E1A0/MYSQL``` 일 경우 아래와 같이 입력

```bash
# Allow data dir access
  /mnt/00B267E9B267E1A0/MYSQL/mysql/ r,
  /mnt/00B267E9B267E1A0/MYSQL/mysql/** rwk,
#  /var/lib/mysql/ r,
#  /var/lib/mysql/** rwk,
```

<br>

**<span style='color:DarkRed'> Configuring AppArmor Access</span>**

```bash
~$ sudo gedit /etc/apparmor.d/abstractions/mysql
```

- 기존에 있던 PATH를 주석으로 처리

```bash
   #/var/lib/mysql{,d}/mysql{,d}.sock rw,
   /ext/mysql{,d}/mysql{,d}.sock rw,
   /{var/,}run/mysql{,d}/mysql{,d}.sock rw,
   /usr/share/{mysql,mysql-community-server,mariadb}/charsets/ r,
   /usr/share/{mysql,mysql-community-server,mariadb}/charsets/*.xml r,,
```

<br>

**<span style='color:DarkRed'> AppArmor profile</span>**
- 변경사항 적용

```bash
~$ sudo /etc/init.d/apparmor reload
```


<br>

**<span style='color:DarkRed'> MySQL 재 시작</span>**

```bash
~$ sudo service mysql start
```

<br>

**<span style='color:DarkRed'> 변경 후 상태 확인</span>**

```bash
~$ mysql -u root -p
```

```sql
mysql> select @@datadir;
+------------------------------------+
| @@datadir                          |
+------------------------------------+
| /mnt/00B267E9B267E1A0/MYSQL/mysql/ |
+------------------------------------+
1 row in set (0.00 sec)
```

<br>

**<span style='color:DarkRed'> 부팅 후 연결이 안될 때</span>**

- 부팅을 하고 나면 MySQL상태가 disconnection 될때 가 있은데 window에 다음과 같이 연결해 주면 됨

```bash
~$ sudo service mysql start
```


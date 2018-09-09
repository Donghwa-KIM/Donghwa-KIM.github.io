---
layout: post
comments: true
title: Spark install on Ubuntu
categories: Spark

tags:
- Spark
---

우분투에서 Spark 2.3.1을 설치하는 방법을 알아보자. 설치 단계는 다음과 같다.

1. Java 설치
2. Spark 다운로드 및 설치
3. Spark란 폴더이름으로 library에 저장
4. 실행 Path 저장


<br>

**<span style='color:DarkRed'>Java 설치</span>**
- 자바가 이미 설치되어 있을 경우 생략

```bash
$ sudo apt-get install default-jdk
```

<br>

**<span style='color:DarkRed'>Spark 다운로드 및 설치</span>**

- ```wget```으로 Spark 2.3.1 zip파일을 받아서 ```tar```명령어로 압축해제

```bash
$ wget http://mirror.navercorp.com/apache/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz

$ tar -xzvf spark-2.3.1-bin-hadoop2.7.tgz
```

<br>

**<span style='color:DarkRed'>Spark란 폴더이름으로 library에 저장</span>**

```bash
# spark란 폴더이름으로 현재 경로에 저장
$ mv spark-2.2.0-bin-hadoop2.7/ spark 

# spark란 폴더이름을 /usr/lib에 이동
$ sudo mv spark/ /usr/lib/
```

<br>

**<span style='color:DarkRed'>실행 Path 저장</span>**

- bashrc 실행

```bash
# open the bashrc 
$ vi ~/.bashrc
```

- 아래의 Path정보들을 ```bashrc```에 입력

```bash
export JAVA_HOME=/usr/lib/jvm/java-8-oracle/jre
export SPARK_HOME=/usr/lib/spark
export PATH=$PATH:SPARK_HOME
```


- 본 저자는 ```java-8-oracle```란 이름으로 java가 설치되어 있기 때문에 위와 같이 입력
- ```default-java```란 이름으로 설치되어 있을 경우, 설치된 이름으로 바꿔줘야 함
- 위에서 설치된 spark의 Path정보도 입력 후 ```$PATH```에 append 

<br>

**<span style='color:DarkRed'>Spark 실행</span>**

- 스파크를 실행시키기 위해 현재경로를 설치된 ```/usr/lib/spark/bin```으로 이동시킨후 ```./spark-shell``` 실행

```bash
:/usr/lib/spark/bin$ ./spark-shell
```
- 아래의 내용이 출력된다면 설치가 잘 완료된 것임

```
Spark session available as 'spark'.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.3.1
      /_/

Using Scala version 2.11.8 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_181)
Type in expressions to have them evaluated.
Type :help for more information.

scala>
```


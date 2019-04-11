---
layout: post
comments: true
title: Scala-Spark install with Jupyter Lab
categories: Spark

tags:
- Spark
---




좀 더 친숙한 Jupyter notebook을 이용하여 scala-based spark를 사용하기 위해 아래와 같이 설치해보자.

<br>

**<span style='color:DarkRed'>Jupyter lab에 scala-spark 사용하기</span>**


- 패키지 설치

```shell
pip install spylon-kernel
```

<br>

- scala 커널 생성

```shell
python -m spylon_kernel install
```

<br>



- Jupyter lab 실행

```shell
jupyter lab
```

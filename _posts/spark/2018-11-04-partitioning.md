---
layout: post
comments: true
title: Partitioning
categories: Spark

tags:
- Spark
---


**<span style='color:DarkRed'>Partitioning</span>**


- Large operation을 효율적으로 사용하기 위해서 데이터를 partitioning을 하여 분산처리
    - Large operation: join(), reduceByKey(), groupBykey()
- Spark 작동 원리를 다음과 같이 나타낼 수 있다.
	1. Shuffle data
	2. Individual tasks로 구분
	3. Distributed to each node(executor) of the cluster
- 여기서 executor를 더 많이 가져가는 방법이 ```partitioning```이다.
- 하지만 너무 많이가져가도 비효율적일수도 있다.
    - 너무 많이 쪼개고, 붙이면 오히려 비 효율적

<br>

**<span style='color:DarkRed'>Example</span>**

[Collaborative Filtering]({{ site.baseurl }}/similarity.html)에 있는 코드에서 ```partitioning```을 적용하고 싶다면 한줄만 바꾸면 된다.

- No partitioning

```scala
val moviePairs = uniqueJoinedRatings.map(makePairs)
```
<br>

- Partitioning
- ```groupBy```(Large operation)전에 moviePairs를 partitioning함


```scala
val moviePairs = uniqueJoinedRatings.map(makePairs).partitionBy(new HashPartitioner(100))
```

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/12_partitioning.ipynb">notebook</a> 코드 참조)









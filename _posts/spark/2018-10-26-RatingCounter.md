---
layout: post
comments: true
title: Spark 작동 원리(map, countByValue)
categories: Spark

tags:
- Spark
---

**<span style='color:DarkRed'>Spark 작동 원리</span>**

> Spark는 크게 두가지 단계로 나눠진다.
> - stage 1(transformation): 수행하고 싶은 operation으로 RDD transformation($RDD_{read} \rightarrow RDD_{parser} \rightarrow RDD_{selection}$)
> - stage 2(action): 선언된 RDD의 결과값 출력(collect, countByValue(), take(),... )

- 아래의 그림처럼 key에 대한 연산(count)이 수행될 때 각 task(key)별로 수행된다는 것을 확인 할 수 있다.

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/jEt67YW.png"></p>

- stage 1: transformation
	- 각 RDD는 병렬로 동시에 실행됨
    - RDD1: textFile()
        - raw data를 RDD로 변환
    - RDD2: map
        - parse out the text
        - RDD의 입력과 출력이 1대1 대응해서 연산(distributed 방식)
            - (row1 $\rightarrow$ RDD1), (row2 $\rightarrow$ RDD2), ...
- stage 2: action
	- countByValue(key별로 count)

---

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/02_RatingsCounter.ipynb">notebook</a> 코드 참조)

```scala
// spark package
import org.apache.spark._
```

    Intitializing Scala interpreter ...

    Spark Web UI available at http://163.152.---.---:----
    SparkContext available as 'sc' (version = 2.3.1, master = local[*], app id = local-1552910928374)
    SparkSession available as 'spark'

    import org.apache.spark._


<br>


```scala
val DATA = "data/u.data"
```




    DATA: String = data/u.data


<br>

- 처음으로 RDD를 생성하기 위해 ```sc```(spark context)를 사용하게 된다.
- 그리고, ```sc.textFile```를 사용하여 ```txt```파일을 불러온다. 


```scala
val lines = sc.textFile(DATA)
```




    lines: org.apache.spark.rdd.RDD[String] = data/u.data MapPartitionsRDD[1] at textFile at <console>:33

<br>


- 1 col: user id
- 2 col: movie id
- 3 col: rating
- 4 col: timestamp


```scala
lines.take(10).foreach(println)
```

    196	242	3	881250949
    186	302	3	891717742
    22	377	1	878887116
    244	51	2	880606923
    166	346	1	886397596
    298	474	4	884182806
    115	265	2	881171488
    253	465	5	891628467
    305	451	3	886324817
    6	86	3	883603013

<br>

```scala
// pick up 3th col in (0,1,2)  
val ratings = lines.map(line => line.split("\t")(2))
```




    ratings: org.apache.spark.rdd.RDD[String] = MapPartitionsRDD[2] at map at <console>:36



<br>

```scala
ratings.take(10).foreach(println)
```

    3
    3
    1
    2
    1
    4
    2
    5
    3
    3

<br>

- **countByValue**
    - value에 대한 unique set의 count가 key value로 저장이 됨


```scala
val results = ratings.countByValue()
```




    results: scala.collection.Map[String,Long] = Map(4 -> 34174, 5 -> 21201, 1 -> 6110, 2 -> 11370, 3 -> 27145)


<br>


```scala
results("1")
```




    res3: Long = 6110


<br>


```scala
results("2")
```




    res4: Long = 11370


<br>


```scala
results.foreach(println)
```

    (4,34174)
    (5,21201)
    (1,6110)
    (2,11370)
    (3,27145)

<br>

- key 기준으로 sort


```scala
val sortedResults = results.toSeq.sortBy(_._1)
```




    sortedResults: Seq[(String, Long)] = ArrayBuffer((1,6110), (2,11370), (3,27145), (4,34174), (5,21201))

<br>




```scala
sortedResults.foreach(println)
```

    (1,6110)
    (2,11370)
    (3,27145)
    (4,34174)
    (5,21201)

<br>

- 함수로 표현 해보자


```scala
def main(data: String) = {

    val lines = sc.textFile(data)
    val ratings = lines.map(line => line.split("\t")(2))
    val results = ratings.countByValue()
    val sortedResults = results.toSeq.sortBy(_._1)

    sortedResults.foreach(println)
    
  }
```




    main: (data: String)Unit



<br>

```scala
main(DATA)
```

    (1,6110)
    (2,11370)
    (3,27145)
    (4,34174)
    (5,21201)


---
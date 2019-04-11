---
layout: post
comments: true
title: KeyValue(zip, mapValues, reduceByKey, collect, slice)
categories: Spark

tags:
- Spark
---

**<span style='color:DarkRed'>key/values Examples</span>**

- Task: {age: number of people}
- Used functions
	- zip: 두개의 Array를 pair하게 묶을 때 사용 
	- mapValues: 연산되는 대상이 (key,value)가 아니라 value만 선택되어 mapping이 됨
	- reduceByKey: key별로 누적해서 value를 연산
	- collect: RDD를 Array로 바꿈
	- slice: Array 인덱스 기준으로 선택

---

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/03_key_value_RDD.ipynb">notebook</a> 코드 참조)

```scala
import org.apache.spark._
```


    Intitializing Scala interpreter ...

    Spark Web UI available at http://163.152.---.---:----
    SparkContext available as 'sc' (version = 2.3.1, master = local[*], app id = local-1552908203569)
    SparkSession available as 'spark'

    import org.apache.spark._

<br>



```scala
val lines = sc.textFile("./data/fakefriends.csv")
```




    lines: org.apache.spark.rdd.RDD[String] = ./data/fakefriends.csv MapPartitionsRDD[1] at textFile at <console>:28


<br>

- 1 col: user id
- 2 col: name
- 3 col: user age
- 4 col: #friends


```scala
lines.take(10).foreach(println)
```

    0,Will,33,385
    1,Jean-Luc,26,2
    2,Hugh,55,221
    3,Deanna,40,465
    4,Quark,68,21
    5,Weyoun,59,318
    6,Gowron,37,220
    7,Will,54,307
    8,Jadzia,38,380
    9,Hugh,27,181

<br>

```scala
val fields = lines.map(line => line.split(","))
```




    fields: org.apache.spark.rdd.RDD[Array[String]] = MapPartitionsRDD[44] at map at <console>:30


<br>


```scala
val age = fields.map(field => field(2).toInt )
val numFriends = fields.map(field => field(3).toInt )
```




    age: org.apache.spark.rdd.RDD[Int] = MapPartitionsRDD[45] at map at <console>:32
    numFriends: org.apache.spark.rdd.RDD[Int] = MapPartitionsRDD[46] at map at <console>:33


<br>

- Extract two arrays


```scala
age.take(10).foreach(println)
```

    33
    26
    55
    40
    68
    59
    37
    54
    38
    27

<br>

```scala
numFriends.take(10).foreach(println)
```

    385
    2
    221
    465
    21
    318
    220
    307
    380
    181

<br>

- zip the two arrays


```scala
val merged = age.zip(numFriends)
```




    merged: org.apache.spark.rdd.RDD[(Int, Int)] = ZippedPartitionsRDD2[48] at zip at <console>:35

<br>



```scala
merged.take(10).foreach(println)
```

    (33,385)
    (26,2)
    (55,221)
    (40,465)
    (68,21)
    (59,318)
    (37,220)
    (54,307)
    (38,380)
    (27,181)

<br>

- As a function
	- 각 행을 parsing한 후 전처리된 행 결과들을 append
    - append시키는 ```scala``` 문법: ```def = {output}```


```scala
def parseLine(line: String)={ 
    val fields = line.split(",")
    val age = fields(2).toInt
    val numFriends = fields(3).toInt
    (age, numFriends) // will be appended as element
}
```




    parseLine: (line: String)(Int, Int)


<br>



```scala
val rdd = lines.map(parseLine)
```




    rdd: org.apache.spark.rdd.RDD[(Int, Int)] = MapPartitionsRDD[17] at map at <console>:38


<br>


```scala
rdd.take(10).foreach(println)
```

    (33,385)
    (26,2)
    (55,221)
    (40,465)
    (68,21)
    (59,318)
    (37,220)
    (54,307)
    (38,380)
    (27,181)

<br>

- **mapValues**: value의 형태를 바꾸고 싶을 때 사용되는 함수
    - transform value(x) into something(x,1)


```scala
val value_expand = rdd.mapValues(x => (x,1))
```




    value_expand: org.apache.spark.rdd.RDD[(Int, (Int, Int))] = MapPartitionsRDD[41] at mapValues at <console>:40


<br>


```scala
value_expand.take(10).foreach(println)
```

    (33,(385,1))
    (26,(2,1))
    (55,(221,1))
    (40,(465,1))
    (68,(21,1))
    (59,(318,1))
    (37,(220,1))
    (54,(307,1))
    (38,(380,1))
    (27,(181,1))

<br>

- **reduceByKey**: key에 대해서 각 연산을 적용하고 싶을 때 사용
    - key 마다 각각 연산
    - unique key마다 2개의 example들을 선택해 친구의 수와 해당 사람의 수를 누적해서 더해 나감
        - input
            - (33, (385, 1)) 
            - (33, (2,1))
        - output
            - (33, (387, 2))


```scala
//reduceByKey(현재 value, 누적 value) => (현재 value 1 + 누적 value 1, 현재 value 2 + 누적 value 2)
val key_cum = value_expand.reduceByKey((x,y)=> (x._1 + y._1, x._2 + y._2))
```




    key_cum: org.apache.spark.rdd.RDD[(Int, (Int, Int))] = ShuffledRDD[42] at reduceByKey at <console>:43


<br>

- (unique key, (#friends, count key))


```scala
key_cum.take(10).foreach(println)
```

    (34,(1473,6))
    (52,(3747,11))
    (56,(1840,6))
    (66,(2488,9))
    (22,(1445,7))
    (28,(2091,10))
    (54,(3615,13))
    (46,(2908,13))
    (48,(2814,10))
    (30,(2594,11))

<br>

- x = values = (#friends, count key)


```scala
val averageByAge = key_cum.mapValues(x => x._1/x._2)
```




    averageByAge: org.apache.spark.rdd.RDD[(Int, Int)] = MapPartitionsRDD[43] at mapValues at <console>:44



<br>


```scala
averageByAge.take(10).foreach(println)
```

    (34,245)
    (52,340)
    (56,306)
    (66,276)
    (22,206)
    (28,209)
    (54,278)
    (46,223)
    (48,281)
    (30,235)

<br>

- action


```scala
val results = averageByAge.collect()
```




    results: Array[(Int, Int)] = Array((34,245), (52,340), (56,306), (66,276), (22,206), (28,209), (54,278), (46,223), (48,281), (30,235), (50,254), (32,207), (36,246), (24,233), (62,220), (64,281), (42,303), (40,250), (18,343), (20,165), (38,193), (58,116), (44,282), (60,202), (26,242), (68,269), (19,213), (39,169), (41,268), (61,256), (21,350), (47,233), (55,295), (53,222), (25,197), (29,215), (59,220), (65,298), (35,211), (27,228), (57,258), (51,302), (33,325), (37,249), (23,246), (45,309), (63,384), (67,214), (69,235), (49,184), (31,267), (43,230))


<br>

- Array slice


```scala
results(0)
```




    res71: (Int, Int) = (34,245)



<br>


```scala
results.slice(0,2)
```




    res72: Array[(Int, Int)] = Array((34,245), (52,340))




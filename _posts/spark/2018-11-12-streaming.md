---
layout: post
comments: true
title: Streaming
categories: Spark

tags:
- Spark
---


**<span style='color:DarkRed'>Streaming</span>** 

- Streaming을 사용하게 되면 real-time으로 연산결과를 확인/저장할 수 있다.

<br>

**<span style='color:DarkRed'>특징</span>** 


- 연산방식
    - byte단위가 아닌 1초단위로 **<span style='color:blue'>실시간</span>** 연산이 수행
    - 일반적으로 small RDD chunk단위로 수행됨
- 저장
    - checkpoint를 이용하여 저장
- 연산
    - Dstreaming 객체의 형태지만 RDD함수로 연산도 할 수 있음
- 연산범위
    - window operation으로 특정시점에 대한 연산이 가능
        - e.g. reducedByWindow()
    - 세션별로 연산이 가능
        - e.g. updateStateByKey()



---

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/19_streaming.ipynb">notebook</a> 코드 참조)



```scala
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.StreamingContext._
```




    import org.apache.spark._
    import org.apache.spark.SparkContext._
    import org.apache.spark.streaming._
    import org.apache.spark.streaming.StreamingContext._



<br>

- Streaming을 하기 위해 ```sparkContext```, ```sparkSession```가 아닌 ```StreamingContext```를 생성해 주자.

```scala
val ssc = new StreamingContext(sc, Seconds(2))
```




    ssc: org.apache.spark.streaming.StreamingContext = org.apache.spark.streaming.StreamingContext@66c6ae57


<br>

---

**<span style='color:DarkRed'>Setup netcat</span>** 

- Terminal에 입력되는 내용(텍스트)이 실시간으로 스파크에 전달되게 한다.
- Terminal를 열어 아래와 같이 실행
    - OpenBSD netcat(nc)
    - ```-l```: Listen mode, for inbound connects
    - ```-k```: Keep inbound sockets open for multiple connects
    - "5000": port

```bash
# on the terminal
nc -lk 5000
```

---

<br>

- 다시 돌아와서, spark와 netcat의 연결을 선언하는 코드는 아래와 같다.

```scala
//ssc.socketTextStream("IP address", port)
val lines = ssc.socketTextStream("localhost", 5000)
```




    lines: org.apache.spark.streaming.dstream.ReceiverInputDStream[String] = org.apache.spark.streaming.dstream.SocketInputDStream@3c501c53

<br>

- 띄어쓰기로 문장 parsing

```scala
val words = lines.flatMap(_.split(" "))
```




    words: org.apache.spark.streaming.dstream.DStream[String] = org.apache.spark.streaming.dstream.FlatMappedDStream@788fa4fc


<br>

- ```reduceByKeyAndWindow``` 사용법을 살펴보자.
- 아래 예제에서는,
	- ```window length```: 30초만큼의 RDD연산(계산되는 범위)
	- ```sliding interval```: 10초마다 계산(계산되는 주)
	- 계산방식은 단어(```key```)별로 단어갯수(```value```)들을 더하는 방식(```x+y```)

```scala
import org.apache.spark.streaming.StreamingContext._
val pairs = words.map(word => (word, 1))
// Reduce last 30 seconds of data, every 10 seconds
val wordCounts = pairs.reduceByKeyAndWindow((a:Int,b:Int) => (a + b) , Seconds(30), Seconds(10))
```




    import org.apache.spark.streaming.StreamingContext._
    pairs: org.apache.spark.streaming.dstream.DStream[(String, Int)] = org.apache.spark.streaming.dstream.MappedDStream@6a98c5dd
    wordCounts: org.apache.spark.streaming.dstream.DStream[(String, Int)] = org.apache.spark.streaming.dstream.ShuffledDStream@2450eaf5

<br>



```scala
wordCounts.print
```
<br>

- ```ssc.start()```를 사용하여 ```necat```과 최종적으로 연결이 된다.
- 따라서, ```necat```에 입력이 되는 텍스트들이 위에 쓰여진 코드의 결과들로 산출 될 수 있다.


```scala
ssc.start()             // Start the computation
ssc.awaitTermination()  // Wait for the computation to terminate
```

<br>

- 예제로, 아래와 같은 텍스트를 입력해 보았다.



<p align="center"><img width="700" height="auto" src="https://i.imgur.com/mChiud4.png"></p>



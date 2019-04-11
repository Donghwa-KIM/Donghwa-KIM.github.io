---
layout: post
comments: true
title: Broadcast(Source.fromFile)
categories: Spark

tags:
- Spark
---

**<span style='color:DarkRed'>Broadcast Variables</span>**


- 데이터가 굉장히 크다면 메모리가 부족할 수도 있는 문제가 발생
- ```Source.fromFile```을 이용해 데이터를 청크단위로 쪼개 클러스터안에 있는 노드에 보낼 수 있음

<br>

**<span style='color:DarkRed'>Movie ViewCount 예제</span>**: **<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/09_ad_broadcast_variables.ipynb">notebook</a> 코드 참조)





```scala
import org.apache.spark.SparkContext
import scala.io.{Codec, Source}
import java.nio.charset.CodingErrorAction
```


    Intitializing Scala interpreter ...

    Spark Web UI available at http://163.152.---.---:----
    SparkContext available as 'sc' (version = 2.3.1, master = local[*], app id = local-1553092409518)
    SparkSession available as 'spark'

    import org.apache.spark.SparkContext
    import scala.io.{Codec, Source}
    import java.nio.charset.CodingErrorAction





<br>

**<span style='color:blue'>Source.fromFile</span>**
 
- Source.fromFile은 Iterator로 한번 사용되면 그 상태가 변화하게 된다. 
    - Source.fromFile은 character단위로 구분되어 있음
- 문장단위로 불러오고 싶은 때 getlines()를 사용


```scala
val _lines = Source.fromFile("./data/u.item").getLines()
_lines.take(10).foreach(println)
```

    1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0
    3|Four Rooms (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995)|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|1|0|0
    4|Get Shorty (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Get%20Shorty%20(1995)|0|1|0|0|0|1|0|0|1|0|0|0|0|0|0|0|0|0|0
    5|Copycat (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Copycat%20(1995)|0|0|0|0|0|0|1|0|1|0|0|0|0|0|0|0|1|0|0
    6|Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)|01-Jan-1995||http://us.imdb.com/Title?Yao+a+yao+yao+dao+waipo+qiao+(1995)|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0
    7|Twelve Monkeys (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Twelve%20Monkeys%20(1995)|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|1|0|0|0
    8|Babe (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Babe%20(1995)|0|0|0|0|1|1|0|0|1|0|0|0|0|0|0|0|0|0|0
    9|Dead Man Walking (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Dead%20Man%20Walking%20(1995)|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0
    10|Richard III (1995)|22-Jan-1996||http://us.imdb.com/M/title-exact?Richard%20III%20(1995)|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|1|0

    _lines: Iterator[String] = non-empty iterator

<br>



```scala
val _splitted = _lines.map(line => line.split("\\|"))
```

    _splitted: Iterator[Array[String]] = non-empty iterator


<br>

- 10 row만 가져와 array 변환


```scala
val _tmp = _splitted.take(10).toArray
```




    _tmp: Array[Array[String]] = Array(Array(1, Toy Story (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?Toy%20Story%20(1995), 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), Array(2, GoldenEye (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?GoldenEye%20(1995), 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0), Array(3, Four Rooms (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?Four%20Rooms%20(1995), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0), Array(4, Get Shorty (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?Get%20Shorty%20(1995), 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), Array(5, Copycat (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?Copycat%20(1995), 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, ...


<br>

- Print first row


```scala
val first_row = _tmp(0)
val second_row = _tmp(1)
```




    first_row: Array[String] = Array(1, Toy Story (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?Toy%20Story%20(1995), 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    second_row: Array[String] = Array(2, GoldenEye (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?GoldenEye%20(1995), 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)

<br>



```scala
first_row.foreach(println)
```

    1
    Toy Story (1995)
    01-Jan-1995
    
    http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)
    0
    0
    0
    1
    1
    1
    0
    0
    0
    0
    0
    0
    0
    0
    0
    0
    0
    0
    0

<br>

- Token의 갯수가 1이상인 row 


```scala
first_row.length >1
```




    res2: Boolean = true


<br>


- Map(dictionany)
    - 계속적인 append를 위해서, ```val```가 아닌 ```var```(variable: mutable)로 선언
    - row의 첫번째 인덱스를 정수타입으로 변환하여 key로 설정
    - row의 두번째 인덱스를 가져와서 value로 설정
    


```scala
var _movieNames: Map[Int, String] = Map()
```




    _movieNames: Map[Int,String] = Map()

<br>

- python $vs$ scala for dictionary

```python
# in python
dictionary = {}
dictionary[key] = value
```

```scala
// in scala
var dictionary: Map[Int, String] = Map()
dictionary += key -> value
```
<br>


- ```_movieNames```에 (key, value)를 append하는 과정

```scala
_movieNames += first_row(0).toInt -> first_row(1)
```

<br>

```scala
_movieNames
```




    res4: Map[Int,String] = Map(1 -> Toy Story (1995))


<br>



- (key, value)가 누적되는 것을 확인할 수 있다.

```scala
_movieNames += second_row(0).toInt -> second_row(1)
```

```scala
_movieNames
```




    res6: Map[Int,String] = Map(1 -> Toy Story (1995), 2 -> GoldenEye (1995))


<br>

---

- As a function


```scala
def loadMovieTitles(): Map[Int, String] = {

    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

    var movieNames: Map[Int, String] = Map()

    val lines = Source.fromFile("./data/u.item").getLines()
    
    for (line <- lines) {
      var splitted = line.split("\\|")
      if (splitted.length > 1) {
        movieNames += (splitted(0).toInt -> splitted(1))
      }
    }
    //dictionary
    return movieNames
}


def parseLine(line: String): (Int, Int) = {
    (line.split("\t")(1).toInt, 1)
}

```




    loadMovieTitles: ()Map[Int,String]
    parseLine: (line: String)(Int, Int)


<br>


**<span style='color:blue'>broadcast</span>**

- ```var movieNames```를 반환하는 loadMovieTitles를 chunk단위로 계산하게 해주는 spark 함수는 ```sc.broadcast```


```scala
val nameDict = sc.broadcast(loadMovieTitles)
```




    nameDict: org.apache.spark.broadcast.Broadcast[Map[Int,String]] = Broadcast(0)



<br>

```scala
nameDict.value
```




    res7: Map[Int,String] = Map(645 -> Paris Is Burning (1990), 892 -> Flubber (1997), 69 -> Forrest Gump (1994), 1322 -> Metisse (Caf� au Lait) (1993), 1665 -> Brother's Kiss, A (1997), 1036 -> Drop Dead Fred (1991), 1586 -> Lashou shentan (1992), 1501 -> Prisoner of the Mountains (Kavkazsky Plennik) (1996), 809 -> Rising Sun (1993), 1337 -> Larger Than Life (1996), 1411 -> Barbarella (1968), 629 -> Victor/Victoria (1982), 1024 -> Mrs. Dalloway (1997), 1469 -> Tom and Huck (1995), 365 -> Powder (1995), 1369 -> Forbidden Christ, The (Cristo proibito, Il) (1950), 138 -> D3: The Mighty Ducks (1996), 1190 -> That Old Feeling (1997), 1168 -> Little Buddha (1993), 760 -> Screamers (1995), 101 -> Heavy Metal (1981), 1454 -> Angel and the Badman (1947), 1633 -> � k�ldum klaka (Cold Fever) (1994), ...


<br>

```scala
val lines = sc.textFile("./data/u.data")
```




    lines: org.apache.spark.rdd.RDD[String] = ./data/u.data MapPartitionsRDD[1] at textFile at <console>:28


<br>

- parseLine: row별로 ```split("\t")```한 후 movieID에 1를 append하는 함수


```scala
val movies = lines.map(parseLine)
```




    movies: org.apache.spark.rdd.RDD[(Int, Int)] = MapPartitionsRDD[2] at map at <console>:32



<br>



```scala
movies.take(10).foreach(println)
```

    (242,1)
    (302,1)
    (377,1)
    (51,1)
    (346,1)
    (474,1)
    (265,1)
    (465,1)
    (451,1)
    (86,1)

<br>

- movieID별로 counting

```scala
val moviesCount = movies.reduceByKey((x, y) => x + y)
```




    moviesCount: org.apache.spark.rdd.RDD[(Int, Int)] = ShuffledRDD[3] at reduceByKey at <console>:34


<br>

```scala
moviesCount.take(10).foreach(println)
```

    (454,16)
    (1084,21)
    (1410,4)
    (772,49)
    (752,39)
    (586,34)
    (428,121)
    (1328,6)
    (464,27)
    (14,183)


<br>


- count기준으로 sorting(ascending)

```scala
val sortedMoviesCount = moviesCount.sortBy(_._2)
```




    sortedMoviesCount: org.apache.spark.rdd.RDD[(Int, Int)] = MapPartitionsRDD[8] at sortBy at <console>:36


<br>


```scala
// ascending ordering
sortedMoviesCount.take(10).foreach(println)
```

    (1494,1)
    (1414,1)
    (1596,1)
    (1630,1)
    (1632,1)
    (1310,1)
    (1670,1)
    (1320,1)
    (1678,1)
    (1674,1)

<br>

- 지금까지, 총 2개의 RDD 객체를 만들어냄
	- nameDict은 (idx, name)으로 구성되어 있음
	- sortedMoviesCount는 (idx, count)로 구성되어 있음
- sortedMoviesCount의 각 idx에 대해서 대응되는 namedict의 이름을 가져오고, count를 붙이려고 함


```scala
nameDict.value
```




    res11: Map[Int,String] = Map(645 -> Paris Is Burning (1990), 892 -> Flubber (1997), 69 -> Forrest Gump (1994), 1322 -> Metisse (Caf� au Lait) (1993), 1665 -> Brother's Kiss, A (1997), 1036 -> Drop Dead Fred (1991), 1586 -> Lashou shentan (1992), 1501 -> Prisoner of the Mountains (Kavkazsky Plennik) (1996), 809 -> Rising Sun (1:993), 1337 -> Larger Than Life (1996), 1411 -> Barbarella (1968), 629 -> Victor/Victoria (1982), 1024 -> Mrs. Dalloway (1997), 1469 -> Tom and Huck (1995), 365 -> Powder (1995), 1369 -> Forbidden Christ, The (Cristo proibito, Il) (1950), 138 -> D3: The Mighty Ducks (1996), 1190 -> That Old Feeling (1997), 1168 -> Little Buddha (1993), 760 -> Screamers (1995), 101 -> Heavy Metal (1981), 1454 -> Angel and the Badman (1947), 1633 -> � k�ldum klaka (Cold Fever) (1994),...


<br>

- m._1: movieID
- m._2: the count
- ```nameDict.value```을 이용하여 ID를 name으로 바꿔 줌
- 그 결과(name, count)를 두번째에 위치(_._2)하고 있는 count기준으로 decreasing(false) sort

```scala
// m = (idx, count)
val results = sortedMoviesCount.map(m => (nameDict.value(m._1), m._2)).sortBy(_._2,false).collect()
```




    results: Array[(String, Int)] = Array((Star Wars (1977),583), (Contact (1997),509), (Fargo (1996),508), (Return of the Jedi (1983),507), (Liar Liar (1997),485), (English Patient, The (1996),481), (Scream (1996),478), (Toy Story (1995),452), (Air Force One (1997),431), (Independence Day (ID4) (1996),429), (Raiders of the Lost Ark (1981),420), (Godfather, The (1972),413), (Pulp Fiction (1994),394), (Twelve Monkeys (1995),392), (Silence of the Lambs, The (1991),390), (Jerry Maguire (1996),384), (Rock, The (1996),378), (Empire Strikes Back, The (1980),367), (Star Trek: First Contact (1996),365), (Back to the Future (1985),350), (Titanic (1997),350), (Mission: Impossible (1996),344), (Fugitive, The (1993),336), (Indiana Jones and the Last Crusade (1989),331), (Willy Wonka and the Chocolate F...

<br>


```scala
results.take(10).foreach(println)
```

    (Star Wars (1977),583)
    (Contact (1997),509)
    (Fargo (1996),508)
    (Return of the Jedi (1983),507)
    (Liar Liar (1997),485)
    (English Patient, The (1996),481)
    (Scream (1996),478)
    (Toy Story (1995),452)
    (Air Force One (1997),431)
    (Independence Day (ID4) (1996),429)


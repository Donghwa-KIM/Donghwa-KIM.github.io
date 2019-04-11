---
layout: post
comments: true
title: Collaborative Filtering(persist, cache, join, filterDuplicates, cosine similarity)
categories: Spark

tags:
- Spark
---

**<span style='color:DarkRed'>Collaboarative Filtering</span>**

- ```u.item```을 이용해 idx와 movie name을 맵핑해주는 Map생성
	- (userID, name)
- ```u.data```를 이용해 유저별, movie Rating 산출
	- (userID, (movieID, Rating)) 
- filterDuplicates
	- 중복되는 데이터 삭제하는 함수를 정의하여 사용
- makePairs
	- 영화끼리, 평점끼리 묶어주는 함수를 적용
	- (```movie_id1```, ```movie_id2```),(```rating1```, ```rating2```)
- 각 moive에 대해서 groupByKey() 사용하여, movie pair마다 rating pair의 코사인 유사도와 연결정도(movie pair의 길이)를 산출  

<br>

**<span style='color:DarkRed'>persist, Cache</span>**
- 같은 작업을 반복적으로 하지 않게, copy해두는 방식 
	- persist: 디스크에 cache할건지, 메모리에 cache할건지 선택할 수 있음
	- Cache: 위에 작업을 자동으로 알아서 하게 함


---

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/11_cosine_similarity.ipynb">notebook</a> 코드 참조)


```scala
import org.apache.spark._
import scala.io.{Codec, Source}
import java.nio.charset.CodingErrorAction
```


    Intitializing Scala interpreter ...

    Spark Web UI available at http://163.152.---.---:----
    SparkContext available as 'sc' (version = 2.3.1, master = local[*], app id = local-1554382960485)
    SparkSession available as 'spark'

    import org.apache.spark._
    import scala.io.{Codec, Source}
    import java.nio.charset.CodingErrorAction



<br>

```scala
implicit val codec = Codec("UTF-8")
codec.onMalformedInput(CodingErrorAction.REPLACE)
codec.onUnmappableCharacter(CodingErrorAction.REPLACE)
```




    codec: scala.io.Codec = UTF-8
    res0: codec.type = UTF-8

<br>

- ```u.item```을 이용해 idx와 movie name을 맵핑해주는 Map을 만들고자 한다.
- 먼저 데이터를 broadcast로 받는 과정을 살펴보자.


```scala
val lines = Source.fromFile("data/u.item").getLines()
```




    lines: Iterator[String] = non-empty iterator



<br> 

- First row

```scala
val tmp = lines.toList.take(1)(0)
```




    tmp: String = 1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0



<br>


```scala
tmp.split("\\|")
```




    res1: Array[String] = Array(1, Toy Story (1995), 01-Jan-1995, "", http://us.imdb.com/M/title-exact?Toy%20Story%20(1995), 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)



<br>

- 위와 같이, row들 split해서 1번째(idx)를 key, 2번째(name)를 value인 Map을 만들어 보자. 
- append되는 ```movieNames```는 variable로 설정해준다.

```scala
def loadMovieNames() : Map[Int, String] = {    
    
    var movieNames: Map[Int, String] = Map()
    val lines = Source.fromFile("data/u.item").getLines()

    for (line <- lines) {
      val fields = line.split("\\|")
      movieNames += (fields(0).toInt -> fields(1))
    }

    movieNames
}

```




    loadMovieNames: ()Map[Int,String]


<br>


```scala
val movieNames = loadMovieNames()
```




    movieNames: Map[Int,String] = Map(645 -> Paris Is Burning (1990), 892 -> Flubber (1997), 69 -> Forrest Gump (1994), 1322 -> Metisse (Caf� au Lait) (1993), 1665 -> Brother's Kiss, A (1997), 1036 -> Drop Dead Fred (1991), 1586 -> Lashou shentan (1992), 1501 -> Prisoner of the Mountains (Kavkazsky Plennik) (1996), 809 -> Rising Sun (1993), 1337 -> Larger Than Life (1996), 1411 -> Barbarella (1968), 629 -> Victor/Victoria (1982), 1024 -> Mrs. Dalloway (1997), 1469 -> Tom and Huck (1995), 365 -> Powder (1995), 1369 -> Forbidden Christ, The (Cristo proibito, Il) (1950), 138 -> D3: The Mighty Ducks (1996), 1190 -> That Old Feeling (1997), 1168 -> Little Buddha (1993), 760 -> Screamers (1995), 101 -> Heavy Metal (1981), 1454 -> Angel and the Badman (1947), 1633 -> � k�ldum klaka (Cold Fever) (1...



<br>

- ```u.data```를 이용해 유저별, movie Rating 산출
	- (userID, (movieID, Rating)) 


```scala
val data = sc.textFile("data/u.data")
```




    data: org.apache.spark.rdd.RDD[String] = data/u.data MapPartitionsRDD[1] at textFile at <console>:32



<br>

- 1 col: User ID
- 2 col: Movie ID
- 3 col: Rating
- 4 col: Timestamp


```scala
data.take(1)(0).split("\t") //first row
```




    res2: Array[String] = Array(196, 242, 3, 881250949)



<br>

- ```type``` 변수타입을 사전에 정의

```scala
type Rating = (Int, Double) //static

def parseLine(line: String): (Int, Rating) = {
    val fields = line.split("\t")
    (fields(0).toInt, (fields(1).toInt, fields(2).toDouble))
}

val ratings = data.map(parseLine)
```




    defined type alias Rating
    parseLine: (line: String)(Int, Rating)
    ratings: org.apache.spark.rdd.RDD[(Int, Rating)] = MapPartitionsRDD[2] at map at <console>:41

<br>



```scala
ratings
```




    res3: org.apache.spark.rdd.RDD[(Int, Rating)] = MapPartitionsRDD[2] at map at <console>:41



<br>

- (userID, movieID, ratings)

```scala
ratings.take(10).foreach(println)
```

    (196,(242,3.0))
    (186,(302,3.0))
    (22,(377,1.0))
    (244,(51,2.0))
    (166,(346,1.0))
    (298,(474,4.0))
    (115,(265,2.0))
    (253,(465,5.0))
    (305,(451,3.0))
    (6,(86,3.0))

<br>

---



**<span style='color:blue'>join</span>**

- 먼저 ```join``` 함수가 어떻게 작동되는지 살펴보자.

```scala
val shipMap = sc.parallelize(Array((1, "Enterprise"),
                  (1, "Enterprise-D"),
                  (2, "Deep Space Nine"),
                  (2 -> "Voyager")))
```




    shipMap: org.apache.spark.rdd.RDD[(Int, String)] = ParallelCollectionRDD[3] at parallelize at <console>:32

<br>


- ```join```할 경우 같은 key(user)에 대한 모든 조합의 경우의 수(두개의 영화)로 concat됨


```scala
val joined_shipMap = shipMap.join(shipMap)
```




    joined_shipMap: org.apache.spark.rdd.RDD[(Int, (String, String))] = MapPartitionsRDD[6] at join at <console>:34

<br>



```scala
joined_shipMap.foreach(println)
```

    (1,(Enterprise,Enterprise))
    (1,(Enterprise,Enterprise-D))
    (2,(Deep Space Nine,Deep Space Nine))
    (2,(Deep Space Nine,Voyager))
    (2,(Voyager,Deep Space Nine))
    (2,(Voyager,Voyager))
    (1,(Enterprise-D,Enterprise))
    (1,(Enterprise-D,Enterprise-D))

---

<br>


- 다시 돌아와서, rating에 자기자신을 join시키면 아래와 같이 나타낼수 있다.

```scala
ratings.take(10).foreach(println)
```

	//(userID,(movieID, rating))
    (196,(242,3.0))
    (186,(302,3.0))
    (22,(377,1.0))
    (244,(51,2.0))
    (166,(346,1.0))
    (298,(474,4.0))
    (115,(265,2.0))
    (253,(465,5.0))
    (305,(451,3.0))
    (6,(86,3.0))

<br>

```scala
val joinedRatings = ratings.join(ratings)
```




    joinedRatings: org.apache.spark.rdd.RDD[(Int, (Rating, Rating))] = MapPartitionsRDD[9] at join at <console>:38



<br>

- userID별로 가능한 (movieID, rating) 조합을 구할 수 있다는 것을 확인할 수 있다.

```scala
joinedRatings.take(5).foreach(println)
```

    (778,((94,2.0),(94,2.0)))
    (778,((94,2.0),(78,1.0)))
    (778,((94,2.0),(7,4.0)))
    (778,((94,2.0),(1273,3.0)))
    (778,((94,2.0),(265,4.0)))

<br>
---

**<span style='color:blue'>filterDuplicates</span>**

- 다음으로, ```filterDuplicates```함수가 어떻게 작동하는지 살펴보자.
	- ```joinedRatings```에서 rating이 중복되는 것을 삭제하는 함수이다.

```scala
val ratingPair = joinedRatings.take(2)(1)
```




    ratingPair: (Int, (Rating, Rating)) = (778,((94,2.0),(78,1.0)))

<br>



```scala
val firstRating = ratingPair._2._1
val secondRating = ratingPair._2._2
```




    firstRating: Rating = (94,2.0)
    secondRating: Rating = (78,1.0)



<br>


```scala
firstRating._1
```




    res7: Int = 94


<br>


```scala
secondRating._1
```




    res8: Int = 78


<br>

- 다르다면 중복이 되지 않는 것을 확인할 수 있다.

```scala
firstRating._1 < secondRating._1
```




    res9: Boolean = false


<br>

- 함수로 정의하면, 아래와 같다.

```scala
 def filterDuplicates(ratingPair: (Int, (Rating, Rating))): 
    Boolean = {
    val firstRating = ratingPair._2._1
    val secondRating = ratingPair._2._2
    //unique
    firstRating._1 < secondRating._1
  }

val uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)
```



    filterDuplicates: (ratingPair: (Int, (Rating, Rating)))Boolean
    uniqueJoinedRatings: org.apache.spark.rdd.RDD[(Int, (Rating, Rating))] = MapPartitionsRDD[10] at filter at <console>:53


<br>



```scala
uniqueJoinedRatings.take(10).foreach(println)
```

    (778,((94,2.0),(1273,3.0)))
    (778,((94,2.0),(265,4.0)))
    (778,((94,2.0),(239,4.0)))
    (778,((94,2.0),(193,4.0)))
    (778,((94,2.0),(1035,1.0)))
    (778,((94,2.0),(616,4.0)))
    (778,((94,2.0),(230,2.0)))
    (778,((94,2.0),(582,1.0)))
    (778,((94,2.0),(262,4.0)))
    (778,((94,2.0),(238,3.0)))


<br>

**<span style='color:blue'>makePairs</span>**

- 다음으로 영화끼리, 평점끼리 묶어주는 함수를 적용한다.
	- Before: (userID, ((```movie_id1```, ```rating1```)), ((```movie_id2```, ```rating2```)))
	- After: (```movie_id1```, ```movie_id2```),(```rating1```, ```rating2```)

```scala
val _ratings = uniqueJoinedRatings.take(1)(0)
```




    _ratings: (Int, (Rating, Rating)) = (778,((94,2.0),(1273,3.0)))



<br>

```scala
val _firstRating = _ratings._2._1
val _secondRating = _ratings._2._2
```




    _firstRating: Rating = (94,2.0)
    _secondRating: Rating = (1273,3.0)

<br>



```scala
(_firstRating._1, _secondRating._1) -> (_firstRating._2, _secondRating._2)
```




    res11: ((Int, Int), (Double, Double)) = ((94,1273),(2.0,3.0))

<br>

- 위 내용을 함수로 표현해보자.

```scala
def makePairs(ratings: (Int, (Rating, Rating))): ((Int, Int), (Double, Double)) = {
    val firstRating = ratings._2._1
    val secondRating = ratings._2._2

    (firstRating._1, secondRating._1) -> (firstRating._2, secondRating._2)
  }

val moviePairs = uniqueJoinedRatings.map(makePairs)
```



    makePairs: (ratings: (Int, (Rating, Rating)))((Int, Int), (Double, Double))
    moviePairs: org.apache.spark.rdd.RDD[((Int, Int), (Double, Double))] = MapPartitionsRDD[11] at map at <console>:55



<br>


```scala
// ((movie_id1, movie_id2),(rating1, rating2))
moviePairs.take(10).foreach(println)
```

    ((94,1273),(2.0,3.0))
    ((94,265),(2.0,4.0))
    ((94,239),(2.0,4.0))
    ((94,193),(2.0,4.0))
    ((94,1035),(2.0,1.0))
    ((94,616),(2.0,4.0))
    ((94,230),(2.0,2.0))
    ((94,582),(2.0,1.0))
    ((94,262),(2.0,4.0))
    ((94,238),(2.0,3.0))

<br>

- key별로 연산하기 위해, ```groupByKey```를 적용하면, ```((movieID_1, movieID_2), Iterable[(rating_1, rating_2)])```의 형태로 
```moviePairsRatings```이 생성된다.

```scala
val moviePairsRatings = moviePairs.groupByKey()
```




    moviePairsRatings: org.apache.spark.rdd.RDD[((Int, Int), Iterable[(Double, Double)])] = ShuffledRDD[12] at groupByKey at <console>:51




<br>

- cosine similarity를 구하기 전에, ```mapValues```, ```groupby```를 살펴보자.

---

**<span style='color:blue'>mapValues</span>**

- value를 입력으로 받아 (key, value)출력


```scala
val shipMap = sc.parallelize(Array((1, "Enterprise"),
                  (1, "Enterprise-D"),
                  (2, "Deep Space Nine"),
                  (2 -> "Voyager")))
```




    shipMap: org.apache.spark.rdd.RDD[(Int, String)] = ParallelCollectionRDD[13] at parallelize at <console>:32



<br>

- x $=$ tuple(```1```, "Enterprise")


```scala
shipMap.map(x => x._1).collect()
```




    res13: Array[Int] = Array(1, 1, 2, 2)


<br>

- x $=$ value("```E```nterprise")


```scala
shipMap.mapValues(x => x(0)).collect()
```




    res14: Array[(Int, Char)] = Array((1,E), (1,E), (2,D), (2,V))




<br>


**<span style='color:blue'>groupby</span>**





```scala
val things =sc.parallelize(List(("animal", "bear"),
             ("animal", "duck"),
             ("plant", "cactus"), 
             ("vehicle", "speed boat"),
             ("vehicle", "school bus")))
```




    things: org.apache.spark.rdd.RDD[(String, String)] = ParallelCollectionRDD[16] at parallelize at <console>:32

<br>

- 각 unique key마다 관련된 value들을 Iterator로 묶어줌

```scala
things.groupByKey().foreach(println)
```

    (plant,CompactBuffer(cactus))
    (animal,CompactBuffer(bear, duck))
    (vehicle,CompactBuffer(speed boat, school bus))


---
<br>


**<span style='color:DarkRed'>Cosine similarity</span>**

- cosine similarity는 ```mapValues```(function)을 사용해 구해진다.s
- 먼저, 첫번째 iteration에 일어나는 과정을 살펴보자.

```scala
var numPairs: Int = 0
var sum_xx: Double = 0.0
var sum_yy: Double = 0.0
var sum_xy: Double = 0.0
```




    numPairs: Int = 0
    sum_xx: Double = 0.0
    sum_yy: Double = 0.0
    sum_xy: Double = 0.0

<br>



```scala
// moviePairsRatings = (key, iterator)
// iterator = (key, value)
moviePairsRatings
```




    res16: org.apache.spark.rdd.RDD[((Int, Int), Iterable[(Double, Double)])] = ShuffledRDD[12] at groupByKey at <console>:51

<br>


- for example
    - key가 (movieID1: 94, movieID2: 1273)인 것을 하나를 샘플(```filter```사용)해 어떻게 작동되는지 보여줌


```scala
val tmp : (Int, Int) = (94,1273)
val filtered = moviePairs.filter(x =>  x._1 == tmp)
filtered.foreach(println)
```

    ((94,1273),(2.0,3.0))
    ((94,1273),(3.0,2.0))
    ((94,1273),(2.0,2.0))
    ((94,1273),(4.0,2.0))
    ((94,1273),(4.0,3.0))
    ((94,1273),(5.0,2.0))
    ((94,1273),(2.0,2.0))
    ((94,1273),(4.0,3.0))
    ((94,1273),(2.0,2.0))

    tmp: (Int, Int) = (94,1273)
    filtered: org.apache.spark.rdd.RDD[((Int, Int), (Double, Double))] = MapPartitionsRDD[18] at filter at <console>:56


<br>

- x._1: movieID pair
- x._2:	rating pair
- ```ratingPairs```는 코사인 유사도를 구하려는 두개의 벡터로 해석될 수 있음

```scala
val ratingPairs = filtered.map(x=>x._2).collect()
```




    ratingPairs: Array[(Double, Double)] = Array((2.0,3.0), (5.0,2.0), (2.0,2.0), (4.0,3.0), (2.0,2.0), (3.0,2.0), (2.0,2.0), (4.0,2.0), (4.0,3.0))



<br>


```scala
ratingPairs.foreach(println)
```

    (2.0,3.0)
    (5.0,2.0)
    (2.0,2.0)
    (4.0,3.0)
    (2.0,2.0)
    (3.0,2.0)
    (2.0,2.0)
    (4.0,2.0)
    (4.0,3.0)


<br>

- for first example

```scala
val pair = ratingPairs(0)
```




    pair: (Double, Double) = (2.0,3.0)



<br>



```scala
val ratingX = pair._1
val ratingY = pair._2
```




    ratingX: Double = 2.0
    ratingY: Double = 3.0


<br>


```scala
sum_xx += ratingX * ratingX
sum_yy += ratingY * ratingY
sum_xy += ratingX * ratingY
```

<br>

```scala
numPairs += 1
```
<br>

- 위 내용을 모든 example에 적용하면 다음과 같다.

```scala
var numPairs: Int = 0
var sum_xx: Double = 0.0
var sum_yy: Double = 0.0
var sum_xy: Double = 0.0

for (pair <- ratingPairs) {
  val ratingX = pair._1
  val ratingY = pair._2

  sum_xx += ratingX * ratingX
  sum_yy += ratingY * ratingY
  sum_xy += ratingX * ratingY
  numPairs += 1
}
```




    numPairs: Int = 9
    sum_xx: Double = 98.0
    sum_yy: Double = 51.0
    sum_xy: Double = 66.0


<br>

- 벡터의 내적 output에 대하여 코사인 유사도 식으로 정리해보자.

```scala
val numerator: Double = sum_xy
val denominator = Math.sqrt(sum_xx) * Math.sqrt(sum_yy)
```




    numerator: Double = 66.0
    denominator: Double = 70.69653456853455



<br>

```scala
var score:Double = 0.0
if (denominator != 0) {
  score = numerator / denominator
}
```




    score: Double = 0.9335676833780072


<br>

- ```numPairs``` = 벡터(rating pair)의 길이

```scala
(score, numPairs)
```




    res24: (Double, Int) = (0.9335676833780072,9)

<br>


- ```cache()```: 특정 RDD 결과물을 처음부터 계산하는 것이 아니라 캐쉬한 RDD를 재사용 할 수 있다.
	- RDD.cache()

- 위 내용들을 종합하여 함수로 표현해보고 결과물을 cache해보자
	- ```mapValues``` 때문에 value(rating pair)만 입력으로 받는다.

```scala
def computeCosineSimilarity(ratingPairs: Iterable[(Double, Double)]): (Double, Int) = {
    var numPairs: Int = 0
    var sum_xx: Double = 0.0
    var sum_yy: Double = 0.0
    var sum_xy: Double = 0.0

    for (pair <- ratingPairs) {
      val ratingX = pair._1
      val ratingY = pair._2

      sum_xx += ratingX * ratingX
      sum_yy += ratingY * ratingY
      sum_xy += ratingX * ratingY
      numPairs += 1
    }

    val numerator: Double = sum_xy
    val denominator = Math.sqrt(sum_xx) * Math.sqrt(sum_yy)

    var score:Double = 0.0
    if (denominator != 0) {
      score = numerator / denominator
    }

    (score, numPairs)
}
val moviePairSimilarities = moviePairsRatings.mapValues(computeCosineSimilarity).cache() //reuse results
```




    computeCosineSimilarity: (ratingPairs: Iterable[(Double, Double)])(Double, Int)
    moviePairSimilarities: org.apache.spark.rdd.RDD[((Int, Int), (Double, Int))] = MapPartitionsRDD[20] at mapValues at <console>:99


<br>

- ```scoreThreshold```: 코사인 유사도 임계값 
- ```coOccurenceThreshold```: 다른 movieID들과의 연결정도 임계값

```scala
val scoreThreshold = 0.60
val coOccurenceThreshold = 50.0
var movieId = 94 
// x = ((movie_id1, movie_id2), (rating_sim, num))
val filteredResults = moviePairSimilarities.filter(x => {
    //pair = (movie_id1, movie_id2)
    val pair = x._1
    //sim = (rating_sim, num)
    val sim = x._2
    (pair._1 == movieId || pair._2 == movieId) &&
      sim._1 > scoreThreshold && sim._2 > coOccurenceThreshold
})
val results = filteredResults.map(x => (x._2, x._1)).sortByKey(ascending = false).take(10)
```




    scoreThreshold: Double = 0.6
    coOccurenceThreshold: Double = 50.0
    movieId: Int = 94
    filteredResults: org.apache.spark.rdd.RDD[((Int, Int), (Double, Int))] = MapPartitionsRDD[21] at filter at <console>:79
    results: Array[((Double, Int), (Int, Int))] = Array(((0.9586049379734621,66),(73,94)), ((0.9540093745210545,124),(94,204)), ((0.953888417198362,117),(94,174)), ((0.9514502380030145,67),(94,742)), ((0.9513610928045688,80),(94,208)), ((0.9512871292224748,105),(79,94)), ((0.9511736470315294,56),(94,232)), ((0.9511281290554398,65),(94,692)), ((0.9487881988287913,66),(94,258)), ((0.9481276017281098,119),(94,210)))


<br>

- movieName = Map(idx => name)


```scala
println("\nTop similar movies (max of 10) for " + movieNames(movieId))
```

    
    Top similar movies (max of 10) for Home Alone (1990)

<br>

```scala
//(sim, (movie_id1, movie_id2))
results
```




    res26: Array[((Double, Int), (Int, Int))] = Array(((0.9586049379734621,66),(73,94)), ((0.9540093745210545,124),(94,204)), ((0.953888417198362,117),(94,174)), ((0.9514502380030145,67),(94,742)), ((0.9513610928045688,80),(94,208)), ((0.9512871292224748,105),(79,94)), ((0.9511736470315294,56),(94,232)), ((0.9511281290554398,65),(94,692)), ((0.9487881988287913,66),(94,258)), ((0.9481276017281098,119),(94,210)))


<br>


```scala
val result = results(0)
```




    result: ((Double, Int), (Int, Int)) = ((0.9586049379734621,66),(73,94))


<br>


```scala
val _sim = result._1
val _pair = result._2
```




    _sim: (Double, Int) = (0.9586049379734621,66)
    _pair: (Int, Int) = (73,94)


<br>


```scala
var similarMovieId = _pair._1
```




    similarMovieId: Int = 73



<br>


```scala
movieId
```




    res27: Int = 94


<br>


```scala
similarMovieId == movieId
```




    res28: Boolean = false


<br>


```scala
println(movieNames(similarMovieId) + "\tscore: " + _sim._1 + "\tstrength: " + _sim._2)
```

    Maverick (1994)	score: 0.9586049379734621	strength: 66

<br>

```scala
for (result <- results) {
    val sim = result._1
    val pair = result._2

    var similarMovieId = pair._1
    if (similarMovieId == movieId) {
      similarMovieId = pair._2
    }
    println(movieNames(similarMovieId) + "\tscore: " + sim._1 + "\tstrength: " + sim._2)
}
```

    Maverick (1994)	score: 0.9586049379734621	strength: 66
    Back to the Future (1985)	score: 0.9540093745210545	strength: 124
    Raiders of the Lost Ark (1981)	score: 0.953888417198362	strength: 117
    Ransom (1996)	score: 0.9514502380030145	strength: 67
    Young Frankenstein (1974)	score: 0.9513610928045688	strength: 80
    Fugitive, The (1993)	score: 0.9512871292224748	strength: 105
    Young Guns (1988)	score: 0.9511736470315294	strength: 56
    American President, The (1995)	score: 0.9511281290554398	strength: 65
    Contact (1997)	score: 0.9487881988287913	strength: 66
    Indiana Jones and the Last Crusade (1989)	score: 0.9481276017281098	strength: 119



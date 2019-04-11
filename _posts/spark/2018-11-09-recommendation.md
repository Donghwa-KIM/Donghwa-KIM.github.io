---
layout: post
comments: true
title: 추천 알고리즘 (ALS in MLlib)
categories: Spark

tags:
- Spark
---

**<span style='color:DarkRed'>ALS: Alternative Least Square</span>** 

- Spark에 내장되어 있는 추천 알고리즘을 사용 해보자.
- 데이터셋으로 ```#User``` $\times$ ```#Moivie``` 데이터프레임이 있다고 가정해보자.

<br>

**<span style='color:DarkRed'>학습 방법</span>** 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/SxjFa6Q.png"></p>

---

- Initalization: 
	- rank의 차원을 n이라고 하면, 아래의 두개의 데이터프레임으로 initialize 할 수 있다.
		- ```#User``` $\times$ ```n```
		- ```n``` $\times$ ```#Movie```

- 이 두개의 행렬을 곱해주면, 원래  ```#User``` $\times$ ```#Moivie```(inference ratings)를 생성시킬 수 있다.

	- e.g. $$X_{11} = 0.3 \times 1.5 + 1.5 \times 2.1 = 0.66$$

- 실제 ratings 데이터셋과 예측된 ratings 데이터셋의 차이를 구한다.

	- e.g. $(0.66-5)^2 + (0.82-1)^2 + ...$

- 각 iteration마다, 그 차이에 대해서 SGD(stocastic gradient decent) 방법으로 최적화된다.

<br>

**<span style='color:DarkRed'>Model Selection</span>** 

- ```Rank```에 따라 성능이 달라질 수 있기 때문에 model selection이 필요하다.
- 최적의 ```Rank```는 Train/Valid/Test로 나눠 평가해 찾아낼 수 있다.

<p align="center"><img width="350" height="auto" src="https://i.imgur.com/XQhpUib.png"></p>



<br>

**<span style='color:DarkRed'>추천 방법</span>** 

- 위 그림에서 실제로 ```User 2```는 ```Moive 2```와 ```Moive 4```를 한번도 보지 않았다고 해보자.
- 하지만 inference ratings에서 ```Moive 4(1.09)```가 ```Moive 4(10.5)```보다 크므로 ```User 2```에게 ```Moive 4```를 추천하게 된다.



---

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/16_movie%20recommendation.ipynb">notebook</a> 코드 참조)



```scala
import org.apache.spark._
import org.apache.spark.SparkContext._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.mllib.recommendation._
```
<br>


- movieID와 name으로 맵핑하는 Map을 만들어 준다.
- 이전 글과 중복되는 내용으로 ```loadMovieNames```는 [여기]({{ site.baseurl }}/broadcast.html)에 자세히 설명되어 있다.


```scala
/** Load up a Map of movie IDs to movie names. */
def loadMovieNames() : Map[Int, String] = {

// Handle character encoding issues:
implicit val codec = Codec("UTF-8")
codec.onMalformedInput(CodingErrorAction.REPLACE)
codec.onUnmappableCharacter(CodingErrorAction.REPLACE)

// Create a Map of Ints to Strings, and populate it from u.item.
var movieNames:Map[Int, String] = Map()

 val lines = Source.fromFile("data/u.item").getLines()
 for (line <- lines) {
   var fields = line.split('|')
   if (fields.length > 1) {
    movieNames += (fields(0).toInt -> fields(1))
   }
 }

 return movieNames
}
```




    loadMovieNames: ()Map[Int,String]


<br>


```scala
val nameDict = loadMovieNames()
```




    nameDict: Map[Int,String] = Map(645 -> Paris Is Burning (1990), 892 -> Flubber (1997), 69 -> Forrest Gump (1994), 1322 -> Metisse (Caf� au Lait) (1993), 1665 -> Brother's Kiss, A (1997), 1036 -> Drop Dead Fred (1991), 1586 -> Lashou shentan (1992), 1501 -> Prisoner of the Mountains (Kavkazsky Plennik) (1996), 809 -> Rising Sun (1993), 1337 -> Larger Than Life (1996), 1411 -> Barbarella (1968), 629 -> Victor/Victoria (1982), 1024 -> Mrs. Dalloway (1997), 1469 -> Tom and Huck (1995), 365 -> Powder (1995), 1369 -> Forbidden Christ, The (Cristo proibito, Il) (1950), 138 -> D3: The Mighty Ducks (1996), 1190 -> That Old Feeling (1997), 1168 -> Little Buddha (1993), 760 -> Screamers (1995), 101 -> Heavy Metal (1981), 1454 -> Angel and the Badman (1947), 1633 -> � k�ldum klaka (Cold Fever) (199...


<br>

```scala
val data = sc.textFile("data/u.data")
```




    data: org.apache.spark.rdd.RDD[String] = data/u.data MapPartitionsRDD[1] at textFile at <console>:37

<br>



```scala
data.map( x => x.split("\t") ).map(x=>x(0).toInt).take(2)
```




    res0: Array[Int] = Array(196, 186)

<br>


- ```Rating```은 ```org.apache.spark.mllib.recommendation```패키지에서 가져온 것이다.


```scala
// rating in the recommendation
val ratings = data.map( x => x.split("\t") )
    .map( x => Rating(x(0).toInt, x(1).toInt, x(2).toDouble) ).cache()
```




    ratings: org.apache.spark.rdd.RDD[org.apache.spark.mllib.recommendation.Rating] = MapPartitionsRDD[6] at map at <console>:41



<br>

```scala
// Rating(userID, movieID, ratings)
ratings.take(5).foreach(println)
```

    Rating(196,242,3.0)
    Rating(186,302,3.0)
    Rating(22,377,1.0)
    Rating(244,51,2.0)
    Rating(166,346,1.0)

<br>

```scala
println("\nTraining recommendation model...")
```

    
    Training recommendation model...

<br>

- Alternating Least Squares algorithm
    - movieID x userID 행렬이 주어진 sparse matrix를 생각해보자.
    - 랜덤하게 (movieID x rank) x (rank x userID)로 matrix factorization이 가능하다.
    - 추측된 행렬과 실제 행렬 차이를 최소화하는 latent variables을 찾게 된다.
    - 추천방식은 userID의 실제로 보지 않은 movidID중에 rating이 가장 높은 것을 추천하게 된다.


```scala
val rank = 8 // latent dimension
val numIterations = 20 // training iter
val model = ALS.train(ratings, rank, numIterations)
```




    rank: Int = 8
    numIterations: Int = 20
    model: org.apache.spark.mllib.recommendation.MatrixFactorizationModel = org.apache.spark.mllib.recommendation.MatrixFactorizationModel@1c432552


<br>


```scala
model
```




    res6: org.apache.spark.mllib.recommendation.MatrixFactorizationModel = org.apache.spark.mllib.recommendation.MatrixFactorizationModel@1c432552



<br>

- 253번 유저에게 추천하고 싶다고 가정해보자.

```scala
val userID = 253
    println("\nRatings for user ID " + userID + ":")
```

    
    Ratings for user ID 253:

    userID: Int = 253



<br>

- filtering for 253 user

```scala
// x = Rating(userID, movieID, ratings)
val userRatings = ratings.filter(x => x.user == userID)
```




    userRatings: org.apache.spark.rdd.RDD[org.apache.spark.mllib.recommendation.Rating] = MapPartitionsRDD[781] at filter at <console>:43



<br>

```scala
val myRatings = userRatings.collect()
```




    myRatings: Array[org.apache.spark.mllib.recommendation.Rating] = Array(Rating(253,465,5.0), Rating(253,259,2.0), Rating(253,97,4.0), Rating(253,746,3.0), Rating(253,510,5.0), Rating(253,216,4.0), Rating(253,50,4.0), Rating(253,183,5.0), Rating(253,203,4.0), Rating(253,747,3.0), Rating(253,483,5.0), Rating(253,210,4.0), Rating(253,198,5.0), Rating(253,568,4.0), Rating(253,294,4.0), Rating(253,566,4.0), Rating(253,237,4.0), Rating(253,188,4.0), Rating(253,127,5.0), Rating(253,647,3.0), Rating(253,173,5.0), Rating(253,175,2.0), Rating(253,527,5.0), Rating(253,282,4.0), Rating(253,742,4.0), Rating(253,298,3.0), Rating(253,117,5.0), Rating(253,806,4.0), Rating(253,95,4.0), Rating(253,448,2.0), Rating(253,83,4.0), Rating(253,87,5.0), Rating(253,705,5.0), Rating(253,487,4.0), Rating(253,243,2....


<br>

- ```product``` = movieID

```scala
// movieID
myRatings(0).product
```




    res8: Int = 465



<br>

- rating

```scala
// rating
myRatings(0).rating
```




    res9: Double = 5.0


<br>

- 253번 사용자가 본 movieID와 rating


```scala
for (rating <- myRatings) {
  println(nameDict(rating.product.toInt) + ": " + rating.rating.toString)
}
```

    Jungle Book, The (1994): 5.0
    George of the Jungle (1997): 2.0
    Dances with Wolves (1990): 4.0
    Real Genius (1985): 3.0
    Magnificent Seven, The (1954): 5.0
    When Harry Met Sally... (1989): 4.0
    Star Wars (1977): 4.0
    Alien (1979): 5.0
    Unforgiven (1992): 4.0
    Benny & Joon (1993): 3.0
    Casablanca (1942): 5.0
    Indiana Jones and the Last Crusade (1989): 4.0
    Nikita (La Femme Nikita) (1990): 5.0
    Speed (1994): 4.0
    Liar Liar (1997): 4.0
    Clear and Present Danger (1994): 4.0
    Jerry Maguire (1996): 4.0
    Full Metal Jacket (1987): 4.0
    Godfather, The (1972): 5.0
    Ran (1985): 3.0
    Princess Bride, The (1987): 5.0
    Brazil (1985): 2.0
    Gandhi (1982): 5.0
    Time to Kill, A (1996): 4.0
    Ransom (1996): 4.0
    Face/Off (1997): 3.0
    Rock, The (1996): 5.0
    Menace II Society (1993): 4.0
    Aladdin (1992): 4.0
    Omen, The (1976): 2.0
    Much Ado About Nothing (1993): 4.0
    Searching for Bobby Fischer (1993): 5.0
    Singin' in the Rain (1952): 5.0
    Roman Holiday (1953): 4.0
    Jungle2Jungle (1997): 2.0
    Shawshank Redemption, The (1994): 5.0
    It's a Wonderful Life (1946): 5.0
    Blade Runner (1982): 4.0
    Fugitive, The (1993): 5.0
    Game, The (1997): 2.0
    Silence of the Lambs, The (1991): 5.0
    Heat (1995): 3.0
    Conspiracy Theory (1997): 4.0
    Reservoir Dogs (1992): 3.0
    Dave (1993): 4.0
    Cool Hand Luke (1967): 4.0
    GoodFellas (1990): 3.0
    Pulp Fiction (1994): 3.0
    Beauty and the Beast (1991): 5.0
    Braveheart (1995): 5.0
    Edge, The (1997): 3.0
    Con Air (1997): 3.0
    His Girl Friday (1940): 5.0
    Usual Suspects, The (1995): 5.0
    Groundhog Day (1993): 5.0
    Air Force One (1997): 4.0
    Babe (1995): 4.0
    Fargo (1996): 4.0
    Alien: Resurrection (1997): 4.0
    Executive Decision (1996): 2.0
    Mr. Holland's Opus (1995): 4.0
    Primal Fear (1996): 3.0
    Jackal, The (1997): 5.0
    My Fair Lady (1964): 5.0
    Conan the Barbarian (1981): 3.0
    Scream 2 (1997): 4.0
    Mirror Has Two Faces, The (1996): 4.0
    Henry V (1989): 5.0
    Terminator 2: Judgment Day (1991): 5.0
    Phenomenon (1996): 3.0
    Hudsucker Proxy, The (1994): 4.0
    Star Trek: First Contact (1996): 4.0
    Cure, The (1995): 3.0
    Some Like It Hot (1959): 5.0
    Fish Called Wanda, A (1988): 3.0
    Heathers (1989): 3.0
    Withnail and I (1987): 3.0
    Arsenic and Old Lace (1944): 5.0
    Toy Story (1995): 5.0
    Raging Bull (1980): 1.0
    Hamlet (1996): 4.0
    Fire Down Below (1997): 3.0
    To Kill a Mockingbird (1962): 5.0
    Stand by Me (1986): 4.0
    Jaws (1975): 4.0
    To Catch a Thief (1955): 5.0
    Independence Day (ID4) (1996): 5.0
    Little Women (1994): 4.0
    Jurassic Park (1993): 3.0
    Wizard of Oz, The (1939): 5.0
    Monty Python and the Holy Grail (1974): 3.0
    Miller's Crossing (1990): 5.0
    Shining, The (1980): 4.0
    Tomorrow Never Dies (1997): 3.0
    Get Shorty (1995): 4.0
    Affair to Remember, An (1957): 5.0
    Schindler's List (1993): 5.0

<br>

- ALS모델을 이용해, 253번 유저에게 10개 영화를 추천

```scala
val recommendations = model.recommendProducts(userID, 10)
```




    recommendations: Array[org.apache.spark.mllib.recommendation.Rating] = Array(Rating(253,1233,5.690142743366552), Rating(253,1463,5.502143677332252), Rating(253,626,5.312405671340901), Rating(253,867,5.241878376653451), Rating(253,318,5.158754547674508), Rating(253,496,5.1223656401703295), Rating(253,1251,4.956725690799651), Rating(253,133,4.925462977349991), Rating(253,1398,4.915498503817086), Rating(253,1242,4.899403232875574))


<br>

- 253번 사용자가 봤던 영화들과 가장 유사한 새로운 10개의 영화 추천


```scala
for (recommendation <- recommendations) {
  println( nameDict(recommendation.product.toInt) + " score " + recommendation.rating )
}
```

    N�nette et Boni (1996) score 5.690142743366552
    Boys, Les (1997) score 5.502143677332252
    So Dear to My Heart (1949) score 5.312405671340901
    Whole Wide World, The (1996) score 5.241878376653451
    Schindler's List (1993) score 5.158754547674508
    It's a Wonderful Life (1946) score 5.1223656401703295
    A Chef in Love (1996) score 4.956725690799651
    Gone with the Wind (1939) score 4.925462977349991
    Anna (1996) score 4.915498503817086
    Old Lady Who Walked in the Sea, The (Vieille qui marchait dans la mer, La) (1991) score 4.899403232875574


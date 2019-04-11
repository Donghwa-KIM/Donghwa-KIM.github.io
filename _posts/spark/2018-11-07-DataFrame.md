---
layout: post
comments: true
title: DataFrames(RDD.toDF, select, filter)
categories: Spark

tags:
- Spark
---


**<span style='color:DarkRed'>DataFrames</span>**

- 데이터를 불러와 DataFrames을 사용하는 방식은 크게 두가지가 있다.
	1. RDD로 불러와 필요한 전처리 후 DataFrame으로 변환하는 방식
		- ```val colNames = Seq()```
		- ```RDD.toDF(colNames: _*)```
	2. 처음부터 DataFrame으로 받는 방식
		- ```spark.read.schema```
- 데이터프레임은 일반적으로 query에 대한 결과를 재출력하는 비효율성을 줄이기 위해 ```cache```를 사용하게 된다.

---

<br>

**<span style='color:DarkRed'>코드 예시</span>**(<a href="https://github.com/Donghwa-KIM/Spark-scala-jupyter-tutorial/blob/master/14_dataframe.ipynb">notebook</a> 코드 참조)

```scala
import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
```

<br>

**<span style='color:Darkblue'> 1) RDD로 불러와 필요한 부분 전처리 후 DataFrame으로 변환하는 방식 </span>**


```scala
type Person = (Int, String, Int, Int)
val colNames = Seq("ID", "name","age","numFriends")
```




    defined type alias Person
    colNames: Seq[String] = List(ID, name, age, numFriends)



<br>

```scala
def mapper(line:String): Person = {
    val fields = line.split(',')  
    val person:Person = (fields(0).toInt, fields(1), fields(2).toInt, fields(3).toInt)
    return person
}
```




    mapper: (line: String)Person


<br>

- ```cache```를 사용하게 되면 query에 대한 결과를 재출력하는 비효율성을 줄일 수 있다.
- 또한 SparkContext(```sc```)가 아니라 SparkSession(```spark```)를 사용해야 데이터프레임을 사용할 수 있다.

```scala
val lines = spark.sparkContext.textFile("data/fakefriends.csv")
val schemaPeople = lines.map(mapper).toDF(colNames: _*).cache()
```




    lines: org.apache.spark.rdd.RDD[String] = data/fakefriends.csv MapPartitionsRDD[67] at textFile at <console>:41
    schemaPeople: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [ID: int, name: string ... 2 more fields]


<br>


```scala
schemaPeople.show()
```

    +---+--------+---+----------+
    | ID|    name|age|numFriends|
    +---+--------+---+----------+
    |  0|    Will| 33|       385|
    |  1|Jean-Luc| 26|         2|
    |  2|    Hugh| 55|       221|
    |  3|  Deanna| 40|       465|
    |  4|   Quark| 68|        21|
    |  5|  Weyoun| 59|       318|
    |  6|  Gowron| 37|       220|
    |  7|    Will| 54|       307|
    |  8|  Jadzia| 38|       380|
    |  9|    Hugh| 27|       181|
    | 10|     Odo| 53|       191|
    | 11|     Ben| 57|       372|
    | 12|   Keiko| 54|       253|
    | 13|Jean-Luc| 56|       444|
    | 14|    Hugh| 43|        49|
    | 15|     Rom| 36|        49|
    | 16|  Weyoun| 22|       323|
    | 17|     Odo| 35|        13|
    | 18|Jean-Luc| 45|       455|
    | 19|  Geordi| 60|       246|
    +---+--------+---+----------+
    only showing top 20 rows
    

<br>


**<span style='color:Darkblue'>2) 처음부터 DataFrame으로 받는 방식</span>**


```scala
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}
```




    import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}



<br>

```scala
val ID = StructField("ID", IntegerType)
val name = StructField("name", StringType)
val age = StructField("age", IntegerType)
val num = StructField("numFriends", IntegerType)
val schema = StructType(List(ID, name, age, num))
```




    ID: org.apache.spark.sql.types.StructField = StructField(ID,IntegerType,true)
    name: org.apache.spark.sql.types.StructField = StructField(name,StringType,true)
    age: org.apache.spark.sql.types.StructField = StructField(age,IntegerType,true)
    num: org.apache.spark.sql.types.StructField = StructField(numFriends,IntegerType,true)
    schema: org.apache.spark.sql.types.StructType = StructType(StructField(ID,IntegerType,true), StructField(name,StringType,true), StructField(age,IntegerType,true), StructField(numFriends,IntegerType,true))

<br>

- ```cache```를 사용하게 되면 query에 대한 결과를 재출력하는 비효율성을 줄일 수 있다.
- SparkContext(```sc```)가 아니라 SparkSession(```spark```)를 사용해야 데이터프레임을 사용할 수 있다.

```scala
val schemaPeople = spark.read.schema(schema).csv("data/fakefriends.csv").cache()
```




    schemaPeople: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [ID: int, name: string ... 2 more fields]

<br>



```scala
schemaPeople.show()
```

    +---+--------+---+----------+
    | ID|    name|age|numFriends|
    +---+--------+---+----------+
    |  0|    Will| 33|       385|
    |  1|Jean-Luc| 26|         2|
    |  2|    Hugh| 55|       221|
    |  3|  Deanna| 40|       465|
    |  4|   Quark| 68|        21|
    |  5|  Weyoun| 59|       318|
    |  6|  Gowron| 37|       220|
    |  7|    Will| 54|       307|
    |  8|  Jadzia| 38|       380|
    |  9|    Hugh| 27|       181|
    | 10|     Odo| 53|       191|
    | 11|     Ben| 57|       372|
    | 12|   Keiko| 54|       253|
    | 13|Jean-Luc| 56|       444|
    | 14|    Hugh| 43|        49|
    | 15|     Rom| 36|        49|
    | 16|  Weyoun| 22|       323|
    | 17|     Odo| 35|        13|
    | 18|Jean-Luc| 45|       455|
    | 19|  Geordi| 60|       246|
    +---+--------+---+----------+
    only showing top 20 rows
    
<br>


- Schema Info


```scala
schemaPeople.printSchema()
```

    Here is our inferred schema:
    root
     |-- ID: integer (nullable = true)
     |-- name: string (nullable = true)
     |-- age: integer (nullable = true)
     |-- numFriends: integer (nullable = true)
    
<br>


- 변수선택


```scala
schemaPeople.select("name").show()
```

    +--------+
    |    name|
    +--------+
    |    Will|
    |Jean-Luc|
    |    Hugh|
    |  Deanna|
    |   Quark|
    |  Weyoun|
    |  Gowron|
    |    Will|
    |  Jadzia|
    |    Hugh|
    |     Odo|
    |     Ben|
    |   Keiko|
    |Jean-Luc|
    |    Hugh|
    |     Rom|
    |  Weyoun|
    |     Odo|
    |Jean-Luc|
    |  Geordi|
    +--------+
    only showing top 20 rows
    
<br>

- Filter


```scala
schemaPeople.filter(schemaPeople("age") < 21).show()
```

    +---+-------+---+----------+
    | ID|   name|age|numFriends|
    +---+-------+---+----------+
    | 21|  Miles| 19|       268|
    | 48|    Nog| 20|         1|
    | 52|Beverly| 19|       269|
    | 54|  Brunt| 19|         5|
    | 60| Geordi| 20|       100|
    | 73|  Brunt| 20|       384|
    |106|Beverly| 18|       499|
    |115|  Dukat| 18|       397|
    |133|  Quark| 19|       265|
    |136|   Will| 19|       335|
    |225|   Elim| 19|       106|
    |304|   Will| 19|       404|
    |327| Julian| 20|        63|
    |341|   Data| 18|       326|
    |349| Kasidy| 20|       277|
    |366|  Keiko| 19|       119|
    |373|  Quark| 19|       272|
    |377|Beverly| 18|       418|
    |404| Kasidy| 18|        24|
    |409|    Nog| 19|       267|
    +---+-------+---+----------+
    only showing top 20 rows
    
<br>

- for each age, count


```scala
schemaPeople.groupBy("age").count().show()
```

    +---+-----+
    |age|count|
    +---+-----+
    | 31|    8|
    | 65|    5|
    | 53|    7|
    | 34|    6|
    | 28|   10|
    | 26|   17|
    | 27|    8|
    | 44|   12|
    | 22|    7|
    | 47|    9|
    | 52|   11|
    | 40|   17|
    | 20|    5|
    | 57|   12|
    | 54|   13|
    | 48|   10|
    | 19|   11|
    | 64|   12|
    | 41|    9|
    | 43|    7|
    +---+-----+
    only showing top 20 rows
    
<br>

- select multiple columns


```scala
schemaPeople.select(schemaPeople("name"), schemaPeople("age")).show()
```

    +--------+---+
    |    name|age|
    +--------+---+
    |    Will| 33|
    |Jean-Luc| 26|
    |    Hugh| 55|
    |  Deanna| 40|
    |   Quark| 68|
    |  Weyoun| 59|
    |  Gowron| 37|
    |    Will| 54|
    |  Jadzia| 38|
    |    Hugh| 27|
    |     Odo| 53|
    |     Ben| 57|
    |   Keiko| 54|
    |Jean-Luc| 56|
    |    Hugh| 43|
    |     Rom| 36|
    |  Weyoun| 22|
    |     Odo| 35|
    |Jean-Luc| 45|
    |  Geordi| 60|
    +--------+---+
    only showing top 20 rows
    
<br>

- select multiple columns and modify


```scala
schemaPeople.select(schemaPeople("name"), schemaPeople("age") + 10).show()
```

    +--------+----------+
    |    name|(age + 10)|
    +--------+----------+
    |    Will|        43|
    |Jean-Luc|        36|
    |    Hugh|        65|
    |  Deanna|        50|
    |   Quark|        78|
    |  Weyoun|        69|
    |  Gowron|        47|
    |    Will|        64|
    |  Jadzia|        48|
    |    Hugh|        37|
    |     Odo|        63|
    |     Ben|        67|
    |   Keiko|        64|
    |Jean-Luc|        66|
    |    Hugh|        53|
    |     Rom|        46|
    |  Weyoun|        32|
    |     Odo|        45|
    |Jean-Luc|        55|
    |  Geordi|        70|
    +--------+----------+
    only showing top 20 rows
    
<br>


```scala
spark.stop()
```

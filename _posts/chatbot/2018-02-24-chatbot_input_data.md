---
layout: post
comments: true
title:  ChatBot(챗봇)에 학습되는 입력 데이터 
categories: ChatBot

tags:
- ChatBot(챗봇)
---

**<span style='color:DarkRed'>ChatBot Input</span>**
> 챗봇 입력데이터는 질문을 한 사람(parent_id) 응답하는 사람(comment_id)의 paired dataset으로 구성해야 하며, 또한 모델을 평가하기 위해 학습(training), 평가(test)데이터로 구분해야만 한다. ```test.from```는 평가데이터이면서 질문하는 내용을 저장, ```test.to```는 평가데이터이면서 응답하는 내용을 저장, ```train.from``` ```train.to``` 는 학습데이터에 대해서 앞서 한 내용을 적용한다.

- [이전 블로그](https://donghwa-kim.github.io/chatbot_database.html)에서 업로드한 데이터를 이용하여 seq2seq 입력데이터를 만들어 보자.
- 월별 많은 데이터 테이블이 존재하게 되는 데 ```for loop```를 실행하면 되니 하나만 예를 들어보자. 

```python
timeframes = ['2006_01','2006_02',...]
```
<br>

- ```2006_01``` 데이터 테이블 예시

```python
timeData = '2006_01'
```
<br>


- 몇가지 필요한 변수들을 정의해보자
> **```limit```** : 
> 한번에 불어오는 데이터의 행의 갯수이다. 이 크기 만큼 ```training set```, ```test set```파일에 번갈아 가면서 업로드 된다.
> **```last_unix```**: 
> 한번 업데이트 하고나면, 업데이트 된 데이터와 중복이 되지 않는 다른 데이터를 training set, test set파일에 재귀적으로 업로드 해야 하는데 데이터 구별을 위해 업데이트 될때마다 마지막 시간값을 기록한다. 
> **```cur_length```**: 
> 처음에 ```limit```이라는 값으로 초기화 되지만, 계속적의 데이터 테이블의 row의 수로 업데이트 된다. 이 의미는 업데이트를 하려는 데이터가 얼마 안 남았을 때, ```cur_length```는 사전에 정의한 ```limit``` 보다 작을 것이다. 즉, ```cur_length``` < ```limit```이 조건이 되었을 때 반복문을 종료시킨다.
> **```counter```**: 
> 실행순서를 출력을 위해 사용한다.
> **```test_done```**:
> 반복문이 실행될 때마다 ```True```, ```False```값을 교체시켜 ```training set```, ```test set```파일에 저장하는데 쓰인다. 


```python
# limit : 얼마나 많은 행을 가져올 것인가? e.g limit = 2000 => 2000 rows와
limit = 5000
# 끝나는 시점
last_unix = 0

cur_length = limit
counter = 0
test_done = False
```

<br>

- ```2006_01``` 데이터 테이블을 불러오기 위한 MySQL DB연결


```python
connection = MySQLdb.connect(host='localhost',
                             user='root',
                             password='1225')
c = connection.cursor()
c.execute("USE {}_reddit;".format(timeData.split('_')[0]))
```

<br>

- ```pandas```를 활용한 ```2006_01``` 데이터 테이블을 불러오기


```python
df = pd.read_sql("SELECT * FROM {} WHERE unix > {} AND parent IS NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format(timeData, last_unix, limit),
            connection)
df
```

<div>
<style scoped>


    .dataframe tbody tr th {
        vertical-align: middle;
                font-size: 10px;
    }
    .dataframe thead td {
      text-align: right;
            font-size: 5px;
    }
    .dataframe thead th {
        text-align: right;
                font-size: 12px;
    }
</style>
<table border="1" class="dataframe" style="font-size: 15px; text-align: right;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parent_id</th>
      <th>comment_id</th>
      <th>parent</th>
      <th>comment</th>
      <th>subreddit</th>
      <th>unix</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c3982</td>
      <td>c4141</td>
      <td>Too bad Mr Crockford himself misunderstands Ja...</td>
      <td>Hmm, I tried the following, and it worked: new...</td>
      <td>reddit.com</td>
      <td>1136838703</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c3899</td>
      <td>c4186</td>
      <td>Sorry, not basic enough. How about explanation...</td>
      <td>U is a combinator, a function that takes a fun...</td>
      <td>reddit.com</td>
      <td>1136855468</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c3187</td>
      <td>c4239</td>
      <td>well, I'm learning Python and so are all the g...</td>
      <td>The ranking claims to rate how 'mainstream' ea...</td>
      <td>reddit.com</td>
      <td>1136889581</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c3846</td>
      <td>c5141</td>
      <td>Most distributions allow you to update from on...</td>
      <td>That's not the point.  If I install SuSE 9.1 o...</td>
      <td>reddit.com</td>
      <td>1137121161</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c5278</td>
      <td>c5310</td>
      <td>Quite frankly, I'm sick of hearing about the b...</td>
      <td>This isn't about socialism vs capitalism. It's...</td>
      <td>reddit.com</td>
      <td>1137165941</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c5222</td>
      <td>c5355</td>
      <td>why?</td>
      <td>Because Bill Gates is a very intelligent man.</td>
      <td>reddit.com</td>
      <td>1137174845</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c5287</td>
      <td>c5356</td>
      <td>I'm wondering how my comment got 3 negative vo...</td>
      <td>Indeed - you just can't have casual discussion...</td>
      <td>reddit.com</td>
      <td>1137175214</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>c5291</td>
      <td>c5373</td>
      <td>wow, this is the only practical use of fractal...</td>
      <td>A while back, lots of people were excited abou...</td>
      <td>reddit.com</td>
      <td>1137177834</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c5228</td>
      <td>c5411</td>
      <td>'hilarious' news piece from *2002*. Can you ev...</td>
      <td>nope, after I showed the link to my girlfriend...</td>
      <td>reddit.com</td>
      <td>1137183450</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

<br>

- 다음 업데이트 때 순차적인 데이터 셋을 불러오기 위해서 ```last_unix```에 업데이트된 마지막 행의 시간을 기록

```python
last_unix = df.tail(1)['unix'].values[0]
last_unix

1137183450
```

    




```python
cur_length = len(df)
cur_length

9
```

<br>

**<span style='color:DarkRed'> Total Code</span>**



```python
for timeData in timeframes:
    connection = MySQLdb.connect(host='localhost',
                                 user='root',
                                 password='1225')
    c = connection.cursor()
    c.execute("USE {}_reddit;".format(timeData.split('_')[0]))

    # limit : 얼마나 많은 행을 가져올 것인가? e.g limit = 2000 => 2000 rows
    limit = 5000
    last_unix = 0
    cur_length = limit
    counter = 0
    test_done = False

    while cur_length == limit:
        # 5000 rows 씩 데이터를 불러옴
        df = pd.read_sql("SELECT * FROM {} WHERE unix > {} AND parent IS NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format(timeData, last_unix, limit),
                    connection)
        # 가장 늦은 시간
        last_unix = df.tail(1)['unix'].values[0]
        cur_length = len(df)
        if not test_done:
            with open('test.from','a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content+'\n')
            with open('test.to', 'a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(content + '\n')

            test_done = True

        else:
            with open('train.from', 'a', encoding='utf8') as f:
                for content in df['parent'].values:
                    f.write(content + '\n')
            with open('train.to', 'a', encoding='utf8') as f:
                for content in df['comment'].values:
                    f.write(content + '\n')

        counter += 1
        if counter % 20 == 0:
            print('Update:',counter*limit)
```

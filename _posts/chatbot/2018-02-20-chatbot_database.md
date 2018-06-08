---
layout: post
comments: true
title:  ChatBot(챗봇) 데이터베이스 구축 
categories: ChatBot

tags:
- ChatBot(챗봇)
- MySQL
---


**<span style='color:DarkRed'>ChatBot Database</span>**
> 챗봇을 구현하기 위한 데이터는 ```http://files.pushshift.io/reddit/comments/```에 수집하였으며, 이 대용량 데이터를 MySQL에 업로드해 학습에 사용하려고 한다.

<br>

**<span style='color:DarkRed'>Transaction 초기화 </span>**

> Comment 데이터를 데이터베이스에 업로드(insert)하다보면 대기시간이 길어지는 현상이 생긴다. 그러면 transaction을 초기화해 주는 작업이 필요하다.
> ```Getting “Lock wait timeout exceeded; try restarting transaction even though I'm not using a transaction```
> 자동으로 transaction 대기시간이 100초 이상이되면 초기화하는 방법은 다음과 같다.

```bash
sudo gedit /etc/mysql/mysql.cnf
```

- 아래 내용을 ```mysql.cnf```에 삽입

```python
[mysqld]
interactive_timeout=180
wait_timeout=180
```

<br>

**<span style='color:DarkRed'>Install MySQLdb </span>**


```python
 (tensorflow) ~$ sudo apt-get install python-dev libmysqlclient-dev

 (tensorflow) ~$ pip install mysqlclient
```

<br>

**<span style='color:DarkRed'> MySQLdb 패키지 설치</span>**


```bash
~$ sudo apt-get install python-pip python-dev libmysqlclient-dev
~$ pip install mysqlclient
```
<br>

**<span style='color:DarkRed'> MySQLdb 연결</span>**
치
```python
 timeData = '2006_01'
 connection = MySQLdb.connect(host ='localhost',
                     user='root',
                     password='pwd')
c = connection.cursor()
```

<br>

**<span style='color:DarkRed'> Database 생성</span>**

```python
 c.execute("CREATE DATABASE IF NOT EXISTS {};".format(timeData))
```

<p align="center"><img width="700" height="auto" src="https://i.imgur.com/6VqnCjn.png?1"></p>

<br>

**<span style='color:DarkRed'> 변수 정의 </span>**

```python
# 데이터 하나하나 업로드하는 방식은 비효율적이기 때문에, transaction으로 객체 할당
sql_transaction = []
# 처리되는 데이터 offset
data_counter = 0
# 실제로 추가되는 데이터 offset
paired_data = 0
# offset
start_data = 0
# Delete data where parent IS NULL
cleanup = 1000000 
```

<br>

**<span style='color:DarkRed'> transaction_bldr</span>**

- transaction이 1000이상이 되면 하나씩 execute 후 commit
- transaction을 사용하는 방식은 중간의 데이터 손실을 방지
- 사용된 sql_transaction 초기화

```python
def transaction_bldr(sql):
    # global: 함수에 local하게 적용되는 것이 아니라 함수 밖에 있는 sql_transaction울 불러옴; preserved the values
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('START TRANSACTION;')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        # upload
        connection.commit()
        # initialize
        sql_transaction = []
```

<br>

**<span style='color:DarkRed'> Data load from  external disk </span>**

- ```/mnt/00B267E9B267E1A0/chatbot/2006/RC_2006-01```에 데이터들이 저장되어 있음
- ```buffering```: memory에 부담이 안되게 해당 크기 만큼 불러옴
- json파일들을 불러옴

```python
with open('/mnt/00B267E9B267E1A0/chatbot/{}/RC_{}'.format(timeData.split('_')[0], timeData.replace('_','-')), buffering=2000000000) as f:
    data = [json.loads(row) for row in f]
```

```python
example = data[3] 
print(example)

{'author': 'libertas',
 'author_flair_css_class': None,
 'author_flair_text': None,
 'body': "this looks interesting, but it's already aired, and it's not like there's streaming video, so what's the point?",
 'controversiality': 0,
 'created_utc': 1136079346,
 'distinguished': None,
 'edited': False,
 'gilded': 0,
 'id': 'c2719',
 'link_id': 't3_22528',
 'parent_id': 't3_22528',
 'retrieved_on': 1473821517,
 'score': 2,
 'stickied': False,
 'subreddit': 'reddit.com',
 'subreddit_id': 't5_6',
 'ups': 2}
```

<br>

**<span style='color:DarkRed'> 변수명 정의 </span>**


```python
parent_id = example['parent_id']
# 줄 바꿈 정보를 캐릭터 형태로 변환
body = example['body'].replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
comment_id = example['id']
created_utc = example['created_utc']
score = example['score']
subreddit = example['subreddit']
```
<br>

**<span style='color:DarkRed'> find_parent </span>**

- comment(body)정보만 주어져 있는 데이터를 parent_id와 comment_id를 맵핑해서 같은 쌍이 되도록 데이터를 생성
- 업로드된 DB에서 comment_id == parent_id 가 되는 comment를 불러옴


```python
def find_parent(pid):
    try:
        # LIMIT 1 : 출력되는 데이터들의 첫번째 행을 가져옴
        sql = "SELECT comment FROM question_answer WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        # 한 row를 가져옴
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        # print(str(e))옴
        return False
```

<br>

**<span style='color:DarkRed'> find_existing_score </span>**

- 기존의 parent_id 보다 더 많은 score를 가질 때 데이터를 업데이트하는 기준 score 값

```python
def find_existing_score(pid):
    try:
        sql = "SELECT score FROM question_answer WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        return False
```

<br>

**<span style='color:DarkRed'> acceptable </span>**

- 데이터 분석하기 적합한 텍스트를 선별

```python
def acceptable(data):
    # token > 1000 이상 이거나 데이터가 없을 경우
    if len(data.split(' ')) > 1000 or len(data) < 1:
        return False
    # 자소단위가 너무 많을 경우
    elif len(data) > 32000:
        return False
    # 삭제표시 TEXT가 들어 있는 경우
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True
```
<br>

**<span style='color:DarkRed'> sql_insert_replace_comment </span>**

- 더 좋은 score를 가지는 parent_id가 있다면 그 데이터 정보로 업데이트
- 삽입되는 sql 구문: ```UPDATE question_answer SET parent_id = "t3_22528", comment_id = "c2719", parent = "False", comment = "this looks interesting, but it\'s already aired, and it\'s not like there\'s streaming video, so what\'s the point?", subreddit = "reddit.com", unix = "1136079346", score = "2" WHERE parent_id = "t3_22528";```

```python
def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        # UPDATE DATA SET parent_id => what you want to change
        sql = """UPDATE question_answer SET parent_id = "{}", comment_id = "{}", parent = "{}", comment = "{}", subreddit = "{}", unix = "{}", score = "{}" WHERE parent_id = "{}";""".format(
            parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))
```

<br>

**<span style='color:DarkRed'> sql_insert_has_parent </span>**

- parent_data 정보가 있는 데이터 일 때 
- 삽입되는 sql 구문: ```INSERT INTO question_answer (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("t3_22528","c2719","False","this looks interesting, but it\'s already aired, and it\'s not like there\'s streaming video, so what\'s the point?","reddit.com",1136079346,2);```

```python
def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO question_answer (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(
            parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))
```

<br>

**<span style='color:DarkRed'> sql_insert_no_parent </span>**

- parent_data 정보가 있는 데이터 없을 때
- 삽입되는 sql 구문: ```INSERT INTO question_answer (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("t3_22528","c2719","this looks interesting, but it\'s already aired, and it\'s not like there\'s streaming video, so what\'s the point?","reddit.com",1136079346,2);``` 


```python
def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO question_answer (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(
            parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))
```

<br>

**<span style='color:DarkRed'> Total code </span>**

```python
import MySQLdb
import json
from datetime import datetime

# data year
year = list(range(2006,2015))
# data month
month = list(range(1,13))
# 데이터 하나하나 업로드하는 방식은 비효율적 따라서 아래의 방법을 적용
sql_transaction = []
# offset
start_data = 0
# Delete data where parent IS NULL
cleanup = 1000000


connection = MySQLdb.connect(host ='localhost',
                     user='root',
                     password='1225')

c = connection.cursor()


def create_database():
    c.execute("CREATE DATABASE IF NOT EXISTS {}_reddit;".format(timeData.split('_')[0]))



def create_table():
    # create a new table if the table does not exist
    c.execute(
        "CREATE TABLE IF NOT EXISTS {} (parent_id VARCHAR(10) , comment_id VARCHAR(10), parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT, PRIMARY KEY (parent_id), UNIQUE KEY(comment_id));".format(timeData))

def format_data(data):
    # tokenize 할때 정보를 유지하기 위해 다음과 같이 치환
    data = data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
    return data


def transaction_bldr(sql):
    # global: 함수에 local하게 적용되는 것이 아니라 함수 밖에 있는 sql_transaction울 불러옴
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('START TRANSACTION;')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        print('Commit!!')
        connection.commit()
        sql_transaction = []


def sql_insert_replace_comment(timeData, commentid, parentid, parent, comment, subreddit, time, score):
    try:
        # UPDATE DATA SET parent_id => what you want to change
        sql = """UPDATE {} SET parent_id = "{}", comment_id = "{}", parent = "{}", comment = "{}", subreddit = "{}", unix = "{}", score = "{}" WHERE parent_id = "{}";""".format(timeData,
            parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


def sql_insert_has_parent(timeData, commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO {} (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(timeData,
            parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


def sql_insert_no_parent(timeData, commentid, parentid, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO {} (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(timeData,
            parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


def acceptable(data):
    # token > 1000 이상 이거나 데이터가 없을 경우
    if len(data.split(' ')) > 1000 or len(data) < 1:
        return False
    # 자소단위가 너무 많을 경우
    elif len(data) > 32000:
        return False
    # 삭제표시 TEXT가 들어 있는 경우
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True


def find_parent(timeData, pid):
    try:
        # LIMIT 1 : 출력되는 데이터들의 첫번째 행을 가져옴
        sql = "SELECT comment FROM {} WHERE comment_id = '{}' LIMIT 1;".format(timeData, pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        # print(str(e))
        return False


def find_existing_score(timeData,pid):
    try:
        sql = "SELECT score FROM {} WHERE parent_id = '{}' LIMIT 1;".format(timeData,pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        return False

for y in year:
    for m in month:
        mth = str(m).zfill(2)
        timeData = str(y) + '_' + mth

        create_database()
        c.execute("USE {}_reddit;".format(timeData.split('_')[0]))
        create_table()
        data_counter = 0
        paired_data = 0

        # buffer 가 1일 경우 버퍼된 데이터의 한 line이 보여짐; -1일 경우 시스템의 default로 할당// buffer 2,000,000 byte 로 메모리 할당 2MB
        # 메모리에 큰 영향이 없지만 비효율적
        with open('/mnt/00B267E9B267E1A0/chatbot/{}/RC_{}'.format(timeData.split('_')[0], timeData.replace('_','-')), buffering=2000000000) as f:
            for data in f:
                # print(data)

                data_counter += 1

                if data_counter > start_data:
                    try:
                        data = json.loads(data)
                        parent_id = data['parent_id'].split('_')[1]
                        body = format_data(data['body'])
                        comment_id = data['id']
                        created_utc = data['created_utc']
                        score = data['score']

                        subreddit = data['subreddit']
                        parent_data = find_parent(timeData, parent_id)

                        existing_comment_score = find_existing_score(timeData, parent_id)
                        if existing_comment_score:
                            if score > existing_comment_score:
                                # acceptable: 더 유효한 데이터를 뽑아내는 작업
                                if acceptable(body):
                                    if parent_data:
                                        # 데이터 교체
                                        sql_insert_replace_comment(timeData, comment_id, parent_id, parent_data, body, subreddit,
                                                                   created_utc, score)
                        # 불필요한 데이터를 전처리해서 정제시킴
                        else:
                            # acceptable: 더 유효한 데이터를 뽑아내는 작업
                            if acceptable(body):
                                # if exist or not
                                if parent_data:
                                    if score >= 2:
                                        sql_insert_has_parent(timeData, comment_id, parent_id, parent_data, body, subreddit,
                                                              created_utc, score)
                                        paired_data += 1
                                else:
                                    # parent_data가 없어도 다른 정보들을 삽입
                                    sql_insert_no_parent(timeData, comment_id, parent_id, body, subreddit, created_utc, score)
                    # 오류 메시지의 내용까지 알고 싶을 때 사용하는 방법
                    except Exception as e:
                        print(str(e))
                # print
                if data_counter % 100000 == 0:
                    print('Total datas Read: {}, Paired data: {}, Time: {}'.format(data_counter, paired_data,
                                                                                  str(datetime.now())))

                if data_counter > start_data:
                    if data_counter % cleanup == 0:
                        print("Cleanin up!")
                        sql = "DELETE FROM {} WHERE parent IS NULL;".format(timeData)
                        c.execute(sql)
                        connection.commit()


        if data_counter < 1000000:
            print("Cleanin up!")
            sql = "DELETE FROM {} WHERE parent IS NULL;".format(timeData)
            c.execute(sql)
            connection.commit()

        # reset
        sql_transaction = []
        start_data = 0
```

<br>

**<span style='color:DarkRed'> Data upload to MySQL </span>**

 
<p align="center"><img width="700" height="auto" src="https://i.imgur.com/pCkbuwa.png?1"></p>


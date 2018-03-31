import MySQLdb
import json
import time
from datetime import datetime



timeData = '2006_01'
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


if __name__ == '__main__':
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


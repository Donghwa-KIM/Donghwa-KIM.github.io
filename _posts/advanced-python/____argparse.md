---
layout: post
comments: true
title: Argparse 사용법
categories: Advanced Python
tags:
- Advanced Python
---


**<span style='color:DarkRed'>Argparse</span>**

- source가 어떻게 작동되는지 확인할 때, ```print```보다  ```logging``` 패키지를 사용하는것이 보기 좋고 편리한 것 같다.
- Class 또는 function의 recursive한 구조를 가지고 있으면 ```print```의 출력이 제대로 되지 않는 경우가 있어서 본 저자는 ```logging``` 주로 사용한다.




```python
import logging

logger = logging.getLogger()


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


logging.info('Attach ParamsList...')   
```

<br>


- ```basic config```을 이용하여, '시간'-'메세지' 기본 구조를 설정 
- 아래의 메세지는 시스템 로그로 표현된 예시

```shell
2019-07-12 22:24:30,735 - Attach ParamsList...
```
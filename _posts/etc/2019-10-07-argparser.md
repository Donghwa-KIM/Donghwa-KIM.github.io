---
layout: post
comments: true
title: argparse 사용 튜로리얼
categories: ssh
tags:
- etc.
---

> 모델의 하이퍼 파라미터를 argparse를 이용해 쉽게 관리할 수 있습니다. 

 - main.py: 모델 학습/평가가 수행되는 파이썬 코드 (run in python)
 - main.sh: ``main.py``에 특정한 하이퍼 파라미터를 부여하는 코드 (run in shell)


<br>

**<span style='color:DarkRed'>main.py</span>**

```python
import argparse
import json

parser = argparse.ArgumentParser(description='python Implementation')
parser.add_argument('--input_dir', type = str, default =None, help='input_dir')
parser.add_argument('--output_dir', default=None, help='output_dir')
parser.add_argument('--log_file', type=str, default=None, help='log_file')
parser.add_argument('--sim_threshold', type=float, default=0.4 , help='Similarity threshold')
parser.add_argument('--depth', type=int, default=4, help='Depth of all leaf nodes')
parser.add_argument('--sim_method', type=str, default='seqDistOnlyToken', help='similarity methods: seqDist, seqDistOnlyToken, seqDistVarCont)')
parser.add_argument('--alpha', type=float, default=0.9, help='seqDistVarCont <*> weights')
parser.add_argument('--is_training', action='store_true', help='is_training' )
parser.add_argument('--year', type = str, default='2019', help='current year')
parser.add_argument('--data_type', type = str, default=None, help='data type of os')
parser.add_argument('--garbage', type =json.loads, help='preprocessing as <token>')
parser.add_argument('--regex', nargs='*', help='preprocessing as <*>' )
parser.add_argument('--log_format', type = str, default=None, help='column structure')


args = parser.parse_args()

print('is_training:',args.is_training)
print('log_format:',args.log_format=='')
print('regex:',args.regex)
print('garbage:',args.garbage)
print('input:',args.input_dir)
```

위 ``main.py`` 코드는 모델의 어떠한 하이퍼 파라미터가 있는지 정의하는 단계입니다.

- string: ``str`` type으로 입력을 받으면 문제 없이 수행됩니다.
- float:  ``float`` type으로 입력을 받으면 문제 없이 수행됩니다.
- boolean: ``action='store_true'``를 사용하여 해당하는 인자(argument)가 입력되면 True, 입력되지 않으면 False로 인식하게 됩니다.
- dictionary: ``string``으로 입력되는 ``dict``으로 표현하기 위해 ``json.loads`` 타입을 사용하게 됩니다.
- list: 입력되는 시퀀스 값들을 list로 사용하기 위해 ``nargs='*'``를 사용하게 됩니다.



<br>


**<span style='color:DarkRed'>main.sh</span>**


```sh
python main.py \
--is_training \
--log_format='<Date> <Time> <KST> <Postsql> <Num> <PG_LOG>: <Content>' \
--regex '["][^"]*["]' "['][^']*[']" '(?<=\=)[A-z0-9-.\[\]<*>]+' '(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$' \
--garbage='{"SELECT.+":"<SELECT_Query>","select.+":"<SELECT_Query>", "DECLARE.+":"<DECLARE_Query>", "INSERT.+":"<INSERT_Query>","WITH.+":"<WITH_Query>"}'
```


- ``main.py``를 실행하기 위해 ```python``` 명령어과 같이 수행합니다.
  - ``is_training``이라는 argument를 입력했기 때문에 해당하는 argument는 ``True``를 반환하게 됩니다.
  - ``log_format``는 ``str``를 기대함으로 string으로 값을 입력해주면 됩니다.
  - 자주 실수하는 부분을 말씀드리면 ``log_format= 'TEXT'``는 에러가 나고, ``log_format='TEXT'``는 잘 수행됩니다. 띄어쓰기에 주의를 해야합니다. <span style='color:red'>(공백문자 주의)</span>
  - ``regex``는 원하는 ``string``을 순차적으로 띄어쓰기 단위로 입력 해주면 입력되는 값들이 ```list```로 append되는 것을 확인할 수 있습니다.
  - ``garbage``: ``str``의 형태로 ditionary를 입력해주면, ``json.loads``에 의해 ``dict``으로 형변환되는 것을 확인할 수 있습니다.


---
layout: post
comments: true
title: 맥북에서 Python 가상환경 만들기(virtualenv, virtualenvWrapper)
categories: Virtualenv, VirtualenvWrapper, Python
tags:
- etc.
---

> 맥북에서 가상환경 설치를 해보겠습니다.


<br>

- 시스템 파이썬 설치

```shell
brew install python3
```


<br>


- 가상환경 패키지 설치


```shell
pip3 install virtualenv virtualenvwrapper
```

<br>


- 가상환경 만들기

```shell
virtualenv --python=python3.7 `가상환경이름`
```

<br>

- 가상환경 `in`

```shell
source `가상환경이름` /bin/activate
```

<br>

- 가상환경 `out`

```shell
deactivate
```

---

<br>

**<span style='color:DarkRed'>가상환경 workon</span>**

> 보다 더 편리하게 가상환경을 사용해보자.

<br>


- 가상환경 폴더 설정

```shell
mkdir ~/.virtualenvs
```

<br>

- vim을 활용하여 `bash_profile`에 아래 코드(text) 삽입
	- `VIRTUALENVWRAPPER_PYTHON`: 시스템 파이썬 경로
	- `VITRUALENVWRAPPER_VIRTUALENV`: 새로 생성된 가상환경의 경로

```
export WORKON_HOME=~/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON='/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/bin/python3.7'
export VITRUALENVWRAPPER_VIRTUALENV='/Users/`사용자이름`/`가상환경이름`/bin/python'
source /usr/local/bin/virtualenvwrapper.sh
```

```shell
workon `가상환경이름`
```


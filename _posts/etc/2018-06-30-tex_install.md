---
layout: post
comments: true
title:  우분투에서 texlive & texmaker 설치
categories: Ubuntu

tags:
- etc.
---

**<span style='color:DarkRed'>latex </span>**

- latex를 사용하면 수식이나 포맷이 엄청 깔끔해져서 많이 사용하고 있는데, 우분투에서 설치방법과 한글문제 인코딩문제도 같이 살펴보자. 
- 먼저 설치순서는 다음과 같다.
    - texlive 설치: 문서를 작성하는데 있어서 필요한 패키지 및 base
    - texmaker 설치: 사용자의 편집을 용이하게 해주는 tool

<br>

**<span style='color:DarkRed'> texlive 설치 </span>**

- 한글을 작성하기 위해서 ```texlive-full``` 버전을 설치해야 한다. 

```bash
:~$ sudo add-apt-repository ppa:texlive-backports/ppa
:~$ sudo apt-get update
:~$ sudo apt-get install texlive-full
```
<br>

**<span style='color:DarkRed'> texmaker 설치 </span>**

- 설치 및 실행방법은 다음과 같다.

```bash깔
:~$  sudo apt-get install texmaker
:~$  texmaker
```
<br>
**<span style='color:DarkRed'> 한글실행 예시 </span>**

- ```\usepackage{kotex}```을 추가해주면 한글도 인코딩이 잘된다.

```latex
\documentclass{article}
\usepackage{kotex}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}

\author{김동화}
\title{tex test}

\begin{document}
\maketitle
한글작성을 해보자. 
\end{document}
```

<p align="center"><img width="700" height="auto" src='https://imgur.com/Mclv81x.png'></p>

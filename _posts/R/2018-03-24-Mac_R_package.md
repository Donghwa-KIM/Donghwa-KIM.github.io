---
layout: post
comments: true
title:  Mac에서 설치 안되는 R package 설치하기 
categories: R install

tags:
- R
---



**<span style='color:DarkRed'>arulesViz 설치 방법</span>**

> Mac에서 R 프로그램을 사용하다 보면 몇가지 설치안되는 패키지가 있는데, 그 중에 arulesViz 패키지가 그러하다. 일반적으로 컴파일 문제 때문에 설치가 안되는데 Window창에 아래의 코드를 입력해 설치해주면 쉽게 설치가 된다 (몇가지 warnings가 있지만 가볍게 무시해준다).



```bash
curl -O http://r.research.att.com/libs/gfortran-4.8.2-darwin13.tar.bz2
sudo tar fvxj gfortran-4.8.2-darwin13.tar.bz2 -C /
```
> 다시 R에서 패키지 설치 후 라이브러리를 불러오면 다음과 같다.

```R
install.package('arulesViz')
library(arulesViz)
```

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/XdftV6F.png"></p>
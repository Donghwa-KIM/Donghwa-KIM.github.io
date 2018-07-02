---
layout: post
comments: true
title:  우분투에서 밀림없이 한글입력기 설치
categories: Ubuntu
tags:
- etc.
---

**<span style='color:DarkRed'>한글입력기 설치 </span>**

- ibus로 한글을 사용해 보았지만, 밀림현상때문에 너무 불편해서 다른 방법을 소개해 본다.

<br>

**<span style='color:DarkRed'> uim 설치 </span>**

- 한글을 작성하기 위해서 ```uim``` 을 설치해야 한다. 

```bash
:~$ sudo apt-get install uim uim-byeoru
```

- 우측상단에 ```시스템설정>언어지원>키보드 입력기```의 설정을 uim으로 변경해 준다. 그리고 로그오프 후 재 접속을 한다.
<p align="center"><img width="300" height="auto" src='https://imgur.com/GBVqCiB.png'></p>

**<span style='color:DarkRed'> Key 설정변경 </span>**


- 오른쪽 Alt키를 한영변환으로 사용하기 위해 키설정을 바꿔줘야 한다.

```bash
:~$ xmodmap -e 'remove mod1 = Alt_R'
:~$ xmodmap -e 'keycode 108 = Hangul'
```


- 검색에서 ```uim```을 입력해 uim 입력기를 실행시킨다.
- 아래의 순서대로 설정을 변경해 주자

<br>
Step1) '벼루'라는 키 설정 활성화
<p align="center"><img width="500" height="auto" src='https://imgur.com/eY7FIzu.png'></p>
<br>

Step2) 전체 키설정 켜기/끄기 편집 및 제거
<p align="center"><img width="500" height="auto" src='https://imgur.com/y1YSuVY.png'></p>

<br>

Step3) 한글 키 설정
<p align="center"><img width="500" height="auto" src='https://imgur.com/Fw5aklS.png'></p>

- 이제 ```오른쪽 alt```키로 한영 변환을 할 수 있다.
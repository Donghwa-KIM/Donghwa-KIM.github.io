---
layout: post
comments: true
title:  Jupyter Slide
categories: Jupyter

tags:
- Jupyter
---

**<span style='color:DarkRed'>Jupyter Notebook를 PPT Slide처럼 사용하기</span>**

- html의 형태로 슬라이드 뽑아내는 방법이다. 

1) 파일형식을 바꿔주는 git를 다운로드(Window일 경우 git CMD를 설치) ```https://github.com/hakimel/reveal.js.git``` 

```bash
donghwa@DESKTOP-L9A4HBN MINGW64 ~/Downloads/presentation (master) git clone https://github.com/hakimel/reveal.js.git
```
<br>

2) 기본 CMD를 다시열어 그 clone한 폴더안에 ipynb를 생성 e.g. 이 글에서는 ```ppt.ipynb```이라는 이름을 가진 파일 생성 
- 아래의 같이 컨텐츠를 구성하였음
- 위 쪽 상단에 View >> Cell Toolbar >> Slideshow 체크
- Slide Type 설정
- 저장
<p align="center"><img width="700" height="auto" src='https://i.imgur.com/gDpWWJS.png'></p>
<br>
3) jupyter_contrib_nbextensions를 설치 

```bash
C:\Users\donghwa\Downloads\presentation> pip install jupyter_contrib_nbextensions
```
<br>
4) 슬라이드 생성 
- 아래의 코드를 슬라이드화 하고 싶은 파일을 입력 ```ppt.ipynb```

```bash
C:\Users\donghwa\Downloads\presentation> jupyter-nbconvert --to slides ppt.ipynb --reveal-prefix=reveal.js
```
- 실행 후 아래의 그림과 같이 html파일이 생성이 됨
<p align="center"><img width="700" height="auto" src='https://i.imgur.com/19yfI3y.png'></p>

<br>
5) 사용화면
- 완성된 html를 다른 폴더에서 실행하면 애니메이션 작동되지 않는다. git clone한 folder(e.g. ```C:\Users\donghwa\Downloads\presentation```)에서만 실행해야된다는 것을 잊지말자.
<p align="center"><iframe src="https://i.imgur.com/C27Y4yK.mp4" width="600" height="338" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></p>


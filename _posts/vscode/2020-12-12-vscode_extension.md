---
layout: post
comments: true
title: vscode extension 필수 항목 및 설정 동기화
categories: vscode

tags:
- vscode
---

**<span style='color:DarkRed'>vscode extension 필수 항목 및 설정 동기화</span>**

**1) Material Theme: 코드 테마 변경 extension입니다.**

<p align="center"><img width="600" height="auto" src="../assets/figure/material_theme.PNG"></p>

<br>

**2) Material Icon Theme: 파일 확장자 아이콘 변경 extension입니다.**

<p align="center"><img width="600" height="auto" src="../assets/figure/material_icon_theme.PNG"></p>

- 예시)

<p align="center"><img width="600" height="auto" src="../assets/figure/oceanic.png"></p>

<br>

**3) Settings Sync: extension 설정 저장/불러오기 입니다.**
  - ssh연결을 사용해서 하시는 분들에 유용한 tool입니다.

<p align="center"><img width="600" height="auto" src="../assets/figure/sync.PNG"></p>

<br>

- **사용방법**
  - gist에 설정사항들을 업로드해 upload/download하는 방식이다.
  - 따라서, github login 및 gist id, key 설정이 필요하다.
  - extension을 설치하면, 아래의 창이 뜨는데 해당 정보를 기입해주면 된다.
    - `login with github` → `Edit Configuration`

<p align="center"><img width="600" height="auto" src="../assets/figure/sync_config.png"></p>


<br>

  - `gist.github.com/{github id}`에 들어가 아래와 같은 `cloudSettings` 있는지 확인해 본다.

<p align="center"><img width="600" height="auto" src="../assets/figure/gist.png"></p>


<br>

- 이제 `upload`/`download`를 확인해보자.
  - 설정 gist에 올리기: `Sync: Update/Upload Settings` 
  - 아래그림의 output log에서 extension 항목들을 확인할 수 있다.

  <p align="center"><img width="600" height="auto" src="../assets/figure/upload.png"></p>

  - 다른 host pc에서 설정 gist에서 다운받기: `Sync: Download Settings` 
    - 해당 설정들이 자동으로 설치된다.








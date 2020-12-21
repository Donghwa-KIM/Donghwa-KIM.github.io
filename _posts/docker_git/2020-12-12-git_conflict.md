---
layout: post
comments: true
title:  Git conflict 해결방법
categories: Git

tags:
- Docker & Git
---

**<span style='color:DarkRed'>Git conflict 해결방법</span>**
- github `master`의 변경사항이 존재하며, `local`에 변경사항이 존재할때 conflict 발생

```
error: Your local changes to the following files would be overwritten by merge:
        test.py
Please commit your changes or stash them before you merge.
```

<br>

1) `git stash`: save local change

- local 변경사항들을 저장

```
(base) C:git_test>git stash
Saved working directory and index state WIP on master: 3a879aa add py
```

<br>

2) `git pull`: Update the file as upstream

- local 변경사항들을 제거하고, 현재 github `master`버전으로 최신화

<br>

3) `git stash apply`: merge

- local 변경사항들을 불러와 병합

```
<<<<<<< Updated upstream (server)
import numpy as ndsdsddsds
=======

import numpy as npd
import pandas as pd
>>>>>>> Stashed changes (local)
```

<br>

4) 문서 수정
- `updated upstream`: file change in github
- `stashed upstream`: file change in local

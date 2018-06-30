---
layout: post
comments: true
title:  우분투에서 외장하드 연결 error 해결
categories: Ubuntu

tags:
- etc.
---

**<span style='color:DarkRed'>Jekyll install </span>**

- 우분트를 강제종료하다 보면, 외장하드가 마운트가 안될 때가 있는데, ntfs를 fix시키면 된다.
- 아래의 /dev/sd{}는 파티션의 이름이다. e.g. /dev/sda

```bash를
:~$ sudo ntfsfix /dev/sd{}
```


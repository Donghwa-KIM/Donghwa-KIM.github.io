---
layout: post
comments: true
title: Preliminaries
categories: Bayesian NLP
tags:

- Bayesian NLP
---

> 먼저, 베이지안 이론은 확률적인 표현이 많기 때문에 ```notation```을 정리해보았습니다.

<br>

**<span style='color:DarkRed'>Probability Measures</span>**



- sample space($\Omega​$): 확률분포에서 일어날 수 있는 모든 가능한 element($\omega​$)들로 이루어진 집합

  - e.g. $\omega \in \Omega $: 단어

- random variable($X$): $\Omega$에 있는 $\omega$ 들을 real number($x$)로 맵핑하는 함수

  - e.g. $X \in \{ 0, 1, 2\} $: 단어빈도

- event($A$): real number($x$)들의 특정 event를 나타냄

  - e.g. $A = (X \geq  1)$:  특정 단어의 빈도가 1이상인 경우
    - $A$ 라는 사건은 random variable $X$가 1보다 큰 경우
    - $ x = 1​$ or $x=2​$
  - $X(w) \in A​$ 를 만족해야함

- probability measure($p_X​$): real number($x​$)들이 발생할 확률을 구하는 함수

  - e.g. $p(x=0), \ p(x=1),\ p(x=2)​$

- $p_X(A)​$: sample space($\Omega​$)에서 random variable($X​$)를 가정할 때, event($A​$)가 일어날 확률

  - $p_X(A) = p( X \in A) = p(X^{-1}(A))​$

  - $X(\omega) \in A$ 에 $X^{-1}$ 양변에 취하면,  $\omega \in X^{-1}(A)$ 로,  $A$ event에 포함된 sample space의  element($\omega​$)라고 생각할 수 있다.

    

- Sample space가 인지할 수 있을 정도로 유한하다면, 아래와 같이 direct하게 표현할 수 있다.

  - $p(X \in \\{\omega\\}) = p ( X = \omega) ​$: 단어분포에서 특정 단어($\omega​$)가 일어날 확률

<br>

**<span style='color:DarkRed'>Reference</span>**

http://www.morganclaypoolpublishers.com/catalog_Orig/samples/9781627054218_sample.pdf

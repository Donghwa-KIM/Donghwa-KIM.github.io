---
layout: post
comments: true
title: Modeling Sentences
categories: Convolutional Neural Nets for Text

tags:
- Convolutional Neural Nets for Text
---

본 <a href="https://arxiv.org/pdf/1404.2188.pdf">논문</a>은 max pooling을 layer별로 다이나믹하게 조절하므로써, varied input sequence를 더 잘 표현 하는 descriptors를 설명하고 있다.

<br>

**<span style='color:DarkRed'>Time-Delay Neural Networks</span>**

> - 시간정보를 가진 input sequence에 CNN을 적용한 사례
> - Time dimension에 따라 convolve
> - 즉 단어들의 순서정보($x$: $\mathbf{s} \in \mathbb{R}^s$)를 filter(weight: $\mathbf{m} \in \mathbb{R}^m$)들이 convolve하게 됨
> - convolve된 아웃풋은 다음 레이어의 입력으로 사용됨

**<span style='color:DarkRed'>Narrow/Wide convolution</span>**
<p align="center"><img width="400" height="auto" src="https://imgur.com/LF1DHEL.png"></p>

> - **<span style='color:blue'>narrow </span>** type: 순서정보의 길이보다 짧은 filter size로 설정 ($s \geq m$)  
> - e.g. $s-m+1 = 7-5+1=3 $
> - **<span style='color:green'>wide </span>**  type: 순서정보의 길이보다 더 긴 filter size로 설정 ($s \leq m$)   
> - e.g. $s+m-1 = 7+5-1=11 $



**<span style='color:blue'>narrow </span>** convolution은 **<span style='color:green'>wide </span>** convolution의 subsequence라고 할 수 있으며, **<span style='color:green'>wide convolution</span>**으로 CNN을 구성했을 때 filter들 안에 있는 모든 weight들이 전체 문장 고려하게 되므로, 순서정보들이 더 길게 학습이 된다.

<br>

**<span style='color:DarkRed'>Wide convolution</span>**

wide convolution일 경우 convolve된 activation matrix의 사이즈를 살펴보자.
여기서 $\mathbb{R}^{d}$는 word2vec이용한 embedding size가 될 수도 있고, random initialization으로 embedding size만큼 새롭게 학습할 수도 있다.

- sentence matrix가 $\mathbf{s} \in  \mathbb{R}^{d \times s}$ 
- filter(weight) matrix가 $\mathbf{m} \in  \mathbb{R}^{d \times m}$

아래의 그림과 같이 $d \times (s+m-1) = d \times (7+3-1)= d \times 9$의 output이 생성된다. 입력의 길이가 7에서 9로 2만큼 확장되었다.
<p align="center"><img width="500" height="auto" src="https://imgur.com/vEHYR8k.png"></p>


<br>


**<span style='color:DarkRed'>Dynamic $k$-Max Pooling</span>**

먼저 $k$-max pooling이란 convolve된 아웃풋의 각row에서 가장 값이 높은 k개로 압축하는 것을 의미한다.
예를 들어 $k=3$일 경우 아래의 그림처럼 표현할 수 있다.

<p align="center"><img width="500" height="auto" src="https://imgur.com/0ynYfM5.png"></p>

하지만, Dynamic $k$-Max Pooling이란 $k$가 고정된 값이 아니라, 각 layer에 따라서 바뀌는것을 의미한다.

$l^{th} $ layer마다 변화하는 $k_{l}$은,

$$k_{l}=\max\big(k_{top}, \lceil \frac{L-l}{L} \times s \rceil \big)$$

$L$은 총 layer의 갯수를 의미하며, $l$: 현재 layer가 몇번째인지를 나타낸다. 따라서 위 수식은 layer이 얕을수록, $k$를 input sequence 길이 만큼 가져가고, layer가 깊을수록, $k$를 사용자가 정의한 값(hyper-parameters)의 $k_{top}$으로 다이나믹하게 layer마다 변화된다.

<br>
앞에서 언급한 내용을 요약한 flow-chart는 아래의 그림과 같다.
<p align="center"><img width="400" height="auto" src="https://imgur.com/NjHG3tw.png"></p>

> 1. 4차원으로 embedding된 word vector가 7개($s=7$)가 있다.
> 2. 2개의 필터($4 \times 3$)를 input sequence에 convolve하면, 2개의 아웃풋의 길이는 $4 \times  (7+3-1) = 4 \times 9$이 된다.
> 3. 그 아웃풋의 각 row에서 가장 높은 값 $k$개를 단어순서대로 가져온다, 그리고 non-linear함수를 각 element에 씌워준다(activation).
> 4. 다시 전보다 작은 2개의 필터($4 \times 2$)를 사용하여 wide convolve를 한다.
> 5. (2번째 $k$-max pooling 전에) Folding이란 반으로 접는것을 의미하며 convolve 반대 반향으로 2개씩 더 해준다. 아웃풋들의 계산과정이 각 row별과 진행되어왔는데, 다른 row들 사이의 dependency를 고려하기 위해 stride와 수직방향으로 2개씩 더 해줘 아웃풋의 사이즈를 절반으로 만든다. (반으로 종이접기를 하는것과 유사)
> 6. $k$-max pooling을 적용해 총 아웃풋$2 \times 6$가 2개가 형성된다. 
> 7. $2 \times 6 \times 2 = 24$개의 feature descriptors을 입력으로 full-connected layer를 구성해 클래스를 분류할 수 있다.
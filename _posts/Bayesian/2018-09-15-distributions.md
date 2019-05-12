---
layout: post
comments: true
title: Dirichlet distribution(Dirichlet prior)를 사용하는 이유
categories: Bayesian Statistics(베이지안 통계)
tags:
- Bayesian Statistics(베이지안 통계)
---


> 이 글은 Dirichlet distribution(디리클레 분포)를 사용하는 이유에 대해서 제가 주관적으로 작성한 내용이며, 전반적인 분포 추정의 개념이 필요해 서론에 추가 하였습니다.


<br>

**<span style='color:DarkRed'> Distribution Estimation </span>**


- 분포를 추정하는 방법은 크게 두가지로 나눠진다.
	- <span style='color:Darkgreen'>Parametric 분포 추정</span>
		- 모수를 가정
			- 가우시안 분포: 평균($\mu$), 분산($\sigma$)
			- binomial 분포: 샘플 수 ```n```, 확률 ```p```
		- <span style='color:DarkBlue'>
데이터가 적어도, Parametric 분포를 잘 가정하면 좋은 추정이 될 수 있음</span>
		- <span style='color:red'> 모수에 영향을 받기 때문에, 데이터 따른 분포의 업데이트(유지보수)가 어려움 </span>
	- <span style='color:Darkgreen'>Nonparametric 분포 추정</span>
		- 모수를 가정하지 않음
		- <span style='color:DarkBlue'>데이터가 많아질수록 좋은 추정을 할 수 있음</span>
		- <span style='color:DarkBlue'>모수에 영향을 받지 않기 때문에, 데이터 따른 분포의 업데이트가 쉬움</span>
		- <span style='color:Red'>데이터가 적을 때 분포의 모양이  over-fitting될 수 있음</span>


<br>

<table style="width:100%">
  <tr>
    <th></th>
    <th><span style='color:blue'>Parametric 분포 추정</span></th> 
    <th><span style='color:green'>Nonparametric 분포 추정</span></th>
  </tr>
  <tr>
    <th>예시</th>
    <td> 가우시안 분포 </td> 
    <td> Parzen Window</td>
  </tr>
  <tr>
    <th>사용된 모수</th>
    <td>$\mu$ (분포의 위치를 결정), $\sigma$ (분포의 모양)</td> 
    <td> - </td>
  </tr>
   <tr>
    <th> 분포의 다양성 </th>
    <td> smooth한 분포 표현 </td> 
    <td> flexible한 분포 표현 </td>
  </tr>
</table>


<br>



**<span style='color:DarkRed'> Parametric distribtutions </span>**

- 먼저, classification에 사용되는 대표적인 4개의 분포를 소개해드리겠습니다.
	- Binomial 분포: 단변량, 이산형(discrete)분포
		- 베르누이 이산형 확률변수($\in$ {0,1})를 가정할 때, $n$ 시행횟수로 성공확률 $p$를 표현
	- Beta 분포: 단변량, 연속형(continious)분포
		- $\alpha, \beta$를 이용해 continuous 확률변수을 이용한 분포 표현
	- Multinomial: 다변량, 이산형(discrete)분포
		- $n$ 시행횟수에 대하여, k개의 이산형 확률변수에 대응되는 k개의 확률값($p_k$)들을 사용한 분포
	- Dirichlet 분포: 다변량, 연속형(continious)분포
		- $\alpha_k$에 대하여, k개의 연속형 확률변수에 대응되는 k개의 continous values($x_k$, $\sum_k x_k =1, \forall x_k \geq 0$)들을 사용하여 분포를 표현
- 위 성질을 이용하여, 좌변에는 단변량 ```random variable```속성을 가진 분포를 두고, 우변에는 다변량 ```random variable```속성을 가진 비례식을 세울 수 있습니다.

$$Binomial: Beta = Multinomial:Dirichlet$$

<br>

**<span style='color:DarkRed'> Dirichlet distribtutions </span>**

- 여기서 중요한 점은 dirichlet 분포에서 샘플링 했을 때 k개의 ```continuous random variables```를 샘플을 할 수 있습니다.
- 다른 말로 k차원을 가진 ```continuous random variables``` vector라고 생각할 수도 있습니다.
- dirichlet 특성(probabilistic k-simplex)에 따라, 이 ```continuous random variables```은 0보다 크며, 합은 1이 됩니다(확률의 정의: $x_k$, $\sum_k x_k =1, \forall x_k \geq 0$).
- 따라서 이 k차원 vector는 ```sum to 1```를 만족하기 때문에, multinomial 분포의 모수인 $p_k$($\sum p_k = 1$)에 사용될 수도 있습니다.
- 다시 정리하면, dirichlet 분포에서 샘플링된 ```k차원 vector```는 multinomial 분포를 parameterize(control)한다고 생각할 수 있습니다.
- 이  ```k차원 vector```는 고정된 값이 아니므로 ```k차원 vector```의 분포(각 ```continuous random variable```의 분포) 또는 uncetainty를 표현한다고 생각할 수 있습니다.

<br>

**<span style='color:DarkRed'> 활용 예제 </span>**



- LDA 토픽모델링으로 예를 들면, 한 문서에 대한 토픽의 분포는 k개의 토픽의 확률($p_k$)로 표현할 수 있습니다.
- 이 토픽 분포에서 특정 토픽을 샘플을 할때(해당하는 문서에서 각 단어에 대한 topic), 이 토픽 분포는 multinomial 분포를 가정하게 됩니다.
- dirichlet 분포의 샘플링 된 ```k차원 vector```는 ```sum to 1```를 만족하기 때문에, multinomial 분포의 모수 $p_k$에 사용될 수 있습니다.
- 따라서 분포의 분포(distribution over distributions)를 표현한다고 할 수 있습니다.



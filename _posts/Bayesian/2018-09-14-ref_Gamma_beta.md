---
layout: post
comments: true
title: 감마함수(Gamma function) & 베타함수(Beta function)의 관계  
categories: Bayesian Statistic(베이지안 통계)
tags:
- Bayesian Statistic(베이지안 통계)
---

**<span style='color:DarkRed'>Gamma function</span>**

- Gamma function은 factorial function의 확장으로 smooth한 커브를 찾는 특징이 있다.
- $n$이 양의 정수일때, 
	- $\Gamma(n) = (n-1)!$
- $n$(on x-axis)이 complex number일때, 아래와 같은 그림으로 표현된다.

$$\Gamma(n) = \int^{ \infty}_{0}x^{n-1}e^{-x}dx$$

<p align="center"><img width="400" height="auto" src="https://imgur.com/CCbU832.png"></p>

<br>


**<span style='color:DarkRed'>Beta function</span>**

$$ B(x,y) = \int^{1}_{0} t^{x-1}(1-t)^{y-1}dt$$

- 2개의 인자(x,y)로 beta function 값이 아래와 같이 표현된다.
	- [0,1]에서 (x,y)가 symmetric하게 작아지면 양 끝부분의 확률이 높아진다.
	- [0,1]에서 (x,y)가 symmetric하게 커지면 가우시안 분포처럼, 가운데 부분의 확률이 커진다.
	<p align="center"><img width="350" height="auto" src="https://imgur.com/dpeINC3.png"></p>

<br>


**<span style='color:DarkRed'>Beta function과 Gamma function 관계</span>**

- 2개의 Gamma function을 곱하면,

$$\Gamma(x)\Gamma(y) = \int^{\infty}_{u=0} e^{-u}u^{x-1}du \cdot \int^{\infty}_{v=0} e^{-v}v^{y-1}dv$$

$$= \int^{\infty}_{u=0} \int^{\infty}_{v=0} e^{-u-v}u^{x-1}v^{y-1}du \ dv$$

- $u=zt,\ v = z(1-t)$ 이라고 하면, $ u + v =z $ 라는 관계식이 성립한다.
	 - $ 0 < u <\infty \ \text{and} \ 0 < v <\infty \rightarrow 0 < z <\infty \ \text{and} \ 0 < t <1$ 

$$\frac{\partial(u,v)}{\partial(z,t)} = \Biggl| \begin{array}{cc} \frac{\partial u}{\partial z} & \frac{\partial u}{\partial t} \\ 
\frac{\partial v}{\partial z} & \frac{\partial v}{\partial t} \\ \end{array} \Biggr| = \Biggl|\begin{array}{cc} t & z \\ 
1-t & -z \\ \end{array} \Biggr| = -zt - z(1-t) = -z $$

$$ \Biggl| \frac{\partial(u,v)}{\partial(z,t)} \Biggl| =  z$$

$$ \partial(u,v)   =  z \times \partial(z,t)$$

- 위 성질을 적용해주면, 

$$\Gamma(x)\Gamma(y) =\int^{\infty}_{z=0}\int^{1}_{t=0}e^{-z}(zt)^{x-1}(z(1-t))^{y-1} z \  dz \ dt$$

$$=\int^{\infty}_{z=0}e^{-z}z^{x+y-1} \ \times dz \int^{1}_{t=0} t^{x-1}(1-t)^{y-1} dt$$

$$ = \Gamma(x+y) B(x,y)$$

<br>

$$ \therefore B(x,y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$$

<br>

**<span style='color:DarkRed'> Multivariate beta function </span>**
- 2개이상의 인자, 즉 다변량 인자를 가질 경우 아래와 같은 성질을 가지며 Dirichlet distribution 정의이다.

$$ Beta(\alpha_1,\alpha_2,\ldots\alpha_n) = \frac{\Gamma(\alpha_1)\,\Gamma(\alpha_2) \cdots \Gamma(\alpha_n)}{\Gamma(\alpha_1 + \alpha_2 + \cdots + \alpha_n)}$$

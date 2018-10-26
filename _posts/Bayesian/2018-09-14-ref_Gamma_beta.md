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

$$\Gamma(n) = \int^{ \infty}_{0}x^{n-1}e^{-x}dx$$

- $n$이 양의 정수일때, $\Gamma(n) = (n-1)!$이 성립하며, **fatorial의 일반화**한 것으로 생각할 수 있다.

<br>


**<span style='color:DarkRed'>Beta function</span>**

- Beta function은 gamma function의 비율($B(x,y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$
)로 표현되는 2변량$(x,y)$ 함수 이다.

$$ B(x,y) = \int^{1}_{0} t^{x-1}(1-t)^{y-1}dt$$

- 앞선 설명된 감마함수가 factorial의 일반화 였다면, ${n\choose k} = \frac{1}{(n+1)B(n-k+1,k+1)}$의 관계식이 성립하므로 베타함수는 **이항계수의 일반화**로 생각할 수 있다.


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
- 위에서 2개인자를 가진 beta함수에 대해서 살펴보았다
- 더 많은 인자를 가질 때 어떻게 될까?
- 2개이상의 인자, 즉 다변량 인자를 가질 경우 아래와 같은 성질을 가지며 Dirichlet distribution 정의이다.

$$ Beta(\alpha_1,\alpha_2,\ldots\alpha_n) = \frac{\Gamma(\alpha_1)\,\Gamma(\alpha_2) \cdots \Gamma(\alpha_n)}{\Gamma(\alpha_1 + \alpha_2 + \cdots + \alpha_n)}$$

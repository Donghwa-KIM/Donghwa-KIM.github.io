---
layout: post
comments: true
title: 파이썬으로 Covariance 구하기
categories: Bayesian NLP, Covariance, Python
tags:
- Bayesian with python (통계, 베이지안)
---


> 이 글은 파이썬을 활용한 Covariacne를 구하는 과정을 설명한 글입니다. Covariacne는 크게 3가지(`full`, `diag`, `spherical`)를 방식으로 구할 수 가 있습니다. 아래의 random으로 생성된 데이터로 예를 들어 보겠습니다.

```python
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
X
```

```
array([[ 0.68026725, -0.01634235],
       [ 3.80951844,  0.79848348],
       [-1.6613724 , -0.57766695],
       ...,
       [-5.86152028,  3.06842556],
       [-5.01893361,  3.11090369],
       [-6.79933099,  2.08232074]])
```

<br>



**<span style='color:DarkRed'> Full Covariance</span>**

- Covariance를 산출

```python
# np.atleast_2d: 1-dim array가 들어오면 2-dimensional로 바꿔주는 함수
# np.cov: covariance
np.atleast_2d(np.cov(X.T))
```

```
array([[10.51401566, -4.13489296],
       [-4.13489296,  2.57580056]])
```

<br>

**<span style='color:DarkRed'> Diagonal Variance</span>**

- 변수별로 variance를 산출

```python
# axis=0: 변수별로 variance를 산출
# ddof: Degrees of Freedom
np.var(X, axis=0, ddof=1)
```

```
array([10.51401566,  2.57580056])
```

<br>

**<span style='color:DarkRed'> Spherical Variance</span>**

- 변수별로 산출된 variance에 대한 평균

```python
# ddof: Degrees of Freedom
np.var(X, axis=0, ddof=1).mean()
```

```
6.544908108895438
```

<br>




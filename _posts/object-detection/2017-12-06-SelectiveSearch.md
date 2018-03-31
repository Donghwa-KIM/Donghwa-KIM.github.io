---
layout: post
comments: true
title: Selective Search (선택적 탐색)
categories: Object Detection
tags:
- Object Detection
---

**<span style='color:DarkRed'>Selective Search</span>**

Bounding boxe들을 찾아주는 super pixel기반(**<span style='color:grey'>엣지를 잘 표현하는 patch</span>**)의 selective search는 hierarchical grouping algorithm 방식을 사용
<p align="center">
  <img width="500" height="auto" src="https://i.imgur.com/DPEJcwI.png">
</p>

<hr>


**<span style='color:DarkRed'>학습 방법</span>**
>**<span style='color:blue'>Definition</span>**\\
$R$: 선택된 region 후보들 {$r_{1}, r_{2},...$} \\
$S$: region들의 유사도 집합 {$s(r_{i}, r_{j})$,...}
<p align="center">
  <img width="400" height="auto" src="https://i.imgur.com/Rb1CpW7.png">
</p>
>1. $r_{1}, r_{2},...r_{n}$들을 초기화
2. 가장 유사성이 높은 $s($<span style="color:Blue">$r_{i}$</span>, <span style="color:Red">$r_{j}$</span>$)$을 선택
3. 선택된 <span style="color:Blue">$r_{i}$</span>, <span style="color:Red">$r_{j}$</span>의 영역을 <span style="color:green"> $r_{t}$</span>로 합침
4. 유사성 집합 $S$에서 이미 합쳐진 $s($<span style="color:Blue">$r_{i}$</span>, $r_{\ast})$, $s(r_{\ast}$, <span style="color:Red">$r_{j}$</span>$)$을 제거
5. 합쳐진 <span style="color:green"> $r_{t}$</span>와 나머지 region들의 새로운 유사성 집합($S_t$)를 계산
6. 새로운 유사성 집합($S_t$), 합쳐진 region(<span style="color:green"> $r_{t}$</span>)을 원래 집합($S$, $R$)에 포함시킴
7. 하나의 region이 될때까지 반복

<hr>

**<span style='color:DarkRed'>어떻게 유사성이 높다고 판단할 수 있는가?</span>**
>**<span style='color:blue'>Region similarity</span>**
>: Selective Search의 유사성은 $[0,1]$ 사이로 정규화된 4가지 요소(Color, Texture, Size, Fill)들의 가중합으로 계산됨
><p align="center">$$S(r_{i},r_{j})=a_{1}s_{colour}(r_{i},r_{j})+a_{2}s_{texture}(r_{i},r_{j})+a_{3}s_{size}(r_{i},r_{j})+a_{4}s_{fill}(r_{i},r_{j})$$</p><p align="center">$$A=\{a_{1}, a_{2}, a_{3}, a_{4}\}, \quad 0 \le a_i \le 1 $$</p>
>A값 설정은 다양한 전략들이 존재하여 본 논문에서는 모든 요소들의 가중치가 같다고 봄
<hr>

- **Color**: 이미지의 색깔
    - 각 컬러 채널을 25개 bin으로 설정
    - 각 region 마다 컬러 히스토그램 생성 $C_i=\{c^1_i,...,c^n_i\} $
    - 차원 수 $n$ = 75 (RGB 3채널 $\times$ 25개의 Bin)
    - $L_1$ norm 정규화 $[0,1]$
    - 인접한 regions의 교집합을 유사도로 측정
    - $s_{colour}(r_{i},r_{j}) = \sum\limits^{n}\limits_{k=1}min(c^k_i,c^k_j)$
    - 유사성 척도의 min function은 두 히스토그램의 교집합을 나타냄

<p align="center">
  <img width="500" height="auto" src="https://i.imgur.com/G7w6Ngf.png">
</p>

<hr>
**<span style='color:DarkRed'>합쳐진 Region(<span style="color:green">$r_{t}$</span>)의 $c_t$은 어떻게 구할것인가?</span>**
>초기에 생성한 $C_i=\{c^1_i,...,c^n_i\}$을 이용하여 업데이트함 (efficiently propagated)
><p align="center">$$C_t = \frac{size(r_i) \times C_i + size(r_j) \times C_j}{size(r_i)+size(r_j)}$$</p><p align="center">$$r_t = size(r_i)+size(r_j)$$</p>
<hr>
- **Texture**: 주변 Pixel값 들의 변화량 <a href="{{ site.baseurl }}hog.html">[참고자료: HoG]</a>
    - $\sigma = 1$인 8방향의 가우시안 미분을 적용
    - 10개의 Bin으로 히스토그램 도출 $T_i=\{t^1_i,...,t^n_i\} $
    - $L_1$ norm 정규화 $[0,1]$
    - 80차원(8방향 $\times$ 10차원의 Bin)의 벡터로 인접한 region들의 유사성을 평가
    - RGB(3차원)의 경우: 8방향 $\times$ Bin의 수(10차원) $\times$ RGB(3차원) = 240차원
    - $s_{texture}(r_{i},r_{j}) = \sum\limits^{n}\limits_{k=1}min(t^k_i,t^k_j)$

<br>

- **Size**: Region들의 사이즈
    - 사이즈가 작을 수록 유사도가 높음
    - $s_{size}(r_{i},r_{j}) = 1-\frac{size(r_i)+size(r_j)}{size(im)}$
    - $im$은 원 이미지를 나타냄

<br>
- **Fill**: candidate Bounding Box 크기와의 차이 
    - candidate Bounding Box와 Region들의 사이즈의 차이가 적을수록 유사도가 높음
    - $s_{fill}(r_{i},r_{j}) = 1-\frac{size(BB_{ij})-size(r_i)-size(r_j)}{size(im)}$
    - $im$은 원 이미지를 나타냄
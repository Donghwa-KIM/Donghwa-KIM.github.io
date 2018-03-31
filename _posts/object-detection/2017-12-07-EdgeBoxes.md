---
layout: post
comments: true
title:  Edge Boxes (엣지 박스)
categories: Object Detection
tags:
- Object Detection
---


**<span style='color:DarkRed'>Edge Boxes</span>**

Gradient orientation을 기반으로 edge group을 표현하고, 이를 이용하여 bounding box score을 계산하는 방법

<p align="center"><img width="auto" height="auto" src="https://i.imgur.com/mXVLnjE.png"></p>
> - 첫번째 행은 원래 이미지를 나타냄
> - 두번째 행은 gradient magnitude와 orientation을 사용하여 **<span style='color:green'>edge group</span>**들을 표현한 그림
> - 세번째 행은 위에서 구해낸 **<span style='color:green'>edge group</span>**에 **<span style='color:blue'>affinity(유사성)</span>**을 적용하여 bounding box를 찾아내는 예제임
<hr>
**<span style='color:DarkRed'>Selective Search와 Edge Boxes 비교</span>**

<table style="width:100%">
  <tr>
    <th></th>
    <th><span style='color:blue'>Seletive Search</span></th> 
    <th><span style='color:green'>Edge Boxes</span></th>
  </tr>
  <tr>
    <th>학습기반</th>
    <td>super pixel </td> 
    <td>edge group</td>
  </tr>
  <tr>
    <th>비교대상</th>
    <td>color, texture, size, fill</td> 
    <td>affinity</td>
  </tr>
   <tr>
    <th> 탐색방법 </th>
    <td> hierachical </td> 
    <td>path of edge groups</td>
  </tr>
</table>
<hr>
**<span style='color:DarkRed'>Edge Response: </span>** **Structured Edge Detection**

- **<span style='color:blue'>Random Forests For Edge Detection</span>**

<p align="center"> <img width="500" height="auto" src="https://i.imgur.com/8PVow01.png"></p>
>   - **<span style='color:Gold'>이미지의 Patch</span>**가 주어지면, 클래스가 edge patch들(**<span style='color:green'>target distribution</span>**)인 Random Forests 분류 문제로 생각해 볼 수 있음
>   - 분류 기준은 낮은 entropy를 가지도록 node가 분리 됨

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/Y1BUvgc.png"></p>

>- Leaf 노드는 **<span style='color:green'>target</span>**(edge patch)의 확률 분포로 표현 되며 확률이 가장 높은 edge를 검출하게 됨
>- 각 트리에서 검출된 edge들을 majority vote방식으로 최종 label(edge)를 선택

- **<span style='color:blue'>Structured Edge Detection</span>**

  - segmentation masks: structured labels
  - colors: 유사한 structured labels (clustered)
<p align="center">
  <img width="300" height="auto" src="https://i.imgur.com/zPn7s5n.png">
</p>
> - $ y \in Y$: 16$\times$16 segmentation masks (structured labels)
> - 후보 feature의 수: **<span style='color:blue'>[</span>**(image patch size(32 $\times$ 32)$\times$(3 color   $+$ 2 magnitude $+$ 8 orientation) $\div$ radius 2 triangle filter $\div$ downsample by a factor of 2 **<span style='color:blue'>]</span>** + **<span style='color:blue'>[</span>** Pairwise difference features(${5 \times 5}\choose{2}$) $\times$(3 color $+$ 2 magnitude $+$ 8 orientation) **<span style='color:blue'>]</span>** **<span style='color:blue'> $=7228$</span>**  
> - binary vector로 edge들 간의 유사성(거리)을 확인하기에는 너무 많은 pixel pair들이 존재함
<p align="center">
  <img width="600" height="auto" src="https://i.imgur.com/aet0YJw.png">
</p>
>- $Z:$ binary vector로 edge들 간(**pair**)의 유사성(거리)
>- PCA(256차원 $\rightarrow$ 5차원으로 축소)와 $k$-means 클러스터링을 사용한 그룹화
>- $C$ = {$ 1, . . . , k$}: 유사한 structed labels ($k$ cluster)
>- $z_k= \arg\min\limits_{z_k} \sum\limits_{i,j} (z_{kj}-z_{i,j})^2$를 만족하는 $z_k$(거리의 차이가 최소)의 $y_k$(mask segmatation)를 선택 
>- 직관적으로, 클러스터된 $z_k$와 다른 edge들과의 거리가 최소가 되는 $z_k$에 segmentation masks($y_k$)를 선택

<hr>

**<span style='color:DarkRed'>Edge Groups</span>**
>- 위에서 설명한 Structured Edge Detection으로 edge를 검출
>- edge들의 연결이 직선(straight contour)으로 이루여져 있다면 서로 높은 **<span style='color:blue'>affinity</span>**를 가짐
- edge들의 연결이 곡선(curvature contour)으로 이루여져 있다면 서로 낮은 **<span style='color:blue'>affinity</span>**를 가짐
- edge group은 높은 affinity를 가진다고 가정
- 따라서, **<span style='color:Red'>각 edge</span>**들의 **<span style='color:blue'>affinity(유사성)</span>**를 고려하는 것이 아니라, **<span style='color:Red'>edge group 사이</span>**의 **<span style='color:blue'>affinity(유사성)</span>**를 고려
- 각 픽셀에 대해서 edge maginitude($m_p$), edge orientation($\theta_p$)를 구함 **<a href="{{ site.baseurl }}hog.html" style="color:Darkgrey">(참조)</a>**
- 아래의 그림처럼 gradient(edge) maginitude가 0.1 이상인 것만 edge로 식별(Non-Maximal Suppression 방식)

<p align="center">
  <img width="auto" height="auto" src="https://i.imgur.com/itZzrLW.png">
</p>
> - Orientation의 차이가 $\frac{\pi}{2}$ 이상이면 이동 경로를 멈추는 greedy approach를 사용
> - 아래 그림과 같이 8-connected edges로 Group을 형성

<p align="center">
  <img width="auto" height="auto" src="https://i.imgur.com/NRqQwVk.png?1">
</p>

<hr>
**<span style='color:DarkRed'>Affinity</span>**
> - Group들 간의 유사성을 확인(affinity)해보고 싶음
> - $a(s_i,s_j)$은 edge group들 간의 유사성(affinity)을 나타내며 아래 식과 같음 
   - $s_i \in S$ : edge group은 $s_i$로 표기하며,  $S$는 $s_i$들의 집합
   - **<span style='color:Red'>$\theta_{ij}$</span>**$:$ 각 group안에 평균 pixel(position) $x_i$ 와 $x_j$의 각도
   - **<span style='color:blue'>$\theta_{i}$</span>**$:$ $i^{th}$ group안에 평균 pixel(position)의 orientation
   - **<span style='color:green'>$\theta_{j}$</span>**$:$ $j^{th}$ group안에 평균 pixel(position)의 orientation

 <p align='center'> $a(s_i,s_j) =$ $|cos($<span style='color:blue'>$\theta_{i}$</span>$-$ <span style='color:Red'>$\theta_{ij}$</span>$) \ cos($<span style='color:green'>$\theta_{j}$</span>$-$ <span style='color:Red'>$\theta_{ij}$</span>$)|^r$ </p>

> - 각 group안에 평균 pixel들의 각도와 그 group들의 orientation이 유사하다면 유사성(affinity)은 높다고 할 수 있음 
> - $r$는 affinity의 민감도(sensitivity)를 조절하기 위한 값

<hr>
**<span style='color:DarkRed'>Bounding box scoring</span>**
> - $b$: candidate bounding box 
> - $m_{i}:$ $i^{th}$ edge group($s_i$)에 있는 모든 edge $p$의 magnitudes $m_p$의 합 ($\sum_{p}maginitude_{p}$)
> - $\overline{x}:$ $i^{th}$ edge group($s_i$)에서 pixel $p$를 랜덤하게 추출
> - $S_{b}:$ $i^{th}$ edge group($s_i$)이 box boundary($b$)에 overlap되는  집합
> - $w_{b}(s_{i})=1$ 이면 **edge group들**이 box $b$에 완전히 담겨져 있는 것을 의미함  

<p align="center">
  <img width="auto" height="auto" src="https://i.imgur.com/H413l8e.png">
</p>
> - **<span style='color:DarkRed'>Edge boxes의 핵심방법론은 edge group $s_i$들을 bounding box와 겹치는 $S_b$안에 포함 되도록, 즉 아래의 식 같이 affinity를 최대화 하는 path of edge group을 구하는 것임 </span>**

<p align='center'>$w_{b}(s_{i}) = 1- \max\limits_{T} \prod\limits_{j}\limits^{|T|-1} a(t_{j}, t_{j+1})$</p>
> - 위에서 정의한 edges $p$ magnitude의 합 $m_i$에 대해서 affinity를 최대화 시키는 path에서 선택된 가중치 $w_{b}(s_{i})$를 적용하며 아래와 같은 bounding box scoring을 정의할 수 있음
  - $b_w$: box의 width
  - $b_h$: box의 height
  - 아래의 식의 분모는 box의 둘레를 표현
  - $k(=1.5)$는 더 큰 window 사이즈를 가지도록(더 많은 edge들을 가지도록)조절해주는 파라미터

<p align='center'>$$h_b=\frac{\sum_{i}w_b(s_i)m_i}{2(b_w+b_h)^{k}}$$</p>

> - box의 중심부에 위치하는 edge들은 덜 중요한 정보를 가지고 있으므로, 이 부분에 해당하는 edge magnitudes을 빼주어서 $b^{in}$으로 재 정의

<p align='center'> $$h_{b}^{in} = h_b - \frac{\sum_{p \in b^{in}}m_p}{2(b_w+b_h)^{k}}$$</p>

> - 아래의 그림은 bounding score를 계산한 그림
> - 이미지안의 **<span style='color:blue'>파란 box</span>**는 box의 차원을 보여주고 있음
> - 완만한 contour한 부분을 제거한 경우(right), 그렇지 않은 경우(middle)보다, straight contour한 edge를 가지며 high affinity를 가져 보다 나은 edge boxes를 검출 할 수 있음
<p align="center">
  <img width="700" height="auto" src="https://i.imgur.com/P6FTz9y.png">
</p>

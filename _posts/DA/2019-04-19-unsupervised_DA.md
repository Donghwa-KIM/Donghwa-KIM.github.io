---
layout: post
comments: true
title: Unsupervised Domain Adaptation by Backpropagation
categories: Domain Adaptation

tags:
- Domain Adaptation
---




**<span style='color:DarkRed'>Domain Adaptation</span>**


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/BcCAeFw.png?1"></p>


- 위 그림 같이, 예측하고자하는 label은 같지만, 그 source와 target domain이 다를 경우 예측된 label은 domain에 의해서 상당히 다를 수 있다. 
	- 딥러닝은 label-set이 많아야 잘 학습됨
	- label이 많아도 실제(test)데이터와 다를 수 있음
	- ```domain adaptation```방법론을 사용하면 ```train```와 ```test```분포를 유사하게 학습할 수 있음

<br>


**<span style='color:DarkRed'>관련 연구</span>**

- 과거 연구에는 일반적으로 파라미터가 고정된 features을 사용(```shallow learning```)
	- source domain(train)에서 유효한 것만 샘플링
	- source domain feature를 target domain에 잘 맞도록 transformation(re-weighted)
		- 제안된 논문은 ***feature 자체를 변형***
	- 두 domain에 대한 유사성을 학습
		- 제안된 논문은 ***두 분포의 차이***를 학습

---

한 해결책으로 <a href="http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf">Unsupervised Domain Adaptation by Backpropagation</a>가 제안 되었다.

- target domain의 label이 없어도 학습가능	
- end-to-end learning
- backpropagation만 조금 바꾼 단순한 구조

<br>

**<span style='color:DarkRed'>학습 구조</span>**

- label predictor($G_y$): label을 잘 예측하는 feed-forward 레이어 학습
- domain classifier($G_d$): domain을 잘 예측하는 feed-forward 레이어 학습
- feature extractor($G_f$): feature 임베딩 gradient 학습은 직관적으로 두가지 gradient로 나눠진다.



$$ \theta_f \leftarrow \theta_f - \alpha (\frac{\partial{L^{i}_{y}}}{\partial{\theta_f}}-\lambda \frac{\partial{L^{i}_{y}}}{\partial{\theta_f}})$$



- (1) label을 잘 맞추는 방향($\frac{\partial{L^{i}_{y}}}{\partial{\theta_f}}$)으로 업데이트 
- (2) domain을 잘 맞추는 방향과 반대 방향($-\lambda \frac{\partial{L^{i}_{y}}}{\partial{\theta_f}}$)으로 업데이트
- $\lambda$: label prediction과 domain adaptation의 상대적 중요도
- $\alpha$: learning rate



<p align="center"><img width="500" height="auto" src="https://i.imgur.com/tl9MnEy.png"></p>


<br>



**<span style='color:DarkRed'>요약</span>**

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-0pky">source(train) domain</th>
    <th class="tg-0pky">target(test) domain</th>
  </tr>
  <tr>
    <td class="tg-c3ow">Objectives</td>
    <td class="tg-0pky">주요 Task를 잘 맞추는 것<br>(discriminativeness)</td>
    <td class="tg-0pky">서로 다른 domain의 차이를 줄이는 것<br>(domain-invariance)</td>
  </tr>
  <tr>
    <td class="tg-0pky">Optimization</td>
    <td class="tg-0pky">min label loss</td>
    <td class="tg-0pky">max domain loss</td>
  </tr>
</table>


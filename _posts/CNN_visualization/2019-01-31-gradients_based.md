---
layout: post
comments: true
title: Gradient-based Visualization with Tensorflow
categories: CNN Visualization
tags:
- CNN Visualization
---




```python
import tensorflow as tf
from tensorflow.python.framework import ops # for gradient
from tensorflow.python.ops import gen_nn_ops # compute gradient
```
<p align="center"><img width="400" height="auto" src="https://i.imgur.com/KdTec6f.png"></p>

<br>

- 위 예제를 텐서플로우 함수를 이용해 구해 보자. 

- 전체 코드는 <a href="https://github.com/Donghwa-KIM/grad_CAM/blob/master/%40RegisterGradient.ipynb">여기</a>에 참조되어 있다.


```python
tf.reset_default_graph()
```


```python
feature_out = tf.constant([[1,-1,5],[2,-5,-7],[-3,2,4]], dtype=tf.float32, name ='features')
feature_out
```




    <tf.Tensor 'features:0' shape=(3, 3) dtype=float32>




```python
grad = tf.constant([[-2,3,-1],[6,-3,1],[2,-1,3]], dtype=tf.float32, name ='gradients')
grad
```




    <tf.Tensor 'gradients:0' shape=(3, 3) dtype=float32>




```python
# feature output > 0
BackpropRelu = gen_nn_ops.relu_grad(grad, feature_out)

# grad > 0
DeconvRelu = tf.where(0. < grad, grad, tf.zeros(grad.get_shape()))

# (feature output > 0 & grad > 0)
GuidedReluGrad  = tf.where(0. < grad, gen_nn_ops.relu_grad(grad, feature_out), tf.zeros(grad.get_shape()))
```


```python
sess = tf.InteractiveSession()
print('BackpropRelu: \n',BackpropRelu.eval())
print('DeconvRelu: \n',DeconvRelu.eval())
print('GuidedReluGrad: \n',GuidedReluGrad.eval())
```

    BackpropRelu: 
     [[-2.  0. -1.]
     [ 6. -0.  0.]
     [ 0. -1.  3.]]
    DeconvRelu: 
     [[0. 3. 0.]
     [6. 0. 1.]
     [2. 0. 3.]]
    GuidedReluGrad: 
     [[0. 0. 0.]
     [6. 0. 0.]
     [0. 0. 3.]]

<br>

**<span style='color:DarkRed'>텐서플로우에서 Gradient 변환 </span>**

- unused_op: 학습되는 train_op(output)
- gradient > 0인 값들에 grad 산출

- decorator(@ops.RegisterGradient): 함수(_GuidedReluGrad)를 입력으로 받아 특정 함수(function)를 반환함
- 아래의 function은 사용자가 직접 정의한 op로 'Relu'와 바꿀 op가 됨
- 아래의 python decorator의 개념은 [여기]({{ site.baseurl }}/decorators.html)에 추가 설명되어 있다.

<br>

- For ```BackpropRelu```

```python
@ops.RegisterGradient("BackpropRelu")
def _BackpropRelu(unused_op, grad):
    return gen_nn_ops.relu_grad(grad, unused_op.outputs[0])
```

<br>

- For ```DeconvRelu```

```python
@ops.RegisterGradient("DeconvRelu")
def _DeconvRelu(unused_op, grad):
    return tf.where(0. < grad, grad, tf.zeros(tf.shape(grad)))
```
<br>

- For ```GuidedRelu```

```python
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(unused_op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, unused_op.outputs[0]),
                    tf.zeros(tf.shape(grad)))
```
<br>

- 현재 default_graph에 정의된 op들을 g라는 sub-graph로 정의
- ```Relu```함수의 gradient를 ```<method>```라고 정의된 gradient 함수로 정의

```python

g = tf.get_default_graph() 
with g.gradient_override_map({"Relu": "<method>"}):
    ...graph...
```

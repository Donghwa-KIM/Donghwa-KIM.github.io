---
layout: post
comments: true
title: Backpropagation Grad-CAM
categories: CNN Visualization
tags:
- CNN Visualization
---

<p align="center"><img width="400" height="auto" src="https://i.imgur.com/cQ2VP6S.png"></p>

- Gradients-based backpropagation은 특정 class(확률이 가장 높은)에 대해서 relu output이 0보다 큰 gradients(w.r.t last conv)를 feature map에 곱하는 방식(weighted-average)

$$\text{gradients}= \frac{\partial y_{label_i}}{\partial \text{ last conv layer}} (\text{relu output} > 0)$$

- gen_nn_ops.relu_grad는 0보다 큰 relu output의 gradients를 계산하는 함수이다.

	```python
	@ops.RegisterGradient("BackpropRelu")
	def _BackpropRelu(unused_op, grad):
	    return gen_nn_ops.relu_grad(grad, unused_op.outputs[0])
	```

---
**<span style='color:DarkRed'>Implementation</span>**

- 전체 코드는 <a href="https://github.com/Donghwa-KIM/grad_CAM/blob/master/backprop.ipynb">여기</a>에 참조되어 있다.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import cv2
from imagenet_classes import class_names

# image pre-processing
from imageio import imread
from PIL import Image

# relu gradient
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
```

<br>

- pretraining weights 불러오기
- ```vgg16_weights.npz```파일은 <a href="https://www.dropbox.com/s/npxaybogajkgl6b/vgg16_weights.npz?dl=0">여기</a>에 첨부되어 있다.

```python
# Assign weight file.
weight_file_path = 'vgg16_weights.npz'
# number of classes
n_labels = 1000      
```


```python
pretrained_weights = dict(np.load(weight_file_path, encoding='bytes'))
```


```python
print(class_names[0:5],'...','\n')
print('number of classes: ', len(class_names))
```
	['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark'] ... 

	number of classes:  1000

<br>

---

- 이미지 전처리
    - 학습 네트워크의 이미지 사이즈는 $224 \times 224$ 로 고정되어 있다.


```python
img = imread('tmp.jpeg')
print('shape:',img.shape)
```

    shape: (960, 720, 3)



```python
img = imread('tmp.jpeg')
# to PIL
img = Image.fromarray(img).resize((224, 224))
plt.imshow(img)
plt.axis('off')
```




    (-0.5, 223.5, 223.5, -0.5)




<p align="center"><img width="300" height="auto" src="https://i.imgur.com/0OY2SQP.png"></p>



```python
# to numpy
img = np.array(img)
img.shape
```




    (224, 224, 3)

<br>

---
**<span style='color:DarkRed'> VGG net</span>**

<p align="center"><img width="600" height="auto" src="https://i.imgur.com/IE9EasS.png"></p>


- default 그래프 생성
- Pre-trained VGG net을 bulid

```python
graph = tf.get_default_graph()
```

- 현재 graph에는 아무것도 없는 상태이지만 앞으로 생성되는 operation들은 이 그래프에 추가가 된다.


```python
graph.get_operations()
```




    []

<br>

- **<span style='color:blue'>Backpropagation-based</span>**:
relu output이 양수인 부분에만 gradient를 계산하는 방식
- {'```Relu```': '```BackpropRelu```'}: ```Relu```의 그래디언트 계산방식을 ```BackpropRelu```로 변환

```python
@ops.RegisterGradient("BackpropRelu")
def _BackpropRelu(unused_op, grad):
    return gen_nn_ops.relu_grad(grad, unused_op.outputs[0])
```


```python
def conv_layer(graph, inputs, name, stride = 1):    

    with tf.variable_scope(name) as scope:
        
        # The weights are retrieved according to how they are stored in arrays
        w = pretrained_weights[name+'_W']
        b = pretrained_weights[name+'_b']
        
        conv_weights = tf.get_variable(
                "W",
                shape=w.shape,
                initializer=tf.constant_initializer(w)
                )
        conv_biases = tf.get_variable(
                "b",
                shape=b.shape,
                initializer=tf.constant_initializer(b)
                )

        conv = tf.nn.conv2d(inputs, conv_weights, [1,stride,stride,1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)
        
        with graph.gradient_override_map({'Relu': 'BackpropRelu'}):
            relu = tf.nn.relu(bias, name=name)
        
    return relu  
```

<br>

```python
image_mean = [103.939, 116.779, 123.68]
epsilon = 1e-4
```


```python
# Define Placeholders for images and labels
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int32, [None], name='labels')
```


```python
r, g, b = tf.split(images_tf,[1,1,1] , 3)
print(r)
print(g)
print(b)
```

    Tensor("split:0", shape=(?, 224, 224, 1), dtype=float32)
    Tensor("split:1", shape=(?, 224, 224, 1), dtype=float32)
    Tensor("split:2", shape=(?, 224, 224, 1), dtype=float32)

<br>

```python
image = tf.concat([b-image_mean[0],g-image_mean[1], r-image_mean[2]],3)
image
```




    <tf.Tensor 'concat:0' shape=(?, 224, 224, 3) dtype=float32>

<br>

- ```Conv1_1``` output 계산 식: $(224-1)/1 +1 = 224$


```python
relu1_1 = conv_layer(graph, image, "conv1_1" )
relu1_1
```




    <tf.Tensor 'conv1_1/conv1_1:0' shape=(?, 224, 224, 64) dtype=float32>

<br>

- ```Conv1_2``` output 계산 식: $(224-1)/1 +1 = 224$


```python
relu1_2 = conv_layer(graph, relu1_1, "conv1_2" )
relu1_2
```




    <tf.Tensor 'conv1_2/conv1_2:0' shape=(?, 224, 224, 64) dtype=float32>

<br>

- ```pool1``` output 계산 식: $(224-2)/2 +1 = 112$


```python
pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
pool1
```




    <tf.Tensor 'pool1:0' shape=(?, 112, 112, 64) dtype=float32>

<br>

- ```Conv2_1``` output 계산 식: $(112-1)/1 +1 = 112$


```python
relu2_1 = conv_layer(graph, pool1, "conv2_1")   
relu2_1
```




    <tf.Tensor 'conv2_1/conv2_1:0' shape=(?, 112, 112, 128) dtype=float32>

<br>

- ```Conv2_2``` output 계산 식: $(112-1)/1 +1 = 112$


```python
relu2_2 = conv_layer(graph, relu2_1, "conv2_2")
relu2_2
```




    <tf.Tensor 'conv2_2/conv2_2:0' shape=(?, 112, 112, 128) dtype=float32>


<br>

- ```pool2``` output 계산 식: $(112-2)/2 +1 = 56$



```python
pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool2')
pool2
```




    <tf.Tensor 'pool2:0' shape=(?, 56, 56, 128) dtype=float32>

<br>


- ```Conv3_1``` output 계산 식: $(56-1)/1 +1 = 56$


```python
relu3_1 = conv_layer(graph, pool2, "conv3_1")
relu3_1
```




    <tf.Tensor 'conv3_1/conv3_1:0' shape=(?, 56, 56, 256) dtype=float32>

<br>

- ```Conv3_2``` output 계산 식: $(56-1)/1 +1 = 56$


```python
relu3_2 = conv_layer(graph, relu3_1, "conv3_2")
relu3_2
```




    <tf.Tensor 'conv3_2/conv3_2:0' shape=(?, 56, 56, 256) dtype=float32>

<br>

- ```Conv3_3``` output 계산 식: $(56-1)/1 +1 = 56$


```python
relu3_3 = conv_layer(graph, relu3_2, "conv3_3")
relu3_3
```




    <tf.Tensor 'conv3_3/conv3_3:0' shape=(?, 56, 56, 256) dtype=float32>


<br>

- ```pool3``` output 계산 식: $(56-2)/2 +1 = 28$


```python
pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME', name='pool3')
pool3
```




    <tf.Tensor 'pool3:0' shape=(?, 28, 28, 256) dtype=float32>


<br>

- ```Conv4_1``` output 계산 식: $(28-1)/1 +1 = 28$


```python
relu4_1 = conv_layer(graph, pool3, "conv4_1")
relu4_1
```




    <tf.Tensor 'conv4_1/conv4_1:0' shape=(?, 28, 28, 512) dtype=float32>




```python
relu4_2 = conv_layer(graph, relu4_1, "conv4_2")
relu4_2
```




    <tf.Tensor 'conv4_2/conv4_2:0' shape=(?, 28, 28, 512) dtype=float32>




```python
relu4_3 = conv_layer(graph, relu4_2, "conv4_3")
relu4_3
```




    <tf.Tensor 'conv4_3/conv4_3:0' shape=(?, 28, 28, 512) dtype=float32>


<br>

- ```pool4``` output 계산 식: $(28-2)/2 +1 = 14$

```python
pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME', name='pool4')
pool4
```




    <tf.Tensor 'pool4:0' shape=(?, 14, 14, 512) dtype=float32>


<br>

- ```Conv5_1``` output 계산 식: $(14-1)/1 +1 = 14$


```python
relu5_1 = conv_layer(graph, pool4, "conv5_1")
relu5_1
```




    <tf.Tensor 'conv5_1/conv5_1:0' shape=(?, 14, 14, 512) dtype=float32>

<br>

- ```Conv5_2``` output 계산 식: $(14-1)/1 +1 = 14$


```python
relu5_2 = conv_layer(graph, relu5_1, "conv5_2")
relu5_2
```




    <tf.Tensor 'conv5_2/conv5_2:0' shape=(?, 14, 14, 512) dtype=float32>

<br>

- ```Conv5_3``` output 계산 식: $(14-1)/1 +1 = 14$


```python
relu5_3 = conv_layer(graph, relu5_2, "conv5_3")
relu5_3
```




    <tf.Tensor 'conv5_3/conv5_3:0' shape=(?, 14, 14, 512) dtype=float32>

<br>

- ```pool5``` output 계산 식: $(14-2)/2 +1 = 7$

```python
pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                       padding='SAME', name='pool5')
pool5
```




    <tf.Tensor 'pool5:0' shape=(?, 7, 7, 512) dtype=float32>


<br>

- ```Fully connected Layer 1``` 

```python
with tf.variable_scope('fc1') as scope:                        

    w = pretrained_weights['fc6_W']
    b = pretrained_weights['fc6_b']

    fc_weights = tf.get_variable("W", shape=w.shape, 
    	initializer=tf.constant_initializer(w))
    fc_biases  = tf.get_variable("b", shape=b.shape, 
    	initializer=tf.constant_initializer(b))           

    # flatten dim 
    shape = int(np.prod(pool5.get_shape()[1:])) #25088 
    pool5_flat = tf.reshape(pool5, [-1, shape])

    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc_weights),
     fc_biases)

    with graph.gradient_override_map({'Relu': 'BackpropRelu'}):
        fc1 = tf.nn.relu(fc1l)

    fc1 = tf.nn.dropout(fc1, keep_prob = 1.0) # option
```


```python
fc1
```




    <tf.Tensor 'fc1/Relu:0' shape=(?, 4096) dtype=float32>

<br>

- ```Fully connected Layer 2``` 


```python
with tf.variable_scope('fc2') as scope: 

    w = pretrained_weights['fc7_W']
    b = pretrained_weights['fc7_b']

    fc_weights = tf.get_variable("W", shape=w.shape, 
    	initializer=tf.constant_initializer(w))
    fc_biases  = tf.get_variable("b", shape=b.shape, 
    	initializer=tf.constant_initializer(b))           

    fc2l = tf.nn.bias_add(tf.matmul(fc1, fc_weights), 
    	fc_biases)

    with graph.gradient_override_map({'Relu': 'BackpropRelu'}):
        fc2 = tf.nn.relu(fc2l)
    fc2 = tf.nn.dropout(fc2, keep_prob = 1.0)
```


```python
fc2
```




    <tf.Tensor 'fc2/Relu:0' shape=(?, 4096) dtype=float32>

<br>

- ```Fully connected Layer 3``` 


```python

with tf.variable_scope('fc3') as scope:

    w = pretrained_weights['fc8_W']
    b = pretrained_weights['fc8_b']

    fc_weights = tf.get_variable("W", shape=w.shape, 
    	initializer=tf.constant_initializer(w))
    fc_biases  = tf.get_variable("b", shape=b.shape, 
    	initializer=tf.constant_initializer(b))         

    output = tf.nn.bias_add(tf.matmul(fc2, fc_weights), 
    	fc_biases)
```


```python
output
```




    <tf.Tensor 'fc3/BiasAdd:0' shape=(?, 1000) dtype=float32>


<br>
---

**<span style='color:DarkRed'> Inference</span>**


- 다시 불러온 graph의 마지막 Conv와 최종 output를 가져옴


```python
last_conv_layer = graph.get_tensor_by_name('conv5_3/conv5_3:0')
last_conv_layer
```




    <tf.Tensor 'conv5_3/conv5_3:0' shape=(?, 14, 14, 512) dtype=float32>


<br>

```python
output = graph.get_tensor_by_name('fc3/BiasAdd:0')
output
```




    <tf.Tensor 'fc3/BiasAdd:0' shape=(?, 1000) dtype=float32>


<br>

- 가장 확률이 높은 class와 연결된 gradients (w.r.t last conv feature maps)를 계산

$$\text{gradients}= \frac{\partial y_{label_i}}{\partial \text{ last conv layer}} (\text{relu output} > 0)$$


```python
gradient = tf.gradients(output[:,tf.squeeze(labels_tf,-1)], last_conv_layer)[0]

gradient
```




    <tf.Tensor 'gradients/pool5_grad/MaxPoolGrad:0' shape=(?, 14, 14, 512) dtype=float32>



$$ norm_i = \frac{grad_i}{\sqrt{\frac{1}{n}\sum grad_i^2}} $$


```python
norm_grads = tf.div(gradient, tf.sqrt(tf.reduce_mean(tf.square(gradient))) + tf.constant(1e-5))
norm_grads
```




    <tf.Tensor 'div:0' shape=(?, 14, 14, 512) dtype=float32>


<br>

```python
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
```


```python
logits_classes = sess.run(output, 
                          feed_dict={images_tf: np.expand_dims(img, axis = 0)}
                         )
```


```python
logits_classes.shape
```




    (1, 1000)


<br>

- Remove redundant axis 

```python
pred = np.squeeze(logits_classes, axis=0)
pred.shape
```




    (1000,)

<br>

- 작은값부터 큰값 순서를 나타내는 index
- $[::-1] \rightarrow $ 큰값부터 작은값으로 바꾸는 인덱스 표현
- 확률(logit)이 가장 큰 5개의 class 선택


```python
pred = (np.argsort(pred)[::-1])[0:6]
pred
```




    array([258, 279, 222, 257, 270, 250])




```python
label_1 = pred[0]
label_1
```




    258




```python
label_2 = pred[5]
label_2
```




    250


<br>

---

**<span style='color:DarkRed'> Gradient Class Activation Maps</span>**


```python
fmaps = last_conv_layer
gradients= norm_grads 
height = 224 # upsampled height
width = 224 # upsampled width
num_fmaps = 512 # number of feature map for last conv
```

<br>

```python
gradients.shape.as_list()
```




    [None, 14, 14, 512]

<br>

- **<span style='color:blue'>각 채널 별로 평균(global average pooling)
</span>**

```python
weights = tf.reduce_mean(gradients, axis=(1,2))
weights.shape.as_list()
```




    [None, 512]


<br>

- Resize bilinear

	 - [None, 14, 14, 512] -> [None, 224, 224, 512]


```python
fmaps_resized = tf.image.resize_bilinear(fmaps, [height, width] )
fmaps_resized
```




    <tf.Tensor 'ResizeBilinear:0' shape=(?, 224, 224, 512) dtype=float32>

<br>

- 4D tensor $\rightarrow$ 3D tensor

```python
fmaps_reshaped = tf.reshape(fmaps_resized, [-1, height*width, num_fmaps]) 
fmaps_reshaped
```




    <tf.Tensor 'Reshape:0' shape=(?, 50176, 512) dtype=float32>

<br>

- 2D tensor $\rightarrow$ 3D tensor


```python
label_w = tf.reshape( weights, [-1, num_fmaps, 1])
label_w
```




    <tf.Tensor 'Reshape_1:0' shape=(?, 512, 1) dtype=float32>


<br>

- Batch multiplication
	- **<span style='color:blue'>Last feature maps $\times$ gradients GAP for a class </span>**

```python
classmap = tf.matmul(fmaps_reshaped, label_w )
classmap
```




    <tf.Tensor 'MatMul:0' shape=(?, 50176, 1) dtype=float32>


<br>

- Image size 원복

```python
classmap = tf.reshape( classmap, [-1, height, width] )
classmap
```




    <tf.Tensor 'Reshape_2:0' shape=(?, 224, 224) dtype=float32>


<br>


- Class map 산출

```python
class_map1 = sess.run(classmap, feed_dict={ images_tf: np.expand_dims(img, axis = 0),labels_tf: [label_1]})
class_map2 = sess.run(classmap, feed_dict={ images_tf: np.expand_dims(img, axis = 0),labels_tf: [label_2]})

print(class_map1.shape)
print(class_map2.shape)

```

    (1, 224, 224)
    (1, 224, 224)

<br>

```python
class_map1 = np.squeeze(class_map1, axis= 0)
class_map2 = np.squeeze(class_map2, axis= 0)

print(class_map1.shape)
print(class_map2.shape)
```

    (224, 224)
    (224, 224)

<br>


```python
class_map1
```




    array([[ 46.46514 ,  44.421066,  42.37698 , ..., -79.55383 , -79.55383 ,
            -79.55383 ],
           [ 44.747826,  42.729706,  40.711582, ..., -73.02732 , -73.02732 ,
            -73.02732 ],
           [ 43.030506,  41.038338,  39.046173, ..., -66.50081 , -66.50081 ,
            -66.50081 ],
           ...,
           [ 53.97239 ,  49.972347,  45.972294, ..., 137.87003 , 137.87003 ,
            137.87003 ],
           [ 53.97239 ,  49.972347,  45.972294, ..., 137.87003 , 137.87003 ,
            137.87003 ],
           [ 53.97239 ,  49.972347,  45.972294, ..., 137.87003 , 137.87003 ,
            137.87003 ]], dtype=float32)


<br>


```python
class_map2
```




    array([[ 40.845783,  39.666195,  38.486614, ..., -19.882212, -19.882212,
            -19.882212],
           [ 40.23872 ,  39.053905,  37.86909 , ..., -20.355597, -20.355597,
            -20.355597],
           [ 39.631653,  38.441616,  37.25157 , ..., -20.82899 , -20.82899 ,
            -20.82899 ],
           ...,
           [ 55.261864,  51.680935,  48.100006, ...,  81.27644 ,  81.27644 ,
             81.27644 ],
           [ 55.261864,  51.680935,  48.100006, ...,  81.27644 ,  81.27644 ,
             81.27644 ],
           [ 55.261864,  51.680935,  48.100006, ...,  81.27644 ,  81.27644 ,
             81.27644 ]], dtype=float32)


<br>

---
**<span style='color:DarkRed'> Visualize</span>**



```python
def normalize(img):
    """Normalize the image range for visualization"""
    return np.uint8((img - img.min()) / (img.max()-img.min())*255)
```


```python
fig, axs = plt.subplots(1,2, figsize=(10,10))
axs[0].imshow(img)
axs[0].imshow(normalize(class_map1), cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
axs[0].set_title('1st class: %s' %class_names[label_1])
axs[0].axis('off')

axs[1].imshow(img)
axs[1].imshow(normalize(class_map2), cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
axs[1].set_title('2nd class: %s' %class_names[label_2])
axs[1].axis('off')
```




    (-0.5, 223.5, 223.5, -0.5)


<p align="center"><img width="600" height="auto" src="https://i.imgur.com/OALfg16.png"></p>




- Contour를 찾아 가장 넓은 면적을 가지는 부분에 대하서 Bounding Box를 취함

```python
heatmap = class_map1
threshold = 0.3
```


```python
# Binarize the heatmap
_, thresholded_heatmap = cv2.threshold(heatmap, threshold * heatmap.max(), 1, cv2.THRESH_BINARY)
plt.imshow(thresholded_heatmap)
```




    <matplotlib.image.AxesImage at 0x158bfab00>



<p align="center"><img width="300" height="auto" src="https://i.imgur.com/hKjfYzO.png"></p>


```python
# Required for converting image to uint8
print('Before:',thresholded_heatmap.dtype)
thresholded_heatmap = cv2.convertScaleAbs(thresholded_heatmap)
print('After:',thresholded_heatmap.dtype)
```

    Before: float32
    After: uint8

<br>

```python
contours, _ = cv2.findContours(thresholded_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('number of contours:',len(contours))
```

    number of contours: 2

<br>

- Contours 면적 계산


```python
contour_areas = []   
for i, c in enumerate(contours):
    contour_areas.append(cv2.contourArea(c))
```


```python
# contour 면적이 큰 순서대로 정렬
sorted_contours = sorted(zip(contour_areas, contours), key=lambda x:x[0], reverse=True)
```


```python
# # contour 면적이 큰 contours 선택
biggest_contour= sorted_contours[0][1]
```


```python
# -1 : represent entire contours
# (255, 255, 255): color
# 3 : thinkness 
contour_image = cv2.drawContours(img.copy(), biggest_contour, -1, (255, 255, 255), 3)
plt.imshow(contour_image)
```




    <matplotlib.image.AxesImage at 0x15c3bf5c0>



<p align="center"><img width="300" height="auto" src="https://i.imgur.com/2LSW4FR.png"></p>


```python
x,y,w,h = cv2.boundingRect(biggest_contour)
x,y,w,h
```




    (88, 24, 136, 200)


<br>


```python
box_image = cv2.rectangle(img.copy(), (x,y), (x+w, y+h), (0, 255,0), 2)
```


```python
plt.imshow(box_image)
plt.axis('off')
```




    (-0.5, 223.5, 223.5, -0.5)



<p align="center"><img width="300" height="auto" src="https://i.imgur.com/B6WMFYG.png"></p>

---

**<span style='color:DarkRed'>Saliency map</span>**

- 마지막 Conv가 아닌 input image에 대해서 gradients를 계산

$$\frac{\partial y_{label_i}}{\partial \text{ input $\textbf{x}$}}$$

```python
gradient_bp = tf.gradients(output[:,tf.squeeze(labels_tf,-1)], images_tf)[0]
gradient_bp
```




    <tf.Tensor 'gradients_1/split_grad/concat:0' shape=(?, 224, 224, 3) dtype=float32>

<br>

- gradient normalize


```python
norm_grads_bp = tf.div(gradient_bp, tf.sqrt(tf.reduce_mean(tf.square(gradient))) + tf.constant(1e-5))
norm_grads_bp
```




    <tf.Tensor 'div_1:0' shape=(?, 224, 224, 3) dtype=float32>



<br>

```python
# Gradients computation
grads_weights1 = sess.run(norm_grads_bp, feed_dict={images_tf: np.expand_dims(img, axis = 0),
                                               labels_tf: [label_1]})
grads_weights2 = sess.run(norm_grads_bp, feed_dict={images_tf: np.expand_dims(img, axis = 0),
                                               labels_tf: [label_2]})

```


```python
grads_weights1 = np.squeeze(grads_weights1)
grads_weights2 = np.squeeze(grads_weights2)
```
<br>

- MinMax uint8 normalize


```python
fig, axs = plt.subplots(1,2, figsize=(10,10))
axs[0].imshow(normalize(grads_weights1), cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
axs[0].set_title('1st class: %s' %class_names[label_1])
axs[0].axis('off')

axs[1].imshow(normalize(grads_weights2), cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
axs[1].set_title('2nd class: %s' %class_names[label_2])
axs[1].axis('off')
```




    (-0.5, 223.5, 223.5, -0.5)


<p align="center"><img width="600" height="auto" src="https://i.imgur.com/Ysto3Nq.png"></p>

- Saliency map에 사용되는 변수

    - gradients w.r.t ```image```
    - gradients w.r.t ```last convs```


```python
def saliency_normalize(img):
    """Normalize the image range for visualization"""
    return (img - img.min()) / (img.max()-img.min())
```


```python
# range 0 ~ 1
saliency_normalize(class_map1)
```




    array([[0.18836465, 0.18551339, 0.18266214, ..., 0.01258321, 0.01258321,
            0.01258321],
           [0.1859692 , 0.18315415, 0.1803391 , ..., 0.02168692, 0.02168692,
            0.02168692],
           [0.18357372, 0.18079488, 0.17801607, ..., 0.03079062, 0.03079062,
            0.03079062],
           ...,
           [0.19883634, 0.19325678, 0.18767717, ..., 0.31586355, 0.31586355,
            0.31586355],
           [0.19883634, 0.19325678, 0.18767717, ..., 0.31586355, 0.31586355,
            0.31586355],
           [0.19883634, 0.19325678, 0.18767717, ..., 0.31586355, 0.31586355,
            0.31586355]], dtype=float32)


<br>

- $0 \sim 1$사이로 정규화 시킨 class activation map($\frac{\partial \textbf{y}}{\partial \text{last conv}}$)을 grad-RGB($\frac{\partial \textbf{y}}{\partial \textbf{x}}$)에 채널별로 각각 곱해줌



```python
gradBGR1 = normalize(grads_weights1)
# VGG16 use BGR internally, so we manually change BGR to RGB
gradRGB_cam1 = np.dstack((
    normalize(gradBGR1[:, :, 2]* saliency_normalize(class_map1)),
    normalize(gradBGR1[:, :, 1]* saliency_normalize(class_map1)),
    normalize(gradBGR1[:, :, 0]* saliency_normalize(class_map1))
))

gradBGR2 = normalize(grads_weights2)
# VGG16 use BGR internally, so we manually change BGR to RGB
gradRGB_cam2 = np.dstack((
    normalize(gradBGR2[:, :, 2]* saliency_normalize(class_map2)),
    normalize(gradBGR2[:, :, 1]* saliency_normalize(class_map2)),
    normalize(gradBGR2[:, :, 0]* saliency_normalize(class_map2))
))
```


```python
fig, axs = plt.subplots(1,2, figsize=(10,10))
axs[0].imshow(gradRGB_cam1, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
axs[0].set_title('1st class: %s' %class_names[label_1])
axs[0].axis('off')

axs[1].imshow(gradRGB_cam2, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
axs[1].set_title('2nd class: %s' %class_names[label_2])
axs[1].axis('off')
```




    (-0.5, 223.5, 223.5, -0.5)




<p align="center"><img width="600" height="auto" src="https://i.imgur.com/KEUNBOm.png"></p>


---

**<span style='color:DarkRed'>Fine-tunning</span>** 

```python
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels= labels_tf ), name='loss_tf')
loss_tf
```




    <tf.Tensor 'loss_tf:0' shape=() dtype=float32>


<br>

```python
tf.trainable_variables() 
```




    [<tf.Variable 'conv1_1/W:0' shape=(3, 3, 3, 64) dtype=float32_ref>,
     <tf.Variable 'conv1_1/b:0' shape=(64,) dtype=float32_ref>,
     <tf.Variable 'conv1_2/W:0' shape=(3, 3, 64, 64) dtype=float32_ref>,
     <tf.Variable 'conv1_2/b:0' shape=(64,) dtype=float32_ref>,
     <tf.Variable 'conv2_1/W:0' shape=(3, 3, 64, 128) dtype=float32_ref>,
     <tf.Variable 'conv2_1/b:0' shape=(128,) dtype=float32_ref>,
     <tf.Variable 'conv2_2/W:0' shape=(3, 3, 128, 128) dtype=float32_ref>,
     <tf.Variable 'conv2_2/b:0' shape=(128,) dtype=float32_ref>,
     <tf.Variable 'conv3_1/W:0' shape=(3, 3, 128, 256) dtype=float32_ref>,
     <tf.Variable 'conv3_1/b:0' shape=(256,) dtype=float32_ref>,
     <tf.Variable 'conv3_2/W:0' shape=(3, 3, 256, 256) dtype=float32_ref>,
     <tf.Variable 'conv3_2/b:0' shape=(256,) dtype=float32_ref>,
     <tf.Variable 'conv3_3/W:0' shape=(3, 3, 256, 256) dtype=float32_ref>,
     <tf.Variable 'conv3_3/b:0' shape=(256,) dtype=float32_ref>,
     <tf.Variable 'conv4_1/W:0' shape=(3, 3, 256, 512) dtype=float32_ref>,
     <tf.Variable 'conv4_1/b:0' shape=(512,) dtype=float32_ref>,
     <tf.Variable 'conv4_2/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv4_2/b:0' shape=(512,) dtype=float32_ref>,
     <tf.Variable 'conv4_3/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv4_3/b:0' shape=(512,) dtype=float32_ref>,
     <tf.Variable 'conv5_1/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv5_1/b:0' shape=(512,) dtype=float32_ref>,
     <tf.Variable 'conv5_2/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv5_2/b:0' shape=(512,) dtype=float32_ref>,
     <tf.Variable 'conv5_3/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv5_3/b:0' shape=(512,) dtype=float32_ref>,
     <tf.Variable 'fc1/W:0' shape=(25088, 4096) dtype=float32_ref>,
     <tf.Variable 'fc1/b:0' shape=(4096,) dtype=float32_ref>,
     <tf.Variable 'fc2/W:0' shape=(4096, 4096) dtype=float32_ref>,
     <tf.Variable 'fc2/b:0' shape=(4096,) dtype=float32_ref>,
     <tf.Variable 'fc3/W:0' shape=(4096, 1000) dtype=float32_ref>,
     <tf.Variable 'fc3/b:0' shape=(1000,) dtype=float32_ref>]

<br>

- filter의 결과값은 한번 출력되면 사라지는 것을 주의


```python
weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
list(weights_only)
```




    [<tf.Variable 'conv1_1/W:0' shape=(3, 3, 3, 64) dtype=float32_ref>,
     <tf.Variable 'conv1_2/W:0' shape=(3, 3, 64, 64) dtype=float32_ref>,
     <tf.Variable 'conv2_1/W:0' shape=(3, 3, 64, 128) dtype=float32_ref>,
     <tf.Variable 'conv2_2/W:0' shape=(3, 3, 128, 128) dtype=float32_ref>,
     <tf.Variable 'conv3_1/W:0' shape=(3, 3, 128, 256) dtype=float32_ref>,
     <tf.Variable 'conv3_2/W:0' shape=(3, 3, 256, 256) dtype=float32_ref>,
     <tf.Variable 'conv3_3/W:0' shape=(3, 3, 256, 256) dtype=float32_ref>,
     <tf.Variable 'conv4_1/W:0' shape=(3, 3, 256, 512) dtype=float32_ref>,
     <tf.Variable 'conv4_2/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv4_3/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv5_1/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv5_2/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'conv5_3/W:0' shape=(3, 3, 512, 512) dtype=float32_ref>,
     <tf.Variable 'fc1/W:0' shape=(25088, 4096) dtype=float32_ref>,
     <tf.Variable 'fc2/W:0' shape=(4096, 4096) dtype=float32_ref>,
     <tf.Variable 'fc3/W:0' shape=(4096, 1000) dtype=float32_ref>]



<br>

```python
weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
[tf.nn.l2_loss(x) for x in list(weights_only)]
```




    [<tf.Tensor 'L2Loss:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_1:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_2:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_3:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_4:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_5:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_6:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_7:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_8:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_9:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_10:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_11:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_12:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_13:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_14:0' shape=() dtype=float32>,
     <tf.Tensor 'L2Loss_15:0' shape=() dtype=float32>]


<br>

```python
weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * 0.0005 # decay rate
weight_decay
```




    <tf.Tensor 'mul:0' shape=() dtype=float32>


<br>

- 아래의 loss를 손실함수로 두고 학습하면 된다.

```python
loss_tf += weight_decay
```

<br>

- 정확도에 대한 op는 아래와 같이 정의 할 수 있다.


```python
tf.argmax(output, 1)
```




    <tf.Tensor 'ArgMax:0' shape=(?,) dtype=int64>

<br>

- argmax의 output 자료형 또한 변환시킬 수 있다.


```python
correct_pred = tf.equal(tf.argmax(output, 1, output_type=tf.int32), labels_tf)
correct_pred
```




    <tf.Tensor 'Equal:0' shape=(?,) dtype=bool>

<br>


```python
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy
```




    <tf.Tensor 'Mean_3:0' shape=() dtype=float32>


---

**<span style='color:DarkRed'>Reference</span>** 

https://github.com/waleedgondal/weakly_supervised_localizations_tf

https://github.com/insikk/Grad-CAM-tensorflow/blob/master/utils.py


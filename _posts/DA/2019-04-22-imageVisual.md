---
layout: post
comments: true
title: Feature Map과 Image를 활용한 텐서보드 시각화-(3)
categories: Domain Adaptation

tags:
- Domain Adaptation
---

**<span style='color:DarkRed'>시각화</span>**

이 글은 텐서보드를 활용해, 임베딩 벡터를 ```t-SNE``` 또는 ```PCA```로 시각화하고, 그 위치에 해당 이미지를 덮어 씌우는 시각화 방법입니다.

---

```python
import tensorflow as tf
import os
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import csv
import math
from PIL import Image
```

<br>

```python
LOG_DIR = os.path.join(os.getcwd(),'TFboard/projection')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR) # makedirs can make it with subdir
```

<br>

```python
imagesPath = "./results/SourceTargetImages.npy"
embedVectorPath = "./results/SourceTargetEmbed.npy"
domainsPath = "./results/SourceTargetDomain.npy"
labelsPath = "./results/SourceTargetLabels.npy"
```

<br>

```python
images = np.load(imagesPath)
# N examples, dimension
embedVector = np.load(embedVectorPath)
Domains = np.load(domainsPath)
Labels = np.load(labelsPath)
```

<br>

---

- label를 시각화하기 위한 meta파일을 생성


```python
temp = []
for domain, label in zip(Domains, Labels):
    if domain[0] == 1:
        temp.append(['Source', np.argmax(label)] )
    else:
        temp.append(['Target', np.argmax(label)] )    
```


```python
print(temp[0:5])
print(temp[500:505])
```

    [['Source', 7], ['Source', 2], ['Source', 1], ['Source', 0], ['Source', 4]]
    [['Target', 7], ['Target', 2], ['Target', 1], ['Target', 0], ['Target', 4]]

<br>

```python
## make metafiles
with open(os.path.join(LOG_DIR,'metadata.tsv'), 'wt', encoding='utf-8') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['domain', 'label'])
    for ix in range(len(temp)):
        tsv_writer.writerow(temp[ix])
```

<br>

- image를 시각화하기 위한 sprite이미지 생성


```python
imageShape = np.array(images.shape[1:3])
imageShape
```




    array([28, 28])


<br>


```python
# the bigger
zoom = 6
visualSize = 60
```

<br>

- 이미지를 한번 확장 시키고 다시 줄임, 더 명확한 형태를 뽑아낼 수 있음


```python
# resizing
IMAGE=[]
for i in range(len(images)):#i=1
    imageExpansion = np.kron(images[i:i+1,], np.ones((1, zoom, zoom, 1))) # float type,  (1, 168, 168, 3)
    img = np.squeeze(imageExpansion, axis=[1,2,3]) # (168, 168, 3)
    if i < 500: #RGB
        img = Image.fromarray(img.astype('uint8'))
    else: # grey scale
        img = Image.fromarray(img.astype('uint8')*255)

    img = img.resize((visualSize,visualSize), Image.ANTIALIAS) # shrinking
    img = np.array(img) # as array
    IMAGE.append(img)
```

<br>

```python
IMAGE = np.array(IMAGE)
```

<br>

```python
def images_to_sprite(data, N):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """

    # if not exist channel
    if len(data.shape) == 3:
        # duplicated as 3 channel
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    # normalize
    data = data.astype(np.float32)

    # reshape as (n,-1)
    _min = data.min(axis=(0,1,2),keepdims=True)
    _max = data.max(axis=(0,1,2), keepdims=True)

    if np.max(_max) !=0:
        data = (data- _min) / (_max-_min)

    # columns of grid = sqrt(N) = n
    # 사각형틀에 이미지를 채우기 위해 아래와 같이 작성
    n = math.ceil(np.sqrt(N))
    
    # ((N), (width+more), (height), (channel))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    # (n, n, width, height, channel).transpose((0, 2, 1, 3, 4))
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    
    # (1920, 1920, 3)concat된 정방형 이미지를 생성
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # norm -> actual pixels
    data = (data * 255).astype(np.uint8)
    return data

```

<br>

- sprite 이미지 저장


```python
sprite = images_to_sprite(IMAGE,len(IMAGE))
IMAGE = Image.fromarray(sprite, mode='RGB')
IMAGE.save(os.path.join(LOG_DIR, 'sprite.png'))
```

<p align="center"><img width="700" height="auto" src='https://i.imgur.com/oBqWZvF.png'></p>

<br>

---

- ```metadata.tsv```와 ```sprite.png```가 있으면 이제 텐서보드로 시각화 할 수 있다.


```python
metadata_file = os.path.join(LOG_DIR, 'metadata.tsv')
sprite_image_path = os.path.join(LOG_DIR, "sprite.png")
```

<br>


```python
## TensorFlow Variable from data
embeddingVector = tf.Variable(embedVector, name='DA_features')
```
<br>

```python
## Running TensorFlow Session
with tf.Session() as sess:
    saver = tf.train.Saver([embeddingVector])
    sess.run(embeddingVector.initializer)

    # './project-tensorboard/tf_data.ckpt'
    saver.save(sess, LOG_DIR+'tf_data.ckpt')

    # adding into projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddingVector.name
    # Link this tensor to its metadata(Labels) file
    embedding.metadata_path = metadata_file
    embedding.sprite.image_path = sprite_image_path
    embedding.sprite.single_image_dim.extend([int(visualSize), int(visualSize)])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
```

<br>

- ```Jupyter Notebook```에서 ```!```를 사용하면 터미널처럼 사용할 수 있다.

```python
! tensorboard --logdir="./TFboard/projection/" --host 888.888.888.888
```

    TensorBoard 1.12.2 at http://888.888.888.888:8888 (Press CTRL+C to quit)

<br>

- 실행 동영상 

<p align="center"><video width="700" height="auto" controls="controls">
  <source src="https://drive.google.com/uc?export=download&id=1yQ8Ms0nZFYBq3cLBYU4e0-mnpJAZBBFv" type="video/mp4" />
  Your browser does not support the video tag.
  /* instead of the last line you could also add the flash player*/
</video></p>

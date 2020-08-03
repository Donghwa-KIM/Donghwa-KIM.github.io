---
layout: post
comments: true
title: Performance Reproduction (seed, virtualenv, requirement.txt)
categories: ssh
tags:
- etc.
---



> 모델의 성능을 복원하기 위해서는 ```python```, ```cudnn```, ```cuda```에 대한 seed 및 패키지 버전관리가 필요합니다. 추가적으로 jupyter lab으로 테스트하는 예제를 보여드리겠습니다.

 - Seed 설정
 - Package 저장
 - 새로운 가상환경 생성
 - Package 복원

<br>

**<span style='color:DarkRed'>Seed 설정</span>**

- 딥러닝 패키지에 따라 seed 부여 코드가 달라집니다.
	- 일반적으로 ```python```, ```cuda```, ```cudnn```에 seed에 부여됩니다.

	- tensorflow==2.0.2
 
		```python
		from tfdeterminism import patch # to be installed
		import random 

		patch()

		# Seed value (can actually be different for each attribution step)
		SEED= 7

		os.environ['PYTHONHASHSEED']=str(SEED)
		random.seed(SEED)
		np.random.seed(SEED)
		tf.random.set_seed(SEED)
		os.environ['HOROVOD_FUSION_THRESHOLD']='0'
		physical_devices = tf.config.experimental.list_physical_devices('GPU')
		for device in physical_devices:
		    tf.config.experimental.set_memory_growth(device, True)	
        ```
    
    - Pytorch

		```python
		import random
		import numpy as np
		import torch
		def set_seed(SEED):
		    random.seed(SEED)
		    np.random.seed(SEED)
		    torch.manual_seed(SEED)
		    torch.backends.cudnn.deterministic = True
		    torch.backends.cudnn.benchmark = False
		    if args.n_gpu > 0:
		        torch.cuda.manual_seed_all(SEED)
		set_seed(7)
		```

<br>

**<span style='color:DarkRed'>사용된 Package 저장</span>**
- 모델이 학습된 가상환경(e.g. `py-env`)을 activate한 후 아래에 코드 입력
- 현재경로로 `requirements.txt`가 생성된 것 을 확인할 수 있습니다. 나중에 이 패키지들을 새로운 가상환경에 설치를 하게 됩니다.  
```
(py-env) pip freeze > requirements.txt
```

<br>

**<span style='color:DarkRed'>새로운 가상환경 생성</span>**

- Virtual env 생성
	- ```env_name```에 원하는 가상환경 이름을 입력

	```python
	conda create -n 'env_name' python=3.6
	```
<br>

- 예시) 가상환경이름을 ```lab```이라고 하면 아래와 같음

	```python
	conda create -n lab python=3.6
	```
<br>

- 실수로 잘못만들었다면, 아래와 같이 삭제할수 있습니다.

	```python
	conda remove --name lab --all
	```

<br>


**<span style='color:DarkRed'>Package 복원</span>**

- 새롭게 생성된 가상환경(e.g. `lab`)에 아래의 패키지들을 설치해줍니다.

	```
	(lab) pip install -r requirements.txt
	```

<br>

**<span style='color:DarkRed'>Test in juypter</span>**

- 먼저 `jupyter` 패키지들을 설치해줍니다.

	```
	conda install ipykernel
	```

<br>

- `jupyter`에 사용할 커널을 설치해줍니다.
	- `--name`에는 사용할 가상환경이름을 입력해 줍니다.
	- `--display-name`에는 jupyter에 표시되는 이름을 설정해줍니다.(본 예제에서는 가상환경이름과 동일한 이름으로 작성하였습니다.)

	```
	python -m ipykernel install --user --name lab --display-name "lab"
	```

<br>
**<span style='color:DarkRed'>Tips</span>**

> 오래사용된 가상환경(패키지가 이것저것 깔려 있는)에 모델를 학습하고 새로운 가상환경을 테스트하는데 생각보다 어려움이 있을 수 있습니다. 의도하지 않는 dependency가 엉켜 있어 복원이 잘 안되는 경험도 해봤기 때문입니다. 가장 좋은 작업환경은 새로운 task를 수행할때 새로운 가상환경을 만들어서 학습 및 기초 환경을 build하는 것이, 테스트할때 패키지들을 많이 설치할 필요도 없고 정확한 reproduction이 가능한 방법인것 같습니다.
---
layout: post
comments: true
title: FreeSurfer를 이용한 Hippocampal Segmentation 방법
categories: Medical Analysis

tags:
- Medical Analysis
---

**<span style='color:DarkRed'>FreeSurfer</span>**

> FreeSurfer는 cortical 및 subcortical anatomy를 위한 소프트웨어 도구이다. 이 도구는 pial surface를 생성해 내고, white matter와 gray matter 사이의 boundary, 즉 white surface의 모델을 구축하는 데에 사용된다. 기본적으로는 T1 이미지를 사용하지만 pial surface들을 좀 더 잘 구분해 내기 위해서 T2나 flair 이미지들을 사용할 수도 있다. (https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all#Fully-AutomatedDirective )
Grey matter는 cortex에도 있고 subcortex에도 있는데 cortex에서 구분해 내는 작업(cortical extraction and labelling)은 cortical parcellation (surface-based, aparc), subcortex에서 구분해 내는 작업은 subcortical segmentation (volume-based, aseg)이라고 부른다.

<p align="center"><img width="600" height="auto" src="https://i.imgur.com/gwE5Qt3.png"></p>


> 본격적으로 우분투 운영체제를 이용하여 Segmetation하는 방법을 살펴보겠다. http://freesurfer.net/fswiki/DownloadAndInstall 에 접속하여 ```freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz``` 을 다운 받는다. 가이드라인은 CentOS로 설명되어 있는데 몇가지 수정만 하면 우분투에서도 쉽게 사용할 수 있다.


<br>

**<span style='color:DarkRed'>FreeSurfer 설치</span>**

 - 다운로드 파일을 ```/usr/local```에 이동시켜 압축을 해제
```bash
~$ cd Downloads
~/Downloads$ sudo tar -C /usr/local -xzvf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz
```

<br>

- 다운받은 Setup.sh을 실행
```bash
~$ export FREESURFER_HOME=/usr/local/freesurfer
~$ source $FREESURFER_HOME/SetUpFreeSurfer.sh
-------- freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0-2beb96c --------
Setting up environment for FreeSurfer/FS-FAST (and FSL)
FREESURFER_HOME   /usr/local/freesurfer
FSFAST_HOME       /usr/local/freesurfer/fsfast록
FSF_OUTPUT_FORMAT nii.gz
SUBJECTS_DIR      /usr/local/freesurfer/subjects
INFO: /home/donghwa/matlab/startup.m does not exist ... creating
MNI_DIR           /usr/local/freesurfer/mni
```
<br>

>Hippocampal을 추출하기 위해서는 관련 패키지를 더 받아야 한다. ```curl```을 이용해서 FREESURFER_HOME ```/usr/local/freesurfer```에 ```runtime2012b.tar.gz```을 받아 그 경로에 압축을 푼다.

```bash
~$ cd $FREESURFER_HOME
/usr/local/freesurfer$ sudo apt-get install curl
/usr/local/freesurfer$ sudo curl "https://surfer.nmr.mgh.harvard.edu/fswiki/MatlabRuntime?action=AttachFile&do=get&target=runtime2012bLinux.tar.gz" -o "runtime2012b.tar.gz"

/usr/local/freesurfer$ sudo tar xvf runtime2012b.tar.gz
/usr/local/freesurfer$ sudo rm $FREESURFER_HOME/runtime2012b.tar.gz
```
- 다시 Source 해주면 INFO에 명시된 matlab 에러를 해결

```bash
source $FREESURFER_HOME/SetUpFreeSurfer.sh
-------- freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0-2beb96c --------
Setting up environment for FreeSurfer/FS-FAST (and FSL)
FREESURFER_HOME   /usr/local/freesurfer
FSFAST_HOME       /usr/local/freesurfer/fsfast
FSF_OUTPUT_FORMAT nii.gz
SUBJECTS_DIR      /usr/local/freesurfer/subjects
MNI_DIR           /usr/local/freesurfer/mni
```
<br>

**<span style='color:DarkRed'>라이센스 등록</span>**

- recon 함수를 실행하기 위해서는 라이센스 등록해야만 함

- https://surfer.nmr.mgh.harvard.edu/registration.html에서 기본정보들을 등록하여 메일로 license.txt를 다운받는다.

```bash
cd ~/Downloads
sudo cp -r license.txt /usr/local/freesurfer/
```
<br>

**<span style='color:DarkRed'>분석할 데이터 경로 설정</span>**

```bash
export SUBJECTS_DIR=/home/donghwa/Documents/Brown_26001_1/scans/anat_1/NIfTI
```
- SUBJECTS_DIR가 ```/usr/local/freesurfer/subjects```에서 ```/home/donghwa/Documents/Brown_26001_1/scans/anat_1/NIfTI```로 Path가 변경된 걸 확인할 수 있음


```bash
~$ cd SUBJECTS_DIR
~/Documents/Brown_26001_1/scans/anat_1/NIfTI$ 
```


<br>

**<span style='color:DarkRed'>recon-all 실행</span>**
- ```/bin/tcsh: bad interpreter ```문제해결을 위한 tcsh를 설치

```bash
~/Documents/Brown_26001_1/scans/anat_1/NIfTI$ sudo apt-get install tcsh
```
```
recon-all -i `분석할 파일.nii` -s `저장할 폴더` -all -hippocampal-subfields-T1

# e.g 
recon-all -i rest.nii -s rest -all -hippocampal-subfields-T1
```
<br>

**<span style='color:DarkRed'>시각화</span>**
- 시각화 패키지를 사용하기 위해 아래와 같은 jpeg관련 패키지도 설치
```bash
~/Documents/Brown_26001_1/scans/anat_1/NIfTI$ sudo apt-get install libjpeg62
```

- freeview -v 에 이미지를 덧붙이는 방법으로 해당 그림파일들의 경로를 추가
```bash
~/Documents/Brown_26001_1/scans/anat_1/NIfTI$ freeview -v `저장된 폴더이름`/mri/T1.mgz `저장된 폴더이름`/mri/lh.hippoSfLabels-T1.v10.mgz:colormap=lut `저장된 폴더이름`/mri/rh.hippoSfLabels-T1.v10.mgz:colormap=lut `저장된 폴더이름`/mri/brainmask.mgz -f `저장된 폴더이름`/surf/lh.white:edgecolor=blue `저장된 폴더이름`/surf/lh.pial:edgecolor=red `저장된 폴더이름`/surf/rh.white:edgecolor=blue `저장된 폴더이름`/surf/rh.pial:edgecolor=red
```
<br>

- pial(빨간색)와 white(파란색)를 구분

<p align="center"><img width="600" height="auto" src="https://i.imgur.com/iGtTMw0.png"></p>

<br>

- Hippocampal Segmentation

<p align="center"><img width="600" height="auto" src="https://i.imgur.com/CrZy29K.png"></p>

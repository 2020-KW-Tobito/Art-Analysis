# AI for ArtTherapy
## 미술치료를 위한 그림 분석 인공지능 프로그램

2020학년도 2학기 광운대학교 참빛설계 프로젝트 

# Sketch captioning model 

![captioningmodel](https://user-images.githubusercontent.com/50594187/102038227-105c1f80-3e0a-11eb-811f-1d3b7d683d61.png)

Figure 1. Sketch image captioning model

# Pre processing
## 1. Stroke Color gradient
## 2. Connected Component Labeling Alogorithm 
## 3. Distance transform

![preprocessing](https://user-images.githubusercontent.com/50594187/102038456-a7c17280-3e0a-11eb-9a1a-d77c47588b51.PNG)

Figure 2. (a) original image, (b) stroke color gradient image, (c) Connected component algorithm image, (d) distance transformed image

# Sketch captioning model BLEU Experiment Result

![captioningresult](https://user-images.githubusercontent.com/50594187/102038071-acd1f200-3e09-11eb-93e9-303cec8e813b.png)

Table 1. Compare Preprocessing algorithm : BLEU–N scores on TU-Berlin captioning dataset (N=1,2,3,4)

# Demo program 

![demo구성도](https://user-images.githubusercontent.com/50594187/102038628-1999bc00-3e0b-11eb-8c94-62e610620b7c.png)

Figure 3. System architecture diagram

![demoUI](https://user-images.githubusercontent.com/50594187/102038653-2a4a3200-3e0b-11eb-8e1f-82747d0809d0.png)

Figure 4. Demo program UI

# Stacked-Convolutional-Auto-Encoders-for-Hierarchical-Feature-Extraction
본 공간은 [Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction](http://people.idsia.ch/~ciresan/data/icann2011.pdf) 논문을 pytorch로 구현한 코드를 공유하기 위해 설립된 공간입니다.

## Code 개요
- Mnist 데이터와 CIFAR10 데이터를 로드하여 미니 배치를 만드는 코드인 `loader.py`
- 모델 architecture를 구성한 코드 `model.py`
- 훈련을 진행하는 `train.py`
- 논문에 소개된 여러 실험을 한번에 돌릴 수 있는 `experiment.py`

## Paper Fig 1.
----------------------------------
<div>
  <img width="569" alt="fig1" src="https://user-images.githubusercontent.com/53327766/95677312-9d62dc80-0bff-11eb-943e-71ee8d30fa9d.PNG">
 </div>  
 
---------------------------------- 

- 논문이 공개한 Figure로 CAE(convolutional auto-encoder) 모델이 학습한 이미지 필터를 시각화한 그림입니다.  
- (a),(b),(c),(d) 그림은 `model.py`의 네 가지 모델 `CAE1,CAE2,CAE3,CAR4`를 MNIST 데이터에 순서대로 적용한 것으로 `train_autoencoder.py`를 통해 결과를 재연할 수 있습니다.
- 재연 결과는 **Paper Fig 1. reproduction**과 같습니다.


## Paper Fig 1. reproduction 
----------------------------------
<div>
  <img width="569" alt="fig2" src="https://user-images.githubusercontent.com/53327766/95677435-85d82380-0c00-11eb-916b-72a20c5d6bff.png">
 </div> 

----------------------------------
  

## Paper Fig 2.
----------------------------------
<div>
  <img width="259" alt="fig2" src="https://user-images.githubusercontent.com/53327766/95677665-ecaa0c80-0c01-11eb-92d6-92be880cf68b.PNG">
</div> 
 
---------------------------------- 
 
- 논문이 공개한 CAE(convolutional auto-encoder) 모델이 학습한 이미지 필터를 시각화한 그림입니다. 
- (a),(b),(c),(d) 그림은 `model.py`의 네 가지 모델 `CAE1,CAE2,CAE3,CAR4`를 CIFAR10 데이터에 순서대로 적용한 것으로 `train_autoencoder.py`를 통해 결과를 재연할 수 있습니다.

## Table 1 and 2.
- 논문이 제안한 CNN 모델을 CAE image filter로 fine-tuning 했을 때, 성능 차이를 보고한 자료입니다.
- 데이터 개수에 따라 fine-tuning 효과가 얼마만큼 차이나는지 파악할 수 있는 자료입니다.
- Table 1에서 사용한 데이터는 MNIST이고, Table 2에서 사용한 데이터는 CIFAR 10입니다.

### CAE 모델 학습
- 논문이 제안한 모델을 CAE image filter로 fine-tuning 하기 위해서는 먼저 CAE를 학습해야합니다.
- 이것은 `model.py`의 `CNN_for_cae` 모델과 `train_autoencoder_fine.py`를 통해 구현할 수 있습니다.

### CNN 모델 학습
- fine-tuning을 거치지 않은 CNN 모델 학습은 `model.py`의 `CNN` 모델과 `train_classifier.py` 코드로 구현할 수 있습니다.

### CNN 모델 fine-tuning 학습
- fine-tuning을 거친 CNN 모델 학습은 `model.py`의 `CNN` 모델과 `train_classifier_fine.py` 코드로 구현할 수 있습니다.

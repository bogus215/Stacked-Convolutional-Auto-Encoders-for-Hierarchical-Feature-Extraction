# Stacked-Convolutional-Auto-Encoders-for-Hierarchical-Feature-Extraction
본 공간은 [Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction](http://people.idsia.ch/~ciresan/data/icann2011.pdf) 논문을 pytorch로 구현한 코드를 공유하기 위해 설립된 공간입니다.

## Code 개요
- Mnist 데이터와 CIFAR10 데이터를 로드하여 미니 배치를 만드는 코드인 `loader.py`
- 모델 architecture를 구성한 코드 `model.py`
- 훈련을 진행하는 `train.py`
- 논문에 소개된 여러 실험을 한번에 돌릴 수 있는 experiment.py

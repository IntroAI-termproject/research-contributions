# 모델 개요
이 리포지토리에는 UNETR: Transformers for 3D Medical Image Segmentation[1]용 코드가 포함되어 있습니다. UNETR는 특징 추출을 위해 CNN에 의존하지 않고 순수한 비전 변환기를 인코더로 사용하는 최초의 3D 분할 네트워크입니다.
이 코드는 BTCV 챌린지 데이터 세트를 사용하는 체적(3D) 다기관 분할 응용 프로그램을 제공합니다.
![image](https://lh3.googleusercontent.com/pw/AM-JKLU2eTW17rYtCmiZP3WWC-U1HCPOHwLe6pxOfJXwv2W-00aHfsNy7jeGV1dwUq0PXFOtkqasQ2Vyhcu6xkKsPzy3wx7O6yGOTJ7ZzA01S6LSh8szbjNLfpbuGgMe6ClpiS61KGvqu71xXFnNcyvJNFjN=w1448-h496-no?authuser=0)

### 종속성 설치
종속성은 다음을 사용하여 설치할 수 있습니다.

``` bash
pip install -r requirements.txt
```

### 훈련

다기관 의미론적 분할(BTCV 데이터 세트) 작업을 위한 표준 하이퍼 매개변수가 있는 UNETR 네트워크는 다음과 같이 정의할 수 있습니다.

``` bash
model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed='perceptron',
    norm_name='instance',
    conv_block=True,
    res_block=True,
    dropout_rate=0.0)
```

위의 UNETR 모델은 CT 영상(1채널 입력)과 14등급 분할 출력에 사용됩니다. 네트워크가 기대하는
크기가 ```(96, 96, 96)```인 리샘플링된 입력 이미지는 ```(16, 16, 16)``` 크기의 겹치지 않는 패치로 변환됩니다.
위치 임베딩은 퍼셉트론 레이어를 사용하여 수행됩니다. ViT 인코더는 [2]에 소개된 표준 하이퍼 매개변수를 따릅니다.
디코더는 컨벌루션 및 잔차 블록과 인스턴스 정규화를 사용합니다. 자세한 내용은 [1]에서 확인할 수 있습니다.

하이퍼 매개변수의 기본값을 사용하면 다음 명령을 사용하여 PyTorch 기본 AMP 패키지를 사용하여 교육을 시작할 수 있습니다.

``` bash
python main.py
--feature_size=32 
--batch_size=1
--logdir=unetr_test
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5 
--save_checkpoint
--data_dir=/dataset/dataset0/
```

```--data_dir```을 사용하여 데이터세트 디렉토리의 위치를 제공해야 합니다.

분산 다중 GPU 교육을 시작하려면 교육 명령에 ```--distributed```를 추가해야 합니다.

AMP를 비활성화하려면 훈련 명령에 ```--noamp```를 추가해야 합니다.

UNETR가 분산 다중 GPU 교육에 사용되는 경우 학습률을 높이는 것이 좋습니다(예: ```--optim_lr```).
GPU 수에 따라 예를 들어, ```--optim_lr=4e-4```는 4개의 GPU로 훈련하는 데 권장됩니다.


### 미세 조정
우리는 BTCV 데이터 세트를 사용하여 UNETR의 최첨단 사전 훈련된 체크포인트 및 TorchScript 모델을 제공합니다.

사전 훈련된 체크포인트를 사용하려면 다음 디렉토리에서 가중치를 다운로드하십시오.

https://drive.google.com/file/d/1kR5QuRAuooYcTNLMnMj80Z9IgSs8jtLO/view?usp=sharing

다운로드가 완료되면 체크포인트를 다음 디렉토리에 배치하거나 ```--pretrained_dir```을 사용하여 모델이 배치된 주소를 제공하세요.

```./pretrained_models```

다음 명령은 사전 훈련된 체크포인트를 사용하여 미세 조정을 시작합니다.
``` bash
python main.py
--batch_size=1
--logdir=unetr_pretrained
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5 
--save_checkpoint
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--pretrained_model_name='UNETR_model_best_acc.pth'
--resume_ckpt
``` 

사전 훈련된 TorchScript 모델을 사용하려면 다음 디렉토리에서 모델을 다운로드하십시오.

https://drive.google.com/file/d/1_YbUE0abQFJUR4Luwict6BB8S77yUaWN/view?usp=sharing

다운로드가 완료되면 TorchScript 모델을 다음 디렉토리에 배치하거나 ```--pretrained_dir```을 사용하여 모델이 배치된 주소를 제공하세요.

```./pretrained_models```

다음 명령은 TorchScript 모델을 사용하여 미세 조정을 시작합니다.
``` bash
python main.py
--batch_size=1
--logdir=unetr_pretrained
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5 
--save_checkpoint
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--noamp
--pretrained_model_name='UNETR_model_best_acc.pt'
--resume_jit
``` 

제공된 TorchScript 모델의 미세 조정은 AMP를 지원하지 않습니다.


### 테스트
사전 훈련된 최신 TorchScript 모델 또는 UNETR의 체크포인트를 사용하여 자체 데이터에서 테스트할 수 있습니다.

사전 훈련된 가중치가 다운로드되면 위의 링크를 사용하여 TorchScript 모델을 다음 디렉토리에 배치하거나
```--pretrained_dir```을 사용하여 모델이 배치된 주소를 제공합니다.

```./pretrained_models```

다음 명령은 제공된 체크포인트를 사용하여 추론을 실행합니다.
``` bash
python test.py
--infer_overlap=0.5
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--saved_checkpoint=ckpt
``` 

```--infer_overlap```은 슬라이딩 창 패치 간의 겹침을 결정합니다. 값이 높을수록 일반적으로 더 정확한 세분화 출력이 생성되지만 추론 시간이 길어집니다.

사전 훈련된 TorchScript 모델을 사용하려면 ```--saved_checkpoint=torchscript```를 사용해야 합니다.

### 튜토리얼
BTCV 데이터 세트를 사용한 다기관 세분화 작업에 대한 자습서는 다음에서 찾을 수 있습니다.

https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d.ipynb

또한 PyTorch Lightning을 활용하는 자습서는 다음에서 찾을 수 있습니다.

https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb

## 데이터셋
![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)

학습데이터는 [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) 에 있습니다.

IRB(Institutional Review Board) 감독하에 진행 중인 대장암 화학 요법 시험과 후향적 복측 탈장 연구를 조합하여 50건의 복부 CT 스캔을 무작위로 선택했습니다. 50개의 스캔은 가변 부피 크기(512 x 512 x 85 - 512 x 512 x 198) 및 시야(약 280 x 280 x 280 mm3 - 500 x 500 x 650 mm3)로 문맥 정맥 조영 단계 동안 캡처되었습니다. 면내 분해능은 0.54 x 0.54 mm2에서 0.98 x 0.98 mm2까지 다양하며 슬라이스 두께는 2.5 mm에서 5.0 mm까지 다양합니다.

- 대상: 1. 비장 2. 오른쪽 신장 3. 왼쪽 신장 4. 담낭 5. 식도 6. 간 7. 위 8. 대동맥 9. IVC 10. 문맥 및 비장 정맥 11. 췌장 12.오른쪽 부신을 포함한 13개의 복부 장기 13.왼쪽 부신.
- 작업: 세분화
- 형식: CT
- 크기: 30개의 3D 볼륨(24개의 교육 + 6개의 테스트)
- 규모 : BTCV 미카이 챌린지

다음 링크에서 모델을 훈련하는 데 사용되는 json 파일을 제공합니다.

https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing

json 파일이 다운로드되면 데이터셋과 같은 폴더에 넣어주세요.

## 인용
이 리포지토리가 유용하다고 생각되면 UNETR 논문을 인용하는 것을 고려하십시오.

```
@article{hatamizadeh2021unetr,
  title={Unetr: Transformers for 3d medical image segmentation},
  author={Hatamizadeh, Ali and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2103.10504},
  year={2021}
}
```

## References
[1] Hatamizadeh, Ali, et al. "UNETR: Transformers for 3D Medical Image Segmentation", 2021. https://arxiv.org/abs/2103.10504.

[2] Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
", 2020. https://arxiv.org/abs/2010.11929.

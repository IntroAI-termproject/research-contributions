# LAMP: 이미지 분할을 위한 자동화된 모델 병렬 처리가 있는 대형 딥 넷

<p>
<img src="./fig/acc_speed_han_0_5hor.png" alt="LAMP on Head and Neck Dataset" width="500"/>
</p>


> 이 작업을 연구에 사용하는 경우 논문을 인용하십시오.

원래 제안된 LAMP 시스템의 재구현:

Wentao Zhu, Can Zhao, Wenqi Li, Holger Roth, Ziyue Xu, and Daguang Xu (2020)
"LAMP: Large Deep Nets with Automated Model Parallelism for Image Segmentation."
MICCAI 2020 (Early Accept, paper link: https://arxiv.org/abs/2006.12575)

## 데모를 실행하려면:

### 전제 조건
- `pip install monai==0.2.0`
- `pip install torchgpipe`

나머지 단계에서는 이 리포지토리가 로컬 파일 시스템에 복제되고 현재 디렉터리가 이 README 파일의 폴더라고 가정합니다.

### 데이터
```bash
mkdir ./data;
cd ./data;
```

Head and Neck CT 데이터 세트를 다운로드하여 `./data` 폴더에 압축을 풉니다.

- `HaN.zip`: https://drive.google.com/file/d/1A2zpVlR3CkvtkJPvtAF3-MH0nr1WZ2Mn/view?usp=sharing
```bash
unzip HaN.zip;  # 압축 해제는 다른 외부 도구로 수행할 수 있습니다.
```

데이터 세트에 대한 자세한 내용은 https://github.com/wentaozhu/AnatomyNet-for-anatomical-segmentation.git 에서 확인하세요.


### 전체 이미지 교육을 위한 최소 하드웨어 요구 사항
- U-Net (`n_feat=32`): 2x 16Gb GPUs
- U-Net (`n_feat=64`): 4x 16Gb GPUs
- U-Net (`n_feat=128`): 2x 32Gb GPUs

### 명령어
첫 번째 블록(`--n_feat`)의 기능 수는 32, 64 또는 128일 수 있습니다.
```bash
mkdir ./log;
python train.py --n_feat=128 --crop_size='64,64,64' --bs=16 --ep=4800  --lr=0.001 > ./log/YOURLOG.log
python train.py --n_feat=128 --crop_size='128,128,128' --bs=4 --ep=1200 --lr=0.001 --pretrain='./HaN_32_16_1200_64,64,64_0.001_*'  > ./log/YOURLOG.log
python train.py --n_feat=128 --crop_size='-1,-1,-1' --bs=1 --ep=300 --lr=0.001 --pretrain='./HaN_32_16_1200_64,64,64_0.001_*' > ./log/YOURLOG.log
```

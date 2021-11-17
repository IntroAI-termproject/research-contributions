# COVID-19 폐렴 병변 분할을 위한 COPLE-Net

<p>
<img src="./fig/img.png" width="30%" alt='lung-ct'>
<img src="./fig/seg.png" width="30%" alt='lung-ct-seg'>
</p>

> 이 작업을 연구에 사용하는 경우 논문을 인용하십시오.

원래 제안한 COPLE-Net의 재구현:

G. Wang, X. Liu, C. Li, Z. Xu, J. Ruan, H. Zhu, T. Meng, K. Li, N. Huang, S. Zhang. (2020)
"A Noise-robust Framework for Automatic Segmentation of COVID-19 Pneumonia Lesions from CT Images."
IEEE Transactions on Medical Imaging. 2020. DOI: [10.1109/TMI.2020.3000314](https://doi.org/10.1109/TMI.2020.3000314)


이 연구 프로토타입은 다음에서 수정되었습니다.
- [The `HiLab-git/COPLE-Net` GitHub repo](https://github.com/HiLab-git/COPLE-Net/)
- [PyMIC, a Pytorch-based toolkit for medical image computing.](https://github.com/HiLab-git/PyMIC)

추론 데모를 실행하려면:

- MONAI 0.2.0 설치:
```bash
pip install "monai[nibabel]==0.2.0"
```

나머지 단계에서는 이 리포지토리가 로컬 파일 시스템에 복제되고 현재 디렉터리가 이 README 파일의 폴더라고 가정합니다.
- [google 드라이브 폴더](https://drive.google.com/drive/folders/1pIoSSc4Iq8R9_xXo0NzaOhIHZ3-PqqDC)에서 입력 예제를 `./images`로 다운로드합니다.
- 적응된 사전 훈련된 모델을 [google 드라이브 폴더](https://drive.google.com/drive/folders/1HXlYJGvTF3gNGOL0UFBeHVoA6Vh_GqEw)에서 `./model`로 다운로드합니다.
- `python run_inference.py`를 실행하면 분할 결과가 `./output`에 저장됩니다.

_(자신의 이미지에서 COVID-19 폐렴 병변을 분할하려면 이미지가 폐 영역으로 잘렸는지 확인하십시오.
  강도는 1500/-650의 창 너비/레벨을 사용하여 [0, 1]로 정규화되었습니다.)_

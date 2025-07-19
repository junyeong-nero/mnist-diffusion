# Diffusion Model Toy Project (MNIST)

MNIST 데이터셋을 이용하여 diffusion model을 학습시키는 토이 프로젝트입니다. DDPM, DDIM, Classifier-Free Guidance 등 핵심적인 디퓨전 모델 기술들을 구현하고 실험했습니다.

이 프로젝트는 단순한 구현을 넘어, 설정 파일 기반의 관리, 다양한 모델 아키텍처(UNet, DiT) 지원, 코드 포매팅 및 단위 테스트 적용 등 확장성과 유지보수성을 고려하여 개선되었습니다.

## ✨ 주요 특징 (Features)

- **다양한 모델 아키텍처**: 전통적인 `UNet`과 최신 `Diffusion Transformer (DiT)` 모델을 모두 지원하며, 커맨드라인 인자로 쉽게 전환할 수 있습니다.
- **설정 파일 기반 관리**: `config.yaml`을 통해 모델 파라미터와 학습 설정을 관리하여, 코드 변경 없이 다양한 실험을 진행할 수 있습니다.
- **Apple Silicon (MPS) 지원**: Apple M-시리즈 칩의 GPU(MPS)를 자동으로 감지하여 가속을 지원합니다.
- **테스트 및 문서화**: `pytest`를 이용한 단위 테스트와 코드 내 `docstring`을 통해 코드의 안정성과 가독성을 높였습니��.
- **스크립트화된 실행**: `train.sh`를 통해 UNet 및 DiT 모델 학습을 간편하게 실행할 수 있습니다.

## ⚙️ 설치 (Setup)

1.  **저장소 복제 (Clone Repository)**
    ```bash
    git clone https://github.com/your-username/MyDiffusion.git
    cd MyDiffusion
    ```

2.  **의존성 설치 (Install Dependencies)**
    프로젝트에 필요한 라이브러리들을 `requirements.txt`를 통해 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 학습 (Training)

`train.py` 스크립트를 사용하여 모델을 학습시킵니다. `--model-type` 인자를 통해 `UNet` 또는 `DiT`를 선택할 수 있습니다. 모델의 세부 구조는 `config.yaml` 파일에 정의되어 있습니다.

**예시 명령어:**

- **UNet 모델 학습**
  ```bash
  python3 train.py --model-type UNet --epochs 30 --batch-size 16 --lr 0.0001
  ```

- **DiT 모델 학습**
  ```bash
  python3 train.py --model-type DiT --epochs 50 --batch-size 4 --lr 0.0002
  ```

### 쉘 스크립트 사용

`train.sh` 파일에 다양한 학습 예시가 포함되어 있습니다. 8GB RAM 환경에 최적화된 배치 사이즈가 설정되어 있습니다.

```bash
# 실행 권한 부여
chmod +x train.sh

# 스크립트 실행 (UNet과 DiT 학습이 순차적으로 진행됨)
./train.sh
```

학습된 모델(`*.pt`)과 손실 기록(`history.pt`)은 프로젝트 루트 디렉토리에 저장됩니다.

## 🎨 샘플링 (Sampling)

학습된 모델을 사용하여 새로운 이미지를 생성하려면 `sampling.py`를 사용합니다. `--model-type`과 `--model-path` 인자를 통해 사용할 모델의 종류와 가중치 파일 경로를 지정해야 합니다.

**예시 명령어:**

```bash
# 학습된 UNet 모델로 샘플링
python3 sampling.py --model-type UNet --model-path "UNet_T1000_E30.pt"
```

## 📊 결과 (Result)

- DDIM with `len(τ) = 10`
- `w` = 1 (CFG sampling ratio)

![result1](./images/result.png)
![result2](./images/result2.png)
![result3](./images.png)

### 학습 손실 (Training Loss)
![loss](./images/loss.png)

## 📚 참고 자료 (Reference Papers)

- [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

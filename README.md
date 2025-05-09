# HMKD-ICMR:Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation

Welcome to the official repository for the paper "Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation", ICMR, 2025.

## Requirement
Ubuntu 20.04.4 LTS
Python 3.8.10
CUDA 11.3
Pytorch 1.11.0
NCCL for CUDA 2.10.3

Install python packages:
pip install timm==
pip install mmcv-full==
pip install opencv-python==

Backbones pretrained on ImageNet:
| CNN | Segformer |
| -- | -- |
|[resnet101-imagenet.pth](https://drive.google.com/file/d/1V8-E4wm2VMsfnNiczSIDoSM7JJBMARkP/view?usp=sharing)| [mit_b0.pth](https://pan.baidu.com/s/1Figp042rc9VNtPc_fkNW3g?pwd=swor )|
|[resnet18-imagenet.pth](https://drive.google.com/file/d/1_i0n3ZePtQuh66uQIftiSwN7QAUlFb8_/view?usp=sharing) | [mit_b1.pth](https://pan.baidu.com/s/1OUblLHQbq18DvXGzRU58jA?pwd=03yb)|
|[mobilenetv2-imagenet.pth](https://drive.google.com/file/d/12EDZjDSCuIpxPv-dkk1vrxA7ka0b0Yjv/view?usp=sharing) | [mit_b4.pth](https://pan.baidu.com/s/1j8pXjZZ-YSi2JXpsaQSSTQ?pwd=cvpd )|

Support datasets:
| Dataset | Train Size | Val Size | Test Size | Class |
| -- | -- | -- |-- |-- |
| Cityscapes | 2975 | 500 | 1525 |19|
| CamVid | 367 | 101 | 233 | 11 |






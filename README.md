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

## The initiallization Weights for Training
(https://download.pytorch.org/models/resnet18-5c106cde.pth), [DeepLabV3 - ResNet-18 - resnet18.pth]
(https://download.pytorch.org/models/resnet101-5d3b4d8f.pth), [DeepLabV3 - ResNet-101 - resnet101.pth]
(https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing), [Segformer - mit-b0 - segformerb0.pth]
(https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw), [Segformer - mit-b4 - segformerb4.pth]

## Trained Weights of HMKD for Testing
[Download](https://pan.baidu.com/s/1xw_6ts5VNV73vXeOLAokwQ?pwd=jvx8)

## Support datasets:
| Dataset | Train Size | Val Size | Test Size | Class |
| -- | -- | -- |-- |-- |
| Cityscapes | 2975 | 500 | 1525 |19|
| CamVid | 367 | 101 | 233 | 11 |

## Train
Please download the pre-trained model weights and dataset first. Next, generate the path lists of the training set and the test set, and change the dataset path in the code to the path of the dataset listing file (.txt) you specified.

~~~python
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 train_NEW_AEU_kd.py > H_V_MSE_fixed.file 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_NEW_AEU_kd.py
~~~

## Citation
If it helps your research,  please use the information below to cite our work, thank you. 

~~~
@ARTICLE{HMKD,
  author={Xu, Mingzhu and Wang, Jing and Wang, Mingcai and Li, Yiping and Hu, Yupeng and Song, Xuemeng and Guan, Weili},
  journal={ICMR}, 
  title={Heterogeneous Model Knowledge Distillation via Dual Alignment for Semantic Segmentation}, 
  year={2025}}
~~~












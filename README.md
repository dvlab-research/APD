# APD
**Adaptive Perspective Distillation for Semantic Segmentation**

Zhuotao Tian*; Pengguang Chen*; Xin Lai; Li Jiang; Shu Liu; Hengshuang Zhao; Bei Yu; Ming-Chang Yang; Jiaya Jia

This project provides an implementation for the TPAMI 2022 paper "[Adaptive Perspective Distillation for Semantic Segmentation](https://ieeexplore.ieee.org/document/9736597)"

## Environment

We verify our code on 
* 4x3090 GPUs
* CUDA 11.1
* python 3.9
* torch 1.12.1
* torchvision 0.13.1

Other similar environments should also work properly.

## Installation

```
git clone https://github.com/dvlab-research/APD.git
cd APD/
```

## Results

| Dataset        | Student    | Teacher     | Baseline | Ours  |
|----------------|------------|-------------|----------|-------|
| ade20k         | [PSPNet-R18](https://github.com/akuxcw/APD/releases/download/v1.0/ade20k_psp18_psp101.pth) | [PSPNet-R101](https://github.com/akuxcw/APD/releases/download/v1.0/pspnet101_fc_ade20k.pth) | 37.19    | 39.25 |
| cityscapes     | [PSPNet-R18](https://github.com/akuxcw/APD/releases/download/v1.0/cityscapes_psp18_psp101.pth) | [PSPNet-R101](https://github.com/akuxcw/APD/releases/download/v1.0/pspnet101_fc_cityscapes.pth) | 74.15    | 75.68 |
| pascal context | [PSPNet-R18](https://github.com/akuxcw/APD/releases/download/v1.0/context_psp18_psp101.pth) | [PSPNet-R101](https://github.com/akuxcw/APD/releases/download/v1.0/pspnet101_fc_context.pth) | 42.29    | 43.96 |



## Training

Use the following command to train PSPNet-R18 on ade20k with APD
```
bash ./tool/train.sh ade20k release_psp18_psp101
```

## <a name="Citation"></a>Citation

Please consider citing ReviewKD in your publications if it helps your research.

```bib
@article{APD,
  author = {Zhuotao Tian and
            Pengguang Chen and
            Xin Lai and
            Li Jiang and
            Shu Liu and
            Hengshuang Zhao and
            Bei Yu and
            Ming{-}Chang Yang and
            Jiaya Jia},
  title = {Adaptive Perspective Distillation for Semantic Segmentation},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  pages = {1372--1387},
  year = {2023}
}}
```

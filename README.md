# BiRSwinT

This repo is the official implementation of ["BiRSwinT: Bilinear Full-Scale Residual Swin-Transformer for Fine-Grained Driver Behavior Recognition"]. It currently includes code and models for the following tasks:

> **Image Classification**: Included in this repo. See [get_started.md](get_started.md) for a quick start.


## Updates
***12/26/2022***

Initial commits:

1. Pretrained models for Swin-Transformer-S on ImageNet-1K ([Swin-T-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth), [Swin-S-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth), [Swin-B-IN1K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth)) and ImageNet-22K ([Swin-B-IN22K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth), [Swin-L-IN22K](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)) are provided.
2. The supported code for AUC and StateFarm image classification are provided.

## Introduction

**BiRSwinT** 	The bilinear fusion method can solve the fine-grained recognition problem to a certain extent.
	After getting the deep descriptors of an image, bilinear pooling computes the sum of the outer 
	product of those deep descriptors. Bilinear pooling captures all pairwise descriptor interactions,
	 i.e., interactions of different part.

	This project aims at solving the problem of poor fine-grained characterization of a single 
	Swin-Transformer model in driver distraction tasks.We interpret the two branches of the bilinear 
	model as the global feature branch and the local feature branch, respectively, with the global branch 
	applying the Swin-Transformer-S model and the local branch applying the Dense-Swin-Transformer model 
	with residuals between Stages.

![teaser](figures/teaser.png)


## Citing BiRSwinT

```
@inproceedings{Yang2022BiRSwinT,
  title={BiRSwinT: Bilinear Full-Scale Residual Swin-Transformer for Fine-Grained Driver Behavior Recognition},
  author={Yang, Wenxuan and Tan, Chenghao and Chen, Yuxin and Xia, Huang and Tang, Xuexi and Cao, Yifan and Zhou, Wenhui and Lin, Lili},
  booktitle={Journal of the Franklin Institute},
  year={2022}
}
```

## Getting Started

- For **Image Classification**, please see [get_started.md](get_started.md) for detailed instructions.